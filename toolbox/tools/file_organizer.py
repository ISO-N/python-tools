"""文件整理工具页（PyQt6 界面）。

按扩展名、修改日期（年-月）或文件大类，把文件移动到目标根目录下的分类
子文件夹。支持预览、递归扫描、隐藏文件、同名文件跳过/自动改名、后台执行、
取消、撤销以及空目录清理。

本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from toolbox.common import (
    natural_path_key,
    natural_sort_key,
    open_directory,
    unique_path,
)
from toolbox.fileops import prune_empty_dirs, remove_empty_dirs
from toolbox.page import ToolPage
from toolbox.ui import add_path_row, make_form
from toolbox.workers import Cancelled, SingleTaskWorker, shutdown_worker

MODE_EXTENSION = "extension"
MODE_MTIME = "mtime"
MODE_CATEGORY = "category"
VALID_MODES = {MODE_EXTENSION, MODE_MTIME, MODE_CATEGORY}

CONFLICT_SKIP = "skip"
CONFLICT_RENAME = "rename"
VALID_CONFLICT_POLICIES = {CONFLICT_SKIP, CONFLICT_RENAME}

CATEGORY_MAP: dict[str, set[str]] = {
    "图片": {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".heic",
        ".svg",
    },
    "文档": {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".txt",
        ".md",
        ".epub",
        ".rtf",
        ".csv",
    },
    "视频": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"},
    "音频": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
    "压缩包": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
}
OTHER_CATEGORY = "其他"
NO_EXTENSION_BUCKET = "无扩展名"

_CATEGORY_BY_EXTENSION = {
    extension: category
    for category, extensions in CATEGORY_MAP.items()
    for extension in extensions
}

STATUS_OK = "ok"
STATUS_CONFLICT = "conflict"
STATUS_SKIPPED = "skipped"

STATUS_LABELS = {
    STATUS_OK: "可移动",
    STATUS_CONFLICT: "目标已存在",
    STATUS_SKIPPED: "跳过",
}

MODE_LABELS = (
    ("按扩展名", MODE_EXTENSION),
    ("按修改日期", MODE_MTIME),
    ("按文件大类", MODE_CATEGORY),
)


# --------------------------------------------------------------------- 纯逻辑


@dataclass
class OrganizeOptions:
    """一次整理操作的选项。"""

    mode: str = MODE_EXTENSION
    recursive: bool = False
    include_hidden: bool = False
    cleanup_empty: bool = True
    conflict_policy: str = CONFLICT_SKIP


@dataclass
class OrganizeItem:
    """一个待移动文件。"""

    src: Path
    dst: Path
    bucket: str
    status: str = STATUS_OK


@dataclass
class OrganizePlan:
    """一次整理操作的完整计划。"""

    root: Path
    target_root: Path
    items: list[OrganizeItem] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def move_items(self) -> list[OrganizeItem]:
        """返回真正会移动的条目（状态为 ``ok``）。"""
        return [item for item in self.items if item.status == STATUS_OK]

    @property
    def conflict_count(self) -> int:
        """返回目标已存在、执行时将跳过的条目数。"""
        return sum(
            1 for item in self.items if item.status == STATUS_CONFLICT
        )

    @property
    def skipped_count(self) -> int:
        """返回预览中定义为跳过的文件总数。"""
        status_skipped = sum(
            1 for item in self.items if item.status == STATUS_SKIPPED
        )
        return self.conflict_count + status_skipped + len(self.skipped)

    @property
    def buckets(self) -> list[str]:
        """返回计划中出现的分类桶（自然排序）。"""
        return sorted(
            {item.bucket for item in self.items}, key=natural_sort_key
        )


@dataclass
class OrganizeResult:
    """执行或撤销整理后的结果。"""

    moved: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _resolved(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:  # pragma: no cover - 极端路径解析失败
        return Path(path).absolute()


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(_resolved(first))) == os.path.normcase(
        str(_resolved(second))
    )


def _destination_taken(path: Path) -> bool:
    """目标位置是否已被占用（包括失效的符号链接）。"""
    return path.exists() or path.is_symlink()


def classify_file(path: Path | str, mode: str) -> str:
    """按 ``mode`` 返回文件所属分类桶。

    - ``extension``：小写扩展名（不含点）；无扩展名返回「无扩展名」。
    - ``mtime``：本地修改时间 ``%Y-%m``。
    - ``category``：按 :data:`CATEGORY_MAP` 归类；无扩展名或未知扩展名
      归入「其他」。
    """
    if mode not in VALID_MODES:
        raise ValueError(f"不支持的分类方式：{mode}")

    path = Path(path)
    suffix = path.suffix.lower()
    if mode == MODE_EXTENSION:
        return suffix[1:] if suffix else NO_EXTENSION_BUCKET
    if mode == MODE_MTIME:
        stamp = datetime.fromtimestamp(path.stat().st_mtime)
        return stamp.strftime("%Y-%m")
    if not suffix:
        return OTHER_CATEGORY
    return _CATEGORY_BY_EXTENSION.get(suffix, OTHER_CATEGORY)


def _validate_plan_inputs(
    root: Path, target_root: Path, options: OrganizeOptions
) -> None:
    if options.mode not in VALID_MODES:
        raise ValueError(f"不支持的分类方式：{options.mode}")
    if options.conflict_policy not in VALID_CONFLICT_POLICIES:
        raise ValueError(f"不支持的同名文件处理方式：{options.conflict_policy}")
    if not root.is_dir():
        raise ValueError(f"源目录不存在或不是文件夹：{root}")
    if target_root.exists() and not target_root.is_dir():
        raise ValueError(f"目标根路径不是文件夹：{target_root}")


def build_organize_plan(
    root: Path | str,
    target_root: Path | str,
    options: OrganizeOptions,
) -> OrganizePlan:
    """扫描源目录并生成整理计划（不会移动任何文件）。

    - 只处理普通文件；符号链接、目录和其他特殊文件会被忽略。
    - ``recursive`` 为假时只扫描第一层；为真时递归扫描且不跟随符号链接。
    - ``include_hidden`` 为假时跳过以 ``.`` 开头的文件和目录。
    - 目标路径为 ``target_root / 分类 / [相对父目录/] 文件名``；递归时保留
      相对父目录，非递归时不保留。
    - 如果 ``target_root`` 位于源目录内部，扫描时排除该子树。
    - 递归且 ``target_root`` 就是源目录时，跳过已经位于同名分类目录内的
      文件，避免二次整理。
    - 目标已存在时按 ``conflict_policy`` 标记为冲突或自动改名。
    """
    root = Path(root)
    target_root = Path(target_root)
    _validate_plan_inputs(root, target_root, options)

    root_resolved = _resolved(root)
    target_resolved = _resolved(target_root)
    target_is_root = root_resolved == target_resolved
    target_inside = (
        not target_is_root
        and target_resolved.is_relative_to(root_resolved)
    )

    items: list[OrganizeItem] = []
    skipped: list[tuple[Path, str]] = []

    def record_skip(path: Path, reason: str) -> None:
        skipped.append((path, reason))

    def process_file(path: Path) -> None:
        try:
            if path.is_symlink():
                record_skip(path, "符号链接")
                return
            if not path.is_file():
                if not path.is_dir():
                    record_skip(path, "不是普通文件")
                return
        except OSError as exc:
            record_skip(path, f"无法读取文件信息：{exc}")
            return

        if not options.include_hidden and path.name.startswith("."):
            record_skip(path, "隐藏文件")
            return

        try:
            bucket = classify_file(path, options.mode)
        except OSError as exc:
            record_skip(path, f"无法读取修改时间：{exc}")
            return

        relative_parent = (
            path.parent.relative_to(root) if options.recursive else Path()
        )
        if (
            options.recursive
            and target_is_root
            and bucket in relative_parent.parts
        ):
            record_skip(path, "已在分类目录中")
            return

        destination = target_root / bucket
        if relative_parent.parts:
            destination = destination / relative_parent
        destination = destination / path.name

        if _same_path(destination, path):
            record_skip(path, "目标与源相同")
            return

        status = STATUS_OK
        if _destination_taken(destination):
            if options.conflict_policy == CONFLICT_SKIP:
                status = STATUS_CONFLICT
            else:
                destination = unique_path(destination)

        items.append(
            OrganizeItem(
                src=path, dst=destination, bucket=bucket, status=status
            )
        )

    def on_walk_error(error: OSError) -> None:
        filename = getattr(error, "filename", None)
        target = Path(filename) if filename else root
        record_skip(target, f"无法读取目录：{error}")

    if options.recursive:
        for dirpath, dirnames, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=on_walk_error,
        ):
            current = Path(dirpath)
            kept_dirs: list[str] = []
            for name in dirnames:
                if not options.include_hidden and name.startswith("."):
                    continue
                if target_inside and _same_path(
                    current / name, target_root
                ):
                    continue
                kept_dirs.append(name)
            dirnames[:] = sorted(kept_dirs, key=natural_sort_key)

            for name in sorted(filenames, key=natural_sort_key):
                process_file(current / name)
    else:
        for entry in sorted(root.iterdir(), key=natural_path_key):
            process_file(entry)

    items.sort(
        key=lambda item: (item.bucket, natural_path_key(item.src))
    )
    skipped.sort(key=lambda entry: natural_path_key(entry[0]))

    return OrganizePlan(
        root=root,
        target_root=target_root,
        items=items,
        skipped=skipped,
    )


def execute_organize(
    plan: OrganizePlan,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> OrganizeResult:
    """按计划移动状态为 ``ok`` 的文件。

    逐项执行，单项失败不影响后续文件，也不会覆盖目标位置的文件（执行时
    才出现的同名文件计为跳过）。取消时抛出
    :class:`toolbox.workers.Cancelled`，其 ``result`` 携带已完成的部分结果。
    """
    moved: list[tuple[Path, Path]] = []
    skipped = sum(1 for item in plan.items if item.status != STATUS_OK)
    errors: list[str] = []
    pending = [item for item in plan.items if item.status == STATUS_OK]
    total = len(pending)
    processed = 0

    for item in pending:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(
                OrganizeResult(
                    moved=list(moved), skipped=skipped, errors=list(errors)
                )
            )

        try:
            if _destination_taken(item.dst) or _same_path(
                item.src, item.dst
            ):
                skipped += 1
            else:
                item.dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item.src), str(item.dst))
                moved.append((item.src, item.dst))
        except Exception as exc:  # noqa: BLE001 - 单项失败继续
            errors.append(f"{item.src} → {item.dst}：{exc}")

        processed += 1
        if on_progress is not None:
            on_progress(processed, total)

    return OrganizeResult(moved=moved, skipped=skipped, errors=errors)


def _cleanup_after_undo(
    moved: Sequence[tuple[Path, Path]], target_root: Path | None
) -> None:
    """撤销成功后尽力清理变空的目录。"""
    try:
        if target_root is not None:
            directory = Path(target_root)
            if directory.is_dir():
                prune_empty_dirs(directory)
            return

        parents = {current.parent for current, _original in moved}
        for directory in parents:
            prune_empty_dirs(directory)
            remove_empty_dirs([directory])
    except Exception:  # noqa: BLE001 - 清理失败不影响撤销结果
        return


def undo_organize(
    records: Sequence[tuple[Path, Path]],
    target_root: Path | None = None,
    cleanup_empty: bool = True,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> OrganizeResult:
    """把 ``records`` 中的（原路径, 移动后路径）逐项移回原位置。

    不覆盖原位置已有的文件；目标文件不存在或原名已被占用时计为跳过。
    ``target_root`` 用于在撤销成功后递归清理目标分类下的空目录；未提供时
    只清理每个文件原目标位置所在的目录。移动结果中保存实际执行的方向
    （移动后路径, 原路径）。每项开始前检查取消状态，取消时抛出
    :class:`toolbox.workers.Cancelled`，其 ``result`` 携带已完成的部分结果。
    """
    moved: list[tuple[Path, Path]] = []
    skipped = 0
    errors: list[str] = []
    total = len(records)
    processed = 0

    for original, current in records:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(
                OrganizeResult(
                    moved=list(moved), skipped=skipped, errors=list(errors)
                )
            )

        original = Path(original)
        current = Path(current)
        try:
            if not _destination_taken(current):
                skipped += 1
            elif _destination_taken(original) or _same_path(original, current):
                skipped += 1
            else:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), str(original))
                moved.append((current, original))
        except Exception as exc:  # noqa: BLE001 - 单项失败继续
            errors.append(f"{current} → {original}：{exc}")

        processed += 1
        if on_progress is not None:
            on_progress(processed, total)

    if cleanup_empty and moved:
        _cleanup_after_undo(moved, target_root)

    return OrganizeResult(moved=moved, skipped=skipped, errors=errors)


def _cleanup_source_empty_dirs(plan: OrganizePlan) -> int:
    """执行整理后清理源目录中的空文件夹，返回删除数量。"""
    try:
        exclude: list[Path] = []
        if not _same_path(plan.root, plan.target_root):
            exclude.append(plan.target_root)
        return len(prune_empty_dirs(plan.root, exclude=exclude))
    except Exception:  # noqa: BLE001 - 清理失败不影响整理结果
        return 0


def _format_error_lines(errors: Sequence[str], limit: int = 5) -> list[str]:
    """取前 ``limit`` 条错误，并注明省略数量。"""
    lines = list(errors[:limit])
    if len(errors) > limit:
        lines.append(f"…（其余 {len(errors) - limit} 条失败已省略）")
    return lines


# ------------------------------------------------------------------- 界面部分


class FileOrganizerPage(ToolPage):
    """按扩展名 / 修改日期 / 文件大类整理文件的工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._executing_plan: OrganizePlan | None = None
        self._undo_records: list[tuple[Path, Path]] = []
        self._undo_target_root: Path | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        form = make_form()

        self.source_edit: QLineEdit = add_path_row(
            form,
            0,
            "源目录：",
            mode="open_dir",
            placeholder="选择要整理文件的目录",
            dialog_title="选择源目录",
            parent=self,
        )
        self.source_edit.textChanged.connect(self._update_open_button)

        self.target_edit: QLineEdit = add_path_row(
            form,
            1,
            "目标根目录：",
            mode="open_dir",
            placeholder="留空表示整理到源目录",
            dialog_title="选择目标根目录",
            parent=self,
        )

        self.mode_combo = QComboBox()
        for label, value in MODE_LABELS:
            self.mode_combo.addItem(label, value)
        form.addWidget(QLabel("分类方式："), 2, 0)
        form.addWidget(self.mode_combo, 2, 1, 1, 2)

        self.conflict_combo = QComboBox()
        self.conflict_combo.addItem("跳过", CONFLICT_SKIP)
        self.conflict_combo.addItem("自动改名", CONFLICT_RENAME)
        form.addWidget(QLabel("同名文件："), 3, 0)
        form.addWidget(self.conflict_combo, 3, 1, 1, 2)

        options = QHBoxLayout()
        self.recursive_check = QCheckBox("递归子文件夹")
        self.hidden_check = QCheckBox("包含隐藏文件")
        self.cleanup_check = QCheckBox("清理空目录")
        self.cleanup_check.setChecked(True)
        options.addWidget(self.recursive_check)
        options.addWidget(self.hidden_check)
        options.addWidget(self.cleanup_check)
        options.addStretch(1)
        form.addWidget(QLabel("选项："), 4, 0)
        form.addLayout(options, 4, 1, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 5, 1, 1, 2)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("预览")
        self.preview_btn.clicked.connect(self._preview)
        self.execute_btn = QPushButton("执行")
        self.execute_btn.setDefault(True)
        self.execute_btn.clicked.connect(self._start_execute)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("撤销上一次整理（暂无可撤销的操作）")
        self.undo_btn.clicked.connect(self._undo)
        self.open_btn = QPushButton("打开源目录")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_source)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(self.execute_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.undo_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 6, 0, 1, 3)

        layout.addLayout(form)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["源文件", "分类", "目标位置", "状态"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.table, 1)

        self.status_label = QLabel("请选择源目录并点击「预览」")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # -------------------------------------------------------------- 状态维护

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        widgets = (
            self.source_edit,
            self.target_edit,
            self.mode_combo,
            self.conflict_combo,
            self.recursive_check,
            self.hidden_check,
            self.cleanup_check,
        )
        for widget in widgets:
            widget.setEnabled(not busy)
        for button in self.findChildren(QPushButton):
            button.setEnabled(not busy)

        if busy:
            self.cancel_btn.setEnabled(True)
            self.undo_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
        else:
            self.cancel_btn.setEnabled(False)
            self._refresh_undo_button()
            self._update_open_button()

    def _refresh_undo_button(self) -> None:
        count = len(self._undo_records)
        self.undo_btn.setEnabled(count > 0 and self._worker is None)
        if count:
            self.undo_btn.setToolTip(f"撤销上一次整理（{count} 个文件）")
        else:
            self.undo_btn.setToolTip("撤销上一次整理（暂无可撤销的操作）")

    def _update_open_button(self, *_args: object) -> None:
        text = self.source_edit.text().strip()
        self.open_btn.setEnabled(
            bool(text)
            and Path(text).expanduser().is_dir()
            and (self._worker is None or not self._worker.isRunning())
        )

    def _set_undo_records(
        self, records: Sequence[tuple[Path, Path]], target_root: Path | None
    ) -> None:
        self._undo_records = list(records)
        self._undo_target_root = target_root
        self._refresh_undo_button()

    def _clear_table(self) -> None:
        self.table.setRowCount(0)

    # -------------------------------------------------------------- 计划与预览

    def _collect_options(
        self,
    ) -> tuple[Path, Path, OrganizeOptions] | None:
        """校验界面输入；失败时弹窗并返回 None。"""
        source_text = self.source_edit.text().strip()
        if not source_text:
            QMessageBox.warning(
                self, "缺少源目录", "请先选择要整理的源目录。"
            )
            return None
        root = Path(source_text).expanduser()
        if not root.is_dir():
            QMessageBox.warning(
                self, "源目录不存在", f"目录不存在：\n{root}"
            )
            return None

        target_text = self.target_edit.text().strip()
        target_root = Path(target_text).expanduser() if target_text else root
        if not target_root.is_dir():
            QMessageBox.warning(
                self, "目标目录不存在", f"目录不存在：\n{target_root}"
            )
            return None

        mode = str(self.mode_combo.currentData() or MODE_EXTENSION)
        conflict_policy = str(
            self.conflict_combo.currentData() or CONFLICT_SKIP
        )
        options = OrganizeOptions(
            mode=mode,
            recursive=self.recursive_check.isChecked(),
            include_hidden=self.hidden_check.isChecked(),
            cleanup_empty=self.cleanup_check.isChecked(),
            conflict_policy=conflict_policy,
        )
        return root, target_root, options

    def _make_plan(self) -> OrganizePlan | None:
        collected = self._collect_options()
        if collected is None:
            return None
        root, target_root, options = collected
        try:
            return build_organize_plan(root, target_root, options)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "无法生成整理计划", f"{root}\n{exc}"
            )
            return None

    def _display_path(self, path: Path, base: Path) -> str:
        try:
            relative = path.relative_to(base)
        except ValueError:
            return str(path)
        text = relative.as_posix()
        return text if text not in {"", "."} else str(path)

    def _fill_table(self, plan: OrganizePlan) -> None:
        rows = len(plan.items) + len(plan.skipped)
        self.table.setRowCount(rows)

        for row, item in enumerate(plan.items):
            source_item = QTableWidgetItem(
                self._display_path(item.src, plan.root)
            )
            source_item.setToolTip(str(item.src))
            bucket_item = QTableWidgetItem(item.bucket)
            destination_item = QTableWidgetItem(
                self._display_path(item.dst, plan.target_root)
            )
            destination_item.setToolTip(str(item.dst))
            status_item = QTableWidgetItem(
                STATUS_LABELS.get(item.status, item.status)
            )
            if item.status == STATUS_CONFLICT:
                status_item.setToolTip(
                    "目标位置已存在同名文件，执行时将跳过"
                )
            self.table.setItem(row, 0, source_item)
            self.table.setItem(row, 1, bucket_item)
            self.table.setItem(row, 2, destination_item)
            self.table.setItem(row, 3, status_item)

        offset = len(plan.items)
        for index, (path, reason) in enumerate(plan.skipped):
            source_item = QTableWidgetItem(
                self._display_path(path, plan.root)
            )
            source_item.setToolTip(f"{path}\n{reason}")
            status_item = QTableWidgetItem(STATUS_LABELS[STATUS_SKIPPED])
            status_item.setToolTip(reason)
            self.table.setItem(offset + index, 0, source_item)
            self.table.setItem(offset + index, 1, QTableWidgetItem("—"))
            self.table.setItem(offset + index, 2, QTableWidgetItem("—"))
            self.table.setItem(offset + index, 3, status_item)

    def _summary_text(self, plan: OrganizePlan) -> str:
        move_count = len(plan.move_items)
        return (
            f"将移动 {move_count} 个文件，跳过 {plan.skipped_count} 个，"
            f"共 {len(plan.buckets)} 个分类。"
        )

    def _preview(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        plan = self._make_plan()
        if plan is None:
            return
        self._fill_table(plan)
        self.progress_bar.setValue(0)
        self._set_status(f"预览完成：{self._summary_text(plan)}")

    # ------------------------------------------------------------------ 执行

    def _start_execute(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        plan = self._make_plan()
        if plan is None:
            return

        self._fill_table(plan)
        move_count = len(plan.move_items)
        self._set_status(f"预览：{self._summary_text(plan)}")
        if move_count == 0:
            QMessageBox.information(
                self,
                "没有可移动的文件",
                "当前设置下没有可移动的文件，未做任何操作。",
            )
            self._set_status("没有可移动的文件")
            return

        collected = self._collect_options()
        if collected is None:
            return
        _root, _target_root, options = collected

        message = (
            f"将把 {move_count} 个文件移动到：\n{plan.target_root}\n\n"
        )
        if plan.skipped_count:
            message += f"另有 {plan.skipped_count} 个文件将被跳过。\n"
        if options.cleanup_empty:
            message += "执行后将清理源目录中的空文件夹。\n"
        message += "\n是否继续？"
        reply = QMessageBox.question(
            self,
            "确认整理",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消整理")
            return

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> object:
            def report(done: int, total: int) -> None:
                on_progress(done, total)

            try:
                result = execute_organize(
                    plan,
                    cancel_event=cancel_event,
                    on_progress=report,
                )
            except Cancelled as exc:
                partial = exc.result
                if (
                    options.cleanup_empty
                    and isinstance(partial, OrganizeResult)
                    and partial.moved
                ):
                    _cleanup_source_empty_dirs(plan)
                raise
            if options.cleanup_empty and result.moved:
                _cleanup_source_empty_dirs(plan)
            return result

        self._executing_plan = plan
        self._begin_worker(task, move_count)

    def _begin_worker(
        self,
        task: Callable[
            [Callable[[object, object], None], threading.Event], object
        ],
        total: int,
    ) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status(f"正在整理 {total} 个文件…")
        self._set_busy(True)

        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_progress)
        worker.task_finished.connect(self._on_task_finished)
        worker.task_cancelled.connect(self._on_task_cancelled)
        worker.task_failed.connect(self._on_task_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_progress(self, done: object, total: object) -> None:
        done_count, total_count = int(done), int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(min(100, percent))
        self._set_status(f"正在整理：{done_count}/{total_count} 个文件…")

    def _on_task_finished(self, result: object) -> None:
        if not isinstance(result, OrganizeResult):
            self._on_task_failed("后台任务返回了未知结果")
            return

        self.progress_bar.setValue(100)
        plan = self._executing_plan
        if result.moved:
            self._set_undo_records(
                result.moved, plan.target_root if plan is not None else None
            )
        self._clear_table()
        self._set_status(
            f"完成：移动 {len(result.moved)} 个，跳过 {result.skipped} 个，"
            f"失败 {len(result.errors)} 个。"
        )

        if result.errors:
            lines = [
                f"已移动 {len(result.moved)} 个文件，"
                f"跳过 {result.skipped} 个，"
                f"失败 {len(result.errors)} 个。",
                "",
                "失败项（前 5 条）：",
                *_format_error_lines(result.errors),
            ]
            QMessageBox.warning(
                self, "整理完成（部分失败）", "\n".join(lines)
            )
        else:
            QMessageBox.information(
                self,
                "整理完成",
                f"已移动 {len(result.moved)} 个文件，"
                f"跳过 {result.skipped} 个。\n\n目标目录：\n"
                f"{plan.target_root if plan is not None else ''}",
            )

    def _on_task_cancelled(self, result: object) -> None:
        partial = result if isinstance(result, OrganizeResult) else None
        moved = len(partial.moved) if partial is not None else 0
        skipped = partial.skipped if partial is not None else 0
        errors = partial.errors if partial is not None else []

        if partial is not None and partial.moved:
            plan = self._executing_plan
            self._set_undo_records(
                partial.moved, plan.target_root if plan is not None else None
            )
        self._clear_table()
        self._set_status(
            f"已取消：移动 {moved} 个，跳过 {skipped} 个，"
            f"失败 {len(errors)} 个。"
        )

        lines = [f"整理已取消，已移动 {moved} 个文件，跳过 {skipped} 个。"]
        if errors:
            lines.extend(["", "失败项（前 5 条）：", *_format_error_lines(errors)])
            QMessageBox.warning(self, "已取消（部分失败）", "\n".join(lines))
        else:
            QMessageBox.information(self, "已取消", "\n".join(lines))

    def _on_task_failed(self, message: str) -> None:
        self._set_status("整理失败")
        QMessageBox.critical(self, "整理失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self._executing_plan = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_task(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._set_status("正在取消…")

    # ------------------------------------------------------------------ 撤销

    def _undo(self) -> None:
        if not self._undo_records:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        reply = QMessageBox.question(
            self,
            "确认撤销",
            f"将撤销上一次整理，把 {len(self._undo_records)} 个文件"
            "移回原位置（不覆盖已有文件）。\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消撤销")
            return

        records = list(self._undo_records)
        target_root = self._undo_target_root
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(f"正在撤销：移回 {len(records)} 个文件…")

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> OrganizeResult:
            return undo_organize(
                records,
                target_root=target_root,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_undo_progress)
        worker.task_finished.connect(self._on_undo_finished)
        worker.task_cancelled.connect(self._on_undo_cancelled)
        worker.task_failed.connect(self._on_task_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_undo_progress(self, done: object, total: object) -> None:
        done_count, total_count = int(done), int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(min(100, percent))
        self._set_status(f"正在撤销：{done_count}/{total_count} 个文件…")

    def _refresh_undo_after_partial(self) -> None:
        """撤销结束后保留仍未移回原位置的记录。"""
        remaining = [
            (src, dst) for src, dst in self._undo_records if dst.exists()
        ]
        self._undo_records = remaining
        if not remaining:
            self._undo_target_root = None
        self._refresh_undo_button()

    def _on_undo_finished(self, result: object) -> None:
        if not isinstance(result, OrganizeResult):
            self._on_task_failed("撤销返回了无法识别的结果。")
            return

        self.progress_bar.setValue(100)
        self._refresh_undo_after_partial()
        self._clear_table()
        self._set_status(
            f"撤销完成：移回 {len(result.moved)} 个，"
            f"跳过 {result.skipped} 个，失败 {len(result.errors)} 个。"
        )

        if result.errors:
            lines = [
                f"已移回 {len(result.moved)} 个文件，"
                f"跳过 {result.skipped} 个，"
                f"失败 {len(result.errors)} 个。",
                "",
                "失败项（前 5 条）：",
                *_format_error_lines(result.errors),
            ]
            QMessageBox.warning(
                self, "撤销完成（部分失败）", "\n".join(lines)
            )
        else:
            QMessageBox.information(
                self,
                "撤销完成",
                f"已移回 {len(result.moved)} 个文件，"
                f"跳过 {result.skipped} 个。",
            )

    def _on_undo_cancelled(self, result: object) -> None:
        partial = result if isinstance(result, OrganizeResult) else None
        moved = len(partial.moved) if partial is not None else 0
        skipped = partial.skipped if partial is not None else 0
        errors = partial.errors if partial is not None else []

        self.progress_bar.setValue(0)
        self._refresh_undo_after_partial()
        self._clear_table()
        self._set_status(
            f"已取消撤销：移回 {moved} 个，跳过 {skipped} 个，"
            f"失败 {len(errors)} 个。"
        )
        if errors:
            lines = [
                f"撤销已取消，已移回 {moved} 个文件，跳过 {skipped} 个。",
                "",
                "失败项（前 5 条）：",
                *_format_error_lines(errors),
            ]
            QMessageBox.warning(
                self, "已取消（部分失败）", "\n".join(lines)
            )
        else:
            QMessageBox.information(
                self,
                "已取消撤销",
                f"撤销已取消，已移回 {moved} 个文件，"
                f"跳过 {skipped} 个。",
            )

    # --------------------------------------------------------------- 打开目录

    def _open_source(self) -> None:
        text = self.source_edit.text().strip()
        if not text:
            return
        folder = Path(text).expanduser()
        if not folder.is_dir():
            QMessageBox.critical(
                self, "目录不存在", f"目录不存在：\n{folder}"
            )
            return
        try:
            open_directory(folder)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(self, "无法打开目录", f"{folder}\n{exc}")

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；整理进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "文件整理")
