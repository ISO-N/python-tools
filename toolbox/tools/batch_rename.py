"""批量重命名工具页（PyQt6 界面）。

只处理所选文件夹第一层的普通文件（不递归，跳过符号链接），支持
「查找替换」「序号重排」「加前后缀」三种模式。预览阶段会完成名称合法性
检查、目标冲突与重复检测；执行时采用两阶段重命名，避免目标名恰好是
另一个源文件时互相覆盖，并保留记录供「撤销上一次」使用。
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import filesystem_key, natural_sort_key, open_directory
from toolbox.errors import Cancelled
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import SingleTaskWorker

MODE_REPLACE = "replace"
MODE_NUMBER = "number"
MODE_AFFIX = "affix"
VALID_MODES = frozenset({MODE_REPLACE, MODE_NUMBER, MODE_AFFIX})

STATUS_OK = "ok"
STATUS_UNCHANGED = "unchanged"
STATUS_CONFLICT = "conflict"
STATUS_DUPLICATE = "duplicate"
STATUS_INVALID = "invalid"

BLOCKING_STATUSES = frozenset(
    {STATUS_CONFLICT, STATUS_DUPLICATE, STATUS_INVALID}
)

STATUS_TEXT = {
    STATUS_OK: "将重命名",
    STATUS_UNCHANGED: "未变化",
    STATUS_CONFLICT: "目标已存在",
    STATUS_DUPLICATE: "新名重复",
    STATUS_INVALID: "名称非法",
}

MODE_ITEMS = (
    (MODE_REPLACE, "查找替换"),
    (MODE_NUMBER, "序号重排"),
    (MODE_AFFIX, "加前后缀"),
)

# Windows 文件名非法字符及 ASCII 控制字符。
_ILLEGAL_STEM_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED_STEMS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


@dataclass
class RenameOptions:
    """一次重命名所使用的规则。"""

    mode: str = MODE_REPLACE
    find: str = ""
    replace: str = ""
    use_regex: bool = False
    case_sensitive: bool = False
    prefix: str = ""
    suffix: str = ""
    start: int = 1
    digits: int = 3
    extension_filter: str = ""


@dataclass
class RenameItem:
    """单个文件的重命名计划项。"""

    src: Path
    dst: Path
    status: str
    reason: str = ""


@dataclass
class RenamePlan:
    """一个文件夹的完整重命名计划。"""

    folder: Path
    items: list[RenameItem] = field(default_factory=list)

    @cached_property
    def ok_items(self) -> list[RenameItem]:
        """真正会被执行的重命名项。"""
        return [item for item in self.items if item.status == STATUS_OK]

    @cached_property
    def has_blocking(self) -> bool:
        """是否存在会导致无法执行重命名的项。"""
        return any(item.status in BLOCKING_STATUSES for item in self.items)


@dataclass
class RenameResult:
    """重命名或撤销的执行结果。"""

    renamed: list[tuple[Path, Path]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- 纯逻辑


def _clean_stem_text(stem: str) -> str:
    """替换非法字符并去掉结尾空格/句点，保留用于展示的文本。"""
    return _ILLEGAL_STEM_CHARS.sub("_", str(stem)).rstrip(" .")


def _sanitize_cleaned(cleaned: str) -> str:
    """校验已经清洗过的文本；为空或属于 Windows 保留名时返回空字符串。"""
    if not cleaned:
        return ""
    base = cleaned.split(".", 1)[0].strip().upper()
    if base in _RESERVED_STEMS:
        return ""
    return cleaned


def sanitize_stem(stem: str) -> str:
    """清理主文件名；结果为空或属于 Windows 保留名时返回空字符串。

    非法字符与 ASCII 控制字符会被替换为下划线，结尾的空格和句点会被
    去掉。Windows 保留名按第一个点之前的部分判断，因此 ``CON``、
    ``con.txt``、``COM1.abc`` 等都会被视为不可用。
    """
    return _sanitize_cleaned(_clean_stem_text(stem))


def _normalize_extension_filter(value: str) -> str:
    """把界面输入的扩展名规范为 ``.jpg`` 这样的小写形式。"""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("*."):
        text = text[1:]
    if not text.startswith("."):
        text = f".{text}"
    return text.casefold()


def _make_replacer(options: RenameOptions) -> Callable[[str], str]:
    """根据查找替换选项返回作用于主文件名的替换函数。"""
    if not options.find:
        return lambda stem: stem

    flags = 0 if options.case_sensitive else re.IGNORECASE
    if options.use_regex:
        try:
            pattern = re.compile(options.find, flags)
        except re.error as exc:
            raise ValueError(f"查找正则表达式无效：{exc}") from exc

        def replace_regex(stem: str) -> str:
            try:
                return pattern.sub(options.replace, stem)
            except re.error as exc:
                raise ValueError(f"替换文本无效：{exc}") from exc

        return replace_regex

    if options.case_sensitive:
        return lambda stem: stem.replace(options.find, options.replace)

    # 非正则模式下的替换文本必须按字面处理，不能让 \1 等被当作反向引用。
    literal = re.compile(re.escape(options.find), flags)
    return lambda stem: literal.sub(lambda _match: options.replace, stem)


def build_rename_plan(
    folder: Path,
    options: RenameOptions,
    *,
    cancel_event: threading.Event | None = None,
) -> RenamePlan:
    """扫描文件夹第一层并生成重命名计划。

    只处理普通文件，跳过符号链接与子目录，按 ``natural_sort_key`` 排序。
    ``extension_filter`` 非空时只保留指定扩展名（大小写不敏感）。
    查找替换模式的正则错误会以 ``ValueError`` 抛出，供界面提示。
    ``cancel_event`` 在扫描与逐项生成计划时检查，取消时抛出
    :class:`toolbox.errors.Cancelled`。
    """
    folder = Path(folder)
    if options.mode not in VALID_MODES:
        raise ValueError(f"不支持的重命名模式：{options.mode}")

    extension = _normalize_extension_filter(options.extension_filter)
    replacer = _make_replacer(options) if options.mode == MODE_REPLACE else None
    digits = max(1, int(options.digits))

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()

    entries: list[Path] = []
    sources: list[Path] = []
    check_cancelled()
    with os.scandir(folder) as scan:
        for entry in scan:
            check_cancelled()
            path = Path(entry.path)
            entries.append(path)
            try:
                if entry.is_symlink():
                    continue
                if extension and path.suffix.casefold() != extension:
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            sources.append(path)
    sources.sort(key=lambda path: natural_sort_key(path.name))

    items: list[RenameItem] = []
    number = int(options.start)
    for src in sources:
        check_cancelled()
        if options.mode == MODE_REPLACE:
            assert replacer is not None
            raw_stem = replacer(src.stem)
        elif options.mode == MODE_NUMBER:
            raw_stem = (
                f"{options.prefix}{number:0{digits}d}{options.suffix}"
            )
        else:
            raw_stem = f"{options.prefix}{src.stem}{options.suffix}"

        cleaned = _clean_stem_text(raw_stem)
        stem = _sanitize_cleaned(cleaned)
        display_stem = stem or cleaned
        dst = folder / f"{display_stem}{src.suffix}"

        if dst.name == src.name:
            item = RenameItem(
                src=src,
                dst=dst,
                status=STATUS_UNCHANGED,
                reason="文件名未变化",
            )
        elif not stem:
            item = RenameItem(
                src=src,
                dst=dst,
                status=STATUS_INVALID,
                reason="新主文件名为空或为 Windows 保留名",
            )
        else:
            item = RenameItem(src=src, dst=dst, status=STATUS_OK)
        items.append(item)

        if options.mode == MODE_NUMBER and item.status != STATUS_UNCHANGED:
            number += 1

    _mark_duplicates(items)
    _mark_conflicts(entries, items)
    return RenamePlan(folder=folder, items=items)


def _mark_duplicates(items: list[RenameItem]) -> None:
    """把映射到同一目标（按 filesystem_key 比较）的计划项标为重复。"""
    groups: dict[str, list[RenameItem]] = {}
    for item in items:
        if item.status == STATUS_OK:
            key = filesystem_key(item.dst.name)
            groups.setdefault(key, []).append(item)

    for group in groups.values():
        if len(group) < 2:
            continue
        for item in group:
            item.status = STATUS_DUPLICATE
            item.reason = f"多个文件将重命名为同一名称：{item.dst.name}"


def _mark_conflicts(entries: list[Path], items: list[RenameItem]) -> None:
    """标记目标名已被占用、或占用者不会移走的计划项。

    目标恰好是另一个状态为 ``ok`` 的计划源文件时不算冲突，因为两阶段
    重命名会先把它移走；若占用者是未变化/非法/冲突/重复项或计划外文件，
    则该目标不会腾空，仍视为冲突。

    ``_mark_duplicates`` 保证每个目标名最多只有一个 ``ok`` 项，因此可以先
    建立「被依赖项 -> 依赖项」的反向边，再从初始冲突出发一次 BFS 传播，
    等价于旧实现的多轮扫描。环状互指且全部为 ``ok`` 时没有初始冲突，
    仍保持 ``ok``。
    """
    entries_by_key: dict[str, list[str]] = {}
    for entry in entries:
        key = filesystem_key(entry.name)
        entries_by_key.setdefault(key, []).append(entry.name)
    source_by_name = {
        item.src.name: index for index, item in enumerate(items)
    }

    dependents: dict[int, set[int]] = {}
    queue: list[int] = []
    for index, item in enumerate(items):
        if item.status != STATUS_OK:
            continue
        blocked = False
        linked: set[int] = set()
        for name in entries_by_key.get(
            filesystem_key(item.dst.name), ()
        ):
            if name == item.src.name:
                continue
            other_index = source_by_name.get(name)
            if other_index is None:
                blocked = True
                break
            other = items[other_index]
            if other.status != STATUS_OK:
                blocked = True
                break
            if other_index not in linked:
                linked.add(other_index)
                dependents.setdefault(other_index, set()).add(index)
        if blocked:
            item.status = STATUS_CONFLICT
            item.reason = f"目标文件名已被占用：{item.dst.name}"
            queue.append(index)

    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        for dependent_index in dependents.get(current, ()):
            dependent = items[dependent_index]
            if dependent.status != STATUS_OK:
                continue
            dependent.status = STATUS_CONFLICT
            dependent.reason = (
                f"目标文件名已被占用：{dependent.dst.name}"
            )
            queue.append(dependent_index)


def _lexists(path: Path) -> bool:
    """判断路径是否存在（包括失效的符号链接）。"""
    return os.path.lexists(str(path))


def _move_no_overwrite(src: Path, dst: Path) -> None:
    """重命名文件，绝不覆盖已存在的目标。"""
    if _lexists(dst):
        raise FileExistsError(f"目标已存在：{dst}")
    src.rename(dst)


def _make_temp_path(
    folder: Path, tag: str, index: int, used: set[Path]
) -> Path:
    """生成同目录下唯一且不存在的临时文件名。"""
    for _attempt in range(1000):
        candidate = folder / f".{tag}_{uuid.uuid4().hex}_{index}"
        if candidate not in used and not _lexists(candidate):
            used.add(candidate)
            return candidate
    raise OSError(f"无法在 {folder} 中生成唯一的临时文件名")


def _restore_paths(
    pairs: list[tuple[Path, Path]], errors: list[str]
) -> None:
    """尽力把 ``(当前位置, 恢复位置)`` 中的文件恢复到恢复位置。

    先全部暂存到新的临时名，再逐一改回目标名，避免链条或环状改名时
    恢复顺序导致互相覆盖。无法恢复的文件会保留在临时名并记录错误。
    """
    staged: list[tuple[Path, Path]] = []
    used: set[Path] = set()
    for index, (current, original) in enumerate(pairs):
        if not _lexists(current):
            errors.append(f"回滚跳过：文件不存在：{current}")
            continue
        try:
            temp = _make_temp_path(current.parent, "pyrename_rb", index, used)
            _move_no_overwrite(current, temp)
        except OSError as exc:
            errors.append(f"回滚暂存失败：{current.name}：{exc}")
            continue
        staged.append((temp, original))

    for temp, original in staged:
        try:
            _move_no_overwrite(temp, original)
        except OSError as exc:
            errors.append(
                f"回滚失败：{temp.name} → {original.name}：{exc}"
            )


def execute_rename_plan(
    plan: RenamePlan,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> RenameResult:
    """两阶段执行计划中的 ``ok`` 项。

    阶段 1 把所有源文件改成同目录临时名，阶段 2 再改成目标名；任一步
    失败都会尽力回滚已改动的文件。未变化、冲突、重复与非法项不会被改动。
    成功时返回 ``(原路径, 新路径)`` 列表，供撤销使用。

    ``cancel_event`` 在每个文件处理前检查；取消时会先把已暂存或已完成
    的文件恢复原名，再抛出 :class:`toolbox.errors.Cancelled`，不会留下
    半改名状态。``on_progress`` 的总步数为 ``2 * len(ok_items)``。
    """
    result = RenameResult()
    ok_items = plan.ok_items
    if not ok_items:
        return result

    total_steps = len(ok_items) * 2

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def report(done: int) -> None:
        if on_progress is not None:
            on_progress(done, total_steps)

    used: set[Path] = set()
    moved: list[tuple[RenameItem, Path]] = []
    for index, item in enumerate(ok_items):
        if cancelled():
            _restore_paths(
                [(temp_path, other.src) for other, temp_path in moved],
                result.errors,
            )
            raise Cancelled(result)
        try:
            temp = _make_temp_path(
                item.src.parent, "pyrename", index, used
            )
            _move_no_overwrite(item.src, temp)
        except OSError as exc:
            result.errors.append(
                f"重命名失败：{item.src.name} → {item.dst.name}：{exc}"
            )
            _restore_paths(
                [(temp_path, other.src) for other, temp_path in moved],
                result.errors,
            )
            return result
        moved.append((item, temp))
        report(index + 1)

    completed: list[tuple[RenameItem, Path]] = []
    for index, (item, temp) in enumerate(moved):
        if cancelled():
            completed_sources = {done.src for done, _temp in completed}
            restore_pairs = [
                (
                    other.dst if other.src in completed_sources else temp_path,
                    other.src,
                )
                for other, temp_path in moved
            ]
            _restore_paths(restore_pairs, result.errors)
            raise Cancelled(result)
        try:
            _move_no_overwrite(temp, item.dst)
        except OSError as exc:
            result.errors.append(
                f"无法改为目标名：{item.src.name} → {item.dst.name}：{exc}"
            )
            completed_sources = {done.src for done, _temp in completed}
            restore_pairs = [
                (
                    other.dst if other.src in completed_sources else temp_path,
                    other.src,
                )
                for other, temp_path in moved
            ]
            _restore_paths(restore_pairs, result.errors)
            return result
        completed.append((item, temp))
        report(len(ok_items) + index + 1)

    if on_progress is not None:
        on_progress(total_steps, total_steps)

    result.renamed = [(item.src, item.dst) for item in ok_items]
    return result


def undo_rename(
    records: list[tuple[Path, Path]],
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> RenameResult:
    """反向两阶段撤销重命名记录。

    ``records`` 的每项为 ``(原路径, 新路径)``。原路径已被其他文件占用时，
    对应项会报错并保留在新路径处，绝不覆盖任何现有文件。返回的
    ``renamed`` 按 ``(撤销前路径, 撤销后路径)`` 记录实际完成的项。

    ``cancel_event`` 在每个文件处理前检查；取消时会恢复已暂存或已完成
    的文件，再抛出 :class:`toolbox.errors.Cancelled`。``on_progress`` 的
    总步数为 ``2 * len(candidates)``。
    """
    result = RenameResult()
    if not records:
        return result
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled(result)

    normalized = [(Path(old), Path(new)) for old, new in records]
    new_paths = {new for _old, new in normalized}
    candidates: list[tuple[Path, Path]] = []

    for old, new in normalized:
        if old == new:
            result.errors.append(f"新旧路径相同，无法撤销：{new}")
            continue
        if not _lexists(new):
            result.errors.append(f"找不到重命名后的文件：{new}")
            continue
        if _lexists(old):
            # 原路径由另一条待撤销记录占用时，两阶段执行会先让它腾空。
            if old in new_paths:
                candidates.append((old, new))
                continue
            # 大小写-only 改名在大小写不敏感的文件系统上可能是同一个文件。
            try:
                same_file = new.samefile(old)
            except OSError:
                same_file = False
            if same_file:
                candidates.append((old, new))
                continue
            result.errors.append(f"原路径已被占用，未覆盖：{old}")
            continue
        candidates.append((old, new))

    if not candidates:
        return result

    total_steps = len(candidates) * 2

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def report(done: int) -> None:
        if on_progress is not None:
            on_progress(done, total_steps)

    used: set[Path] = set()
    moved: list[tuple[Path, Path, Path]] = []
    for index, (old, new) in enumerate(candidates):
        if cancelled():
            _restore_paths(
                [
                    (temp_path, new_path)
                    for _o, new_path, temp_path in moved
                ],
                result.errors,
            )
            raise Cancelled(result)
        try:
            temp = _make_temp_path(
                new.parent, "pyrename_undo", index, used
            )
            _move_no_overwrite(new, temp)
        except OSError as exc:
            result.errors.append(f"撤销失败：{new.name}：{exc}")
            _restore_paths(
                [
                    (temp_path, new_path)
                    for _o, new_path, temp_path in moved
                ],
                result.errors,
            )
            return result
        moved.append((old, new, temp))
        report(index + 1)

    completed: list[tuple[Path, Path, Path]] = []
    for index, (old, new, temp) in enumerate(moved):
        if cancelled():
            completed_new = {done_new for _o, done_new, _t in completed}
            restore_pairs = [
                (
                    old_path if new_path in completed_new else temp_path,
                    new_path,
                )
                for old_path, new_path, temp_path in moved
            ]
            _restore_paths(restore_pairs, result.errors)
            raise Cancelled(result)
        try:
            _move_no_overwrite(temp, old)
        except OSError as exc:
            result.errors.append(
                f"撤销失败：{new.name} → {old.name}：{exc}"
            )
            completed_new = {done_new for _o, done_new, _t in completed}
            restore_pairs = [
                (
                    old_path if new_path in completed_new else temp_path,
                    new_path,
                )
                for old_path, new_path, temp_path in moved
            ]
            _restore_paths(restore_pairs, result.errors)
            return result
        completed.append((old, new, temp))
        report(len(candidates) + index + 1)

    if on_progress is not None:
        on_progress(total_steps, total_steps)

    result.renamed = [(new, old) for old, new, _temp in completed]
    return result


# ------------------------------------------------------------------- 界面部分


class BatchRenamePage(ToolPage):
    """批量重命名工具页。"""

    shutdown_label = "重命名"

    def __init__(self) -> None:
        super().__init__()
        self._undo_records: list[tuple[Path, Path]] = []
        self._plan: RenamePlan | None = None
        self._plan_stale = True
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._pending_execute_plan: RenamePlan | None = None
        self._build_ui()

    def current_worker(self) -> QThread | None:
        return self._worker

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_MARGINS)
        layout.setSpacing(10)

        form = make_form()
        self.folder_edit = add_path_row(
            form,
            0,
            "文件夹：",
            mode="open_dir",
            placeholder="选择要重命名的文件夹（只处理第一层文件）",
            dialog_title="选择文件夹",
            parent=self,
        )

        self.ext_edit = QLineEdit()
        self.ext_edit.setPlaceholderText("如 .jpg，留空表示全部")
        form.addWidget(QLabel("扩展名过滤："), 1, 0)
        form.addWidget(self.ext_edit, 1, 1, 1, 2)

        self.mode_combo = QComboBox()
        for mode, label in MODE_ITEMS:
            self.mode_combo.addItem(label, mode)
        form.addWidget(QLabel("重命名方式："), 2, 0)
        form.addWidget(self.mode_combo, 2, 1, 1, 2)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_replace_page())
        self.mode_stack.addWidget(self._build_number_page())
        self.mode_stack.addWidget(self._build_affix_page())
        form.addWidget(self.mode_stack, 3, 0, 1, 3)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("预览")
        self.preview_btn.clicked.connect(self._preview)
        self.execute_btn = QPushButton("执行重命名")
        self.execute_btn.setDefault(True)
        self.execute_btn.setEnabled(False)
        self.execute_btn.clicked.connect(self._execute)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.undo_btn = QPushButton("撤销上一次")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("撤销上一次重命名（暂无可撤销的操作）")
        self.undo_btn.clicked.connect(self._undo)
        self.open_btn = QPushButton("打开文件夹")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(self.execute_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.undo_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 4, 0, 1, 3)

        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["序号", "原文件名", "新文件名", "状态"]
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
        header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.table, 1)

        self.status_label = QLabel("请选择文件夹并设置重命名规则")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._connect_option_signals()
        self._refresh_undo_button()
        self._update_open_button()

    def _build_replace_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("要查找的文本")
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("替换为（可留空）")
        grid.addWidget(QLabel("查找："), 0, 0)
        grid.addWidget(self.find_edit, 0, 1, 1, 3)
        grid.addWidget(QLabel("替换："), 1, 0)
        grid.addWidget(self.replace_edit, 1, 1, 1, 3)

        options = QHBoxLayout()
        self.regex_check = QCheckBox("使用正则表达式")
        self.case_check = QCheckBox("区分大小写")
        options.addWidget(self.regex_check)
        options.addWidget(self.case_check)
        options.addStretch(1)
        grid.addLayout(options, 2, 1, 1, 3)
        return page

    def _build_number_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.number_prefix_edit = QLineEdit()
        self.number_prefix_edit.setPlaceholderText("新文件名前缀（可留空）")
        self.start_spin = QSpinBox()
        self.start_spin.setRange(1, 999999)
        self.start_spin.setValue(1)
        self.digits_spin = QSpinBox()
        self.digits_spin.setRange(1, 8)
        self.digits_spin.setValue(3)

        grid.addWidget(QLabel("前缀："), 0, 0)
        grid.addWidget(self.number_prefix_edit, 0, 1, 1, 3)
        grid.addWidget(QLabel("起始序号："), 1, 0)
        grid.addWidget(self.start_spin, 1, 1)
        grid.addWidget(QLabel("位数："), 1, 2)
        grid.addWidget(self.digits_spin, 1, 3)
        return page

    def _build_affix_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.affix_prefix_edit = QLineEdit()
        self.affix_prefix_edit.setPlaceholderText("新文件名前缀（可留空）")
        self.affix_suffix_edit = QLineEdit()
        self.affix_suffix_edit.setPlaceholderText("新文件名后缀（可留空）")
        grid.addWidget(QLabel("前缀："), 0, 0)
        grid.addWidget(self.affix_prefix_edit, 0, 1, 1, 3)
        grid.addWidget(QLabel("后缀："), 1, 0)
        grid.addWidget(self.affix_suffix_edit, 1, 1, 1, 3)
        return page

    def _connect_option_signals(self) -> None:
        self.folder_edit.textChanged.connect(self._on_options_changed)
        self.ext_edit.textChanged.connect(self._on_options_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        for edit in (
            self.find_edit,
            self.replace_edit,
            self.number_prefix_edit,
            self.affix_prefix_edit,
            self.affix_suffix_edit,
        ):
            edit.textChanged.connect(self._on_options_changed)
        for check in (self.regex_check, self.case_check):
            check.toggled.connect(self._on_options_changed)
        for spin in (self.start_spin, self.digits_spin):
            spin.valueChanged.connect(self._on_options_changed)

    def _on_mode_changed(self, index: int) -> None:
        self.mode_stack.setCurrentIndex(index)
        self._on_options_changed()

    def _on_options_changed(self, *_args: object) -> None:
        self._plan_stale = True
        self._update_open_button()
        if self._plan is None:
            return
        can_execute = (
            not self._plan.has_blocking
            and bool(self._plan.ok_items)
            and not self._busy
        )
        self.execute_btn.setEnabled(can_execute)
        if can_execute:
            # 表格过期时仍允许执行，届时会按当前设置重新生成计划。
            self._set_status(
                "设置已更改，表格为旧预览；点击「执行重命名」"
                "会重新生成计划。"
            )
        else:
            self._set_status("设置已更改，请重新预览后再执行。")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @staticmethod
    def _folder_from_text(text: str) -> Path | None:
        """把输入框内容解析为路径；无法展开 ``~user`` 时返回 None。"""
        try:
            return Path(text).expanduser()
        except (OSError, RuntimeError):
            return None

    def _update_open_button(self) -> None:
        text = self.folder_edit.text().strip()
        folder = self._folder_from_text(text) if text else None
        self.open_btn.setEnabled(
            folder is not None and folder.is_dir() and not self._busy
        )

    def _refresh_undo_button(self) -> None:
        count = len(self._undo_records)
        self.undo_btn.setEnabled(count > 0 and not self._busy)
        if count:
            self.undo_btn.setToolTip(f"撤销上一次重命名（{count} 个文件）")
        else:
            self.undo_btn.setToolTip("撤销上一次重命名（暂无可撤销的操作）")

    def _refresh_buttons(self) -> None:
        if self._busy:
            self.execute_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
            return
        if self._plan is None:
            self.execute_btn.setEnabled(False)
        else:
            self.execute_btn.setEnabled(
                not self._plan.has_blocking and bool(self._plan.ok_items)
            )
        self._refresh_undo_button()
        self._update_open_button()

    def _set_busy(self, busy: bool) -> None:
        """任务期间禁用输入与执行类按钮，仅保留取消按钮。"""
        self._busy = busy
        for widget in (
            self.folder_edit,
            self.ext_edit,
            self.mode_combo,
            self.mode_stack,
            self.preview_btn,
        ):
            widget.setEnabled(not busy)
        for button in self.findChildren(QPushButton):
            button.setEnabled(not busy)
        if busy:
            self.cancel_btn.setEnabled(True)
        else:
            self.cancel_btn.setEnabled(False)
            self._refresh_buttons()

    # --------------------------------------------------------------- 预览流程

    def _collect_options(self) -> tuple[Path, RenameOptions] | None:
        """校验界面输入并收集 RenameOptions；失败时弹窗并返回 None。"""
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            QMessageBox.warning(
                self, "缺少文件夹", "请先选择要重命名的文件夹。"
            )
            return None
        folder = self._folder_from_text(folder_text)
        if folder is None:
            QMessageBox.warning(
                self, "路径无效", f"无法解析文件夹路径：\n{folder_text}"
            )
            return None
        if not folder.is_dir():
            QMessageBox.warning(
                self, "文件夹不存在", f"目录不存在：\n{folder}"
            )
            return None

        mode = str(self.mode_combo.currentData() or MODE_REPLACE)
        if mode == MODE_REPLACE and not self.find_edit.text():
            QMessageBox.warning(
                self,
                "缺少查找文本",
                "查找替换模式需要填写要查找的文本。",
            )
            return None

        if mode == MODE_NUMBER:
            prefix = self.number_prefix_edit.text()
            suffix = ""
        elif mode == MODE_AFFIX:
            prefix = self.affix_prefix_edit.text()
            suffix = self.affix_suffix_edit.text()
        else:
            prefix = ""
            suffix = ""

        options = RenameOptions(
            mode=mode,
            find=self.find_edit.text(),
            replace=self.replace_edit.text(),
            use_regex=self.regex_check.isChecked(),
            case_sensitive=self.case_check.isChecked(),
            prefix=prefix,
            suffix=suffix,
            start=self.start_spin.value(),
            digits=self.digits_spin.value(),
            extension_filter=self.ext_edit.text().strip(),
        )
        return folder, options

    def _build_plan(self) -> RenamePlan | None:
        """按当前界面设置生成计划，并刷新表格、按钮与状态。"""
        collected = self._collect_options()
        if collected is None:
            return None
        folder, options = collected
        try:
            plan = build_rename_plan(folder, options)
        except ValueError as exc:
            self._plan = None
            self._plan_stale = True
            self.table.setRowCount(0)
            self._refresh_buttons()
            self._set_status("重命名规则无效")
            QMessageBox.warning(self, "重命名规则错误", str(exc))
            return None
        except OSError as exc:
            self._plan = None
            self._plan_stale = True
            self.table.setRowCount(0)
            self._refresh_buttons()
            self._set_status("无法读取文件夹")
            QMessageBox.critical(
                self, "无法读取文件夹", f"{folder}\n{exc}"
            )
            return None

        self._plan = plan
        self._plan_stale = False
        self._fill_table(plan)
        self._refresh_buttons()
        self._report_plan(plan)
        return plan

    def _fill_table(self, plan: RenamePlan) -> None:
        self.table.setRowCount(0)
        self.table.setRowCount(len(plan.items))
        for row, item in enumerate(plan.items):
            number_item = QTableWidgetItem(str(row + 1))
            number_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            src_item = QTableWidgetItem(item.src.name)
            if item.dst == plan.folder:
                new_name = "（空）"
            else:
                new_name = item.dst.name
            dst_item = QTableWidgetItem(new_name)
            dst_item.setToolTip(str(item.dst))
            status_item = QTableWidgetItem(
                STATUS_TEXT.get(item.status, item.status)
            )
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setToolTip(item.reason)

            if item.status in {
                STATUS_CONFLICT,
                STATUS_DUPLICATE,
                STATUS_INVALID,
            }:
                status_item.setForeground(QColor("#c62828"))
            elif item.status == STATUS_UNCHANGED:
                status_item.setForeground(QColor("#888888"))

            self.table.setItem(row, 0, number_item)
            self.table.setItem(row, 1, src_item)
            self.table.setItem(row, 2, dst_item)
            self.table.setItem(row, 3, status_item)

    def _report_plan(self, plan: RenamePlan) -> None:
        total = len(plan.items)
        counts = Counter(item.status for item in plan.items)
        ok_count = counts[STATUS_OK]
        unchanged = counts[STATUS_UNCHANGED]
        conflict = counts[STATUS_CONFLICT]
        duplicate = counts[STATUS_DUPLICATE]
        invalid = counts[STATUS_INVALID]

        if plan.has_blocking:
            parts: list[str] = []
            if conflict:
                parts.append(f"目标已存在 {conflict} 个")
            if duplicate:
                parts.append(f"新名重复 {duplicate} 个")
            if invalid:
                parts.append(f"名称非法 {invalid} 个")
            summary = "、".join(parts)
            self._set_status(
                f"预览完成：共 {total} 个文件，其中 {ok_count} 个可重命名；"
                f"存在阻塞项（{summary}），已禁用执行。"
                "请调整规则后重新预览。"
            )
            return

        if ok_count:
            self._set_status(
                f"预览完成：共 {total} 个文件，将重命名 {ok_count} 个，"
                f"未变化 {unchanged} 个。"
            )
        else:
            self._set_status(
                f"预览完成：共 {total} 个文件，没有需要重命名的文件。"
            )

    def _preview(self) -> None:
        if self._busy:
            return
        self._build_plan()

    # --------------------------------------------------------------- 执行流程

    def _execute(self) -> None:
        if self._busy:
            return
        if self._plan is None or self._plan_stale:
            self._start_rebuild_plan()
            return
        self._confirm_and_execute(self._plan)

    def _start_rebuild_plan(self) -> None:
        """计划过期时在后台重新扫描并生成计划。"""
        collected = self._collect_options()
        if collected is None:
            return
        folder, options = collected

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> object:
            try:
                plan = build_rename_plan(
                    folder, options, cancel_event=cancel_event
                )
            except (OSError, ValueError) as exc:
                return exc
            if cancel_event.is_set():
                raise Cancelled(plan)
            return plan

        self._pending_execute_plan = None
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)
        self._set_status("正在重新生成重命名计划…")
        worker = SingleTaskWorker(task, self)
        worker.task_finished.connect(self._on_rebuild_finished)
        worker.task_cancelled.connect(self._on_rebuild_cancelled)
        worker.task_failed.connect(self._on_rebuild_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_rebuild_finished(self, result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        if isinstance(result, Exception):
            self._plan = None
            self._plan_stale = True
            self.table.setRowCount(0)
            self._refresh_buttons()
            self._set_status("无法生成重命名计划")
            if isinstance(result, ValueError):
                QMessageBox.warning(
                    self, "重命名规则错误", str(result)
                )
            else:
                QMessageBox.critical(
                    self, "无法读取文件夹", str(result)
                )
            return
        if not isinstance(result, RenamePlan):
            self._on_rebuild_failed("后台任务返回了未知结果")
            return
        self._plan = result
        self._plan_stale = False
        self._fill_table(result)
        self._refresh_buttons()
        self._report_plan(result)
        # 等 worker 的 finished 信号后再弹出确认对话框，避免连续启动
        # 两个 worker 时覆盖 self._worker。
        self._pending_execute_plan = result

    def _on_rebuild_cancelled(self, _result: object) -> None:
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("已取消重命名")

    def _on_rebuild_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        self._set_status("无法生成重命名计划")
        QMessageBox.critical(self, "无法生成重命名计划", message)

    def _confirm_and_execute(self, plan: RenamePlan) -> None:
        if plan.has_blocking:
            self._set_status("存在阻塞项，无法执行重命名。")
            QMessageBox.warning(
                self,
                "存在阻塞项",
                "当前计划中存在目标已存在、新名重复或名称非法的文件，"
                "请调整规则并重新预览。",
            )
            return

        ok_items = plan.ok_items
        if not ok_items:
            QMessageBox.information(
                self,
                "没有可重命名的文件",
                "当前规则下没有需要改变名称的文件。",
            )
            self._set_status("没有需要重命名的文件")
            return

        reply = QMessageBox.question(
            self,
            "确认重命名",
            f"将重命名 {len(ok_items)} 个文件。\n\n"
            f"位置：{plan.folder}\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消重命名")
            return

        self._start_execute_worker(plan)

    def _start_execute_worker(self, plan: RenamePlan) -> None:
        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> RenameResult:
            return execute_rename_plan(
                plan,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(f"正在重命名 {len(plan.ok_items)} 个文件…")
        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_execute_progress)
        worker.task_finished.connect(self._on_execute_finished)
        worker.task_cancelled.connect(self._on_execute_cancelled)
        worker.task_failed.connect(self._on_execute_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_execute_progress(self, done: object, total: object) -> None:
        done_count, total_count = int(done), int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(min(100, percent))
        self._set_status(f"正在重命名：{done_count}/{total_count} 步…")

    def _on_execute_finished(self, result: object) -> None:
        if not isinstance(result, RenameResult):
            self._on_execute_failed("后台任务返回了未知结果")
            return

        self.progress_bar.setValue(100)
        if result.renamed:
            self._undo_records = list(result.renamed)

        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        self._refresh_buttons()

        if result.errors:
            self._set_status(
                f"重命名未完成：{len(result.errors)} 个错误（已尽力回滚）"
            )
            QMessageBox.critical(
                self,
                "重命名失败",
                "重命名未完成，已尽力回滚。\n\n"
                + "\n".join(result.errors),
            )
            return

        count = len(result.renamed)
        self._set_status(
            f"完成：已重命名 {count} 个文件。"
            "可点击「撤销上一次」恢复原名。"
        )
        QMessageBox.information(
            self, "重命名完成", f"已重命名 {count} 个文件。"
        )

    def _on_execute_cancelled(self, result: object) -> None:
        partial = result if isinstance(result, RenameResult) else None
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        if partial is not None and partial.renamed:
            self._undo_records = [
                (old, new)
                for old, new in partial.renamed
                if _lexists(new)
            ]
        else:
            self._undo_records = [
                (old, new)
                for old, new in self._undo_records
                if _lexists(new)
            ]
        self._refresh_buttons()
        self._set_status("已取消重命名（已尽力恢复原名）")

    def _on_execute_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        self._refresh_buttons()
        self._set_status("重命名失败")
        QMessageBox.critical(
            self, "重命名失败", f"发生未预期的错误：\n{message}"
        )

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        pending = self._pending_execute_plan
        self._pending_execute_plan = None
        if pending is not None:
            self._confirm_and_execute(pending)

    def _cancel_task(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._set_status("正在取消…")

    def _undo(self) -> None:
        if self._busy or not self._undo_records:
            return
        count = len(self._undo_records)
        reply = QMessageBox.question(
            self,
            "确认撤销",
            f"将撤销上一次重命名，恢复 {count} 个文件的原名。\n\n"
            "是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消撤销")
            return

        records = list(self._undo_records)

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> RenameResult:
            return undo_rename(
                records,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(f"正在撤销：恢复 {len(records)} 个文件…")
        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_undo_progress)
        worker.task_finished.connect(self._on_undo_finished)
        worker.task_cancelled.connect(self._on_undo_cancelled)
        worker.task_failed.connect(self._on_undo_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_undo_progress(self, done: object, total: object) -> None:
        done_count, total_count = int(done), int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(min(100, percent))
        self._set_status(f"正在撤销：{done_count}/{total_count} 步…")

    def _on_undo_finished(self, result: object) -> None:
        if not isinstance(result, RenameResult):
            self._on_undo_failed("后台任务返回了未知结果")
            return

        self.progress_bar.setValue(100)
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)

        if result.errors:
            self._undo_records = [
                (old, new)
                for old, new in self._undo_records
                if _lexists(new)
            ]
            self._refresh_buttons()
            self._set_status(
                f"撤销未完全成功：恢复 {len(result.renamed)} 个，"
                f"失败 {len(result.errors)} 个"
            )
            QMessageBox.critical(
                self,
                "撤销未完全成功",
                "\n".join(result.errors),
            )
            return

        self._undo_records = []
        self._refresh_buttons()
        self._set_status(
            f"撤销完成：已恢复 {len(result.renamed)} 个文件的原名"
        )
        QMessageBox.information(
            self,
            "撤销完成",
            f"已恢复 {len(result.renamed)} 个文件的原名。",
        )

    def _on_undo_cancelled(self, _result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        self._undo_records = [
            (old, new)
            for old, new in self._undo_records
            if _lexists(new)
        ]
        self._refresh_buttons()
        self._set_status("已取消撤销（已尽力恢复）")

    def _on_undo_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._plan = None
        self._plan_stale = True
        self.table.setRowCount(0)
        self._refresh_buttons()
        self._set_status("撤销失败")
        QMessageBox.critical(
            self, "撤销失败", f"发生未预期的错误：\n{message}"
        )

    def _open_folder(self) -> None:
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            return
        folder = self._folder_from_text(folder_text)
        if folder is None:
            QMessageBox.critical(
                self, "路径无效", f"无法解析文件夹路径：\n{folder_text}"
            )
            return
        if not folder.is_dir():
            QMessageBox.critical(
                self, "目录不存在", f"目录不存在：\n{folder}"
            )
            return
        try:
            open_directory(folder)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(self, "无法打开目录", f"{folder}\n{exc}")
