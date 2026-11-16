"""PDF 合并 / 拆分工具页（PyQt6 界面）。

页面使用 ``QTabWidget`` 提供「合并」与「拆分」两个页签，底部共用进度条、
状态标签和操作按钮。合并按列表顺序拼接多个 PDF；拆分支持每页一份、
每 N 页一份与自定义页码范围。

本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import bisect
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import UniquePathAllocator, open_directory
from toolbox.page import ToolPage
from toolbox.pdf_utils import PdfError, open_pdf, save_pdf_atomic
from toolbox.ui import add_path_row, make_form
from toolbox.workers import Cancelled, SingleTaskWorker, shutdown_worker

MODE_EACH_PAGE = "each"
MODE_EVERY_N_PAGES = "every"
MODE_CUSTOM_RANGES = "custom"

MIN_MERGE_FILES = 2
EVERY_N_MIN, EVERY_N_MAX = 1, 9999

TaskCallable = Callable[
    [Callable[[object, object], None], threading.Event], object
]


@dataclass
class MergeResult:
    """一次合并操作的结果。"""

    output: Path
    files: int
    pages: int


@dataclass
class SplitChunk:
    """一个待拆分的页码范围（1 基，含两端）。"""

    start: int
    end: int


@dataclass
class SplitResult:
    """一次拆分操作的结果。"""

    files: int
    pages: int
    skipped: int


# --------------------------------------------------------------------- 纯逻辑

_RANGE_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "、": ",",
        "；": ",",
        ";": ",",
        "～": "-",
        "~": "-",
        "—": "-",
        "–": "-",
        "－": "-",
        "‐": "-",
        "‑": "-",
        "−": "-",
        "至": "-",
    }
)


def parse_range_groups(text: str, total_pages: int) -> list[SplitChunk]:
    """把 ``1-3,5,8-10`` 形式的范围解析为拆分块。

    兼容全角逗号、顿号、分号、波浪号、各种破折号，忽略空白。
    每个块使用 1 基页码且包含两端；保留输入顺序；拒绝越界与重叠区间。
    """
    if total_pages < 1:
        raise ValueError("PDF 页数必须大于 0")

    normalized = (text or "").translate(_RANGE_TRANSLATION)
    normalized = "".join(normalized.split())
    if not normalized:
        raise ValueError("拆分范围不能为空")

    chunks: list[SplitChunk] = []
    # 已接受区间按 start 排序且互不重叠；只需检查前驱与后继即可判定重叠。
    accepted: list[tuple[int, int]] = []
    for token in normalized.split(","):
        if not token:
            continue
        pieces = token.split("-")
        if len(pieces) == 1:
            if not pieces[0].isdigit():
                raise ValueError(f"无法识别的页码范围片段：{token}")
            start = end = int(pieces[0])
        elif len(pieces) == 2:
            start_text, end_text = pieces
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"无法识别的页码范围片段：{token}")
            start, end = int(start_text), int(end_text)
        else:
            raise ValueError(f"无法识别的页码范围片段：{token}")

        if start > end:
            raise ValueError(f"页码范围起点大于终点：{token}")
        if start < 1 or end > total_pages:
            raise ValueError(f"页码超出范围（1-{total_pages}）：{token}")

        position = bisect.bisect_left(accepted, (start,))
        if position and accepted[position - 1][1] >= start:
            raise ValueError(f"页码范围重叠：{token}")
        if position < len(accepted) and accepted[position][0] <= end:
            raise ValueError(f"页码范围重叠：{token}")
        accepted.insert(position, (start, end))
        chunks.append(SplitChunk(start, end))

    if not chunks:
        raise ValueError("拆分范围不能为空")
    return chunks


def _normalize_split_mode(mode: str) -> str:
    """把内部值或中文显示名统一为拆分模式常量。"""
    key = (
        "".join(str(mode).split())
        .casefold()
        .replace("_", "")
        .replace("-", "")
    )
    aliases = {
        "each": MODE_EACH_PAGE,
        "perpage": MODE_EACH_PAGE,
        "每页": MODE_EACH_PAGE,
        "每页一份": MODE_EACH_PAGE,
        "every": MODE_EVERY_N_PAGES,
        "everyn": MODE_EVERY_N_PAGES,
        "everynpages": MODE_EVERY_N_PAGES,
        "每n页": MODE_EVERY_N_PAGES,
        "每n页一份": MODE_EVERY_N_PAGES,
        "custom": MODE_CUSTOM_RANGES,
        "customranges": MODE_CUSTOM_RANGES,
        "range": MODE_CUSTOM_RANGES,
        "ranges": MODE_CUSTOM_RANGES,
        "自定义": MODE_CUSTOM_RANGES,
        "自定义范围": MODE_CUSTOM_RANGES,
    }
    try:
        return aliases[key]
    except KeyError:
        raise ValueError(f"无法识别的拆分方式：{mode}") from None


def build_split_chunks(
    total_pages: int,
    mode: str,
    every_n: int,
    ranges: list[SplitChunk] | None = None,
) -> list[SplitChunk]:
    """根据模式生成拆分块。

    - 每页一份：生成 ``total_pages`` 个单页块。
    - 每 N 页一份：最后一块可以不足 N 页。
    - 自定义范围：使用并校验传入的 ``ranges``，保持其顺序。
    """
    if total_pages < 1:
        raise ValueError("PDF 页数必须大于 0")

    normalized_mode = _normalize_split_mode(mode)
    if normalized_mode == MODE_EACH_PAGE:
        return [SplitChunk(page, page) for page in range(1, total_pages + 1)]

    if normalized_mode == MODE_EVERY_N_PAGES:
        if every_n < 1:
            raise ValueError("每份页数必须大于 0")
        chunks: list[SplitChunk] = []
        start = 1
        while start <= total_pages:
            end = min(start + every_n - 1, total_pages)
            chunks.append(SplitChunk(start, end))
            start = end + 1
        return chunks

    if not ranges:
        raise ValueError("拆分范围不能为空")
    chunks = []
    for item in ranges:
        start, end = int(item.start), int(item.end)
        if start < 1 or end > total_pages or start > end:
            raise ValueError(
                f"拆分范围无效：{start}-{end}"
                f"（有效范围 1-{total_pages}，且起点不能大于终点）"
            )
        chunks.append(SplitChunk(start, end))
    return chunks


def chunk_filename(stem: str, chunk: SplitChunk) -> str:
    """生成拆分块的文件名：单页 ``stem_p001.pdf``，多页 ``stem_p001-010.pdf``。"""
    if chunk.start < 1 or chunk.end < chunk.start:
        raise ValueError(f"无效的页码范围：{chunk.start}-{chunk.end}")
    if chunk.start == chunk.end:
        return f"{stem}_p{chunk.start:03d}.pdf"
    return f"{stem}_p{chunk.start:03d}-{chunk.end:03d}.pdf"


def merge_pdfs(
    paths: Sequence[Path],
    output: Path,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> MergeResult:
    """按顺序合并多个 PDF。

    合并分两遍：第一遍逐文件打开、记录页数并收集全部错误；任一文件不可用
    时抛出 :class:`PdfError`（消息中列出文件名），且不会产生输出。第二遍
    逐个重新打开并按顺序插入，最后原子保存。``on_progress`` 以
    ``(已处理文件数, 文件总数)`` 报告进度，取消在文件之间检查。
    """
    path_list = [Path(path) for path in paths]
    output = Path(output)
    if len(path_list) < MIN_MERGE_FILES:
        raise ValueError(f"至少需要 {MIN_MERGE_FILES} 个 PDF 文件才能合并")

    resolved_output = output.resolve()
    for path in path_list:
        if path.resolve() == resolved_output:
            raise ValueError(f"输出文件不能与输入文件相同：{path}")

    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()

    # 第一遍：只校验并记录页数，不在内存中保留 Document。
    validated: list[tuple[Path, int]] = []
    errors: list[str] = []
    for path in path_list:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        if not path.is_file():
            errors.append(f"{path.name}：文件不存在")
            continue
        try:
            source = open_pdf(path)
            try:
                page_count = source.page_count
            finally:
                source.close()
        except Exception as exc:  # noqa: BLE001 - 汇总后统一转中文提示
            errors.append(f"{path.name}：{exc}")
        else:
            validated.append((path, page_count))

    if errors:
        raise PdfError("以下 PDF 文件无法处理：\n" + "\n".join(errors))
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()

    # 第二遍：逐个重新打开并插入，插入后立即关闭以限制打开文件数。
    document = pymupdf.open()
    done_pages = 0
    try:
        for index, (path, page_count) in enumerate(validated):
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            source = open_pdf(path)
            try:
                document.insert_pdf(
                    source,
                    from_page=0,
                    to_page=page_count - 1,
                    links=True,
                    annots=True,
                )
            finally:
                source.close()
            done_pages += page_count
            if on_progress is not None:
                on_progress(index + 1, len(validated))
        save_pdf_atomic(document, output)
    finally:
        document.close()

    return MergeResult(
        output=output, files=len(validated), pages=done_pages
    )


def split_pdf(
    src: Path,
    out_dir: Path,
    chunks: Sequence[SplitChunk],
    *,
    overwrite: bool,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> SplitResult:
    """按块拆分 PDF，每个块生成一个 PDF 文件。

    ``overwrite`` 为假时，同名目标文件通过
    :class:`~toolbox.common.UniquePathAllocator` 自动改名；为真时原子覆盖。
    取消时抛出 ``Cancelled``，其 ``result`` 为已完成部分
    的 :class:`SplitResult`，已经生成的文件保留。``on_progress`` 以
    ``(已处理页数, 待处理页数)`` 报告进度。
    """
    src = Path(src)
    out_dir = Path(out_dir)
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("拆分范围不能为空")
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled(SplitResult(files=0, pages=0, skipped=0))

    source = open_pdf(src)
    src_resolved = src.resolve()
    out_dir_resolved = out_dir.resolve()
    done_files = 0
    done_pages = 0
    skipped = 0
    try:
        total_pages = source.page_count
        validated: list[SplitChunk] = []
        for chunk in chunk_list:
            start, end = int(chunk.start), int(chunk.end)
            if start < 1 or end > total_pages or start > end:
                raise ValueError(
                    f"拆分范围无效：{start}-{end}"
                    f"（有效范围 1-{total_pages}，且起点不能大于终点）"
                )
            validated.append(SplitChunk(start, end))

        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(SplitResult(files=0, pages=0, skipped=0))

        if out_dir.exists() and not out_dir.is_dir():
            raise PdfError(f"输出目录不是文件夹：{out_dir}")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PdfError(f"无法创建输出目录：{out_dir}\n{exc}") from exc

        allocator = UniquePathAllocator(reserve_planned=True)
        src_in_out_dir = src_resolved.parent == out_dir_resolved
        total_chunk_pages = sum(
            chunk.end - chunk.start + 1 for chunk in validated
        )
        for chunk in validated:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled(
                    SplitResult(
                        files=done_files,
                        pages=done_pages,
                        skipped=skipped,
                    )
                )

            target = out_dir / chunk_filename(src.stem, chunk)
            if target.exists():
                # 目标不存在时不可能与源文件相同，无需 resolve。
                same_as_source = (
                    src_in_out_dir
                    and target.name == src_resolved.name
                ) or target.resolve() == src_resolved
                if same_as_source or not overwrite:
                    target = allocator.reserve(target)

            document = pymupdf.open()
            try:
                try:
                    document.insert_pdf(
                        source,
                        from_page=chunk.start - 1,
                        to_page=chunk.end - 1,
                    )
                    save_pdf_atomic(document, target, preset="fast")
                except Exception as exc:  # noqa: BLE001 - 统一转中文提示
                    raise PdfError(
                        f"拆分第 {chunk.start}-{chunk.end} 页失败："
                        f"{target}\n{exc}"
                    ) from exc
            finally:
                document.close()

            done_files += 1
            done_pages += chunk.end - chunk.start + 1
            if on_progress is not None:
                on_progress(done_pages, total_chunk_pages)
    finally:
        source.close()

    return SplitResult(files=done_files, pages=done_pages, skipped=skipped)


# ------------------------------------------------------------------- 界面部分


class PdfMergeSplitPage(ToolPage):
    shutdown_label = "PDF 处理"
    """PDF 合并 / 拆分工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._job_kind: str | None = None
        self._last_output_dir: Path | None = None
        self._build_ui()
        self._update_split_mode_widgets()
        self._on_tab_changed()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.merge_tab = self._build_merge_tab()
        self.split_tab = self._build_split_tab()
        self.tabs.addTab(self.merge_tab, "合并")
        self.tabs.addTab(self.split_tab, "拆分")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始合并")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_task)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output_dir)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status_label = QLabel("请添加 PDF 文件或选择拆分源文件")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_merge_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("待合并 PDF（按列表顺序合并）："))
        self.merge_list = QListWidget()
        self.merge_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.merge_list.setMinimumHeight(180)
        layout.addWidget(self.merge_list, 1)

        list_buttons = QHBoxLayout()
        add_btn = QPushButton("添加文件…")
        add_btn.clicked.connect(self._add_merge_files)
        remove_btn = QPushButton("删除选中")
        remove_btn.clicked.connect(self._remove_merge_files)
        up_btn = QPushButton("上移")
        up_btn.clicked.connect(self._move_merge_up)
        down_btn = QPushButton("下移")
        down_btn.clicked.connect(self._move_merge_down)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(remove_btn)
        list_buttons.addWidget(up_btn)
        list_buttons.addWidget(down_btn)
        list_buttons.addStretch(1)
        layout.addLayout(list_buttons)

        form = make_form()
        self.merge_output_edit: QLineEdit = add_path_row(
            form,
            0,
            "输出 PDF：",
            mode="save_file",
            placeholder="选择合并后的 PDF 保存位置",
            dialog_title="保存合并后的 PDF",
            file_filter="PDF 文件 (*.pdf)",
            parent=self,
        )
        layout.addLayout(form)

        hint = QLabel(
            "至少需要 2 个 PDF；合并前会检查文件是否存在、加密或损坏。"
        )
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return tab

    def _build_split_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = make_form()
        self.split_src_edit: QLineEdit = add_path_row(
            form,
            0,
            "源 PDF：",
            mode="open_file",
            placeholder="选择要拆分的 PDF 文件",
            dialog_title="选择 PDF 文件",
            file_filter="PDF 文件 (*.pdf);;所有文件 (*)",
            parent=self,
        )
        self.split_outdir_edit: QLineEdit = add_path_row(
            form,
            1,
            "输出目录：",
            mode="open_dir",
            placeholder="选择拆分结果的保存目录",
            dialog_title="选择输出目录",
            parent=self,
        )

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("每页一份", MODE_EACH_PAGE)
        self.mode_combo.addItem("每 N 页一份", MODE_EVERY_N_PAGES)
        self.mode_combo.addItem("自定义范围", MODE_CUSTOM_RANGES)
        self.mode_combo.currentIndexChanged.connect(
            self._update_split_mode_widgets
        )
        form.addWidget(QLabel("拆分方式："), 2, 0)
        form.addWidget(self.mode_combo, 2, 1, 1, 2)

        self.split_every_spin = QSpinBox()
        self.split_every_spin.setRange(EVERY_N_MIN, EVERY_N_MAX)
        self.split_every_spin.setValue(10)
        self.split_every_spin.setFixedWidth(110)
        form.addWidget(QLabel("每份页数："), 3, 0)
        form.addWidget(self.split_every_spin, 3, 1)

        self.split_range_edit = QLineEdit()
        self.split_range_edit.setPlaceholderText("如 1-3,5,8-10")
        form.addWidget(QLabel("页码范围："), 4, 0)
        form.addWidget(self.split_range_edit, 4, 1, 1, 2)
        layout.addLayout(form)

        hint = QLabel(
            "自定义范围支持全角逗号、顿号、波浪号、破折号，空白会被忽略；"
            "同名输出文件将自动改名，不覆盖已有文件。"
        )
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return tab

    def _on_tab_changed(self, _index: int | None = None) -> None:
        if self.tabs.currentIndex() == 0:
            self.start_btn.setText("开始合并")
        else:
            self.start_btn.setText("开始拆分")

    def _update_split_mode_widgets(self, *_args: object) -> None:
        mode = self.mode_combo.currentData()
        self.split_every_spin.setEnabled(mode == MODE_EVERY_N_PAGES)
        self.split_range_edit.setEnabled(mode == MODE_CUSTOM_RANGES)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_running(self, running: bool) -> None:
        self.tabs.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if running:
            self.open_btn.setEnabled(False)
        else:
            self._update_split_mode_widgets()

    # --------------------------------------------------------------- 合并输入

    def _add_merge_files(self) -> None:
        start_dir = str(Path.home())
        if self.merge_list.count():
            last_item = self.merge_list.item(self.merge_list.count() - 1)
            if last_item is not None:
                start_dir = str(Path(last_item.text()).parent)
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "添加 PDF 文件",
            start_dir,
            "PDF 文件 (*.pdf);;所有文件 (*)",
        )
        if not selected:
            return
        for file_name in selected:
            self.merge_list.addItem(str(Path(file_name)))
        self._set_status(f"已添加 {len(selected)} 个 PDF 文件")

    def _remove_merge_files(self) -> None:
        selected = self.merge_list.selectedItems()
        if not selected:
            return
        for item in selected:
            self.merge_list.takeItem(self.merge_list.row(item))
        self._set_status(f"已删除 {len(selected)} 个文件")

    def _move_merge_up(self) -> None:
        self._move_merge_item(-1)

    def _move_merge_down(self) -> None:
        self._move_merge_item(1)

    def _move_merge_item(self, offset: int) -> None:
        row = self.merge_list.currentRow()
        target = row + offset
        if row < 0 or target < 0 or target >= self.merge_list.count():
            return
        item = self.merge_list.takeItem(row)
        if item is None:
            return
        self.merge_list.insertItem(target, item)
        self.merge_list.setCurrentRow(target)

    def _merge_paths(self) -> list[Path]:
        return [
            Path(self.merge_list.item(index).text())
            for index in range(self.merge_list.count())
            if self.merge_list.item(index) is not None
        ]

    # --------------------------------------------------------------- 启动任务

    def _start_task(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self.tabs.currentIndex() == 0:
            self._start_merge()
        else:
            self._start_split()

    def _collect_merge_job(self) -> tuple[list[Path], Path] | None:
        paths = self._merge_paths()
        if len(paths) < MIN_MERGE_FILES:
            QMessageBox.warning(
                self,
                "文件数量不足",
                f"请至少添加 {MIN_MERGE_FILES} 个 PDF 文件。",
            )
            return None

        output_text = self.merge_output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(
                self, "缺少输出文件", "请指定合并后的 PDF 保存位置。"
            )
            return None
        output = Path(output_text).expanduser()
        if not output.name:
            QMessageBox.warning(
                self, "输出路径无效", f"无效的输出路径：\n{output}"
            )
            return None
        if output.exists() and output.is_dir():
            QMessageBox.warning(
                self, "输出路径无效", f"输出路径是一个目录：\n{output}"
            )
            return None
        if not output.suffix:
            output = output.with_name(f"{output.name}.pdf")
            self.merge_output_edit.setText(str(output))

        resolved_output = output.resolve()
        for path in paths:
            if path.expanduser().resolve() == resolved_output:
                QMessageBox.warning(
                    self,
                    "输出路径冲突",
                    f"输出文件不能与输入文件相同：\n{path}",
                )
                return None

        if output.exists():
            reply = QMessageBox.question(
                self,
                "输出文件已存在",
                f"输出文件已存在：\n{output}\n\n继续将覆盖该文件，是否继续？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None
        return paths, output

    def _collect_split_job(
        self,
    ) -> tuple[Path, Path, list[SplitChunk]] | None:
        src_text = self.split_src_edit.text().strip()
        if not src_text:
            QMessageBox.warning(
                self, "缺少源 PDF", "请先选择要拆分的 PDF 文件。"
            )
            return None
        src = Path(src_text).expanduser()
        if not src.is_file():
            QMessageBox.warning(
                self, "PDF 文件不存在", f"找不到文件：\n{src}"
            )
            return None

        out_text = self.split_outdir_edit.text().strip()
        if not out_text:
            QMessageBox.warning(
                self, "缺少输出目录", "请先选择拆分结果的输出目录。"
            )
            return None
        out_dir = Path(out_text).expanduser()
        if out_dir.exists() and not out_dir.is_dir():
            QMessageBox.warning(
                self, "输出路径无效", f"输出路径不是目录：\n{out_dir}"
            )
            return None

        try:
            document = open_pdf(src)
            try:
                total_pages = document.page_count
            finally:
                document.close()
        except Exception as exc:  # noqa: BLE001 - 统一转中文提示
            QMessageBox.critical(
                self, "无法打开 PDF", f"{src.name}\n{exc}"
            )
            return None

        mode = self.mode_combo.currentData()
        if not isinstance(mode, str):
            mode = MODE_EACH_PAGE
        ranges: list[SplitChunk] | None = None
        if mode == MODE_CUSTOM_RANGES:
            try:
                ranges = parse_range_groups(
                    self.split_range_edit.text(), total_pages
                )
            except ValueError as exc:
                QMessageBox.warning(self, "页码范围错误", str(exc))
                return None

        try:
            chunks = build_split_chunks(
                total_pages, mode, self.split_every_spin.value(), ranges
            )
        except ValueError as exc:
            QMessageBox.warning(self, "拆分设置错误", str(exc))
            return None
        return src, out_dir, chunks

    def _start_merge(self) -> None:
        job = self._collect_merge_job()
        if job is None:
            return
        paths, output = job

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> object:
            def report(done: int, total: int) -> None:
                on_progress(done, total)

            return merge_pdfs(
                paths,
                output,
                cancel_event=cancel_event,
                on_progress=report,
            )

        self._job_kind = "merge"
        self._last_output_dir = output.resolve().parent
        self._begin_worker(task, f"正在合并 {len(paths)} 个文件…")

    def _start_split(self) -> None:
        job = self._collect_split_job()
        if job is None:
            return
        src, out_dir, chunks = job

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> object:
            def report(done: int, total: int) -> None:
                on_progress(done, total)

            return split_pdf(
                src,
                out_dir,
                chunks,
                overwrite=False,
                cancel_event=cancel_event,
                on_progress=report,
            )

        total_pages = sum(chunk.end - chunk.start + 1 for chunk in chunks)
        self._job_kind = "split"
        self._last_output_dir = out_dir.resolve()
        self._begin_worker(
            task,
            f"正在拆分：共 {total_pages} 页，生成 {len(chunks)} 个文件…",
        )

    def _begin_worker(self, task: TaskCallable, status: str) -> None:
        self.progress_bar.setValue(0)
        self._set_status(status)
        self._set_running(True)

        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_progress)
        worker.task_finished.connect(self._on_task_finished)
        worker.task_cancelled.connect(self._on_task_cancelled)
        worker.task_failed.connect(self._on_task_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    # --------------------------------------------------------------- 任务回调

    def _on_progress(self, done: object, total: object) -> None:
        done_count, total_count = int(done), int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(min(100, percent))
        if self._job_kind == "split":
            self._set_status(f"正在拆分：{done_count}/{total_count} 页…")
        else:
            self._set_status(
                f"正在合并：{done_count}/{total_count} 个文件…"
            )

    def _on_task_finished(self, result: object) -> None:
        if isinstance(result, MergeResult):
            self._finish_merge(result)
        elif isinstance(result, SplitResult):
            self._finish_split(result)
        else:
            self._on_task_failed("后台任务返回了未知结果")

    def _finish_merge(self, result: MergeResult) -> None:
        self.progress_bar.setValue(100)
        if self._last_output_dir is None:
            self._last_output_dir = result.output.resolve().parent
        if self._last_output_dir.is_dir():
            self.open_btn.setEnabled(True)
        self._set_status(
            f"合并完成：{result.files} 个文件，共 {result.pages} 页"
            f" → {result.output}"
        )
        QMessageBox.information(
            self,
            "合并完成",
            f"已合并 {result.files} 个文件，共 {result.pages} 页。\n\n"
            f"输出文件：\n{result.output}",
        )

    def _finish_split(self, result: SplitResult) -> None:
        self.progress_bar.setValue(100)
        if self._last_output_dir is not None and self._last_output_dir.is_dir():
            self.open_btn.setEnabled(True)
        skipped_text = (
            f"，跳过 {result.skipped} 个同名文件" if result.skipped else ""
        )
        self._set_status(
            f"拆分完成：已生成 {result.files} 个文件，共 {result.pages} 页"
            f"{skipped_text} → {self._last_output_dir}"
        )
        QMessageBox.information(
            self,
            "拆分完成",
            f"已生成 {result.files} 个 PDF 文件，共 {result.pages} 页"
            f"{skipped_text}。\n\n输出目录：\n{self._last_output_dir}",
        )

    def _on_task_cancelled(self, result: object) -> None:
        if self._job_kind == "split":
            files = result.files if isinstance(result, SplitResult) else 0
            pages = result.pages if isinstance(result, SplitResult) else 0
            self._set_status(
                f"已取消：已完成 {files} 个文件，共 {pages} 页"
            )
            if (
                files
                and self._last_output_dir is not None
                and self._last_output_dir.is_dir()
            ):
                self.open_btn.setEnabled(True)
            QMessageBox.information(
                self,
                "已取消",
                f"拆分已取消，已完成 {files} 个文件，共 {pages} 页。\n"
                "已经生成的文件已保留。",
            )
        else:
            self._set_status("已取消：未生成输出文件，已完成 0 个文件")
            QMessageBox.information(
                self,
                "已取消",
                "合并已取消，未生成输出文件。\n已完成文件数：0",
            )

    def _on_task_failed(self, message: str) -> None:
        self._set_status("处理失败")
        QMessageBox.critical(self, "处理失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)

    def _cancel_task(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._set_status("正在取消…")

    def _open_output_dir(self) -> None:
        if self._last_output_dir is None:
            return
        if not self._last_output_dir.is_dir():
            QMessageBox.critical(
                self,
                "目录不存在",
                f"输出目录不存在：\n{self._last_output_dir}",
            )
            return
        try:
            open_directory(self._last_output_dir)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(
                self, "无法打开目录", f"{self._last_output_dir}\n{exc}"
            )

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；任务进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "PDF 处理")
