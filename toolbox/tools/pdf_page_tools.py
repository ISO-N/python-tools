"""PDF 页码、目录书签与文字水印工具页（PyQt6 界面）。

一个页面共用源 / 输出 PDF 路径，通过页签分别配置三项可组合的操作：

* 「页脚页码」：按模板为指定页范围绘制页码；
* 「目录书签」：把 ``{"章节标题": 起始页码}`` 配置写入 PDF 目录；
* 「文字水印」：在页面中心或平铺绘制带透明度的旋转文字。

处理在后台线程中执行，支持进度报告与取消。本模块作为「Python 工具箱」
的一个工具页使用。
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from toolbox.chapters import BOOKMARK_AI_PROMPT, ConfigError, parse_chapter_config
from toolbox.common import find_cjk_font_file, open_directory
from toolbox.page import ToolPage
from toolbox.pdf_utils import (
    PageNumberTemplate,
    TextStamp,
    add_text_stamps,
    format_page_number,
    load_pdf_font_resource,
    open_pdf,
    parse_page_number_template,
    register_cjk_font,
    save_pdf_atomic,
)
from toolbox.ui import make_form
from toolbox.workers import Cancelled, SingleTaskWorker, shutdown_worker

MM_TO_PT = 72.0 / 25.4

PAGE_NUMBER_POSITIONS = (
    "底居中",
    "底右",
    "底左",
    "顶居中",
    "顶右",
    "顶左",
)

PAGE_NUMBER_TEMPLATE_DEFAULT = "第 {page} 页 / 共 {total} 页"
BOOKMARK_OFFSET_MIN, BOOKMARK_OFFSET_MAX = -9999, 9999
FONT_FALLBACK_HINT = "使用内置字体，未嵌入 PDF，换阅读器可能缺字"


# --------------------------------------------------------------------- 数据模型


@dataclass
class PageNumberOptions:
    """页码绘制参数。"""

    template: str = PAGE_NUMBER_TEMPLATE_DEFAULT
    start_number: int = 1
    first_page: int = 1
    position: str = "底居中"
    font_size: float = 10.0
    margin_mm: float = 12.0
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class WatermarkOptions:
    """水印绘制参数。"""

    text: str
    font_size: float = 36.0
    opacity: float = 0.25
    angle: float = 45.0
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    tiled: bool = False
    gap: float = 80.0
    margin_mm: float = 0.0


@dataclass
class PageToolsResult:
    """一次 PDF 页面处理的结果统计。"""

    output: Path
    pages: int
    page_numbers: int
    bookmarks: int
    skipped_bookmarks: list[tuple[str, int]] = field(default_factory=list)
    watermarks: int = 0
    font_embedded: bool = False


# --------------------------------------------------------------------- 纯逻辑


def _validate_page_number_options(
    options: PageNumberOptions,
    total_pages: int,
    template: PageNumberTemplate | None = None,
) -> None:
    """校验页码参数；模板可传入已解析结果以避免重复解析。"""
    parsed = (
        template
        if template is not None
        else parse_page_number_template(options.template)
    )
    parsed.render(options.start_number, total_pages)
    if options.start_number < 1:
        raise ValueError("起始编号必须大于等于 1")
    if options.first_page < 1:
        raise ValueError("起始页必须大于等于 1")
    if options.position not in PAGE_NUMBER_POSITIONS:
        raise ValueError(f"不支持的页码位置：{options.position}")
    if options.font_size <= 0:
        raise ValueError("页码字号必须大于 0")
    if options.margin_mm < 0:
        raise ValueError("页码边距不能为负数")


def _page_number_layout(
    page: pymupdf.Page,
    options: PageNumberOptions,
    page_number: int,
    total: int,
    font: pymupdf.Font,
    template: PageNumberTemplate | None = None,
) -> tuple[str, tuple[float, float]]:
    """计算某一页的页码文本与绘制中心，同时校验文本是否过宽。"""
    parsed = (
        template
        if template is not None
        else parse_page_number_template(options.template)
    )
    text = parsed.render(page_number, total)
    margin = options.margin_mm * MM_TO_PT
    width = font.text_length(text, fontsize=options.font_size)
    drawable_width = page.rect.width - 2 * margin
    if width > drawable_width:
        raise ValueError(
            f"页码文字过宽（{width:.1f} pt），超出可绘制宽度"
            f"（{drawable_width:.1f} pt），请缩小字号、边距或缩短模板"
        )

    if options.position.endswith("左"):
        x = page.rect.x0 + margin + width / 2
    elif options.position.endswith("右"):
        x = page.rect.x1 - margin - width / 2
    else:
        x = (page.rect.x0 + page.rect.x1) / 2

    if options.position.startswith("底"):
        y = page.rect.y1 - margin - options.font_size / 2
    else:
        y = page.rect.y0 + margin + options.font_size / 2
    return text, (x, y)


def apply_page_numbers(
    doc: pymupdf.Document,
    options: PageNumberOptions,
    font_path: Path | str | None,
) -> int:
    """为 ``doc`` 从 ``first_page``（1 基）开始绘制页码，返回绘制页数。

    第一页使用 ``start_number``，之后逐页递增；``{total}`` 始终是文档总页数。
    每页绘制前都会尝试嵌入 ``font_path`` 字体，失败时回退内置 ``china-s``。
    """
    total = doc.page_count
    template = parse_page_number_template(options.template)
    _validate_page_number_options(options, total, template)
    if options.first_page > total:
        return 0

    resource = load_pdf_font_resource(font_path)
    font = resource.font
    drawn = 0
    for index in range(options.first_page - 1, total):
        page = doc.load_page(index)
        page_number = options.start_number + index - (options.first_page - 1)
        text, center = _page_number_layout(
            page, options, page_number, total, font, template
        )
        fontname = register_cjk_font(
            page, font_path, resource=resource
        )
        add_text_stamps(
            page,
            [
                TextStamp(
                    center,
                    text,
                    options.font_size,
                    color=options.color,
                )
            ],
            fontname=fontname,
            font=font,
        )
        drawn += 1
    return drawn


def build_toc_entries(
    config: list[tuple[str, int]], offset: int, page_count: int
) -> tuple[list[list], list[tuple[str, int]]]:
    """把章节配置换算为 PyMuPDF 目录项，并收集越界条目。

    PDF 页码 = 起始页码 + ``offset``；只保留 ``1 <= 页码 <= page_count``
    的条目。返回值依次为 ``[级别, 标题, 页码]`` 列表与
    ``(标题, 越界后的 PDF 页码)`` 列表。
    """
    entries: list[list] = []
    skipped: list[tuple[str, int]] = []
    for title, start_page in config:
        pdf_page = int(start_page) + int(offset)
        if 1 <= pdf_page <= page_count:
            entries.append([1, title, pdf_page])
        else:
            skipped.append((title, pdf_page))
    return entries, skipped


def apply_bookmarks(
    doc: pymupdf.Document,
    config: list[tuple[str, int]],
    offset: int,
) -> tuple[int, list[tuple[str, int]]]:
    """写入一级书签，返回（写入数量, 越界跳过列表）。"""
    entries, skipped = build_toc_entries(config, offset, doc.page_count)
    if entries:
        doc.set_toc(entries)
    return len(entries), skipped


def _validate_watermark_options(options: WatermarkOptions) -> None:
    if not options.text or not options.text.strip():
        raise ValueError("水印文字不能为空")
    if options.font_size <= 0:
        raise ValueError("水印字号必须大于 0")
    if not 0.0 <= options.opacity <= 1.0:
        raise ValueError("水印透明度必须在 0 到 1 之间")
    if options.gap <= 0:
        raise ValueError("水印间距必须大于 0")
    if options.margin_mm < 0:
        raise ValueError("水印边距不能为负数")


def _watermark_centers(
    page: pymupdf.Page,
    options: WatermarkOptions,
    text_width: float,
) -> list[tuple[float, float]]:
    """返回水印文字的绘制中心点列表（非平铺时只有一个页面中心）。"""
    rect = page.rect
    center_x = (rect.x0 + rect.x1) / 2
    center_y = (rect.y0 + rect.y1) / 2
    if not options.tiled:
        return [(center_x, center_y)]

    margin = options.margin_mm * MM_TO_PT
    x0, y0 = rect.x0 + margin, rect.y0 + margin
    x1, y1 = rect.x1 - margin, rect.y1 - margin
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

    angle = math.radians(options.angle)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    # 把页面矩形旋转到文字自身的坐标系，再按行列步长铺满。
    grid_x: list[float] = []
    grid_y: list[float] = []
    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        dx, dy = px - center_x, py - center_y
        grid_x.append(dx * cos_a + dy * sin_a)
        grid_y.append(-dx * sin_a + dy * cos_a)
    min_x, max_x = min(grid_x), max(grid_x)
    min_y, max_y = min(grid_y), max(grid_y)

    step_x = text_width + options.gap
    step_y = options.font_size + options.gap
    span_x = max_x - min_x
    span_y = max_y - min_y
    columns = max(1, math.ceil(span_x / step_x)) if span_x > 0 else 1
    rows = max(1, math.ceil(span_y / step_y)) if span_y > 0 else 1
    start_x = (min_x + max_x) / 2 - (columns - 1) * step_x / 2
    start_y = (min_y + max_y) / 2 - (rows - 1) * step_y / 2

    centers: list[tuple[float, float]] = []
    for row in range(rows):
        gy = start_y + row * step_y
        for column in range(columns):
            gx = start_x + column * step_x
            centers.append(
                (
                    center_x + gx * cos_a - gy * sin_a,
                    center_y + gx * sin_a + gy * cos_a,
                )
            )
    return centers


def apply_watermark(
    doc: pymupdf.Document,
    options: WatermarkOptions,
    font_path: Path | str | None,
) -> int:
    """为每一页绘制水印，返回绘制页数。"""
    _validate_watermark_options(options)
    resource = load_pdf_font_resource(font_path)
    font = resource.font
    text_width = font.text_length(options.text, fontsize=options.font_size)
    drawn = 0
    for index in range(doc.page_count):
        page = doc.load_page(index)
        fontname = register_cjk_font(
            page, font_path, resource=resource
        )
        stamps = [
            TextStamp(
                center,
                options.text,
                options.font_size,
                color=options.color,
                fill_opacity=options.opacity,
                angle=options.angle,
                text_width=text_width,
            )
            for center in _watermark_centers(page, options, text_width)
        ]
        add_text_stamps(page, stamps, fontname=fontname, font=font)
        drawn += 1
    return drawn


def process_pdf(
    src: Path | str,
    dst: Path | str,
    page_number_options: PageNumberOptions | None,
    bookmark_config: list[tuple[str, int]] | None,
    bookmark_offset: int,
    watermark_options: WatermarkOptions | None,
    *,
    font_path: Path | str | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> PageToolsResult:
    """在源 PDF 上依次应用启用的操作并保存到 ``dst``。

    ``page_number_options`` / ``bookmark_config`` / ``watermark_options``
    为 ``None`` 表示未启用对应操作，至少需要启用一项。``font_path`` 省略
    （``None``）时会自动查找系统中文字体；传入空字符串则明确使用内置字体。
    每页处理前检查取消，进度总量为页数 × 页面操作数 + 书签操作数。
    """
    src_path = Path(src).expanduser()
    dst_path = Path(dst).expanduser()
    if not src_path.is_file():
        raise ValueError(f"源 PDF 不存在：{src_path}")
    if src_path.resolve() == dst_path.resolve():
        raise ValueError("输出文件不能与源 PDF 相同，请选择其他输出路径")
    if dst_path.is_dir():
        raise ValueError(f"输出路径是目录：{dst_path}")

    has_page_numbers = page_number_options is not None
    has_bookmarks = bookmark_config is not None
    has_watermark = watermark_options is not None
    if not (has_page_numbers or has_bookmarks or has_watermark):
        raise ValueError("请至少启用一项操作")

    if font_path is None:
        font_path = find_cjk_font_file()

    total_pages = 0
    page_numbers_drawn = 0
    bookmark_count = 0
    skipped_bookmarks: list[tuple[str, int]] = []
    watermark_pages = 0
    font_embedded = False

    document = open_pdf(src_path)
    try:
        total_pages = document.page_count
        page_operations = int(has_page_numbers) + int(has_watermark)
        total_steps = total_pages * page_operations + (
            1 if has_bookmarks else 0
        )
        done = 0

        def report_progress() -> None:
            if on_progress is not None:
                on_progress(done, total_steps)

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()

        def advance() -> None:
            nonlocal done
            done += 1
            report_progress()

        # 先做全部参数校验，避免处理到一半才发现配置错误。
        page_number_template: PageNumberTemplate | None = None
        if has_page_numbers:
            page_number_template = parse_page_number_template(
                page_number_options.template
            )
            _validate_page_number_options(
                page_number_options, total_pages, page_number_template
            )
        if has_watermark:
            _validate_watermark_options(watermark_options)
        toc_entries: list[list] = []
        if has_bookmarks:
            if not bookmark_config:
                raise ValueError("书签配置为空")
            toc_entries, skipped_bookmarks = build_toc_entries(
                bookmark_config, bookmark_offset, total_pages
            )
            if not toc_entries:
                raise ValueError(
                    f"所有章节换算后的 PDF 页码都超出 1-{total_pages}，"
                    "未执行书签操作"
                )

        font_resource = (
            load_pdf_font_resource(font_path)
            if has_page_numbers or has_watermark
            else None
        )
        font = font_resource.font if font_resource is not None else None
        watermark_text_width = (
            font.text_length(
                watermark_options.text,
                fontsize=watermark_options.font_size,
            )
            if has_watermark
            else 0.0
        )
        report_progress()

        for index in range(total_pages if page_operations else 0):
            check_cancelled()
            page = document.load_page(index)
            draw_page_number = has_page_numbers and (
                index + 1 >= page_number_options.first_page
            )
            fontname: str | None = None
            if draw_page_number or has_watermark:
                fontname = register_cjk_font(
                    page, font_path, resource=font_resource
                )
                if fontname != "china-s":
                    font_embedded = True

            stamps: list[TextStamp] = []
            if has_page_numbers:
                if draw_page_number:
                    page_number = (
                        page_number_options.start_number
                        + index
                        - (page_number_options.first_page - 1)
                    )
                    text, center = _page_number_layout(
                        page,
                        page_number_options,
                        page_number,
                        total_pages,
                        font,
                        page_number_template,
                    )
                    stamps.append(
                        TextStamp(
                            center,
                            text,
                            page_number_options.font_size,
                            color=page_number_options.color,
                        )
                    )
                    page_numbers_drawn += 1

            if has_watermark:
                for center in _watermark_centers(
                    page, watermark_options, watermark_text_width
                ):
                    stamps.append(
                        TextStamp(
                            center,
                            watermark_options.text,
                            watermark_options.font_size,
                            color=watermark_options.color,
                            fill_opacity=watermark_options.opacity,
                            angle=watermark_options.angle,
                            text_width=watermark_text_width,
                        )
                    )
                watermark_pages += 1

            if stamps and fontname is not None and font is not None:
                add_text_stamps(
                    page, stamps, fontname=fontname, font=font
                )

            if has_page_numbers:
                advance()
                check_cancelled()

            if has_watermark:
                advance()
                check_cancelled()

        if has_bookmarks:
            check_cancelled()
            document.set_toc(toc_entries)
            bookmark_count = len(toc_entries)
            advance()

        check_cancelled()
        if (
            font_embedded
            and font_resource is not None
            and font_resource.file_backed
        ):
            try:
                document.subset_fonts()
            except Exception:  # noqa: BLE001 - 字体子集化失败不影响保存
                pass
        save_pdf_atomic(document, dst_path, preset="balanced")
    finally:
        document.close()

    return PageToolsResult(
        output=dst_path,
        pages=total_pages,
        page_numbers=page_numbers_drawn,
        bookmarks=bookmark_count,
        skipped_bookmarks=skipped_bookmarks,
        watermarks=watermark_pages,
        font_embedded=font_embedded,
    )


# --------------------------------------------------------------------- 界面部分


# 浏览与开始处理会重复读取同一个源 PDF；按（路径, mtime, 大小）缓存结果，
# 文件改动或换路径后键自然失效。
_SOURCE_INFO_CACHE: dict[
    tuple[str, int, int], tuple[int, list[list] | None]
] = {}
_SOURCE_INFO_CACHE_LIMIT = 16


def _source_info(
    path: Path, *, with_toc: bool = False
) -> tuple[int, list[list] | None]:
    """返回缓存中的页数与书签；``with_toc`` 为假时不读取目录。"""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _SOURCE_INFO_CACHE.get(key)
    if cached is not None and (not with_toc or cached[1] is not None):
        return cached

    document = open_pdf(path)
    try:
        page_count = document.page_count
        toc = document.get_toc() if with_toc else None
    finally:
        document.close()

    if toc is None and cached is not None:
        toc = cached[1]
    if len(_SOURCE_INFO_CACHE) >= _SOURCE_INFO_CACHE_LIMIT:
        _SOURCE_INFO_CACHE.clear()
    _SOURCE_INFO_CACHE[key] = (page_count, toc)
    return page_count, toc


class PdfPageToolsPage(ToolPage):
    shutdown_label = "PDF 处理"
    """PDF 页码 / 目录书签 / 文字水印工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._last_output: Path | None = None
        self._page_count: int | None = None
        self._auto_output: str | None = None
        self._page_number_color = (0.0, 0.0, 0.0)
        self._watermark_color = (0.5, 0.5, 0.5)
        self._build_ui()
        self._update_font_hint()
        self._update_template_hint()
        self._update_enabled_states()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        form = make_form()

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择要处理的源 PDF 文件")
        self.source_browse_btn = QPushButton("浏览…")
        self.source_browse_btn.clicked.connect(self._browse_source)
        form.addWidget(QLabel("源 PDF："), 0, 0)
        form.addWidget(self.source_edit, 0, 1)
        form.addWidget(self.source_browse_btn, 0, 2)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择处理后 PDF 的保存位置")
        self.output_browse_btn = QPushButton("浏览…")
        self.output_browse_btn.clicked.connect(self._browse_output)
        form.addWidget(QLabel("输出 PDF："), 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(self.output_browse_btn, 1, 2)

        self.font_edit = QLineEdit()
        default_font = find_cjk_font_file()
        if default_font is not None:
            self.font_edit.setText(str(default_font))
        self.font_edit.setPlaceholderText("选择中文字体文件（留空则使用内置字体）")
        self.font_browse_btn = QPushButton("浏览…")
        self.font_browse_btn.clicked.connect(self._browse_font)
        form.addWidget(QLabel("中文字体："), 2, 0)
        form.addWidget(self.font_edit, 2, 1)
        form.addWidget(self.font_browse_btn, 2, 2)

        self.font_hint = QLabel()
        self.font_hint.setWordWrap(True)
        form.addWidget(self.font_hint, 3, 1, 1, 2)
        self.font_edit.textChanged.connect(self._update_font_hint)

        layout.addLayout(form)

        self.tabs = QTabWidget()
        self.page_number_tab = self._build_page_number_tab()
        self.bookmark_tab = self._build_bookmark_tab()
        self.watermark_tab = self._build_watermark_tab()
        self.tabs.addTab(self.page_number_tab, "页脚页码")
        self.tabs.addTab(self.bookmark_tab, "目录书签")
        self.tabs.addTab(self.watermark_tab, "文字水印")
        layout.addWidget(self.tabs, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m 项")
        layout.addWidget(self.progress_bar)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_processing)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output_dir)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status_label = QLabel("请选择源 PDF，并至少启用一项操作")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_page_number_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.page_number_enabled = QCheckBox("启用此操作")
        self.page_number_enabled.toggled.connect(self._update_enabled_states)
        layout.addWidget(self.page_number_enabled)

        body = QWidget()
        form = make_form()

        self.template_edit = QLineEdit(PAGE_NUMBER_TEMPLATE_DEFAULT)
        form.addWidget(QLabel("页码模板："), 0, 0)
        form.addWidget(self.template_edit, 0, 1, 1, 2)

        self.template_hint = QLabel()
        self.template_hint.setWordWrap(True)
        form.addWidget(self.template_hint, 1, 1, 1, 2)
        self.template_edit.textChanged.connect(self._update_template_hint)

        self.start_number_spin = QSpinBox()
        self.start_number_spin.setRange(1, 99999)
        self.start_number_spin.setValue(1)
        form.addWidget(QLabel("起始编号："), 2, 0)
        form.addWidget(self.start_number_spin, 2, 1)

        self.first_page_spin = QSpinBox()
        self.first_page_spin.setRange(1, 99999)
        self.first_page_spin.setValue(1)
        form.addWidget(QLabel("从第几页开始："), 3, 0)
        form.addWidget(self.first_page_spin, 3, 1)

        self.position_combo = QComboBox()
        self.position_combo.addItems(PAGE_NUMBER_POSITIONS)
        form.addWidget(QLabel("位置："), 4, 0)
        form.addWidget(self.position_combo, 4, 1)

        self.page_font_size_spin = QDoubleSpinBox()
        self.page_font_size_spin.setRange(6.0, 72.0)
        self.page_font_size_spin.setDecimals(1)
        self.page_font_size_spin.setSingleStep(0.5)
        self.page_font_size_spin.setValue(10.0)
        self.page_font_size_spin.setSuffix(" pt")
        form.addWidget(QLabel("字号："), 5, 0)
        form.addWidget(self.page_font_size_spin, 5, 1)

        self.page_margin_spin = QDoubleSpinBox()
        self.page_margin_spin.setRange(0.0, 50.0)
        self.page_margin_spin.setDecimals(1)
        self.page_margin_spin.setSingleStep(1.0)
        self.page_margin_spin.setValue(12.0)
        self.page_margin_spin.setSuffix(" mm")
        form.addWidget(QLabel("边距："), 6, 0)
        form.addWidget(self.page_margin_spin, 6, 1)

        self.page_color_btn = QPushButton()
        self.page_color_btn.setFixedWidth(110)
        self.page_color_btn.clicked.connect(self._choose_page_number_color)
        self._update_color_button(
            self.page_color_btn, self._page_number_color
        )
        form.addWidget(QLabel("颜色："), 7, 0)
        form.addWidget(self.page_color_btn, 7, 1)

        body.setLayout(form)
        layout.addWidget(self._scroll_body(body), 1)
        self.page_number_body = body
        return tab

    def _build_bookmark_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.bookmark_enabled = QCheckBox("启用此操作")
        self.bookmark_enabled.toggled.connect(self._update_enabled_states)
        layout.addWidget(self.bookmark_enabled)

        body = QWidget()
        form = make_form()

        config_label = QLabel(
            '章节配置（JSON，格式：{"章节标题": 起始页码}）：'
        )
        form.addWidget(config_label, 0, 0, 1, 3)

        self.bookmark_config_edit = QPlainTextEdit()
        self.bookmark_config_edit.setPlaceholderText(
            '{\n  "第一章": 1,\n  "第二章": 15,\n  "第三章": 31\n}'
        )
        self.bookmark_config_edit.setMinimumHeight(120)
        self.bookmark_config_edit.setTabChangesFocus(True)
        form.addWidget(self.bookmark_config_edit, 1, 0, 1, 3)

        buttons = QHBoxLayout()
        self.copy_prompt_btn = QPushButton("复制 AI 提示词")
        self.copy_prompt_btn.setToolTip(
            "复制一段提示词模板到剪贴板，粘贴给 AI 并附上书籍目录图片，"
            "让 AI 返回本工具需要的 JSON 配置"
        )
        self.copy_prompt_btn.clicked.connect(self._copy_ai_prompt)
        buttons.addWidget(self.copy_prompt_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 2, 0, 1, 3)

        self.bookmark_offset_spin = QSpinBox()
        self.bookmark_offset_spin.setRange(
            BOOKMARK_OFFSET_MIN, BOOKMARK_OFFSET_MAX
        )
        self.bookmark_offset_spin.setValue(0)
        self.bookmark_offset_spin.setFixedWidth(120)
        form.addWidget(QLabel("页码偏移："), 3, 0)
        form.addWidget(self.bookmark_offset_spin, 3, 1)

        offset_hint = QLabel(
            "PDF 页码 = 章节起始页码 + 偏移；"
            "例如章节起始页为 1、它在 PDF 中实际是第 11 页时，偏移填 10"
        )
        offset_hint.setWordWrap(True)
        offset_hint.setStyleSheet("color: #666;")
        offset_hint.setToolTip("偏移 = PDF 页码 − 章节起始页码")
        form.addWidget(offset_hint, 4, 1, 1, 2)

        body.setLayout(form)
        layout.addWidget(self._scroll_body(body), 1)
        self.bookmark_body = body
        return tab

    def _build_watermark_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.watermark_enabled = QCheckBox("启用此操作")
        self.watermark_enabled.toggled.connect(self._update_enabled_states)
        layout.addWidget(self.watermark_enabled)

        body = QWidget()
        form = make_form()

        self.watermark_text_edit = QLineEdit()
        self.watermark_text_edit.setPlaceholderText("例如：内部资料 / 机密")
        form.addWidget(QLabel("水印文字："), 0, 0)
        form.addWidget(self.watermark_text_edit, 0, 1, 1, 2)

        self.watermark_font_size_spin = QSpinBox()
        self.watermark_font_size_spin.setRange(6, 200)
        self.watermark_font_size_spin.setValue(36)
        self.watermark_font_size_spin.setSuffix(" pt")
        form.addWidget(QLabel("字号："), 1, 0)
        form.addWidget(self.watermark_font_size_spin, 1, 1)

        opacity_row = QHBoxLayout()
        self.watermark_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.watermark_opacity_slider.setRange(1, 100)
        self.watermark_opacity_slider.setValue(25)
        self.watermark_opacity_spin = QSpinBox()
        self.watermark_opacity_spin.setRange(1, 100)
        self.watermark_opacity_spin.setValue(25)
        self.watermark_opacity_spin.setSuffix(" %")
        self.watermark_opacity_slider.valueChanged.connect(
            self.watermark_opacity_spin.setValue
        )
        self.watermark_opacity_spin.valueChanged.connect(
            self.watermark_opacity_slider.setValue
        )
        opacity_row.addWidget(self.watermark_opacity_slider, 1)
        opacity_row.addWidget(self.watermark_opacity_spin)
        form.addWidget(QLabel("透明度："), 2, 0)
        form.addLayout(opacity_row, 2, 1, 1, 2)

        self.watermark_angle_spin = QSpinBox()
        self.watermark_angle_spin.setRange(-180, 180)
        self.watermark_angle_spin.setValue(45)
        self.watermark_angle_spin.setSuffix("°")
        form.addWidget(QLabel("角度："), 3, 0)
        form.addWidget(self.watermark_angle_spin, 3, 1)

        self.watermark_color_btn = QPushButton()
        self.watermark_color_btn.setFixedWidth(110)
        self.watermark_color_btn.clicked.connect(
            self._choose_watermark_color
        )
        self._update_color_button(
            self.watermark_color_btn, self._watermark_color
        )
        form.addWidget(QLabel("颜色："), 4, 0)
        form.addWidget(self.watermark_color_btn, 4, 1)

        self.watermark_tiled_check = QCheckBox("平铺（铺满整页）")
        self.watermark_tiled_check.setChecked(False)
        form.addWidget(QLabel("平铺："), 5, 0)
        form.addWidget(self.watermark_tiled_check, 5, 1)

        self.watermark_gap_spin = QSpinBox()
        self.watermark_gap_spin.setRange(10, 500)
        self.watermark_gap_spin.setValue(80)
        self.watermark_gap_spin.setSuffix(" pt")
        self.watermark_gap_spin.setToolTip("仅在启用平铺时生效")
        form.addWidget(QLabel("间距："), 6, 0)
        form.addWidget(self.watermark_gap_spin, 6, 1)
        self.watermark_tiled_check.toggled.connect(
            self._update_enabled_states
        )

        body.setLayout(form)
        layout.addWidget(self._scroll_body(body), 1)
        self.watermark_body = body
        return tab

    def _scroll_body(self, body: QWidget) -> QScrollArea:
        """把页签内容放进无边框滚动区域，避免窗口较小时被裁切。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        return scroll

    # ------------------------------------------------------------ 界面辅助

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _font_may_fall_back(self) -> bool:
        """当前启用项需要字体，但字体路径为空或不存在。"""
        if not (
            self.page_number_enabled.isChecked()
            or self.watermark_enabled.isChecked()
        ):
            return False
        text = self.font_edit.text().strip()
        return not text or not Path(text).expanduser().is_file()

    def _update_font_hint(self, *_args: object) -> None:
        text = self.font_edit.text().strip()
        if not text:
            self.font_hint.setText(
                f"未指定字体文件：将{FONT_FALLBACK_HINT}"
            )
            self.font_hint.setStyleSheet("color: #b3261e;")
        elif not Path(text).expanduser().is_file():
            self.font_hint.setText(
                f"字体文件不存在：将{FONT_FALLBACK_HINT}"
            )
            self.font_hint.setStyleSheet("color: #b3261e;")
        else:
            self.font_hint.setText(
                f"将嵌入字体：{Path(text).expanduser().name}"
            )
            self.font_hint.setStyleSheet("color: #666;")

    def _update_template_hint(self, *_args: object) -> None:
        try:
            preview = format_page_number(
                self.template_edit.text(), 1, 1
            )
        except ValueError as exc:
            self.template_hint.setText(f"模板错误：{exc}")
            self.template_hint.setStyleSheet("color: #b3261e;")
        else:
            self.template_hint.setText(f"预览：{preview}")
            self.template_hint.setStyleSheet("color: #666;")

    def _update_color_button(
        self, button: QPushButton, color: tuple[float, float, float]
    ) -> None:
        red, green, blue = (round(value * 255) for value in color)
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        text_color = "#ffffff" if luminance < 140 else "#000000"
        button.setText("颜色…")
        button.setStyleSheet(
            f"background-color: rgb({red}, {green}, {blue});"
            f" color: {text_color};"
        )

    def _choose_color(
        self, current: tuple[float, float, float]
    ) -> tuple[float, float, float] | None:
        chosen = QColorDialog.getColor(
            QColor.fromRgbF(*current), self, "选择颜色"
        )
        if not chosen.isValid():
            return None
        return (chosen.redF(), chosen.greenF(), chosen.blueF())

    def _choose_page_number_color(self) -> None:
        color = self._choose_color(self._page_number_color)
        if color is None:
            return
        self._page_number_color = color
        self._update_color_button(self.page_color_btn, color)

    def _choose_watermark_color(self) -> None:
        color = self._choose_color(self._watermark_color)
        if color is None:
            return
        self._watermark_color = color
        self._update_color_button(self.watermark_color_btn, color)

    def _update_enabled_states(self, *_args: object) -> None:
        busy = self._busy
        for widget in (
            self.source_edit,
            self.output_edit,
            self.font_edit,
            self.source_browse_btn,
            self.output_browse_btn,
            self.font_browse_btn,
        ):
            widget.setEnabled(not busy)
        self.tabs.setEnabled(not busy)
        self.page_number_body.setEnabled(
            not busy and self.page_number_enabled.isChecked()
        )
        self.bookmark_body.setEnabled(
            not busy and self.bookmark_enabled.isChecked()
        )
        self.watermark_body.setEnabled(
            not busy and self.watermark_enabled.isChecked()
        )
        self.watermark_gap_spin.setEnabled(
            not busy
            and self.watermark_enabled.isChecked()
            and self.watermark_tiled_check.isChecked()
        )
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.open_btn.setEnabled(
            not busy
            and self._last_output is not None
            and self._last_output.is_file()
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_enabled_states()

    def _browse_source(self) -> None:
        current = self.source_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择源 PDF",
            start_dir,
            "PDF 文件 (*.pdf);;所有文件 (*)",
        )
        if not selected:
            return
        source = Path(selected)
        self.source_edit.setText(selected)

        default_output = source.with_name(f"{source.stem}_annotated.pdf")
        current_output = self.output_edit.text().strip()
        if not current_output or current_output == self._auto_output:
            self.output_edit.setText(str(default_output))
            self._auto_output = str(default_output)
        self._load_source_page_count(source)

    def _load_source_page_count(self, path: Path) -> bool:
        try:
            page_count, _toc = _source_info(path)
        except Exception as exc:  # noqa: BLE001 - 统一提示读取失败
            self._page_count = None
            self.first_page_spin.setRange(1, 99999)
            self._set_status("无法读取 PDF 页数")
            QMessageBox.critical(
                self, "无法打开 PDF", f"读取 PDF 页数失败：\n{exc}"
            )
            return False

        self._page_count = page_count
        self.first_page_spin.setRange(1, max(1, page_count))
        self._set_status(f"已选择：{path.name}（共 {page_count} 页）")
        return True

    def _browse_output(self) -> None:
        current = self.output_edit.text().strip()
        if current:
            start_path = current
        elif self.source_edit.text().strip():
            source = Path(self.source_edit.text().strip())
            start_path = str(source.with_name(f"{source.stem}_annotated.pdf"))
        else:
            start_path = str(Path.home() / "output_annotated.pdf")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出 PDF",
            start_path,
            "PDF 文件 (*.pdf)",
        )
        if not selected:
            return
        output = Path(selected)
        if not output.suffix:
            output = output.with_suffix(".pdf")
        self.output_edit.setText(str(output))
        self._auto_output = None

    def _browse_font(self) -> None:
        current = self.font_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择中文字体文件",
            start_dir,
            "字体文件 (*.ttf *.ttc *.otf);;所有文件 (*)",
        )
        if selected:
            self.font_edit.setText(selected)

    def _copy_ai_prompt(self) -> None:
        QApplication.clipboard().setText(BOOKMARK_AI_PROMPT)
        self._set_status(
            "已复制 AI 提示词。请粘贴到 AI 对话并上传书籍目录图片，"
            "再把返回的 JSON 粘贴到章节配置框。"
        )

    # --------------------------------------------------------------- 处理流程

    def _prepare_job(self):
        """校验界面输入并弹窗确认，返回处理参数；取消或失败时返回 None。"""
        source_text = self.source_edit.text().strip()
        if not source_text:
            QMessageBox.warning(
                self, "缺少源 PDF", "请先选择要处理的源 PDF 文件。"
            )
            return None
        source = Path(source_text).expanduser()
        if not source.is_file():
            QMessageBox.warning(
                self, "源 PDF 不存在", f"找不到文件：\n{source}"
            )
            return None

        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(
                self, "缺少输出路径", "请先选择处理后 PDF 的保存位置。"
            )
            return None
        output = Path(output_text).expanduser()
        if source.resolve() == output.resolve():
            QMessageBox.warning(
                self,
                "输出路径无效",
                "输出文件不能与源 PDF 相同，请选择其他输出路径。",
            )
            return None
        if output.is_dir():
            QMessageBox.warning(
                self, "输出路径无效", f"输出路径是一个目录：\n{output}"
            )
            return None

        page_number_enabled = self.page_number_enabled.isChecked()
        bookmark_enabled = self.bookmark_enabled.isChecked()
        watermark_enabled = self.watermark_enabled.isChecked()
        if not (
            page_number_enabled or bookmark_enabled or watermark_enabled
        ):
            QMessageBox.warning(
                self, "未启用操作", "请至少启用一项操作后再开始处理。"
            )
            return None

        page_number_options: PageNumberOptions | None = None
        if page_number_enabled:
            template = self.template_edit.text()
            try:
                format_page_number(template, 1, 1)
            except ValueError as exc:
                QMessageBox.warning(self, "页码模板错误", str(exc))
                return None
            page_number_options = PageNumberOptions(
                template=template,
                start_number=self.start_number_spin.value(),
                first_page=self.first_page_spin.value(),
                position=self.position_combo.currentText(),
                font_size=float(self.page_font_size_spin.value()),
                margin_mm=float(self.page_margin_spin.value()),
                color=self._page_number_color,
            )

        bookmark_config: list[tuple[str, int]] | None = None
        if bookmark_enabled:
            try:
                bookmark_config = parse_chapter_config(
                    self.bookmark_config_edit.toPlainText(),
                    validate_names=False,
                )
            except ConfigError as exc:
                QMessageBox.warning(self, "章节配置错误", str(exc))
                return None

        watermark_options: WatermarkOptions | None = None
        if watermark_enabled:
            text = self.watermark_text_edit.text()
            if not text.strip():
                QMessageBox.warning(
                    self, "缺少水印文字", "请填写要绘制的水印文字。"
                )
                return None
            watermark_options = WatermarkOptions(
                text=text,
                font_size=float(self.watermark_font_size_spin.value()),
                opacity=self.watermark_opacity_spin.value() / 100.0,
                angle=float(self.watermark_angle_spin.value()),
                color=self._watermark_color,
                tiled=self.watermark_tiled_check.isChecked(),
                gap=float(self.watermark_gap_spin.value()),
            )

        try:
            page_count, toc = _source_info(
                source, with_toc=bookmark_config is not None
            )
            existing_toc = toc or []
        except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
            QMessageBox.critical(self, "无法打开 PDF", str(exc))
            return None

        self._page_count = page_count
        self.first_page_spin.setRange(1, max(1, page_count))

        if (
            page_number_options is not None
            and page_number_options.first_page > page_count
        ):
            QMessageBox.warning(
                self,
                "起始页超出范围",
                f"「从第几页开始」为 {page_number_options.first_page}，"
                f"但源 PDF 只有 {page_count} 页。",
            )
            return None

        offset = self.bookmark_offset_spin.value()
        if bookmark_config is not None:
            entries, _skipped = build_toc_entries(
                bookmark_config, offset, page_count
            )
            if not entries:
                QMessageBox.warning(
                    self,
                    "书签全部越界",
                    f"所有章节换算后的 PDF 页码都不在 1-{page_count} "
                    "范围内，未开始处理。请检查起始页码与偏移。",
                )
                return None
            if existing_toc:
                reply = QMessageBox.question(
                    self,
                    "替换原有书签",
                    f"源 PDF 已有 {len(existing_toc)} 个书签，"
                    "继续处理将替换原有书签。\n\n是否继续？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return None

        if output.exists():
            reply = QMessageBox.question(
                self,
                "输出文件已存在",
                f"文件已存在：\n{output}\n\n是否覆盖？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None

        font_text = self.font_edit.text().strip()
        font_path: Path | str = (
            Path(font_text).expanduser() if font_text else ""
        )
        return (
            source,
            output,
            page_number_options,
            bookmark_config,
            offset,
            watermark_options,
            page_count,
            font_path,
        )

    def _start_processing(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        job = self._prepare_job()
        if job is None:
            return
        (
            source,
            output,
            page_number_options,
            bookmark_config,
            offset,
            watermark_options,
            page_count,
            font_path,
        ) = job

        page_operations = int(page_number_options is not None) + int(
            watermark_options is not None
        )
        total_steps = page_count * page_operations + (
            1 if bookmark_config is not None else 0
        )
        self._last_output = output
        self.progress_bar.setRange(0, max(1, total_steps))
        self.progress_bar.setValue(0)
        self._set_busy(True)

        message = "正在处理…"
        if self._font_may_fall_back():
            message += f"（{FONT_FALLBACK_HINT}）"
        self._set_status(message)

        worker = SingleTaskWorker(
            lambda on_progress, cancel_event: process_pdf(
                source,
                output,
                page_number_options,
                bookmark_config,
                offset,
                watermark_options,
                font_path=font_path,
                cancel_event=cancel_event,
                on_progress=on_progress,
            ),
            self,
        )
        worker.progress_changed.connect(self._on_progress)
        worker.task_finished.connect(self._on_finished)
        worker.task_cancelled.connect(self._on_cancelled)
        worker.task_failed.connect(self._on_failed)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        worker.start()

    def _on_progress(self, done: object, total: object) -> None:
        try:
            completed = int(done)
            total_steps = max(1, int(total))
        except (TypeError, ValueError):
            return
        completed = max(0, min(completed, total_steps))
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(completed)
        message = f"正在处理：{completed}/{total_steps} 项…"
        if self._font_may_fall_back():
            message += f"（{FONT_FALLBACK_HINT}）"
        self._set_status(message)

    def _on_finished(self, result: object) -> None:
        self.progress_bar.setValue(self.progress_bar.maximum())
        if not isinstance(result, PageToolsResult):
            self._set_status("处理完成")
            QMessageBox.information(self, "处理完成", "处理已完成。")
            return

        self._last_output = result.output
        font_used = (
            self.page_number_enabled.isChecked()
            or self.watermark_enabled.isChecked()
        )
        summary_parts = [
            f"页码 {result.page_numbers} 页",
            f"书签 {result.bookmarks} 个",
            f"水印 {result.watermarks} 页",
        ]
        if result.skipped_bookmarks:
            summary_parts.append(
                f"跳过书签 {len(result.skipped_bookmarks)} 个"
            )
        self._set_status(
            "完成：" + "，".join(summary_parts) + f" → {result.output}"
        )
        if font_used and not result.font_embedded:
            self._set_status(
                self.status_label.text() + f"；{FONT_FALLBACK_HINT}"
            )

        message_lines = [
            f"输出文件：{result.output}",
            "",
            f"已绘制页码：{result.page_numbers} 页",
            f"已添加书签：{result.bookmarks} 个",
            f"已绘制水印：{result.watermarks} 页",
        ]
        if result.skipped_bookmarks:
            message_lines.append("")
            message_lines.append(
                f"以下 {len(result.skipped_bookmarks)} 个章节越界，已跳过："
            )
            message_lines.extend(
                f"- {title}（PDF 页码 {page}）"
                for title, page in result.skipped_bookmarks
            )
        if font_used and not result.font_embedded:
            message_lines.extend(
                ["", f"注意：{FONT_FALLBACK_HINT}。"]
            )
        QMessageBox.information(
            self, "处理完成", "\n".join(message_lines)
        )

    def _on_cancelled(self, _result: object) -> None:
        self.progress_bar.setValue(0)
        self._set_status("已取消处理，未保存输出文件")
        QMessageBox.information(
            self, "已取消", "处理已取消，未保存输出文件。"
        )

    def _on_failed(self, message: str) -> None:
        self._set_status("处理失败")
        QMessageBox.critical(self, "处理失败", message)

    def _on_thread_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_processing(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._set_status("正在取消…")

    def _open_output_dir(self) -> None:
        if self._last_output is None:
            return
        folder = self._last_output.parent
        if not folder.is_dir():
            QMessageBox.critical(
                self, "目录不存在", f"输出目录不存在：\n{folder}"
            )
            return
        try:
            open_directory(folder)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(
                self, "无法打开目录", f"{folder}\n{exc}"
            )

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；处理进行中时确认、取消并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "PDF 处理")
