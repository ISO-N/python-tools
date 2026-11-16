"""图片拼接工具页（PyQt6 界面）。

把图片文件夹（可递归）中的图片按文件名自然顺序拼接为纵向长图，或排成
网格拼图。第一遍只读取图片尺寸并排除损坏 / 动画图片，第二遍逐张解码粘贴，
内存中只保留画布和当前图片。本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import open_directory
from toolbox.images import (
    fit_size,
    image_has_alpha,
    is_animated,
    load_oriented,
    probe_image,
    save_image_atomic,
    scan_image_files,
)
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import (
    Cancelled,
    MAX_IMAGE_WORKERS,
    SingleTaskWorker,
    shutdown_worker,
)

MAX_CANVAS_PIXELS = 200_000_000
MAX_CANVAS_SIDE = 30_000
# 超大画布本身已占用大量内存，并行解码只会推高内存峰值，强制串行。
COLLAGE_PARALLEL_PIXEL_LIMIT = 50_000_000
MAX_WEBP_SIDE = 16_383
TARGET_WIDTH_MIN, TARGET_WIDTH_MAX = 200, 20_000
COLUMNS_MIN, COLUMNS_MAX = 1, 12
SPACING_MIN, SPACING_MAX = 0, 200
MARGIN_MIN, MARGIN_MAX = 0, 200
QUALITY_MIN, QUALITY_MAX = 1, 95

_FORMAT_SUFFIXES = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
_KNOWN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


@dataclass
class CollageOptions:
    """拼图选项。"""

    mode: str = "long"
    target_width: int = 1200
    columns: int = 3
    spacing: int = 10
    margin: int = 10
    background: tuple[int, int, int] = (255, 255, 255)
    upscale: bool = True
    format: str = "png"
    quality: int = 90


@dataclass
class CollageLayout:
    """拼图画布尺寸与每张图片的放置区域 (x, y, w, h)。"""

    width: int
    height: int
    placements: list[tuple[int, int, int, int]]


@dataclass
class CollageResult:
    """一次拼接的结果。"""

    output: Path
    count: int
    skipped: list[tuple[Path, str]]
    width: int
    height: int


@dataclass
class _TileOutcome:
    """第二遍解码并缩放后的瓦片；``image`` 由消费方负责关闭。"""

    image: Image.Image | None
    reason: str | None = None


# --------------------------------------------------------------------- 纯逻辑


def _validate_options(options: CollageOptions) -> None:
    if options.mode not in {"long", "grid"}:
        raise ValueError(f"不支持的拼接模式：{options.mode}")
    if options.format.lower() not in {"png", "jpeg", "webp"}:
        raise ValueError(f"不支持的输出格式：{options.format}")
    if options.target_width <= 0:
        raise ValueError("目标宽度必须大于 0")
    if options.margin < 0 or options.spacing < 0:
        raise ValueError("边距和间距不能为负数")
    if options.mode == "grid" and not (
        COLUMNS_MIN <= options.columns <= COLUMNS_MAX
    ):
        raise ValueError(
            f"网格列数必须在 {COLUMNS_MIN}-{COLUMNS_MAX} 之间"
        )
    if not QUALITY_MIN <= options.quality <= QUALITY_MAX:
        raise ValueError(
            f"图片质量必须在 {QUALITY_MIN}-{QUALITY_MAX} 之间"
        )
    if len(options.background) != 3 or any(
        not 0 <= int(value) <= 255 for value in options.background
    ):
        raise ValueError("背景颜色必须是三个 0-255 的整数")


def _validate_sizes(sizes: Sequence[tuple[int, int]]) -> None:
    for width, height in sizes:
        if width <= 0 or height <= 0:
            raise ValueError(f"图片尺寸无效：{width}×{height}")


def _usable_width(options: CollageOptions) -> int:
    usable = options.target_width - 2 * options.margin
    if usable < 1:
        raise ValueError("目标宽度过小或边距过大，画布没有可用宽度")
    return usable


def _validate_canvas(layout: CollageLayout, options: CollageOptions) -> None:
    if layout.width <= 0 or layout.height <= 0:
        raise ValueError(f"画布尺寸无效：{layout.width}×{layout.height}")
    longest_side = max(layout.width, layout.height)
    if (
        options.format.lower() == "webp"
        and longest_side > MAX_WEBP_SIDE
    ):
        raise ValueError(
            "WebP 格式的图片单边不能超过 16383 像素，"
            "请改用 PNG 或 JPEG。"
            f"（当前画布 {layout.width}×{layout.height}）"
        )
    if longest_side > MAX_CANVAS_SIDE:
        raise ValueError(
            f"画布单边过大（{layout.width}×{layout.height}），"
            f"不能超过 {MAX_CANVAS_SIDE} 像素。"
        )
    total_pixels = layout.width * layout.height
    if total_pixels > MAX_CANVAS_PIXELS:
        raise ValueError(
            f"画布总像素过多（{total_pixels:,}），"
            f"不能超过 {MAX_CANVAS_PIXELS:,} 像素。"
        )


def compute_long_layout(
    sizes: Sequence[tuple[int, int]], options: CollageOptions
) -> CollageLayout:
    """计算纵向长图布局：所有图片缩放到可用宽并逐张向下排列。"""
    _validate_options(options)
    _validate_sizes(sizes)
    usable = _usable_width(options)

    placements: list[tuple[int, int, int, int]] = []
    y = options.margin
    for width, height in sizes:
        scale = usable / width
        if not options.upscale:
            scale = min(scale, 1.0)
        image_width = max(1, round(width * scale))
        image_height = max(1, round(height * scale))
        x = options.margin + (usable - image_width) // 2
        placements.append((x, y, image_width, image_height))
        y += image_height + options.spacing

    if sizes:
        canvas_height = (
            2 * options.margin
            + sum(item[3] for item in placements)
            + (len(placements) - 1) * options.spacing
        )
    else:
        canvas_height = 2 * options.margin

    layout = CollageLayout(
        width=options.target_width,
        height=max(1, canvas_height),
        placements=placements,
    )
    _validate_canvas(layout, options)
    return layout


def compute_grid_layout(
    sizes: Sequence[tuple[int, int]], options: CollageOptions
) -> CollageLayout:
    """计算网格拼图布局：每格 contain 居中，按列数自动换行。"""
    _validate_options(options)
    _validate_sizes(sizes)
    usable = _usable_width(options)
    columns = options.columns
    available = usable - (columns - 1) * options.spacing
    if available < columns:
        raise ValueError("间距或列数过大，网格单元格宽度不足")
    cell = available / columns

    rows = math.ceil(len(sizes) / columns) if sizes else 0
    canvas_height = int(
        round(
            2 * options.margin
            + rows * cell
            + max(0, rows - 1) * options.spacing
        )
    )

    placements: list[tuple[int, int, int, int]] = []
    for index, (width, height) in enumerate(sizes):
        row, column = divmod(index, columns)
        cell_x = options.margin + column * (cell + options.spacing)
        cell_y = options.margin + row * (cell + options.spacing)
        image_width, image_height = fit_size(
            width, height, cell, cell, upscale=options.upscale
        )
        x = round(cell_x + (cell - image_width) / 2)
        y = round(cell_y + (cell - image_height) / 2)
        placements.append((x, y, image_width, image_height))

    layout = CollageLayout(
        width=options.target_width,
        height=max(1, canvas_height),
        placements=placements,
    )
    _validate_canvas(layout, options)
    return layout


def _probe_image(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    """一次打开文件头读取尺寸并检测动画；失败时返回（None, 中文原因）。"""
    try:
        probe = probe_image(path)
    except Exception as exc:  # noqa: BLE001 - 单张损坏时继续处理
        return None, f"无法读取：{exc}"
    if probe.frames > 1:
        return None, "动画图片，已跳过"
    if probe.width <= 0 or probe.height <= 0:
        return None, "图片尺寸无效"
    return (probe.width, probe.height), None


def _paste_ready(
    canvas: Image.Image,
    source: Image.Image,
    x: int,
    y: int,
) -> None:
    """把已缩放到目标尺寸的图片粘贴到画布，透明区域透出画布背景。

    ``source`` 只读不修改；透明图优先复用 RGBA 源。主线程调用时只读取
    瓦片，绝不跨线程共享可变的 PIL 图层。
    """
    if image_has_alpha(source):
        rgba = source if source.mode == "RGBA" else source.convert("RGBA")
        try:
            alpha = rgba.getchannel("A")
            try:
                canvas.paste(rgba, (x, y), alpha)
            finally:
                alpha.close()
        finally:
            if rgba is not source:
                rgba.close()
    elif source.mode == "RGB":
        canvas.paste(source, (x, y))
    else:
        rgb = source.convert("RGB")
        try:
            canvas.paste(rgb, (x, y))
        finally:
            rgb.close()


def _paste_fitted(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    """把图片缩放到 box 指定尺寸后粘贴，透明区域透出画布背景。

    源图只读不修改：RGB 直接粘贴；透明图优先复用 RGBA 源，缩放时使用
    ``reducing_gap=2.0`` 降低大图缩小的开销。
    """
    x, y, width, height = box
    resized: Image.Image | None = None
    source = image
    if image.size != (width, height):
        resized = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
            reducing_gap=2.0,
        )
        source = resized
    try:
        _paste_ready(canvas, source, x, y)
    finally:
        if resized is not None:
            resized.close()


def _prepare_tile(
    path: Path,
    box: tuple[int, int, int, int],
    cancel_event: threading.Event | None,
) -> _TileOutcome:
    """在工作线程中解码并缩放到目标尺寸，返回可交给主线程粘贴的瓦片。

    返回的 ``image`` 所有权转移给调用方；异常或动画图片会在此关闭图片。
    """
    width, height = box[2], box[3]
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    image = load_oriented(path, max_size=(width, height))
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        if is_animated(image):
            image.close()
            return _TileOutcome(None, "动画图片，已跳过")
        if image.size != (width, height):
            resized = image.resize(
                (width, height),
                Image.Resampling.LANCZOS,
                reducing_gap=2.0,
            )
            image.close()
            image = resized
        return _TileOutcome(image)
    except BaseException:
        image.close()
        raise


def _save_format(options: CollageOptions) -> tuple[str, dict[str, object]]:
    fmt = options.format.lower()
    if fmt == "jpeg":
        return "JPEG", {"quality": int(options.quality)}
    if fmt == "webp":
        return "WEBP", {"quality": int(options.quality)}
    return "PNG", {}


def build_collage(
    images: Sequence[Path | str],
    output: Path | str,
    options: CollageOptions,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> CollageResult:
    """把图片拼接为一张长图或网格图，返回结果与跳过项。

    第一遍只读取尺寸并排除损坏 / 动画图片；第二遍逐张解码粘贴。
    取消时抛出 ``Cancelled``，其 ``result`` 为已完成部分的结果。
    """
    _validate_options(options)
    output = Path(output)
    image_paths = [Path(path) for path in images]
    if not image_paths:
        raise ValueError("没有可拼接的图片")
    if output.exists() and output.is_dir():
        raise ValueError(f"输出路径是一个目录：{output}")

    output_resolved = output.resolve()
    for path in image_paths:
        try:
            if path.resolve() == output_resolved:
                raise ValueError(f"输出文件不能覆盖源图片：{path}")
        except OSError:
            continue

    skipped: list[tuple[Path, str]] = []
    valid_paths: list[Path] = []
    sizes: list[tuple[int, int]] = []
    total_files = len(image_paths)
    total_steps = max(1, total_files * 2)

    for index, path in enumerate(image_paths):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(
                CollageResult(output, 0, skipped, 0, 0)
            )
        size, reason = _probe_image(path)
        if size is None:
            skipped.append((path, reason or "无法读取"))
        else:
            valid_paths.append(path)
            sizes.append(size)
        if on_progress is not None:
            on_progress(index + 1, total_steps)

    if not valid_paths:
        sample = ""
        if skipped:
            first_path, first_reason = skipped[0]
            sample = f"\n示例：{first_path.name}：{first_reason}"
        raise ValueError(f"没有可拼接的有效图片。{sample}")

    if options.mode == "long":
        layout = compute_long_layout(sizes, options)
    else:
        layout = compute_grid_layout(sizes, options)

    canvas = Image.new(
        "RGB", (layout.width, layout.height), options.background
    )
    count = 0
    workers = max(1, int(MAX_IMAGE_WORKERS))
    if layout.width * layout.height > COLLAGE_PARALLEL_PIXEL_LIMIT:
        workers = 1
    window = max(1, workers * 2)
    valid_total = len(valid_paths)

    def cancelled_result() -> Cancelled:
        return Cancelled(
            CollageResult(
                output,
                count,
                skipped,
                layout.width,
                layout.height,
            )
        )

    executor = ThreadPoolExecutor(max_workers=workers)
    pending: dict[int, Future[_TileOutcome]] = {}
    next_submit = 0
    try:
        try:
            for index, (path, box) in enumerate(
                zip(valid_paths, layout.placements)
            ):
                if cancel_event is not None and cancel_event.is_set():
                    raise cancelled_result()

                # 预取窗口限制在 workers*2，避免所有原图同时驻留内存。
                while len(pending) < window and next_submit < valid_total:
                    if cancel_event is not None and cancel_event.is_set():
                        raise cancelled_result()
                    pending[next_submit] = executor.submit(
                        _prepare_tile,
                        valid_paths[next_submit],
                        layout.placements[next_submit],
                        cancel_event,
                    )
                    next_submit += 1

                future = pending[index]
                while not future.done():
                    if cancel_event is not None and cancel_event.is_set():
                        raise cancelled_result()
                    wait((future,), timeout=0.1)

                try:
                    outcome = future.result()
                except Cancelled:
                    pending.pop(index, None)
                    raise cancelled_result() from None
                except Exception as exc:  # noqa: BLE001 - 单张失败时继续
                    pending.pop(index, None)
                    skipped.append((path, f"读取失败：{exc}"))
                else:
                    pending.pop(index, None)
                    tile = outcome.image
                    if tile is None:
                        skipped.append((path, outcome.reason or "无法读取"))
                    else:
                        try:
                            # 主线程按布局顺序只读粘贴，瓦片归主线程关闭。
                            _paste_ready(canvas, tile, box[0], box[1])
                            count += 1
                        except Exception as exc:  # noqa: BLE001 - 单张继续
                            skipped.append((path, f"读取失败：{exc}"))
                        finally:
                            tile.close()

                if on_progress is not None:
                    # 读取尺寸与绘制各占一半进度，跳过项不影响最终到达 100%。
                    paste_step = round(
                        (index + 1) * total_files / valid_total
                    )
                    on_progress(total_files + paste_step, total_steps)

            if count == 0:
                raise RuntimeError("所有图片都无法读取，未生成拼接图。")

            image_format, save_kwargs = _save_format(options)
            save_image_atomic(
                canvas,
                output,
                image_format=image_format,
                **save_kwargs,
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            # 取消时可能已有完成但尚未粘贴的瓦片，逐个关闭避免泄漏。
            for future in pending.values():
                if future.cancelled():
                    continue
                try:
                    outcome = future.result()
                except BaseException:  # noqa: BLE001 - 失败任务没有瓦片
                    continue
                if outcome.image is not None:
                    outcome.image.close()
            pending.clear()
    finally:
        canvas.close()

    return CollageResult(
        output=output,
        count=count,
        skipped=skipped,
        width=layout.width,
        height=layout.height,
    )


# ------------------------------------------------------------------- 界面部分


class ImageCollagePage(ToolPage):
    shutdown_label = "拼接"
    """图片拼接工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._last_output: Path | None = None
        self._background = (255, 255, 255)
        self._input_widgets: list[QWidget] = []
        self._browse_buttons: list[QWidget] = []
        self._build_ui()
        self._sync_mode_widgets()
        self._sync_quality()
        self._update_background_button()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_MARGINS)
        layout.setSpacing(10)

        form = make_form()

        self.folder_edit = add_path_row(
            form,
            0,
            "图片文件夹：",
            mode="open_dir",
            placeholder="选择要拼接的图片文件夹",
            dialog_title="选择图片文件夹",
            parent=self,
        )
        self.output_edit = add_path_row(
            form,
            1,
            "输出文件：",
            mode="save_file",
            placeholder="选择拼接结果的保存位置",
            dialog_title="保存拼接图片",
            file_filter="图片文件 (*.png *.jpg *.jpeg *.webp);;所有文件 (*)",
            parent=self,
        )
        for row in (0, 1):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                self._browse_buttons.append(item.widget())

        self.recursive_check = QCheckBox("递归包含子文件夹")
        form.addWidget(QLabel("扫描选项："), 2, 0)
        form.addWidget(self.recursive_check, 2, 1, 1, 2)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("纵向长图", "long")
        self.mode_combo.addItem("网格拼图", "grid")
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_widgets)
        mode_row.addWidget(self.mode_combo)
        mode_row.addSpacing(20)
        mode_row.addWidget(QLabel("目标宽度："))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(TARGET_WIDTH_MIN, TARGET_WIDTH_MAX)
        self.width_spin.setSingleStep(100)
        self.width_spin.setValue(1200)
        self.width_spin.setSuffix(" px")
        mode_row.addWidget(self.width_spin)
        mode_row.addSpacing(20)
        mode_row.addWidget(QLabel("列数："))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(COLUMNS_MIN, COLUMNS_MAX)
        self.columns_spin.setValue(3)
        mode_row.addWidget(self.columns_spin)
        mode_row.addStretch(1)
        form.addWidget(QLabel("拼接模式："), 3, 0)
        form.addLayout(mode_row, 3, 1, 1, 2)

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("间距："))
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(SPACING_MIN, SPACING_MAX)
        self.spacing_spin.setValue(10)
        self.spacing_spin.setSuffix(" px")
        layout_row.addWidget(self.spacing_spin)
        layout_row.addSpacing(20)
        layout_row.addWidget(QLabel("边距："))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(MARGIN_MIN, MARGIN_MAX)
        self.margin_spin.setValue(10)
        self.margin_spin.setSuffix(" px")
        layout_row.addWidget(self.margin_spin)
        layout_row.addSpacing(20)
        layout_row.addWidget(QLabel("背景色："))
        self.background_btn = QPushButton()
        self.background_btn.setToolTip("选择拼图背景颜色")
        self.background_btn.clicked.connect(self._choose_background)
        layout_row.addWidget(self.background_btn)
        layout_row.addSpacing(20)
        self.upscale_check = QCheckBox("允许放大图片")
        self.upscale_check.setChecked(True)
        layout_row.addWidget(self.upscale_check)
        layout_row.addStretch(1)
        form.addWidget(QLabel("布局设置："), 4, 0)
        form.addLayout(layout_row, 4, 1, 1, 2)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("输出格式："))
        self.format_combo = QComboBox()
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("JPEG", "jpeg")
        self.format_combo.addItem("WebP", "webp")
        self.format_combo.currentIndexChanged.connect(
            self._on_format_changed
        )
        format_row.addWidget(self.format_combo)
        format_row.addSpacing(20)
        format_row.addWidget(QLabel("图片质量："))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(QUALITY_MIN, QUALITY_MAX)
        self.quality_spin.setValue(90)
        format_row.addWidget(self.quality_spin)
        format_row.addWidget(QLabel("（仅 JPEG / WebP）"))
        format_row.addStretch(1)
        form.addWidget(QLabel("输出设置："), 5, 0)
        form.addLayout(format_row, 5, 1, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 6, 1, 1, 2)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始拼接")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_build)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_build)
        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output_dir)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 7, 1, 1, 2)

        layout.addLayout(form)
        layout.addStretch(1)

        self.status_label = QLabel("请选择图片文件夹并设置输出文件")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.folder_edit,
            self.output_edit,
            *self._browse_buttons,
            self.recursive_check,
            self.mode_combo,
            self.width_spin,
            self.columns_spin,
            self.spacing_spin,
            self.margin_spin,
            self.background_btn,
            self.upscale_check,
            self.format_combo,
            self.quality_spin,
        ]

    def _current_format(self) -> str:
        data = self.format_combo.currentData()
        return str(data) if data is not None else "png"

    def _current_mode(self) -> str:
        data = self.mode_combo.currentData()
        return str(data) if data is not None else "long"

    def _sync_mode_widgets(self, *_args: object) -> None:
        is_grid = self._current_mode() == "grid"
        self.columns_spin.setEnabled(is_grid and not self._busy)

    def _sync_quality(self, *_args: object) -> None:
        self.quality_spin.setEnabled(
            self._current_format() != "png" and not self._busy
        )

    def _on_format_changed(self, *_args: object) -> None:
        self._sync_quality()
        text = self.output_edit.text().strip()
        if not text:
            return
        path = Path(text)
        suffix = _FORMAT_SUFFIXES[self._current_format()]
        lowered = path.suffix.lower()
        if lowered in {".jpg", ".jpeg"} and suffix == ".jpg":
            return
        if lowered == suffix or not lowered:
            if not lowered:
                try:
                    path = path.with_suffix(suffix)
                except ValueError:
                    return
                self.output_edit.setText(str(path))
            return
        if lowered in _KNOWN_IMAGE_SUFFIXES:
            self.output_edit.setText(str(path.with_suffix(suffix)))

    def _update_background_button(self) -> None:
        red, green, blue = self._background
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        foreground = "#000000" if luminance > 150 else "#ffffff"
        self.background_btn.setText(f"#{red:02X}{green:02X}{blue:02X}")
        self.background_btn.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({red}, {green}, {blue});"
            f"color: {foreground};"
            "border: 1px solid #999; padding: 2px 10px;"
            "}"
        )

    def _choose_background(self) -> None:
        selected = QColorDialog.getColor(
            QColor(*self._background), self, "选择拼图背景颜色"
        )
        if not selected.isValid():
            return
        self._background = (
            selected.red(),
            selected.green(),
            selected.blue(),
        )
        self._update_background_button()

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in self._input_widgets:
            widget.setEnabled(not busy)
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if busy:
            self.open_btn.setEnabled(False)
        else:
            self._sync_mode_widgets()
            self._sync_quality()

    # --------------------------------------------------------------- 拼接流程

    def _collect_inputs(
        self,
    ) -> tuple[list[Path], Path, CollageOptions] | None:
        """校验界面输入并扫描图片；失败时弹窗并返回 None。"""
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            QMessageBox.warning(
                self, "缺少图片文件夹", "请先选择要拼接的图片文件夹。"
            )
            return None
        folder = Path(folder_text).expanduser()
        if not folder.is_dir():
            QMessageBox.warning(
                self, "文件夹不存在", f"目录不存在：\n{folder}"
            )
            return None

        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(
                self, "缺少输出文件", "请先选择拼接结果的保存位置。"
            )
            return None
        output = Path(output_text).expanduser()
        if output.exists() and output.is_dir():
            QMessageBox.warning(
                self,
                "输出路径无效",
                f"输出路径是一个目录，请选择文件路径：\n{output}",
            )
            return None

        output_format = self._current_format()
        suffix = _FORMAT_SUFFIXES[output_format]
        allowed_suffixes = (
            {".jpg", ".jpeg"} if output_format == "jpeg" else {suffix}
        )
        if output.suffix.lower() not in allowed_suffixes:
            output = output.with_suffix(suffix)
            self.output_edit.setText(str(output))

        try:
            images = scan_image_files(
                folder, recursive=self.recursive_check.isChecked()
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "无法读取文件夹", f"{folder}\n{exc}"
            )
            return None
        if not images:
            detail = ""
            if not self.recursive_check.isChecked():
                detail = "\n如需包含子文件夹，请勾选「递归包含子文件夹」。"
            QMessageBox.warning(
                self,
                "没有可拼接的图片",
                "该文件夹中未找到支持的图片文件"
                "（JPG、PNG、WebP、BMP、GIF、TIFF）。" + detail,
            )
            return None

        output_resolved = output.resolve()
        for image in images:
            try:
                if image.resolve() == output_resolved:
                    QMessageBox.warning(
                        self,
                        "输出路径无效",
                        "输出文件不能覆盖源图片，请换一个输出路径。",
                    )
                    return None
            except OSError:
                continue

        if output.exists():
            reply = QMessageBox.question(
                self,
                "文件已存在",
                f"输出文件已存在：\n{output}\n\n"
                "继续将覆盖该文件，是否继续？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None

        options = CollageOptions(
            mode=self._current_mode(),
            target_width=int(self.width_spin.value()),
            columns=int(self.columns_spin.value()),
            spacing=int(self.spacing_spin.value()),
            margin=int(self.margin_spin.value()),
            background=self._background,
            upscale=self.upscale_check.isChecked(),
            format=output_format,
            quality=int(self.quality_spin.value()),
        )
        return images, output, options

    def _start_build(self) -> None:
        if self._busy:
            return
        collected = self._collect_inputs()
        if collected is None:
            return
        images, output, options = collected

        self._last_output = output
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(
            f"正在拼接 {len(images)} 张图片（读取尺寸并绘制）…"
        )

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> CollageResult:
            return build_collage(
                images,
                output,
                options,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_progress)
        worker.task_finished.connect(self._on_finished)
        worker.task_cancelled.connect(self._on_cancelled)
        worker.task_failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_progress(self, done: object, total: object) -> None:
        done_count = int(done)
        total_count = int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(percent)
        self._set_status(f"正在拼接…（{percent}%）")

    def _on_finished(self, result: object) -> None:
        if not isinstance(result, CollageResult):
            self._on_failed("任务返回了无法识别的结果。")
            return

        self.progress_bar.setValue(100)
        self.open_btn.setEnabled(True)
        self._last_output = result.output
        skipped_count = len(result.skipped)
        dimension_text = f"{result.width} × {result.height} 像素"
        if not skipped_count:
            self._set_status(
                f"完成：已拼接 {result.count} 张，尺寸 {dimension_text}"
                f" → {result.output}"
            )
            QMessageBox.information(
                self,
                "拼接完成",
                f"已拼接 {result.count} 张图片。\n"
                f"尺寸：{dimension_text}\n"
                f"输出：{result.output}",
            )
            return

        preview = "\n".join(
            f"{path.name}：{reason}"
            for path, reason in result.skipped[:5]
        )
        if skipped_count > 5:
            preview += f"\n…另有 {skipped_count - 5} 张"
        self._set_status(
            f"完成：已拼接 {result.count} 张，尺寸 {dimension_text}，"
            f"跳过 {skipped_count} 张 → {result.output}"
        )
        QMessageBox.warning(
            self,
            "拼接完成（有跳过项）",
            f"已拼接 {result.count} 张图片。\n"
            f"尺寸：{dimension_text}\n"
            f"输出：{result.output}\n\n"
            f"跳过 {skipped_count} 张：\n{preview}",
        )

    def _on_cancelled(self, result: object) -> None:
        if isinstance(result, CollageResult):
            detail = f"已拼接 {result.count} 张"
        else:
            detail = "已处理 0 张"
        self._set_status(f"已取消：{detail}，未生成输出文件")

    def _on_failed(self, message: str) -> None:
        self._set_status("拼接失败")
        QMessageBox.critical(self, "拼接失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_build(self) -> None:
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
            QMessageBox.critical(self, "无法打开目录", f"{folder}\n{exc}")

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；拼接进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "拼接")
