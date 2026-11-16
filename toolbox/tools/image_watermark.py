"""图片文字水印工具页（PyQt6 界面）。

批量给图片添加文字水印，支持九宫格位置与平铺两种排布。中文字体文件由
用户指定或自动查找；处理在后台线程中逐张进行，输出目录与输入目录自动
隔离，绝不覆盖源图片。本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import find_cjk_font_file, open_directory
from toolbox.images import load_oriented, save_image_atomic, scan_image_files
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import BatchResult, BatchWorker, shutdown_worker

FONT_SIZE_MIN, FONT_SIZE_MAX = 6, 200
OPACITY_MIN, OPACITY_MAX = 1, 100
MARGIN_MIN, MARGIN_MAX = 0, 200
GAP_MIN, GAP_MAX = 10, 500
ANGLE_MIN, ANGLE_MAX = -180, 180

POSITIONS = (
    "左上",
    "上中",
    "右上",
    "左中",
    "居中",
    "右中",
    "左下",
    "下中",
    "右下",
)
TILE_POSITION = "平铺"
POSITION_CHOICES = (*POSITIONS, TILE_POSITION)

_ALIGNMENT = {
    "左上": ("left", "top"),
    "上中": ("center", "top"),
    "右上": ("right", "top"),
    "左中": ("left", "middle"),
    "居中": ("center", "middle"),
    "右中": ("right", "middle"),
    "左下": ("left", "bottom"),
    "下中": ("center", "bottom"),
    "右下": ("right", "bottom"),
}

_SUPPORTED_OUTPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class WatermarkOptions:
    """水印选项。"""

    text: str = ""
    position: str = "左上"
    font_path: str = ""
    font_size: int = 36
    color: tuple[int, int, int] = (128, 128, 128)
    opacity: float = 0.25
    angle: int = 0
    margin: int = 20
    gap: int = 60


@dataclass
class WatermarkResult:
    """单张图片的水印处理结果。"""

    output: Path
    before: int
    after: int
    width: int
    height: int


# --------------------------------------------------------------------- 纯逻辑


def _validate_options(options: WatermarkOptions) -> None:
    if not options.text.strip():
        raise ValueError("水印文字不能为空")
    if options.position not in POSITION_CHOICES:
        raise ValueError(f"不支持的水印位置：{options.position}")
    if options.font_size <= 0:
        raise ValueError("字体大小必须大于 0")
    if len(options.color) != 3 or any(
        not 0 <= int(value) <= 255 for value in options.color
    ):
        raise ValueError("水印颜色必须是三个 0-255 的整数")
    if not 0.0 <= float(options.opacity) <= 1.0:
        raise ValueError("透明度必须在 0-1 之间")
    if options.margin < 0:
        raise ValueError("边距不能为负数")
    if options.gap < 0:
        raise ValueError("间距不能为负数")


def load_watermark_font(
    options: WatermarkOptions,
) -> ImageFont.FreeTypeFont:
    """加载 ``options.font_path`` 指定的字体；失败时抛出中文 ValueError。"""
    path_text = (options.font_path or "").strip()
    if not path_text:
        raise ValueError(
            "未指定字体文件，请选择支持中文的 .ttf / .ttc / .otf 字体。"
        )
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise ValueError(f"字体文件不存在：\n{path}")
    if options.font_size <= 0:
        raise ValueError("字体大小必须大于 0")
    try:
        return ImageFont.truetype(str(path), options.font_size, index=0)
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(
            f"无法加载字体文件：\n{path}\n{exc}"
        ) from exc


def render_text_layer(
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    angle: float,
) -> Image.Image:
    """把文字渲染到透明 RGBA 图层上并旋转，返回由调用方关闭的新图片。

    使用 ``ImageDraw.textbbox`` 计算文字实际外框，四周各留 2 像素边距，
    再按 ``angle`` 以双三次插值旋转并扩展画布。
    """
    if not text.strip():
        raise ValueError("水印文字不能为空")
    if len(color) != 3:
        raise ValueError("水印颜色必须是三个 0-255 的整数")

    probe = Image.new("L", (1, 1))
    try:
        probe_draw = ImageDraw.Draw(probe)
        bbox = probe_draw.textbbox((0, 0), text, font=font)
    finally:
        probe.close()

    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    layer = Image.new("RGBA", (width + 4, height + 4), (0, 0, 0, 0))
    try:
        draw = ImageDraw.Draw(layer)
        draw.text(
            (2 - bbox[0], 2 - bbox[1]),
            text,
            font=font,
            fill=(int(color[0]), int(color[1]), int(color[2]), 255),
        )
        rotated = layer.rotate(
            angle,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )
    finally:
        layer.close()
    return rotated


def _stamp(
    canvas: Image.Image,
    layer: Image.Image,
    x: int,
    y: int,
) -> None:
    """把图层按 ``Image.alpha_composite`` 盖到画布，自动裁剪越界部分。"""
    left = max(0, x)
    top = max(0, y)
    right = min(canvas.width, x + layer.width)
    bottom = min(canvas.height, y + layer.height)
    if right <= left or bottom <= top:
        return
    region = layer.crop((left - x, top - y, right - x, bottom - y))
    try:
        canvas.alpha_composite(region, (left, top))
    finally:
        region.close()


def _position_origin(
    position: str,
    canvas_size: tuple[int, int],
    layer_size: tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    horizontal, vertical = _ALIGNMENT[position]
    canvas_width, canvas_height = canvas_size
    layer_width, layer_height = layer_size
    if horizontal == "left":
        x = margin
    elif horizontal == "center":
        x = (canvas_width - layer_width) // 2
    else:
        x = canvas_width - layer_width - margin

    if vertical == "top":
        y = margin
    elif vertical == "middle":
        y = (canvas_height - layer_height) // 2
    else:
        y = canvas_height - layer_height - margin
    return x, y


def apply_watermark(
    image: Image.Image,
    options: WatermarkOptions,
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """返回添加水印后的新 RGBA 图片，原图保持不变。"""
    _validate_options(options)
    layer = render_text_layer(
        options.text, font, options.color, options.angle
    )
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    try:
        if options.opacity < 1.0:
            alpha = layer.getchannel("A")
            try:
                adjusted = alpha.point(
                    lambda value: int(value * options.opacity)
                )
            finally:
                alpha.close()
            try:
                layer.putalpha(adjusted)
            finally:
                adjusted.close()

        if options.position == TILE_POSITION:
            step_x = layer.width + options.gap
            step_y = layer.height + options.gap
            y = options.margin
            while y < base.height:
                x = options.margin
                while x < base.width:
                    _stamp(overlay, layer, x, y)
                    x += step_x
                y += step_y
        else:
            x, y = _position_origin(
                options.position, base.size, layer.size, options.margin
            )
            _stamp(overlay, layer, x, y)

        return Image.alpha_composite(base, overlay)
    finally:
        overlay.close()
        layer.close()
        if base is not image:
            base.close()


def _output_format_for(src: Path) -> tuple[str, dict[str, object]]:
    suffix = src.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG", {"quality": 92}
    if suffix == ".webp":
        return "WEBP", {"quality": 90}
    return "PNG", {}


def watermark_one(
    src: Path | str,
    dst: Path | str,
    options: WatermarkOptions,
    font: ImageFont.FreeTypeFont,
) -> WatermarkResult:
    """处理单张图片并保存到 dst，返回大小与尺寸信息。

    动画图片抛出 ``ValueError("动画图片，已跳过")``；输出目录自动创建。
    """
    _validate_options(options)
    src = Path(src)
    dst = Path(dst)
    try:
        same_file = src.resolve() == dst.resolve()
    except OSError:
        same_file = src.absolute() == dst.absolute()
    if same_file:
        raise ValueError("输出文件不能覆盖源图片")

    with Image.open(src) as probe:
        if getattr(probe, "n_frames", 1) > 1:
            raise ValueError("动画图片，已跳过")

    before = src.stat().st_size
    image = load_oriented(src)
    try:
        processed = apply_watermark(image, options, font)
        try:
            width, height = processed.width, processed.height
            image_format, save_kwargs = _output_format_for(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_image_atomic(
                processed,
                dst,
                image_format=image_format,
                **save_kwargs,
            )
        finally:
            processed.close()
    finally:
        image.close()

    after = dst.stat().st_size
    return WatermarkResult(
        output=dst,
        before=before,
        after=after,
        width=width,
        height=height,
    )


# ------------------------------------------------------------------- 界面部分


class ImageWatermarkPage(ToolPage):
    """批量图片文字水印工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BatchWorker | None = None
        self._busy = False
        self._font_valid = False
        self._last_output_root: Path | None = None
        self._color = (128, 128, 128)
        self._input_widgets: list[QWidget] = []
        self._browse_buttons: list[QWidget] = []
        self._build_ui()
        self._load_default_font()
        self._sync_position_widgets()
        self._update_color_button()
        self._update_output_hint()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_MARGINS)
        layout.setSpacing(10)

        form = make_form()

        self.folder_edit = add_path_row(
            form,
            0,
            "输入目录：",
            mode="open_dir",
            placeholder="选择包含图片的文件夹",
            dialog_title="选择图片文件夹",
            parent=self,
        )
        self.output_edit = add_path_row(
            form,
            1,
            "输出目录：",
            mode="open_dir",
            placeholder="留空则使用同级的 {目录名}_watermarked",
            dialog_title="选择水印图片输出目录",
            parent=self,
        )
        for row in (0, 1):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                self._browse_buttons.append(item.widget())

        self.output_hint = QLabel()
        self.output_hint.setStyleSheet("color: #666;")
        self.output_hint.setWordWrap(True)
        form.addWidget(self.output_hint, 2, 1, 1, 2)

        self.recursive_check = QCheckBox("递归包含子文件夹")
        form.addWidget(QLabel("扫描选项："), 3, 0)
        form.addWidget(self.recursive_check, 3, 1, 1, 2)

        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("输入要平铺或定位显示的水印文字")
        form.addWidget(QLabel("水印文字："), 4, 0)
        form.addWidget(self.text_edit, 4, 1, 1, 2)

        self.font_edit = QLineEdit()
        self.font_edit.setPlaceholderText(
            "选择支持中文的字体文件（.ttf / .ttc / .otf）"
        )
        self.font_edit.textChanged.connect(self._validate_font_path)
        self.font_browse_btn = QPushButton("浏览…")
        self.font_browse_btn.clicked.connect(self._browse_font)
        form.addWidget(QLabel("字体文件："), 5, 0)
        form.addWidget(self.font_edit, 5, 1)
        form.addWidget(self.font_browse_btn, 5, 2)

        self.font_hint = QLabel()
        self.font_hint.setWordWrap(True)
        form.addWidget(self.font_hint, 6, 1, 1, 2)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("字号："))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(FONT_SIZE_MIN, FONT_SIZE_MAX)
        self.font_size_spin.setValue(36)
        self.font_size_spin.setSuffix(" px")
        style_row.addWidget(self.font_size_spin)
        style_row.addSpacing(20)
        style_row.addWidget(QLabel("颜色："))
        self.color_btn = QPushButton()
        self.color_btn.setToolTip("选择水印文字颜色")
        self.color_btn.clicked.connect(self._choose_color)
        style_row.addWidget(self.color_btn)
        style_row.addSpacing(20)
        style_row.addWidget(QLabel("角度："))
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(ANGLE_MIN, ANGLE_MAX)
        self.angle_spin.setValue(0)
        self.angle_spin.setSuffix("°")
        style_row.addWidget(self.angle_spin)
        style_row.addStretch(1)
        form.addWidget(QLabel("文字样式："), 7, 0)
        form.addLayout(style_row, 7, 1, 1, 2)

        position_row = QHBoxLayout()
        position_row.addWidget(QLabel("位置："))
        self.position_combo = QComboBox()
        for position in POSITION_CHOICES:
            self.position_combo.addItem(position, position)
        self.position_combo.currentIndexChanged.connect(
            self._sync_position_widgets
        )
        position_row.addWidget(self.position_combo)
        position_row.addSpacing(20)
        position_row.addWidget(QLabel("边距："))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(MARGIN_MIN, MARGIN_MAX)
        self.margin_spin.setValue(20)
        self.margin_spin.setSuffix(" px")
        position_row.addWidget(self.margin_spin)
        position_row.addSpacing(20)
        position_row.addWidget(QLabel("平铺间距："))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(GAP_MIN, GAP_MAX)
        self.gap_spin.setValue(60)
        self.gap_spin.setSuffix(" px")
        position_row.addWidget(self.gap_spin)
        position_row.addStretch(1)
        form.addWidget(QLabel("水印排布："), 8, 0)
        form.addLayout(position_row, 8, 1, 1, 2)

        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(OPACITY_MIN, OPACITY_MAX)
        self.opacity_slider.setValue(25)
        self.opacity_slider.setSingleStep(1)
        opacity_row.addWidget(self.opacity_slider, 1)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(OPACITY_MIN, OPACITY_MAX)
        self.opacity_spin.setValue(25)
        self.opacity_spin.setSuffix(" %")
        opacity_row.addWidget(self.opacity_spin)
        self.opacity_slider.valueChanged.connect(
            self.opacity_spin.setValue
        )
        self.opacity_spin.valueChanged.connect(
            self.opacity_slider.setValue
        )
        form.addWidget(QLabel("透明度："), 9, 0)
        form.addLayout(opacity_row, 9, 1, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 10, 1, 1, 2)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始添加水印")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_watermark)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_watermark)
        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output_dir)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 11, 1, 1, 2)

        layout.addLayout(form)
        layout.addStretch(1)

        self.status_label = QLabel("请选择图片文件夹并设置水印选项")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.folder_edit,
            self.output_edit,
            *self._browse_buttons,
            self.recursive_check,
            self.text_edit,
            self.font_edit,
            self.font_browse_btn,
            self.font_size_spin,
            self.color_btn,
            self.angle_spin,
            self.position_combo,
            self.margin_spin,
            self.gap_spin,
            self.opacity_slider,
            self.opacity_spin,
        ]

        self.folder_edit.textChanged.connect(self._update_output_hint)
        self.output_edit.textChanged.connect(self._update_output_hint)

    def _load_default_font(self) -> None:
        default_font = find_cjk_font_file()
        if default_font is not None:
            self.font_edit.blockSignals(True)
            self.font_edit.setText(str(default_font))
            self.font_edit.blockSignals(False)
        self._validate_font_path()

    def _browse_font(self) -> None:
        current = self.font_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择字体文件",
            start_dir,
            "字体文件 (*.ttf *.ttc *.otf);;所有文件 (*)",
        )
        if selected:
            self.font_edit.setText(selected)

    def _validate_font_path(self, *_args: object) -> None:
        path_text = self.font_edit.text().strip()
        if not path_text:
            self._font_valid = False
            self.font_hint.setText(
                "未找到可用的中文字体，请手动选择 .ttf / .ttc / .otf 字体文件。"
            )
            self.font_hint.setStyleSheet("color: #c62828;")
            self._update_start_enabled()
            return

        try:
            load_watermark_font(
                WatermarkOptions(text="测试", font_path=path_text)
            )
        except ValueError as exc:
            self._font_valid = False
            self.font_hint.setText(str(exc))
            self.font_hint.setStyleSheet("color: #c62828;")
        else:
            self._font_valid = True
            self.font_hint.setText(
                f"字体可用：{Path(path_text).expanduser().name}"
            )
            self.font_hint.setStyleSheet("color: #666;")
        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        self.start_btn.setEnabled(not self._busy and self._font_valid)

    def _update_color_button(self) -> None:
        red, green, blue = self._color
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        foreground = "#000000" if luminance > 150 else "#ffffff"
        self.color_btn.setText(f"#{red:02X}{green:02X}{blue:02X}")
        self.color_btn.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({red}, {green}, {blue});"
            f"color: {foreground};"
            "border: 1px solid #999; padding: 2px 10px;"
            "}"
        )

    def _choose_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(*self._color), self, "选择水印颜色"
        )
        if not selected.isValid():
            return
        self._color = (selected.red(), selected.green(), selected.blue())
        self._update_color_button()

    def _sync_position_widgets(self, *_args: object) -> None:
        tiling = self._is_tiling()
        self.margin_spin.setEnabled(not tiling and not self._busy)
        self.gap_spin.setEnabled(tiling and not self._busy)

    def _is_tiling(self) -> bool:
        data = self.position_combo.currentData()
        return (str(data) if data is not None else "左上") == TILE_POSITION

    def _same_path(self, first: Path, second: Path) -> bool:
        try:
            return first.resolve() == second.resolve()
        except OSError:
            return first.absolute() == second.absolute()

    def _fallback_output_root(self, input_root: Path) -> Path:
        name = input_root.name or "images"
        return input_root.parent / f"{name}_watermarked"

    def _effective_output_root(self, input_root: Path) -> Path:
        text = self.output_edit.text().strip()
        fallback = self._fallback_output_root(input_root)
        if not text:
            return fallback
        output_root = Path(text).expanduser()
        if self._same_path(output_root, input_root):
            return fallback
        return output_root

    def _update_output_hint(self, *_args: object) -> None:
        input_text = self.folder_edit.text().strip()
        if not input_text:
            self.output_hint.setText(
                "输出目录留空时，将在输入目录同级创建 "
                "「{目录名}_watermarked」。"
            )
            return
        input_root = Path(input_text).expanduser()
        fallback = self._fallback_output_root(input_root)
        output_text = self.output_edit.text().strip()
        if not output_text:
            self.output_hint.setText(f"将输出到：{fallback}")
            return
        output_root = Path(output_text).expanduser()
        if self._same_path(output_root, input_root):
            self.output_hint.setText(
                f"输出目录与输入相同，将自动改为：{fallback}"
            )
        else:
            self.output_hint.setText(f"将输出到：{output_root}")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in self._input_widgets:
            widget.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if busy:
            self.open_btn.setEnabled(False)
        else:
            self._sync_position_widgets()
        self._update_start_enabled()

    # --------------------------------------------------------------- 水印流程

    def _destinations_for(
        self,
        images: list[Path],
        input_root: Path,
        output_root: Path,
    ) -> list[tuple[Path, Path]]:
        used: set[str] = set()
        items: list[tuple[Path, Path]] = []
        for src in images:
            dst = output_root / src.relative_to(input_root)
            if dst.suffix.lower() not in _SUPPORTED_OUTPUT_SUFFIXES:
                dst = dst.with_suffix(".png")
            key = os.path.normcase(str(dst.resolve(strict=False))).casefold()
            if key in used:
                index = 2
                while True:
                    candidate = dst.with_name(
                        f"{dst.stem} ({index}){dst.suffix}"
                    )
                    candidate_key = os.path.normcase(
                        str(candidate.resolve(strict=False))
                    ).casefold()
                    if candidate_key not in used:
                        dst = candidate
                        key = candidate_key
                        break
                    index += 1
            used.add(key)
            items.append((src, dst))
        return items

    def _collect_inputs(
        self,
    ) -> tuple[Path, Path, list[tuple[Path, Path]], WatermarkOptions] | None:
        """校验界面输入并生成待处理列表；失败时弹窗并返回 None。"""
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            QMessageBox.warning(
                self, "缺少输入目录", "请先选择包含图片的文件夹。"
            )
            return None
        input_root = Path(folder_text).expanduser()
        if not input_root.is_dir():
            QMessageBox.warning(
                self, "输入目录不存在", f"目录不存在：\n{input_root}"
            )
            return None

        text = self.text_edit.text().strip()
        if not text:
            QMessageBox.warning(
                self, "缺少水印文字", "请输入要添加的水印文字。"
            )
            return None

        output_text = self.output_edit.text().strip()
        output_root = self._effective_output_root(input_root)
        if output_text and str(output_root) != output_text:
            # 用户填写的目录与输入目录相同：按约定改为 _watermarked。
            self.output_edit.setText(str(output_root))
        if output_root.exists() and not output_root.is_dir():
            QMessageBox.warning(
                self,
                "输出路径无效",
                f"输出路径不是目录：\n{output_root}",
            )
            return None
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(
                self, "无法创建输出目录", f"{output_root}\n{exc}"
            )
            return None
        if not os.access(output_root, os.W_OK):
            QMessageBox.warning(
                self,
                "输出目录不可写",
                f"没有写入权限：\n{output_root}",
            )
            return None

        try:
            images = scan_image_files(
                input_root,
                recursive=self.recursive_check.isChecked(),
                exclude_roots=(output_root,),
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "无法读取文件夹", f"{input_root}\n{exc}"
            )
            return None
        if not images:
            detail = ""
            if not self.recursive_check.isChecked():
                detail = "\n如需包含子文件夹，请勾选「递归包含子文件夹」。"
            QMessageBox.warning(
                self,
                "没有可处理的图片",
                "该文件夹中未找到支持的图片文件"
                "（JPG、PNG、WebP、BMP、GIF、TIFF）。" + detail,
            )
            return None

        items = self._destinations_for(images, input_root, output_root)
        options = WatermarkOptions(
            text=text,
            position=str(self.position_combo.currentData()),
            font_path=self.font_edit.text().strip(),
            font_size=int(self.font_size_spin.value()),
            color=self._color,
            opacity=self.opacity_slider.value() / 100.0,
            angle=int(self.angle_spin.value()),
            margin=int(self.margin_spin.value()),
            gap=int(self.gap_spin.value()),
        )
        return input_root, output_root, items, options

    def _start_watermark(self) -> None:
        if self._busy:
            return
        if not self._font_valid:
            QMessageBox.warning(
                self,
                "字体不可用",
                "请先选择可用的中文字体文件再开始。",
            )
            return

        collected = self._collect_inputs()
        if collected is None:
            return
        _input_root, output_root, items, options = collected

        try:
            font = load_watermark_font(options)
        except ValueError as exc:
            self._font_valid = False
            self.font_hint.setText(str(exc))
            self.font_hint.setStyleSheet("color: #c62828;")
            self._update_start_enabled()
            QMessageBox.critical(self, "字体不可用", str(exc))
            return

        self._last_output_root = output_root
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(f"正在为 {len(items)} 张图片添加水印…")

        def process(
            item: tuple[Path, Path],
            _cancel_event: threading.Event,
        ) -> WatermarkResult:
            src, dst = item
            return watermark_one(src, dst, options, font)

        worker = BatchWorker(
            items,
            process,
            label_fn=lambda item: item[0].name,
            parent=self,
        )
        worker.item_started.connect(self._on_item_started)
        worker.progress_changed.connect(self._on_progress)
        worker.batch_finished.connect(self._on_batch_finished)
        worker.batch_cancelled.connect(self._on_batch_cancelled)
        worker.batch_failed.connect(self._on_batch_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_item_started(self, label: str) -> None:
        self._set_status(f"正在处理：{label}")

    def _on_progress(self, done: object, total: object) -> None:
        done_count = int(done)
        total_count = int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(percent)

    def _on_batch_finished(self, result: object) -> None:
        if not isinstance(result, BatchResult):
            self._on_batch_failed("任务返回了无法识别的结果。")
            return

        self.progress_bar.setValue(100)
        output_root = self._last_output_root
        if output_root is not None and output_root.is_dir():
            self.open_btn.setEnabled(True)
        failure_count = len(result.errors)
        lines = [
            f"成功：{result.done} 张",
            f"失败：{failure_count} 张",
            f"输出目录：{output_root}",
        ]
        if result.errors:
            preview = "\n".join(
                f"{label}：{message}"
                for label, message in result.errors[:5]
            )
            if failure_count > 5:
                preview += f"\n…另有 {failure_count - 5} 张失败"
            lines.extend(["", "失败项：", preview])

        message = "\n".join(lines)
        self._set_status(
            f"完成：成功 {result.done} 张，失败 {failure_count} 张"
            f" → {output_root}"
        )
        if failure_count:
            QMessageBox.warning(
                self, "处理完成（有失败项）", message
            )
        else:
            QMessageBox.information(self, "处理完成", message)

    def _on_batch_cancelled(self, result: object) -> None:
        if isinstance(result, BatchResult):
            if result.done and self._last_output_root is not None:
                if self._last_output_root.is_dir():
                    self.open_btn.setEnabled(True)
            self._set_status(
                f"已取消：成功 {result.done} 张，"
                f"失败 {len(result.errors)} 张"
            )
        else:
            self._set_status("已取消")

    def _on_batch_failed(self, message: str) -> None:
        self._set_status("水印处理失败")
        QMessageBox.critical(self, "水印处理失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_watermark(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._set_status("正在取消…")

    def _open_output_dir(self) -> None:
        if self._last_output_root is None:
            return
        if not self._last_output_root.is_dir():
            QMessageBox.critical(
                self,
                "目录不存在",
                f"输出目录不存在：\n{self._last_output_root}",
            )
            return
        try:
            open_directory(self._last_output_root)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(
                self,
                "无法打开目录",
                f"{self._last_output_root}\n{exc}",
            )

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；处理进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "水印处理")
