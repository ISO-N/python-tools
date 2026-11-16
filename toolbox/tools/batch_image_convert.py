"""批量图片转换工具页（PyQt6 界面）。

把文件夹中的图片批量转换为 JPEG / PNG / WebP，可限制长边或按百分比缩放，
并保留输入目录的相对目录结构。本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PyQt6.QtWidgets import (
    QCheckBox,
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

from toolbox.common import filesystem_key, open_directory
from toolbox.images import (
    fit_size,
    is_animated,
    load_oriented,
    save_image_atomic,
    scan_image_files,
)
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import (
    BatchResult,
    BatchWorker,
    Cancelled,
    shutdown_worker,
)

QUALITY_MIN, QUALITY_MAX = 1, 100
LONG_EDGE_MIN, LONG_EDGE_MAX = 100, 20000
PERCENT_MIN, PERCENT_MAX = 1, 1000

SUPPORTED_FORMATS = {"jpeg", "png", "webp"}
RESIZE_MODES = {"none", "long_edge", "percent"}


@dataclass
class ConvertOptions:
    """一次批量转换的输出选项。"""

    target_format: str = "jpeg"
    quality: int = 85
    resize_mode: str = "none"
    long_edge: int = 1600
    percent: int = 100
    overwrite: bool = False


@dataclass
class ConvertJob:
    """一张图片的转换任务。"""

    src: Path
    dst: Path


@dataclass
class ConvertPlan:
    """一次批量转换的完整计划。"""

    jobs: list[ConvertJob]
    skipped: list[tuple[Path, str]]
    conflicts: list[Path]
    output_root: Path


@dataclass
class ConvertResult:
    """单张图片的转换结果。"""

    before: int
    after: int
    width: int
    height: int


# --------------------------------------------------------------------- 纯逻辑


def _normalized_format(target_format: str) -> str:
    """把格式名统一为 ``jpeg`` / ``png`` / ``webp``。"""
    fmt = str(target_format).lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的图片格式：{target_format}")
    return fmt


def _validate_options(options: ConvertOptions) -> None:
    _normalized_format(options.target_format)
    if options.resize_mode not in RESIZE_MODES:
        raise ValueError(f"不支持的缩放方式：{options.resize_mode}")
    if not QUALITY_MIN <= int(options.quality) <= QUALITY_MAX:
        raise ValueError(
            f"质量必须在 {QUALITY_MIN}-{QUALITY_MAX} 之间：{options.quality}"
        )
    if options.resize_mode == "long_edge" and int(options.long_edge) < 1:
        raise ValueError("长边必须大于 0")
    if options.resize_mode == "percent" and int(options.percent) < 1:
        raise ValueError("百分比必须大于 0")


def _same_directory(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def default_output_root(input_root: Path) -> Path:
    """输入目录同级的 ``{name}_converted`` 目录。"""
    input_root = Path(input_root)
    return input_root.parent / f"{input_root.name}_converted"


def output_path_for(
    src: Path,
    input_root: Path,
    output_root: Path,
    target_format: str,
) -> Path:
    """计算输出路径：保留相对父目录，替换扩展名为 jpg / png / webp。"""
    fmt = _normalized_format(target_format)
    src = Path(src)
    input_root = Path(input_root)
    output_root = Path(output_root)
    try:
        relative = src.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"源文件不在输入目录中：{src}") from exc
    extension = "jpg" if fmt == "jpeg" else fmt
    return output_root / relative.with_suffix(f".{extension}")


def plan_jobs(
    input_root: Path,
    output_root: Path,
    recursive: bool,
    options: ConvertOptions,
) -> ConvertPlan:
    """扫描输入目录并生成转换计划。

    不同源扩展名映射到同一目标（如 ``a.jpg`` 与 ``a.png`` 都转为
    ``a.webp``）视为冲突；目标文件已存在且 ``overwrite`` 为 False 也加入
    冲突。冲突只记录目标路径，由界面决定覆盖还是跳过。
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    _validate_options(options)
    if _same_directory(input_root, output_root):
        raise ValueError("输出目录不能与输入目录相同")

    sources = scan_image_files(
        input_root,
        recursive=recursive,
        exclude_roots=(output_root,),
    )

    jobs: list[ConvertJob] = []
    conflicts: list[Path] = []
    seen_targets: set[str] = set()
    conflict_keys: set[str] = set()

    for src in sources:
        dst = output_path_for(
            src, input_root, output_root, options.target_format
        )
        key = filesystem_key(str(dst))

        if key in seen_targets:
            if key not in conflict_keys:
                conflict_keys.add(key)
                conflicts.append(dst)
        else:
            seen_targets.add(key)

        if key not in conflict_keys and not options.overwrite and dst.exists():
            conflict_keys.add(key)
            conflicts.append(dst)

        jobs.append(ConvertJob(src=src, dst=dst))

    return ConvertPlan(
        jobs=jobs,
        skipped=[],
        conflicts=conflicts,
        output_root=output_root,
    )


def _source_is_animated(path: Path) -> bool:
    """读取文件头判断是否为多帧图片（exif_transpose 会丢失帧信息）。"""
    with Image.open(path) as image:
        return is_animated(image)


def convert_image(
    src: Path,
    dst: Path,
    options: ConvertOptions,
    cancel_event: threading.Event | None = None,
) -> ConvertResult:
    """转换一张图片，返回前后文件大小与输出尺寸。

    动画图片抛出 ``ValueError("动画图片，已跳过")``；``cancel_event`` 被置位
    时抛出 ``Cancelled``，由 ``BatchWorker`` 作为取消处理。
    """
    _validate_options(options)
    src = Path(src)
    dst = Path(dst)
    fmt = _normalized_format(options.target_format)

    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    before = src.stat().st_size

    if _source_is_animated(src):
        raise ValueError("动画图片，已跳过")

    image = load_oriented(src)
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()

        if options.resize_mode == "long_edge":
            target_size = fit_size(
                image.width,
                image.height,
                options.long_edge,
                options.long_edge,
                upscale=False,
            )
        elif options.resize_mode == "percent":
            target_size = (
                max(1, round(image.width * options.percent / 100)),
                max(1, round(image.height * options.percent / 100)),
            )
        else:
            target_size = None

        if target_size is not None and target_size != image.size:
            resized = image.resize(target_size, Image.Resampling.LANCZOS)
            image.close()
            image = resized

        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()

        dst.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "jpeg":
            save_image_atomic(
                image,
                dst,
                image_format="JPEG",
                quality=options.quality,
                optimize=True,
            )
        elif fmt == "png":
            save_image_atomic(
                image,
                dst,
                image_format="PNG",
                optimize=True,
            )
        else:
            save_image_atomic(
                image,
                dst,
                image_format="WEBP",
                quality=options.quality,
                method=4,
            )

        return ConvertResult(
            before=before,
            after=dst.stat().st_size,
            width=image.width,
            height=image.height,
        )
    finally:
        image.close()


# ------------------------------------------------------------------- 界面部分


class BatchImageConvertPage(ToolPage):
    """批量图片转换工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: BatchWorker | None = None
        self._busy = False
        self._last_output_root: Path | None = None
        self._auto_output: str | None = None
        self._current_label = ""
        self._run_skipped = 0
        self._input_widgets: list[QWidget] = []
        self._build_ui()
        self._sync_format_controls()
        self._sync_resize_controls()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_MARGINS)
        layout.setSpacing(10)

        form = make_form()

        self.input_edit = add_path_row(
            form,
            0,
            "输入目录：",
            mode="open_dir",
            placeholder="选择包含图片的文件夹",
            dialog_title="选择输入目录",
            parent=self,
        )
        self.output_edit = add_path_row(
            form,
            1,
            "输出目录：",
            mode="open_dir",
            placeholder="留空则自动使用「输入目录名_converted」",
            dialog_title="选择输出目录",
            parent=self,
        )

        browse_buttons: list[QWidget] = []
        for row in (0, 1):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                browse_buttons.append(item.widget())

        self.recursive_check = QCheckBox("递归包含子文件夹")
        form.addWidget(QLabel("扫描选项："), 2, 0)
        form.addWidget(self.recursive_check, 2, 1, 1, 2)

        output_options = QHBoxLayout()
        output_options.addWidget(QLabel("目标格式："))
        self.format_combo = QComboBox()
        self.format_combo.addItem("JPEG", "jpeg")
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("WebP", "webp")
        output_options.addWidget(self.format_combo)
        output_options.addSpacing(24)
        output_options.addWidget(QLabel("质量："))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(QUALITY_MIN, QUALITY_MAX)
        self.quality_spin.setValue(85)
        self.quality_spin.setToolTip("JPEG / WebP 有损压缩质量")
        output_options.addWidget(self.quality_spin)
        output_options.addStretch(1)
        form.addWidget(QLabel("输出设置："), 3, 0)
        form.addLayout(output_options, 3, 1, 1, 2)

        resize_options = QHBoxLayout()
        resize_options.addWidget(QLabel("缩放方式："))
        self.resize_combo = QComboBox()
        self.resize_combo.addItem("不缩放", "none")
        self.resize_combo.addItem("限制长边", "long_edge")
        self.resize_combo.addItem("按百分比", "percent")
        resize_options.addWidget(self.resize_combo)
        resize_options.addSpacing(16)
        resize_options.addWidget(QLabel("长边（像素）："))
        self.long_edge_spin = QSpinBox()
        self.long_edge_spin.setRange(LONG_EDGE_MIN, LONG_EDGE_MAX)
        self.long_edge_spin.setValue(1600)
        resize_options.addWidget(self.long_edge_spin)
        resize_options.addSpacing(16)
        resize_options.addWidget(QLabel("百分比（%）："))
        self.percent_spin = QSpinBox()
        self.percent_spin.setRange(PERCENT_MIN, PERCENT_MAX)
        self.percent_spin.setValue(100)
        resize_options.addWidget(self.percent_spin)
        resize_options.addStretch(1)
        form.addWidget(QLabel("尺寸设置："), 4, 0)
        form.addLayout(resize_options, 4, 1, 1, 2)

        resize_hint = QLabel(
            "限制长边只缩小不放大；按百分比可放大或缩小。"
            "PNG 不使用质量参数。"
        )
        resize_hint.setStyleSheet("color: #666;")
        resize_hint.setWordWrap(True)
        form.addWidget(resize_hint, 5, 1, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 6, 1, 1, 2)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始转换")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_conversion)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_conversion)
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

        self.status_label = QLabel("请选择输入目录并设置转换选项")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.input_edit,
            self.output_edit,
            *browse_buttons,
            self.recursive_check,
            self.format_combo,
            self.quality_spin,
            self.resize_combo,
            self.long_edge_spin,
            self.percent_spin,
        ]
        self.input_edit.textChanged.connect(self._on_input_changed)
        self.format_combo.currentIndexChanged.connect(
            self._sync_format_controls
        )
        self.resize_combo.currentIndexChanged.connect(
            self._sync_resize_controls
        )

    def _on_input_changed(self, text: str) -> None:
        """输入目录变化时，在输出目录为空或仍是自动值时更新默认值。"""
        if self._busy:
            return
        text = text.strip()
        if not text:
            return
        input_root = Path(text).expanduser()
        if not input_root.is_dir():
            return
        current = self.output_edit.text().strip()
        if current and current != self._auto_output:
            return
        default = str(default_output_root(input_root))
        self._auto_output = default
        self.output_edit.setText(default)

    def _sync_format_controls(self, *_args: object) -> None:
        is_png = self.format_combo.currentData() == "png"
        self.quality_spin.setEnabled(not self._busy and not is_png)

    def _sync_resize_controls(self, *_args: object) -> None:
        mode = self.resize_combo.currentData()
        self.long_edge_spin.setEnabled(
            not self._busy and mode == "long_edge"
        )
        self.percent_spin.setEnabled(
            not self._busy and mode == "percent"
        )

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
            self._sync_format_controls()
            self._sync_resize_controls()
            self._update_open_button()

    def _update_open_button(self) -> None:
        self.open_btn.setEnabled(
            not self._busy
            and self._last_output_root is not None
            and self._last_output_root.is_dir()
        )

    # --------------------------------------------------------------- 转换流程

    def _current_options(self) -> ConvertOptions:
        return ConvertOptions(
            target_format=str(self.format_combo.currentData()),
            quality=int(self.quality_spin.value()),
            resize_mode=str(self.resize_combo.currentData()),
            long_edge=int(self.long_edge_spin.value()),
            percent=int(self.percent_spin.value()),
        )

    def _ask_use_default_output(
        self, input_root: Path, output_root: Path
    ) -> Path | None:
        """输出目录与输入目录相同时，询问是否改用默认目录。"""
        default = default_output_root(input_root)
        box = QMessageBox(self)
        box.setWindowTitle("输出目录无效")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("输出目录不能与输入目录相同。")
        box.setInformativeText(f"是否改用默认输出目录？\n{default}")
        use_btn = box.addButton("使用默认目录", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(use_btn)
        box.exec()
        if box.clickedButton() is use_btn:
            self._auto_output = str(default)
            self.output_edit.setText(str(default))
            return default
        self._set_status(f"输出目录不能与输入目录相同：{output_root}")
        return None

    def _collect_plan(
        self,
    ) -> tuple[Path, Path, ConvertOptions, ConvertPlan] | None:
        """校验界面输入并生成转换计划；失败时弹窗并返回 None。"""
        input_text = self.input_edit.text().strip()
        if not input_text:
            QMessageBox.warning(
                self, "缺少输入目录", "请先选择包含图片的输入目录。"
            )
            return None
        input_root = Path(input_text).expanduser()
        if not input_root.is_dir():
            QMessageBox.warning(
                self, "输入目录不存在", f"目录不存在：\n{input_root}"
            )
            return None

        output_text = self.output_edit.text().strip()
        if not output_text:
            output_root = default_output_root(input_root)
            self._auto_output = str(output_root)
            self.output_edit.setText(str(output_root))
        else:
            output_root = Path(output_text).expanduser()
        if output_root.exists() and not output_root.is_dir():
            QMessageBox.warning(
                self,
                "输出目录无效",
                f"输出路径已存在且不是目录：\n{output_root}",
            )
            return None

        if _same_directory(input_root, output_root):
            chosen = self._ask_use_default_output(input_root, output_root)
            if chosen is None:
                return None
            output_root = chosen

        options = self._current_options()
        try:
            plan = plan_jobs(
                input_root,
                output_root,
                self.recursive_check.isChecked(),
                options,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "无法生成转换计划", str(exc))
            return None

        return input_root, output_root, options, plan

    def _ask_conflict_policy(self, count: int) -> bool | None:
        """询问冲突处理方式：返回 True=覆盖，False=跳过，None=取消。"""
        box = QMessageBox(self)
        box.setWindowTitle("目标文件冲突")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            f"有 {count} 个目标文件已存在，"
            "或不同格式的源图片会生成同一个目标文件。"
        )
        box.setInformativeText(
            "覆盖：转换结果覆盖这些目标文件（同一目标的多个源图以最后一个为准）。\n"
            "跳过：跳过所有目标冲突的源图片。"
        )
        overwrite_btn = box.addButton("覆盖", QMessageBox.ButtonRole.AcceptRole)
        skip_btn = box.addButton("跳过", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(skip_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite_btn:
            return True
        if clicked is skip_btn:
            return False
        return None

    def _start_conversion(self) -> None:
        if self._busy or (
            self._worker is not None and self._worker.isRunning()
        ):
            return
        collected = self._collect_plan()
        if collected is None:
            return
        _input_root, output_root, options, plan = collected

        if not plan.jobs:
            detail = ""
            if not self.recursive_check.isChecked():
                detail = "\n如需包含子文件夹，请勾选「递归包含子文件夹」。"
            QMessageBox.information(
                self,
                "没有可转换的图片",
                "输入目录中未找到支持的图片文件"
                "（JPG、PNG、WebP、BMP、GIF、TIFF）。" + detail,
            )
            self._set_status("没有可转换的图片")
            return

        jobs = list(plan.jobs)
        skipped = len(plan.skipped)
        if plan.conflicts:
            policy = self._ask_conflict_policy(len(plan.conflicts))
            if policy is None:
                self._set_status("已取消转换")
                return
            if policy:
                options.overwrite = True
            else:
                conflict_keys = {
                    filesystem_key(str(path)) for path in plan.conflicts
                }
                jobs = [
                    job
                    for job in plan.jobs
                    if filesystem_key(str(job.dst)) not in conflict_keys
                ]
                skipped += len(plan.jobs) - len(jobs)
                if not jobs:
                    QMessageBox.information(
                        self,
                        "没有可转换的图片",
                        f"{len(plan.jobs)} 张图片的目标文件均存在冲突，"
                        "已全部跳过。",
                    )
                    self._set_status("没有可转换的图片（目标文件存在冲突）")
                    return

        self._last_output_root = output_root
        self._run_skipped = skipped
        self._current_label = ""
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(f"正在转换 {len(jobs)} 张图片…")

        def process(
            job: ConvertJob, cancel_event: threading.Event
        ) -> ConvertResult:
            return convert_image(job.src, job.dst, options, cancel_event)

        worker = BatchWorker(
            jobs,
            process,
            label_fn=lambda job: job.src.name,
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
        self._current_label = label

    def _on_progress(self, done: object, total: object) -> None:
        done_count = int(done)
        total_count = int(total)
        percent = int(done_count / total_count * 100) if total_count else 0
        self.progress_bar.setValue(percent)
        if total_count and done_count < total_count:
            self._set_status(
                f"正在转换 {done_count + 1}/{total_count}："
                f"{self._current_label}"
            )
        else:
            self._set_status(f"已处理 {done_count}/{total_count} 张图片…")

    def _summary_text(
        self, success: int, failed: int, skipped: int
    ) -> str:
        text = f"成功 {success} 张，失败 {failed} 张，跳过 {skipped} 张。"
        if self._last_output_root is not None:
            text += f"\n输出目录：{self._last_output_root}"
        return text

    @staticmethod
    def _error_preview(errors: list[tuple[str, str]]) -> str:
        lines = [
            f"{label}：{message}" for label, message in errors[:5]
        ]
        if len(errors) > 5:
            lines.append(f"…另有 {len(errors) - 5} 项")
        return "\n".join(lines)

    def _on_batch_finished(self, result: object) -> None:
        if not isinstance(result, BatchResult):
            self._on_batch_failed("任务返回了无法识别的结果。")
            return

        success = result.done
        failed = len(result.errors)
        skipped = self._run_skipped
        self.progress_bar.setValue(100)
        self._update_open_button()
        self._set_status(
            f"完成：成功 {success} 张，失败 {failed} 张，"
            f"跳过 {skipped} 张 → {self._last_output_root}"
        )

        summary = self._summary_text(success, failed, skipped)
        if result.errors:
            details = self._error_preview(result.errors)
            QMessageBox.warning(
                self,
                "转换完成（有失败项）",
                f"{summary}\n\n失败详情：\n{details}",
            )
        else:
            QMessageBox.information(self, "转换完成", summary)

    def _on_batch_cancelled(self, result: object) -> None:
        success = 0
        failed = 0
        if isinstance(result, BatchResult):
            success = result.done
            failed = len(result.errors)
        self._update_open_button()
        self._set_status(
            f"已取消：成功 {success} 张，失败 {failed} 张，"
            f"跳过 {self._run_skipped} 张"
        )

    def _on_batch_failed(self, message: str) -> None:
        self._set_status("转换失败")
        QMessageBox.critical(self, "转换失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_conversion(self) -> None:
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
        """主窗口关闭前调用；转换进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "图片转换")
