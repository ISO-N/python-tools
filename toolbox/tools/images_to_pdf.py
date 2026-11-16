"""图片合成 PDF 工具页（PyQt6 界面）。

选择图片文件夹后，按文件名自然顺序把图片合成为 PDF。支持 A4 页面适配与
原图尺寸两种模式，可递归扫描子文件夹。本模块作为「Python 工具箱」的一个
工具页使用。
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import open_directory
from toolbox.images import (
    ImagePayload,
    fit_size,
    prepare_image_payload,
    scan_image_files,
)
from toolbox.page import ToolPage
from toolbox.pdf_utils import save_pdf_atomic
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import (
    Cancelled,
    MAX_IMAGE_WORKERS,
    SingleTaskWorker,
    shutdown_worker,
)

A4_WIDTH_PT = 595.28
A4_HEIGHT_PT = 841.89
MM_TO_PT = 72 / 25.4
MARGIN_MIN_MM, MARGIN_MAX_MM = 0.0, 50.0
DPI_MIN, DPI_MAX = 72, 600
JPEG_QUALITY = 92
# 大文件单独提交并排空预取窗口，避免多张原始字节同时驻留内存。
LARGE_FILE_THRESHOLD = 32 * 1024 * 1024


@dataclass(frozen=True)
class PageOptions:
    """PDF 页面生成选项。"""

    mode: str = "a4"
    margin_mm: float = 10.0
    dpi: int = 96


@dataclass(frozen=True)
class PageLayout:
    """单页 PDF 的页面尺寸与图片放置区域（单位：pt）。"""

    page_width: float
    page_height: float
    image_rect: tuple[float, float, float, float]


@dataclass
class PdfBuildResult:
    """一次图片合成 PDF 的结果。"""

    output: Path
    count: int
    skipped: list[tuple[Path, str]]


# --------------------------------------------------------------------- 纯逻辑


def _validate_options(options: PageOptions) -> None:
    if options.mode not in {"a4", "original"}:
        raise ValueError(f"不支持的页面模式：{options.mode}")
    if options.margin_mm < 0:
        raise ValueError("页边距不能为负数")
    if options.mode == "original" and options.dpi <= 0:
        raise ValueError("原图 DPI 必须大于 0")
    if options.mode == "a4":
        margin = options.margin_mm * MM_TO_PT
        if margin * 2 >= A4_WIDTH_PT:
            raise ValueError("页边距过大，A4 页面没有可用区域")


def page_geometry(
    image_width: int, image_height: int, options: PageOptions
) -> PageLayout:
    """计算图片在 PDF 页面中的尺寸与居中位置。

    A4 模式：图片等比缩放到扣除页边距后的可用区域并居中，图片比 A4 更宽时
    自动使用横向页面。原图模式：页面尺寸按 ``像素 × 72 / DPI`` 换算，再加
    两倍页边距，图片铺满内边距区域。
    """
    _validate_options(options)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"图片尺寸无效：{image_width}×{image_height}")

    margin = options.margin_mm * MM_TO_PT
    if options.mode == "original":
        scale = 72.0 / options.dpi
        page_width = image_width * scale + 2 * margin
        page_height = image_height * scale + 2 * margin
        return PageLayout(
            page_width=page_width,
            page_height=page_height,
            image_rect=(
                margin,
                margin,
                page_width - margin,
                page_height - margin,
            ),
        )

    portrait_ratio = A4_WIDTH_PT / A4_HEIGHT_PT
    if image_width / image_height > portrait_ratio:
        page_width, page_height = A4_HEIGHT_PT, A4_WIDTH_PT
    else:
        page_width, page_height = A4_WIDTH_PT, A4_HEIGHT_PT

    box_width = page_width - 2 * margin
    box_height = page_height - 2 * margin
    # fit_size 返回整数点尺寸；向下取整可用区域，避免四舍五入后越出页边距。
    fitted_width, fitted_height = fit_size(
        image_width,
        image_height,
        max(1.0, float(math.floor(box_width))),
        max(1.0, float(math.floor(box_height))),
    )
    x0 = (page_width - fitted_width) / 2
    y0 = (page_height - fitted_height) / 2
    return PageLayout(
        page_width=page_width,
        page_height=page_height,
        image_rect=(x0, y0, x0 + fitted_width, y0 + fitted_height),
    )


def _prepare_payload(
    path: Path, prefer_original: bool
) -> ImagePayload:
    """线程池任务：准备一张图片的可嵌入字节。"""
    return prepare_image_payload(
        path,
        jpeg_quality=JPEG_QUALITY,
        prefer_original=prefer_original,
        optimize=False,
    )


def _is_large_file(path: Path) -> bool:
    try:
        return path.stat().st_size > LARGE_FILE_THRESHOLD
    except OSError:
        return False


def _insert_payload(
    document: pymupdf.Document,
    layout: PageLayout,
    payload: ImagePayload,
) -> None:
    """在单线程中新建页面并插入图片；失败时删除页面后抛出。"""
    page = document.new_page(
        width=layout.page_width, height=layout.page_height
    )
    try:
        page.insert_image(
            pymupdf.Rect(*layout.image_rect),
            stream=payload.data,
            keep_proportion=True,
            alpha=0 if not payload.has_alpha else 1,
        )
    except Exception:
        document.delete_page(page.number)
        raise


def build_pdf(
    images: Sequence[Path],
    output: Path,
    options: PageOptions,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> PdfBuildResult:
    """把图片按顺序合成为 PDF，返回输出路径、成功页数与跳过项。

    损坏或无法读取的图片会记入 ``skipped`` 并继续处理后续图片；全部失败时
    抛出 ``RuntimeError``。``cancel_event`` 被置位时在图片之间停止，并抛出
    ``Cancelled``，其 ``result`` 为已处理部分的结果。
    """
    _validate_options(options)
    output = Path(output)
    image_paths = [Path(path) for path in images]
    total = len(image_paths)
    if total == 0:
        raise ValueError("没有可转换的图片")
    if output.exists() and output.is_dir():
        raise ValueError(f"输出路径是一个目录：{output}")
    output_resolved = output.resolve()
    output_absolute = output.absolute()
    for path in image_paths:
        # 先做低成本比较，只有路径相同或位于同一目录时才解析符号链接。
        if path == output or path.absolute() == output_absolute:
            raise ValueError("输出文件不能覆盖源图片")
        if path.parent == output.parent and path.resolve() == output_resolved:
            raise ValueError("输出文件不能覆盖源图片")

    skipped: list[tuple[Path, str]] = []
    count = 0
    workers = max(1, int(MAX_IMAGE_WORKERS))
    window = max(1, workers * 2)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending: dict[int, Future[ImagePayload]] = {}
    next_submit = 0

    def cancel_now() -> None:
        raise Cancelled(PdfBuildResult(output, count, skipped))

    document = pymupdf.open()
    try:
        for index, path in enumerate(image_paths, start=1):
            if cancel_event is not None and cancel_event.is_set():
                cancel_now()

            # 只用 workers*2 的窗口预取；大文件单独提交并排空窗口。
            while next_submit < total and len(pending) < window:
                if cancel_event is not None and cancel_event.is_set():
                    cancel_now()
                candidate = image_paths[next_submit]
                large = _is_large_file(candidate)
                if large and pending:
                    break
                pending[next_submit] = executor.submit(
                    _prepare_payload, candidate, True
                )
                next_submit += 1
                if large:
                    break

            future = pending.pop(index - 1)
            while not future.done():
                if cancel_event is not None and cancel_event.is_set():
                    cancel_now()
                wait((future,), timeout=0.1)
            if cancel_event is not None and cancel_event.is_set():
                cancel_now()

            payload: ImagePayload | None = None
            retry_payload: ImagePayload | None = None
            try:
                try:
                    payload = future.result()
                    layout = page_geometry(
                        payload.width, payload.height, options
                    )
                except Cancelled:
                    cancel_now()
                except Exception as exc:  # noqa: BLE001 - 单张损坏继续
                    skipped.append((path, str(exc)))
                else:
                    insert_failed = False
                    try:
                        _insert_payload(document, layout, payload)
                    except Exception:  # noqa: BLE001 - 原字节失败重编码一次
                        insert_failed = True
                    if not insert_failed:
                        count += 1
                    else:
                        # 先离开异常处理块并释放原字节，再准备回退编码，
                        # 避免大文件重试时原始与重编码数据同时驻留。
                        payload = None
                        if (
                            cancel_event is not None
                            and cancel_event.is_set()
                        ):
                            cancel_now()
                        retry_future = executor.submit(
                            _prepare_payload, path, False
                        )
                        while not retry_future.done():
                            if (
                                cancel_event is not None
                                and cancel_event.is_set()
                            ):
                                cancel_now()
                            wait((retry_future,), timeout=0.1)
                        if (
                            cancel_event is not None
                            and cancel_event.is_set()
                        ):
                            cancel_now()
                        try:
                            retry_payload = retry_future.result()
                            retry_layout = page_geometry(
                                retry_payload.width,
                                retry_payload.height,
                                options,
                            )
                        except Cancelled:
                            cancel_now()
                        except Exception as exc:  # noqa: BLE001 - 单张继续
                            skipped.append((path, str(exc)))
                        else:
                            try:
                                _insert_payload(
                                    document, retry_layout, retry_payload
                                )
                            except Exception as exc:  # noqa: BLE001
                                skipped.append((path, str(exc)))
                            else:
                                count += 1
            finally:
                payload = None
                retry_payload = None

            if on_progress is not None:
                on_progress(index, total)

        if count == 0:
            sample = ""
            if skipped:
                first_path, first_reason = skipped[0]
                sample = f"\n示例：{first_path.name}：{first_reason}"
            raise RuntimeError(f"所有图片都无法读取，未生成 PDF。{sample}")

        save_pdf_atomic(document, output)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        pending.clear()
        document.close()

    return PdfBuildResult(output=output, count=count, skipped=skipped)


# ------------------------------------------------------------------- 界面部分


class ImagesToPdfPage(ToolPage):
    shutdown_label = "转换"
    """图片合成 PDF 工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._last_output: Path | None = None
        self._input_widgets: list[QWidget] = []
        self._browse_buttons: list[QWidget] = []
        self._build_ui()
        self._sync_dpi_enabled()

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
            placeholder="选择包含图片的文件夹",
            dialog_title="选择图片文件夹",
            parent=self,
        )
        self.output_edit = add_path_row(
            form,
            1,
            "输出 PDF：",
            mode="save_file",
            placeholder="选择 PDF 保存位置",
            dialog_title="保存 PDF",
            file_filter="PDF 文件 (*.pdf)",
            parent=self,
        )
        for row in (0, 1):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                self._browse_buttons.append(item.widget())

        self.recursive_check = QCheckBox("递归包含子文件夹")
        form.addWidget(QLabel("扫描选项："), 2, 0)
        form.addWidget(self.recursive_check, 2, 1, 1, 2)

        page_options = QHBoxLayout()
        page_options.addWidget(QLabel("页面模式："))
        self.a4_radio = QRadioButton("A4 适配")
        self.a4_radio.setChecked(True)
        self.original_radio = QRadioButton("原图尺寸")
        page_options.addWidget(self.a4_radio)
        page_options.addWidget(self.original_radio)
        page_options.addSpacing(24)
        page_options.addWidget(QLabel("页边距（mm）："))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(MARGIN_MIN_MM, MARGIN_MAX_MM)
        self.margin_spin.setDecimals(1)
        self.margin_spin.setSingleStep(1.0)
        self.margin_spin.setValue(10.0)
        page_options.addWidget(self.margin_spin)
        page_options.addStretch(1)
        form.addWidget(QLabel("页面设置："), 3, 0)
        form.addLayout(page_options, 3, 1, 1, 2)

        dpi_options = QHBoxLayout()
        dpi_options.addWidget(QLabel("DPI："))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(DPI_MIN, DPI_MAX)
        self.dpi_spin.setSingleStep(1)
        self.dpi_spin.setValue(96)
        dpi_options.addWidget(self.dpi_spin)
        self.dpi_hint = QLabel("仅「原图尺寸」模式使用")
        self.dpi_hint.setStyleSheet("color: #666;")
        dpi_options.addWidget(self.dpi_hint)
        dpi_options.addStretch(1)
        form.addWidget(QLabel("原图输出："), 4, 0)
        form.addLayout(dpi_options, 4, 1, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 5, 1, 1, 2)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始生成")
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
        form.addLayout(buttons, 6, 1, 1, 2)

        layout.addLayout(form)
        layout.addStretch(1)

        self.status_label = QLabel("请选择图片文件夹并设置输出 PDF")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.folder_edit,
            self.output_edit,
            *self._browse_buttons,
            self.recursive_check,
            self.a4_radio,
            self.original_radio,
            self.margin_spin,
            self.dpi_spin,
        ]
        self.original_radio.toggled.connect(self._sync_dpi_enabled)

    def _sync_dpi_enabled(self, *_args: object) -> None:
        self.dpi_spin.setEnabled(
            self.original_radio.isChecked() and not self._busy
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
            self._sync_dpi_enabled()

    # --------------------------------------------------------------- 生成流程

    def _collect_inputs(
        self,
    ) -> tuple[list[Path], Path, PageOptions] | None:
        """校验界面输入并扫描图片；失败时弹窗并返回 None。"""
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            QMessageBox.warning(
                self, "缺少图片文件夹", "请先选择要转换为 PDF 的图片文件夹。"
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
                self, "缺少输出 PDF", "请先选择输出 PDF 的保存位置。"
            )
            return None
        output = Path(output_text).expanduser()
        if output.exists() and output.is_dir():
            QMessageBox.warning(
                self,
                "输出路径无效",
                f"输出路径是一个目录，请选择 PDF 文件路径：\n{output}",
            )
            return None
        if output.suffix.lower() != ".pdf":
            QMessageBox.warning(
                self,
                "输出文件格式错误",
                "输出文件必须使用 .pdf 扩展名。",
            )
            return None

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
                "没有可转换的图片",
                "该文件夹中未找到支持的图片文件"
                "（JPG、PNG、WebP、BMP、GIF、TIFF）。" + detail,
            )
            return None

        if output.exists():
            reply = QMessageBox.question(
                self,
                "文件已存在",
                f"输出文件已存在：\n{output}\n\n继续将覆盖该文件，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None

        options = PageOptions(
            mode="original" if self.original_radio.isChecked() else "a4",
            margin_mm=float(self.margin_spin.value()),
            dpi=int(self.dpi_spin.value()),
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
        self._set_status(f"正在生成 PDF（共 {len(images)} 张图片）…")

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> PdfBuildResult:
            return build_pdf(
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
        self._set_status(f"正在处理第 {done_count}/{total_count} 张图片…")

    def _on_finished(self, result: object) -> None:
        if not isinstance(result, PdfBuildResult):
            self._on_failed("任务返回了无法识别的结果。")
            return

        self.progress_bar.setValue(100)
        self.open_btn.setEnabled(True)
        self._last_output = result.output
        skipped_count = len(result.skipped)
        if not skipped_count:
            self._set_status(
                f"完成：已生成 {result.count} 页 PDF → {result.output}"
            )
            QMessageBox.information(
                self,
                "生成完成",
                f"已生成 {result.count} 页 PDF：\n{result.output}",
            )
            return

        preview = "\n".join(
            f"{path.name}：{reason}"
            for path, reason in result.skipped[:5]
        )
        if skipped_count > 5:
            preview += f"\n…另有 {skipped_count - 5} 张"
        self._set_status(
            f"完成：已生成 {result.count} 页 PDF，"
            f"跳过 {skipped_count} 张图片 → {result.output}"
        )
        QMessageBox.warning(
            self,
            "生成完成（有跳过项）",
            f"已生成 {result.count} 页 PDF：\n{result.output}\n\n"
            f"跳过 {skipped_count} 张图片：\n{preview}",
        )

    def _on_cancelled(self, result: object) -> None:
        detail = ""
        if isinstance(result, PdfBuildResult):
            detail = f"（已处理 {result.count} 张图片"
            if result.skipped:
                detail += f"，跳过 {len(result.skipped)} 张"
            detail += "）"
        self._set_status(f"已取消，未生成 PDF{detail}")

    def _on_failed(self, message: str) -> None:
        self._set_status("生成失败")
        QMessageBox.critical(self, "生成失败", message)

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
        """主窗口关闭前调用；生成进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "转换")
