"""PDF 压缩工具页（PyQt6 界面）。

提供两种压缩方式：
- 无损压缩：清理冗余对象并重新压缩流数据，可选字体子集化；
- 有损压缩：把每页渲染为 JPEG 图片后重建 PDF，体积更小，但文字不可搜索。
本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PyQt6.QtWidgets import (
    QCheckBox,
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

from toolbox.common import human_size, open_directory
from toolbox.page import ToolPage
from toolbox.pdf_utils import open_pdf, save_pdf_atomic
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import Cancelled, SingleTaskWorker, shutdown_worker

DPI_MIN, DPI_MAX = 72, 600
QUALITY_MIN, QUALITY_MAX = 1, 95

_METADATA_KEYS = {
    "author",
    "title",
    "subject",
    "keywords",
    "creator",
    "producer",
    "creationDate",
    "modDate",
    "trapped",
}
_TRAPPED_VALUES = {"True", "False", "Unknown"}
_PDF_DATE = re.compile(
    r"^D:\d{4}(\d{2}(\d{2}(\d{2}(\d{2}(\d{2})?)?)?)?)?([Zz+\-].*)?$"
)


@dataclass(frozen=True)
class CompressionOptions:
    """PDF 压缩选项。"""

    mode: str = "lossless"
    dpi: int = 150
    quality: int = 75
    subset_fonts: bool = True


@dataclass(frozen=True)
class CompressionResult:
    """一次压缩操作的结果（字节数）。"""

    before: int
    after: int
    pages: int


# --------------------------------------------------------------------- 纯逻辑


def _sanitize_metadata(metadata: dict[str, str]) -> dict[str, str]:
    """过滤 PyMuPDF 不接受的元数据字段、空值与格式不合法值。"""
    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        if key not in _METADATA_KEYS or not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        if key in {"creationDate", "modDate"} and not _PDF_DATE.match(value):
            continue
        if key == "trapped" and value not in _TRAPPED_VALUES:
            continue
        cleaned[key] = value
    return cleaned


def _preserve_toc(
    document: pymupdf.Document,
    new_document: pymupdf.Document,
    page_count: int,
) -> None:
    """复制书签目录；``get_toc`` 的页码本身就是 1 基，无需再转换。"""
    try:
        toc = document.get_toc()
    except Exception:  # noqa: BLE001 - 读取书签失败不影响压缩
        return
    if not toc:
        return

    cleaned: list[list[object]] = []
    for entry in toc:
        if len(entry) < 3:
            continue
        level, title, page_number = entry[0], entry[1], entry[2]
        try:
            page_index = int(page_number)
        except (TypeError, ValueError):
            continue
        if page_index < 1:
            continue
        cleaned.append([level, title, min(page_index, page_count)])
    if not cleaned:
        return
    try:
        new_document.set_toc(cleaned)
    except Exception:  # noqa: BLE001 - 写入书签失败不影响压缩
        pass


def _preserve_metadata(
    document: pymupdf.Document, new_document: pymupdf.Document
) -> None:
    try:
        metadata = document.metadata or {}
    except Exception:  # noqa: BLE001 - 读取元数据失败不影响压缩
        return
    cleaned = _sanitize_metadata(metadata)
    if not cleaned:
        return
    try:
        new_document.set_metadata(cleaned)
    except Exception:  # noqa: BLE001 - 元数据不合法时忽略
        pass


def compress_pdf(
    src: Path,
    dst: Path,
    options: CompressionOptions,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> CompressionResult:
    """压缩 PDF 并返回压缩前后的字节数与页数。

    无损模式使用 ``save_pdf_atomic`` 清理对象、重新压缩流数据；有损模式把
    每页渲染为 JPEG 后重建文档，并尽量保留书签与元数据。取消时抛出
    ``Cancelled``，其 ``result`` 为已处理页数（``after`` 为 0，表示未生成
    输出文件）。``dst`` 与 ``src`` 解析后相同会直接拒绝。
    """
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    if options.mode not in {"lossless", "lossy"}:
        raise ValueError(f"不支持的压缩模式：{options.mode}")
    if options.mode == "lossy":
        if options.dpi <= 0:
            raise ValueError("DPI 必须大于 0")
        if not 1 <= options.quality <= 100:
            raise ValueError("JPEG 质量必须在 1-100 之间")

    if src.resolve() == dst.resolve():
        raise ValueError("输出文件不能与源文件相同")
    if dst.exists() and dst.is_dir():
        raise ValueError(f"输出路径是一个目录：{dst}")

    document = open_pdf(src)
    try:
        before = src.stat().st_size
        total = document.page_count
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(CompressionResult(before, 0, 0))

        if options.mode == "lossless":
            if options.subset_fonts:
                try:
                    document.subset_fonts()
                except Exception:  # noqa: BLE001 - 字体子集化失败时静默忽略
                    pass
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled(CompressionResult(before, 0, 0))
            if on_progress is not None:
                on_progress(0, total)
            save_pdf_atomic(
                document,
                dst,
                garbage=4,
                clean=True,
                deflate=True,
                deflate_images=True,
                deflate_fonts=True,
            )
        else:
            pages_done = 0
            new_document = pymupdf.open()
            try:
                for index in range(total):
                    if (
                        cancel_event is not None
                        and cancel_event.is_set()
                    ):
                        raise Cancelled(
                            CompressionResult(before, 0, pages_done)
                        )
                    page = document.load_page(index)
                    pixmap = page.get_pixmap(
                        dpi=options.dpi,
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    jpeg = pixmap.tobytes(
                        "jpeg", jpg_quality=options.quality
                    )
                    new_page = new_document.new_page(
                        width=page.rect.width, height=page.rect.height
                    )
                    new_page.insert_image(page.rect, stream=jpeg)
                    pages_done += 1
                    if on_progress is not None:
                        on_progress(pages_done, total)

                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled(
                        CompressionResult(before, 0, pages_done)
                    )
                _preserve_toc(document, new_document, total)
                _preserve_metadata(document, new_document)
                save_pdf_atomic(new_document, dst)
            finally:
                new_document.close()

        after = dst.stat().st_size
    finally:
        document.close()

    if on_progress is not None:
        on_progress(total, total)
    return CompressionResult(before=before, after=after, pages=total)


# ------------------------------------------------------------------- 界面部分


class PdfCompressPage(ToolPage):
    """PDF 压缩工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._last_output: Path | None = None
        self._auto_output: str | None = None
        self._input_widgets: list[QWidget] = []
        self._browse_buttons: list[QWidget] = []
        self._build_ui()
        self._sync_option_states()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_MARGINS)
        layout.setSpacing(10)

        form = make_form()

        self.source_edit = add_path_row(
            form,
            0,
            "源 PDF：",
            mode="open_file",
            placeholder="选择要压缩的 PDF 文件",
            dialog_title="选择 PDF 文件",
            file_filter="PDF 文件 (*.pdf);;所有文件 (*)",
            parent=self,
        )
        self.output_edit = add_path_row(
            form,
            1,
            "输出 PDF：",
            mode="save_file",
            placeholder="选择压缩后 PDF 的保存位置",
            dialog_title="保存压缩后的 PDF",
            file_filter="PDF 文件 (*.pdf)",
            parent=self,
        )
        for row in (0, 1):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                self._browse_buttons.append(item.widget())

        mode_options = QHBoxLayout()
        mode_options.addWidget(QLabel("压缩模式："))
        self.lossless_radio = QRadioButton("无损压缩")
        self.lossless_radio.setChecked(True)
        self.lossy_radio = QRadioButton("有损压缩（图片化）")
        mode_options.addWidget(self.lossless_radio)
        mode_options.addWidget(self.lossy_radio)
        mode_options.addSpacing(24)
        mode_options.addWidget(QLabel("DPI："))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(DPI_MIN, DPI_MAX)
        self.dpi_spin.setSingleStep(1)
        self.dpi_spin.setValue(150)
        mode_options.addWidget(self.dpi_spin)
        mode_options.addSpacing(16)
        mode_options.addWidget(QLabel("JPEG 质量："))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(QUALITY_MIN, QUALITY_MAX)
        self.quality_spin.setSingleStep(1)
        self.quality_spin.setValue(75)
        mode_options.addWidget(self.quality_spin)
        mode_options.addStretch(1)
        form.addWidget(QLabel("压缩设置："), 2, 0)
        form.addLayout(mode_options, 2, 1, 1, 2)

        font_options = QHBoxLayout()
        self.subset_check = QCheckBox("字体子集化（仅无损压缩）")
        self.subset_check.setChecked(True)
        font_options.addWidget(self.subset_check)
        font_options.addStretch(1)
        form.addWidget(QLabel("字体设置："), 3, 0)
        form.addLayout(font_options, 3, 1, 1, 2)

        self.warning_label = QLabel(
            "注意：有损模式会把每页转为 JPEG 图片，文字将不可搜索/复制，"
            "链接与注释会丢失。"
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #a15c00;")
        form.addWidget(self.warning_label, 4, 0, 1, 3)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 5, 1, 1, 2)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始压缩")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_compression)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_compression)
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

        self.status_label = QLabel("请选择要压缩的 PDF 文件")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.source_edit,
            self.output_edit,
            *self._browse_buttons,
            self.lossless_radio,
            self.lossy_radio,
            self.dpi_spin,
            self.quality_spin,
            self.subset_check,
        ]
        self.source_edit.textChanged.connect(self._update_default_output)
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.lossless_radio.toggled.connect(self._sync_option_states)
        self.lossy_radio.toggled.connect(self._sync_option_states)

    def _on_output_edited(self, _text: str) -> None:
        self._auto_output = None

    def _update_default_output(self, *_args: object) -> None:
        source_text = self.source_edit.text().strip()
        if not source_text:
            return
        source = Path(source_text).expanduser()
        default_output = str(
            source.with_name(f"{source.stem}_compressed.pdf")
        )
        current = self.output_edit.text().strip()
        if not current or current == self._auto_output:
            self._auto_output = default_output
            self.output_edit.setText(default_output)

    def _sync_option_states(self, *_args: object) -> None:
        lossy = self.lossy_radio.isChecked()
        enabled = not self._busy
        self.dpi_spin.setEnabled(enabled and lossy)
        self.quality_spin.setEnabled(enabled and lossy)
        self.subset_check.setEnabled(enabled and not lossy)

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
            self._sync_option_states()

    # --------------------------------------------------------------- 压缩流程

    def _collect_inputs(
        self,
    ) -> tuple[Path, Path, CompressionOptions] | None:
        """校验界面输入；失败或用户取消覆盖时返回 None。"""
        source_text = self.source_edit.text().strip()
        if not source_text:
            QMessageBox.warning(
                self, "缺少源 PDF", "请先选择要压缩的 PDF 文件。"
            )
            return None
        source = Path(source_text).expanduser()
        if not source.is_file():
            QMessageBox.warning(
                self, "源文件不存在", f"找不到文件：\n{source}"
            )
            return None

        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(
                self, "缺少输出 PDF", "请先选择压缩后 PDF 的保存位置。"
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
        if source.resolve() == output.resolve():
            QMessageBox.warning(
                self,
                "输出文件与源文件相同",
                "输出文件不能与源文件相同，请选择其他保存位置。",
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

        options = CompressionOptions(
            mode="lossy" if self.lossy_radio.isChecked() else "lossless",
            dpi=int(self.dpi_spin.value()),
            quality=int(self.quality_spin.value()),
            subset_fonts=self.subset_check.isChecked(),
        )
        return source, output, options

    def _start_compression(self) -> None:
        if self._busy:
            return
        collected = self._collect_inputs()
        if collected is None:
            return
        source, output, options = collected

        self._last_output = output
        self.progress_bar.setValue(0)
        self._set_busy(True)
        mode_text = "有损压缩" if options.mode == "lossy" else "无损压缩"
        self._set_status(f"正在压缩（{mode_text}）：{source.name}…")

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> CompressionResult:
            return compress_pdf(
                source,
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
        self._set_status(f"正在压缩…（{done_count}/{total_count} 页）")

    def _on_finished(self, result: object) -> None:
        if not isinstance(result, CompressionResult):
            self._on_failed("任务返回了无法识别的结果。")
            return

        self.progress_bar.setValue(100)
        self.open_btn.setEnabled(True)
        output = self._last_output
        output_text = str(output) if output is not None else ""

        before = human_size(result.before)
        after = human_size(result.after)
        if result.before <= 0:
            change = "无法计算百分比"
        elif result.after < result.before:
            change = (
                f"减少 "
                f"{(result.before - result.after) / result.before * 100:.1f}%"
            )
        elif result.after > result.before:
            change = (
                f"增大 "
                f"{(result.after - result.before) / result.before * 100:.1f}%"
            )
        else:
            change = "变化 0.0%"

        self._set_status(
            f"完成：{before} → {after}（{change}），"
            f"共 {result.pages} 页 → {output_text}"
        )
        extra = ""
        if result.after > result.before:
            extra = "\n\n压缩后文件反而变大，已保留生成的文件。"
        QMessageBox.information(
            self,
            "压缩完成",
            f"压缩前：{before}\n压缩后：{after}（{change}）\n"
            f"页数：{result.pages}\n输出：{output_text}{extra}",
        )

    def _on_cancelled(self, result: object) -> None:
        processed = result.pages if isinstance(result, CompressionResult) else 0
        self._set_status(
            f"已取消，未生成输出文件（已处理 {processed} 页）"
        )

    def _on_failed(self, message: str) -> None:
        self._set_status("压缩失败")
        QMessageBox.critical(self, "压缩失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_compression(self) -> None:
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
        """主窗口关闭前调用；压缩进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "压缩")
