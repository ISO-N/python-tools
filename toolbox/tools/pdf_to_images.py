"""PDF 转图片工具页（PyQt6 界面）。

选择 PDF 文件后，逐页渲染为 PNG / JPEG 图片，输出到以 PDF 文件名命名的子目录。
本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import open_directory
from toolbox.errors import Cancelled
from toolbox.page import ToolPage
from toolbox.parallel import Chunk, ChunkedCancelled, run_chunked_pool
from toolbox.pdf_parallel import render_pages_chunk
from toolbox.pdf_utils import open_pdf
from toolbox.workers import SingleTaskWorker, shutdown_worker

DPI_MIN, DPI_MAX = 72, 600
JPEG_QUALITY = 90
MAX_CHUNK_BYTES = 128 * 1024 * 1024
MAX_CHUNK_PAGES = 16
MAX_PARALLEL_WORKERS = 4
# 进程池启动成本约 0.2s；小任务（低 DPI / 页数少）串行更快。
MIN_PARALLEL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RenderResult:
    """取消或完成时携带的输出目录与已生成数量。"""

    out_dir: Path
    done: int


def parse_page_range(text: str, total_pages: int) -> list[int]:
    """把用户输入的页码范围解析为 0 基页码列表。

    支持空输入（表示全部页面）、单页 ``8``、区间 ``1-5``、逗号组合 ``1-5,8``，
    兼容全角逗号、顿号、波浪号、破折号等写法。返回值去重并按升序排列。
    """
    normalized = (text or "").strip()
    normalized = (
        normalized.replace("，", ",")
        .replace("、", ",")
        .replace("；", ",")
        .replace("~", "-")
        .replace("～", "-")
        .replace("—", "-")
        .replace("－", "-")
        .replace("至", "-")
        .replace(" ", "")
        .replace("　", "")
    )
    if not normalized or normalized.lower() in {"全部", "所有", "all"}:
        return list(range(total_pages))

    pages: set[int] = set()
    for token in normalized.split(","):
        if not token:
            continue
        pieces = token.split("-")
        if len(pieces) == 2:
            start_text, end_text = pieces
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"无法识别的页码片段：{token}")
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"页码区间起点大于终点：{token}")
        elif len(pieces) == 1 and pieces[0].isdigit():
            start = end = int(pieces[0])
        else:
            raise ValueError(f"无法识别的页码片段：{token}")

        if start < 1 or end > total_pages:
            raise ValueError(f"页码超出范围（1-{total_pages}）：{token}")
        pages.update(range(start - 1, end))

    if not pages:
        raise ValueError("页码范围为空")
    return sorted(pages)


def output_dir_for(pdf_path: Path, output_root: Path) -> Path:
    """返回本次转换使用的子目录：输出目录 / PDF 文件名（不含扩展名）。"""
    return Path(output_root) / Path(pdf_path).stem


def _temp_image_path(
    out_dir: Path, tmp_prefix: str, final_name: str
) -> Path:
    """与进程池约定一致的临时文件名。"""
    return out_dir / f".{tmp_prefix}_{final_name}.tmp"


def _remove_temp_files(out_dir: Path, tmp_prefix: str) -> None:
    """删除本任务遗留的临时图片；最终文件不受影响。"""
    for path in out_dir.glob(f".{tmp_prefix}_*.tmp"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def convert_pdf(
    pdf_path: Path,
    out_dir: Path,
    image_format: str,
    dpi: int,
    pages: Sequence[int],
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, int]:
    """把 PDF 的指定页面渲染成图片，返回（输出目录, 生成数量）。

    ``pages`` 为 0 基页码；``on_progress`` 在每页完成后回调（已完成数, 总数）；
    ``cancel_event`` 被置位时在两页之间停止，并抛出
    ``Cancelled(RenderResult(out_dir, done))``。每页先写临时文件，成功后
    再原子替换为最终文件。
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    fmt = image_format.lower()
    if fmt not in {"png", "jpeg", "jpg"}:
        raise ValueError(f"不支持的图片格式：{image_format}")
    ext = "jpg" if fmt in {"jpeg", "jpg"} else "png"
    stem = pdf_path.stem
    total = len(pages)
    done = 0
    tmp_prefix = uuid.uuid4().hex

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise RuntimeError(f"无法打开 PDF 文件：{exc}") from exc

    try:
        if doc.needs_pass:
            raise RuntimeError("PDF 文件受密码保护，无法转换")
        if doc.page_count == 0:
            raise RuntimeError("PDF 文件不包含任何页面")
        if any(index < 0 or index >= doc.page_count for index in pages):
            raise ValueError(f"页码超出 PDF 页数范围（共 {doc.page_count} 页）")

        out_dir.mkdir(parents=True, exist_ok=True)
        for index in pages:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled(RenderResult(out_dir, done))

            page = doc.load_page(index)
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            output_path = out_dir / f"{stem}_page_{index + 1:03d}.{ext}"
            temp_path = _temp_image_path(
                out_dir, tmp_prefix, output_path.name
            )
            try:
                # 临时文件以 .tmp 结尾，必须显式指定输出格式。
                if ext == "jpg":
                    pixmap.save(
                        str(temp_path),
                        output="jpeg",
                        jpg_quality=JPEG_QUALITY,
                    )
                else:
                    pixmap.save(str(temp_path), output="png")
                os.replace(temp_path, output_path)
            except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"保存图片失败：{output_path}\n{exc}"
                ) from exc

            done += 1
            if on_progress is not None:
                on_progress(done, total)
    finally:
        doc.close()

    return out_dir, done


def _estimate_pixmap_bytes(width: float, height: float, dpi: int) -> int:
    """按 RGB 三通道估算一页 pixmap 的字节数。"""
    return (
        math.ceil(width * dpi / 72) * math.ceil(height * dpi / 72) * 3
    )


def _split_pages_into_chunks(
    page_bytes: dict[int, int],
    pages: Sequence[int],
    *,
    max_pages: int,
) -> list[list[int]]:
    """按预估字节数与页数上限贪心切分，保持页面顺序。"""
    groups: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for index in pages:
        size = page_bytes[index]
        if current and (
            current_bytes + size > MAX_CHUNK_BYTES
            or len(current) >= max_pages
        ):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(index)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def convert_pdf_parallel(
    pdf_path: Path,
    out_dir: Path,
    image_format: str,
    dpi: int,
    pages: Sequence[int],
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    max_workers: int | None = None,
) -> tuple[Path, int]:
    """多进程渲染 PDF 页面；输出文件名与串行路径完全一致。

    父进程只打开 PDF 读取页数与页面尺寸，渲染在 ``spawn`` 子进程中完成；
    子进程写临时文件，父进程按块顺序原子改名为最终文件。取消时只保留
    连续完成块的前缀图片，并抛出 ``Cancelled(RenderResult(out_dir, done))``。
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    fmt = image_format.lower()
    if fmt not in {"png", "jpeg", "jpg"}:
        raise ValueError(f"不支持的图片格式：{image_format}")
    ext = "jpg" if fmt in {"jpeg", "jpg"} else "png"
    page_list = list(pages)
    total = len(page_list)

    document = open_pdf(pdf_path)
    try:
        if any(
            index < 0 or index >= document.page_count
            for index in page_list
        ):
            raise ValueError(
                f"页码超出 PDF 页数范围（共 {document.page_count} 页）"
            )
        page_bytes: dict[int, int] = {}
        for index in page_list:
            rect = document.load_page(index).rect
            page_bytes[index] = _estimate_pixmap_bytes(
                rect.width, rect.height, dpi
            )
    finally:
        document.close()

    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_count = max(1, os.cpu_count() or 1)
    worker_limit = min(MAX_PARALLEL_WORKERS, cpu_count)
    if max_workers is not None:
        worker_limit = min(
            worker_limit, max(1, int(max_workers))
        )

    total_bytes = sum(page_bytes.values())
    if (
        total <= 4
        or worker_limit <= 1
        or total_bytes < MIN_PARALLEL_BYTES
    ):
        return convert_pdf(
            pdf_path,
            out_dir,
            image_format,
            dpi,
            page_list,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

    # 在内存上限内继续按页数均衡切分，让小 PDF 也能用上多个进程。
    page_cap = min(
        MAX_CHUNK_PAGES,
        max(1, math.ceil(total / worker_limit)),
    )
    page_groups = _split_pages_into_chunks(
        page_bytes, page_list, max_pages=page_cap
    )
    workers = min(worker_limit, len(page_groups))
    if workers <= 1:
        return convert_pdf(
            pdf_path,
            out_dir,
            image_format,
            dpi,
            page_list,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

    tmp_prefix = uuid.uuid4().hex
    child_format = "jpeg" if ext == "jpg" else "png"
    chunks = [
        Chunk(
            index=i,
            units=len(group),
            payload={
                "pdf_path": str(pdf_path),
                "pages": group,
                "out_dir": str(out_dir),
                "stem": pdf_path.stem,
                "image_format": child_format,
                "dpi": int(dpi),
                "quality": JPEG_QUALITY,
                "tmp_prefix": tmp_prefix,
            },
        )
        for i, group in enumerate(page_groups)
    ]

    done = 0

    def on_committed(_chunk: Chunk, results: object) -> None:
        nonlocal done
        for item in results:  # type: ignore[union-attr]
            temp_path = Path(item["temp"])
            final_path = Path(item["final"])
            try:
                os.replace(temp_path, final_path)
            except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
                raise RuntimeError(
                    f"保存图片失败：{final_path}\n{exc}"
                ) from exc
            done += 1
            if on_progress is not None:
                on_progress(done, total)

    try:
        run_chunked_pool(
            chunks,
            render_pages_chunk,
            cancel_event=cancel_event,
            on_committed=on_committed,
            max_workers=workers,
        )
    except ChunkedCancelled:
        _remove_temp_files(out_dir, tmp_prefix)
        raise Cancelled(RenderResult(out_dir, done)) from None
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        _remove_temp_files(out_dir, tmp_prefix)
        if isinstance(exc, RuntimeError) and str(exc).startswith(
            "保存图片失败"
        ):
            raise
        raise RuntimeError(f"转换失败：{exc}") from exc

    return out_dir, done


class PdfToImagesPage(ToolPage):
    """PDF 转图片工具页。"""

    shutdown_label = "转换"

    def current_worker(self) -> SingleTaskWorker | None:
        return self._worker

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._last_output_dir: Path | None = None
        self._build_ui()
        self._update_subdir_hint()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setColumnStretch(1, 1)

        self.pdf_edit = QLineEdit()
        self.pdf_edit.setPlaceholderText("选择要转换的 PDF 文件")
        self.pdf_edit.textChanged.connect(self._update_subdir_hint)
        browse_pdf_btn = QPushButton("浏览…")
        browse_pdf_btn.clicked.connect(self._browse_pdf)
        form.addWidget(QLabel("PDF 文件："), 0, 0)
        form.addWidget(self.pdf_edit, 0, 1)
        form.addWidget(browse_pdf_btn, 0, 2)

        self.outdir_edit = QLineEdit()
        self.outdir_edit.setPlaceholderText("选择图片输出位置")
        self.outdir_edit.textChanged.connect(self._update_subdir_hint)
        browse_outdir_btn = QPushButton("浏览…")
        browse_outdir_btn.clicked.connect(self._browse_outdir)
        form.addWidget(QLabel("输出目录："), 1, 0)
        form.addWidget(self.outdir_edit, 1, 1)
        form.addWidget(browse_outdir_btn, 1, 2)

        self.subdir_hint = QLabel()
        self.subdir_hint.setStyleSheet("color: #666;")
        form.addWidget(self.subdir_hint, 2, 1, 1, 2)

        options = QHBoxLayout()
        options.addWidget(QLabel("格式："))
        self.png_radio = QRadioButton("PNG")
        self.png_radio.setChecked(True)
        self.jpeg_radio = QRadioButton("JPEG")
        options.addWidget(self.png_radio)
        options.addWidget(self.jpeg_radio)
        options.addSpacing(24)
        options.addWidget(QLabel("DPI："))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(DPI_MIN, DPI_MAX)
        self.dpi_spin.setSingleStep(25)
        self.dpi_spin.setValue(150)
        options.addWidget(self.dpi_spin)
        options.addStretch(1)
        form.addWidget(QLabel("图片设置："), 3, 0)
        form.addLayout(options, 3, 1, 1, 2)

        self.range_edit = QLineEdit()
        self.range_edit.setPlaceholderText("留空表示全部")
        form.addWidget(QLabel("页码范围："), 4, 0)
        form.addWidget(self.range_edit, 4, 1)
        range_hint = QLabel("示例：1-5,8")
        range_hint.setStyleSheet("color: #666;")
        form.addWidget(range_hint, 4, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 5, 1, 1, 2)

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
        form.addLayout(buttons, 6, 1, 1, 2)

        layout.addLayout(form)
        layout.addStretch(1)

        self.status_label = QLabel("请选择要转换的 PDF 文件")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _browse_pdf(self) -> None:
        current = self.pdf_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 PDF 文件",
            start_dir,
            "PDF 文件 (*.pdf);;所有文件 (*)",
        )
        if not selected:
            return
        self.pdf_edit.setText(selected)
        if not self.outdir_edit.text().strip():
            self.outdir_edit.setText(str(Path(selected).parent))
        self._update_subdir_hint()
        self._set_status(f"已选择：{Path(selected).name}")

    def _browse_outdir(self) -> None:
        current = self.outdir_edit.text().strip()
        start_dir = current if current else str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self, "选择输出目录", start_dir
        )
        if not selected:
            return
        self.outdir_edit.setText(selected)
        self._update_subdir_hint()

    def _update_subdir_hint(self, *_args: object) -> None:
        pdf_text = self.pdf_edit.text().strip()
        out_text = self.outdir_edit.text().strip()
        if pdf_text and out_text:
            target = Path(out_text) / Path(pdf_text).stem
            self.subdir_hint.setText(f"将创建子目录：{target}")
        else:
            self.subdir_hint.setText("将在输出目录下创建以 PDF 文件名命名的子目录")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # --------------------------------------------------------------- 转换流程

    def _collect_options(self) -> tuple[Path, Path, str, int, list[int]] | None:
        """校验界面输入并读取页数、解析页码范围；失败时弹窗并返回 None。"""
        pdf_text = self.pdf_edit.text().strip()
        out_text = self.outdir_edit.text().strip()
        if not pdf_text:
            QMessageBox.warning(self, "缺少 PDF 文件", "请先选择要转换的 PDF 文件。")
            return None
        pdf_path = Path(pdf_text).expanduser()
        if not pdf_path.is_file():
            QMessageBox.warning(
                self, "PDF 文件不存在", f"找不到文件：\n{pdf_path}"
            )
            return None
        if not out_text:
            QMessageBox.warning(self, "缺少输出目录", "请先选择输出目录。")
            return None
        output_root = Path(out_text).expanduser()
        if not output_root.is_dir():
            QMessageBox.warning(
                self, "输出目录不存在", f"目录不存在：\n{output_root}"
            )
            return None
        if not os.access(output_root, os.W_OK):
            QMessageBox.warning(
                self, "输出目录不可写", f"没有写入权限：\n{output_root}"
            )
            return None

        try:
            doc = pymupdf.open(pdf_path)
            try:
                if doc.needs_pass:
                    QMessageBox.warning(
                        self,
                        "PDF 受密码保护",
                        "该 PDF 需要密码才能打开，暂不支持转换。",
                    )
                    return None
                total_pages = doc.page_count
            finally:
                doc.close()
        except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
            QMessageBox.critical(
                self,
                "无法打开 PDF",
                f"文件可能已损坏或格式不受支持：\n{exc}",
            )
            return None
        if total_pages == 0:
            QMessageBox.warning(self, "PDF 为空", "该 PDF 不包含任何页面。")
            return None

        try:
            pages = parse_page_range(self.range_edit.text(), total_pages)
        except ValueError as exc:
            QMessageBox.warning(self, "页码范围错误", str(exc))
            return None

        image_format = "jpeg" if self.jpeg_radio.isChecked() else "png"
        return pdf_path, output_root, image_format, self.dpi_spin.value(), pages

    def _start_conversion(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        options = self._collect_options()
        if options is None:
            return
        pdf_path, output_root, image_format, dpi, pages = options

        out_dir = output_dir_for(pdf_path, output_root)
        if out_dir.exists() and any(out_dir.iterdir()):
            reply = QMessageBox.question(
                self,
                "目录已存在",
                f"目录已存在且不为空：\n{out_dir}\n\n继续将覆盖同名图片，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._last_output_dir = out_dir
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._set_status(f"正在转换 {len(pages)} 页…")

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> RenderResult:
            final_dir, count = convert_pdf_parallel(
                pdf_path,
                out_dir,
                image_format,
                dpi,
                list(pages),
                cancel_event=cancel_event,
                on_progress=on_progress,
            )
            return RenderResult(final_dir, count)

        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_progress)
        worker.task_finished.connect(self._on_done)
        worker.task_cancelled.connect(self._on_cancelled)
        worker.task_failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_progress(self, done: object, total: object) -> None:
        done_count = int(done)
        total_count = int(total)
        percent = (
            int(done_count / total_count * 100) if total_count else 0
        )
        self.progress_bar.setValue(percent)
        self._set_status(
            f"正在处理第 {done_count}/{total_count} 页…"
        )

    def _on_done(self, result: object) -> None:
        if not isinstance(result, RenderResult):
            self._on_failed("任务返回了无法识别的结果。")
            return
        self._last_output_dir = result.out_dir
        self.progress_bar.setValue(100)
        self._set_status(
            f"完成：已生成 {result.done} 张图片 → {result.out_dir}"
        )
        self.open_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "转换完成",
            f"已生成 {result.done} 张图片：\n{result.out_dir}",
        )

    def _on_cancelled(self, result: object) -> None:
        done = result.done if isinstance(result, RenderResult) else 0
        self._set_status(f"已取消，已生成 {done} 张图片")
        if done:
            self.open_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self._set_status("转换失败")
        QMessageBox.critical(self, "转换失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _cancel_conversion(self) -> None:
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
        """主窗口关闭前调用；转换进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self, "转换")
