"""PDF 打开、保存与文字绘制的共享辅助函数（依赖 PyMuPDF）。"""

from __future__ import annotations

import os
import re
import string
import tempfile
from pathlib import Path

import pymupdf


class PdfError(RuntimeError):
    """PDF 无法打开或不满足处理条件时抛出。"""


def open_pdf(path: Path | str) -> pymupdf.Document:
    """打开 PDF 并做统一预检；调用方负责关闭返回的 Document。"""
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise PdfError(f"无法打开 PDF 文件：{exc}") from exc
    if document.needs_pass:
        document.close()
        raise PdfError("PDF 文件受密码保护，无法处理")
    if document.page_count == 0:
        document.close()
        raise PdfError("PDF 文件不包含任何页面")
    return document


def save_pdf_atomic(
    document: pymupdf.Document,
    output: Path | str,
    *,
    garbage: int = 4,
    clean: bool = True,
    deflate: bool = True,
    deflate_images: bool = True,
    deflate_fonts: bool = True,
) -> Path:
    """先写同目录临时文件再原子替换；调用方负责关闭 document。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".pdf", dir=output.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        document.save(
            str(tmp_path),
            garbage=garbage,
            clean=clean,
            deflate=deflate,
            deflate_images=deflate_images,
            deflate_fonts=deflate_fonts,
        )
        os.replace(tmp_path, output)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output


def register_cjk_font(
    page: pymupdf.Page,
    font_path: Path | str | None,
    fontname: str = "TBCJK",
) -> str:
    """尝试把系统字体嵌入页面；失败或无字体时回退内置 ``china-s``。

    返回实际可用的字体名。``china-s`` 不会嵌入 PDF，换阅读器可能缺字。
    """
    if font_path:
        try:
            page.insert_font(fontname=fontname, fontfile=str(font_path))
            return fontname
        except Exception:  # noqa: BLE001 - 字体不可用时回退
            pass
    return "china-s"


def load_pdf_font(
    font_path: Path | str | None = None,
) -> pymupdf.Font:
    """加载字体对象，用于计算文字宽度与基线位置。"""
    if font_path:
        try:
            return pymupdf.Font(fontfile=str(font_path))
        except Exception:  # noqa: BLE001 - 字体不可用时回退
            pass
    try:
        return pymupdf.Font(fontname="china-s")
    except Exception:  # noqa: BLE001 - 最后回退内置西文字体
        return pymupdf.Font(fontname="helv")


def insert_rotated_text(
    page: pymupdf.Page,
    center: tuple[float, float],
    text: str,
    *,
    fontname: str,
    font: pymupdf.Font,
    fontsize: float,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fill_opacity: float = 1.0,
    angle: float = 0.0,
) -> None:
    """以 ``center`` 为中心绘制文字，支持任意旋转与透明度。"""
    if not text:
        return
    width = font.text_length(text, fontsize=fontsize)
    ascender = float(getattr(font, "ascender", 0.8))
    descender = float(getattr(font, "descender", -0.2))
    cx, cy = center
    start = pymupdf.Point(
        cx - width / 2,
        cy + (ascender + descender) / 2 * fontsize,
    )
    kwargs: dict[str, object] = {
        "fontname": fontname,
        "fontsize": fontsize,
        "fill": color,
        "fill_opacity": fill_opacity,
    }
    if angle:
        kwargs["morph"] = (
            pymupdf.Point(cx, cy),
            pymupdf.Matrix(angle),
        )
    page.insert_text(start, text, **kwargs)


_PAGE_NUMBER_SPEC = re.compile(r"0?\d*d\Z")


def format_page_number(template: str, page_number: int, total: int) -> str:
    """渲染页码模板，只允许 ``{page}`` 与 ``{total}``。"""
    if not template:
        raise ValueError("页码模板不能为空")
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"页码模板格式错误：{exc}") from exc
    for _literal, field, spec, conversion in parsed:
        if field is None:
            continue
        if field not in {"page", "total"}:
            raise ValueError(
                f"页码模板只能使用 {{page}} 和 {{total}}，发现：{{{field}}}"
            )
        if conversion:
            raise ValueError("页码模板不支持转换标记")
        if spec and not _PAGE_NUMBER_SPEC.fullmatch(spec):
            raise ValueError(
                f"不支持的格式说明：{{{field}:{spec}}}（可用 0 与 d，如 03d）"
            )
    try:
        return template.format(page=page_number, total=total)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"页码模板无法渲染：{exc}") from exc
