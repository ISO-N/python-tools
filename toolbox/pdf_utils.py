"""PDF 打开、保存与文字绘制的共享辅助函数（依赖 PyMuPDF）。"""

from __future__ import annotations

import os
import re
import string
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
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


@dataclass(frozen=True)
class SavePreset:
    """``Document.save`` 的清理 / 压缩参数组合。"""

    garbage: int
    clean: bool
    deflate: bool
    deflate_images: bool
    deflate_fonts: bool


SAVE_PRESETS = {
    # 与旧默认完全一致。
    "balanced": SavePreset(4, True, True, True, True),
    # 用于拆分产生的单页文档、新建的有损压缩文档：避免重复清理新对象。
    "fast": SavePreset(1, False, True, False, False),
}


def save_pdf_atomic(
    document: pymupdf.Document,
    output: Path | str,
    *,
    preset: str | SavePreset | None = None,
    garbage: int | None = None,
    clean: bool | None = None,
    deflate: bool | None = None,
    deflate_images: bool | None = None,
    deflate_fonts: bool | None = None,
) -> Path:
    """先写同目录临时文件再原子替换；调用方负责关闭 document。

    不传任何参数时等价于原来的默认值；显式参数覆盖预设字段。
    """
    if isinstance(preset, SavePreset):
        base = preset
    elif preset is not None:
        try:
            base = SAVE_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(f"未知的 PDF 保存预设：{preset}") from exc
    else:
        base = SAVE_PRESETS["balanced"]

    options = {
        "garbage": base.garbage if garbage is None else garbage,
        "clean": base.clean if clean is None else clean,
        "deflate": base.deflate if deflate is None else deflate,
        "deflate_images": (
            base.deflate_images
            if deflate_images is None
            else deflate_images
        ),
        "deflate_fonts": (
            base.deflate_fonts if deflate_fonts is None else deflate_fonts
        ),
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".pdf", dir=output.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        document.save(str(tmp_path), **options)
        os.replace(tmp_path, output)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output


@dataclass(frozen=True)
class PdfFontResource:
    """一次加载、可跨页复用的字体资源。"""

    font: pymupdf.Font
    fontname: str
    buffer: bytes | None
    file_backed: bool


def load_pdf_font_resource(
    font_path: Path | str | None = None,
    fontname: str = "TBCJK",
) -> PdfFontResource:
    """加载字体对象与可复用的字体字节。

    ``file_backed`` 为真时 ``register_cjk_font`` 使用 ``fontbuffer``，
    避免每页重新读取并解析字体文件。
    """
    if font_path:
        try:
            font = pymupdf.Font(fontfile=str(font_path))
            buffer = getattr(font, "buffer", None)
            return PdfFontResource(
                font=font,
                fontname=fontname,
                buffer=buffer,
                file_backed=buffer is not None,
            )
        except Exception:  # noqa: BLE001 - 字体不可用时回退
            pass
    try:
        font = pymupdf.Font(fontname="china-s")
        return PdfFontResource(font, "china-s", None, False)
    except Exception:  # noqa: BLE001 - 最后回退内置西文字体
        font = pymupdf.Font(fontname="helv")
        return PdfFontResource(font, "helv", None, False)


def register_cjk_font(
    page: pymupdf.Page,
    font_path: Path | str | None,
    fontname: str = "TBCJK",
    *,
    resource: PdfFontResource | None = None,
) -> str:
    """尝试把系统字体嵌入页面；失败或无字体时回退内置 ``china-s``。

    返回实际可用的字体名。``china-s`` 不会嵌入 PDF，换阅读器可能缺字。
    传入 ``resource`` 时优先使用已缓存的 ``fontbuffer``。
    """
    if resource is not None:
        if resource.buffer:
            try:
                page.insert_font(
                    fontname=resource.fontname,
                    fontbuffer=resource.buffer,
                )
                return resource.fontname
            except Exception:  # noqa: BLE001 - 字体不可用时继续回退
                pass
        if font_path:
            try:
                page.insert_font(
                    fontname=resource.fontname,
                    fontfile=str(font_path),
                )
                return resource.fontname
            except Exception:  # noqa: BLE001 - 字体不可用时回退
                pass
        return resource.fontname

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
    return load_pdf_font_resource(font_path).font


@dataclass(frozen=True)
class TextStamp:
    """一条待绘制的文字（可旋转、可半透明）。"""

    center: tuple[float, float]
    text: str
    fontsize: float
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fill_opacity: float = 1.0
    angle: float = 0.0
    text_width: float | None = None


def add_text_stamps(
    page: pymupdf.Page,
    stamps: Sequence[TextStamp],
    *,
    fontname: str,
    font: pymupdf.Font,
    overlay: bool = True,
) -> None:
    """把同一页上的多条文字合并到一个 Shape，最后只提交一次内容流。"""
    if not stamps:
        return
    ascender = float(getattr(font, "ascender", 0.8))
    descender = float(getattr(font, "descender", -0.2))
    shape = page.new_shape()
    drew = False
    for stamp in stamps:
        if not stamp.text:
            continue
        width = (
            stamp.text_width
            if stamp.text_width is not None
            else font.text_length(stamp.text, fontsize=stamp.fontsize)
        )
        cx, cy = stamp.center
        start = pymupdf.Point(
            cx - width / 2,
            cy + (ascender + descender) / 2 * stamp.fontsize,
        )
        kwargs: dict[str, object] = {
            "fontname": fontname,
            "fontsize": stamp.fontsize,
            "fill": stamp.color,
            "fill_opacity": stamp.fill_opacity,
        }
        if stamp.angle:
            kwargs["morph"] = (
                pymupdf.Point(cx, cy),
                pymupdf.Matrix(stamp.angle),
            )
        shape.insert_text(start, stamp.text, **kwargs)
        drew = True
    if drew:
        shape.commit(overlay=overlay)


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
    text_width: float | None = None,
) -> None:
    """以 ``center`` 为中心绘制文字，支持任意旋转与透明度。"""
    if not text:
        return
    width = (
        text_width
        if text_width is not None
        else font.text_length(text, fontsize=fontsize)
    )
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


@dataclass(frozen=True)
class PageNumberTemplate:
    """已校验的页码模板。"""

    source: str

    def render(self, page_number: int, total: int) -> str:
        try:
            return self.source.format(page=page_number, total=total)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"页码模板无法渲染：{exc}") from exc


@lru_cache(maxsize=64)
def parse_page_number_template(template: str) -> PageNumberTemplate:
    """校验页码模板并缓存解析结果；只允许 ``{page}`` 与 ``{total}``。"""
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
    return PageNumberTemplate(template)


def format_page_number(template: str, page_number: int, total: int) -> str:
    """渲染页码模板，只允许 ``{page}`` 与 ``{total}``。"""
    return parse_page_number_template(template).render(page_number, total)
