"""PDF 逐页渲染的进程池工作函数。

本模块只依赖标准库与 PyMuPDF，**不导入 PyQt6**，供 ``spawn`` 子进程直接
导入。所有函数必须是模块顶层可 pickle 的函数，payload 只使用
str / int / list 等原始类型；PyMuPDF Document 绝不跨线程或跨进程共享。
"""

from __future__ import annotations

import os

import pymupdf


def _open_document(pdf_path: str) -> pymupdf.Document:
    """在子进程内独立打开 PDF，并校验密码与页数。"""
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise RuntimeError(f"无法打开 PDF 文件：{exc}") from exc
    if document.needs_pass:
        document.close()
        raise RuntimeError("PDF 文件受密码保护，无法处理")
    if document.page_count == 0:
        document.close()
        raise RuntimeError("PDF 文件不包含任何页面")
    return document


def _check_page_index(document: pymupdf.Document, index: int) -> None:
    if index < 0 or index >= document.page_count:
        raise RuntimeError(
            f"页码超出 PDF 页数范围（共 {document.page_count} 页）："
            f"{index + 1}"
        )


def render_pages_chunk(payload: dict) -> list[dict]:
    """把一个分块内的页面渲染为临时图片文件。

    payload 字段：``pdf_path``、``pages``（0 基页码）、``out_dir``、
    ``stem``、``image_format``（png / jpeg）、``dpi``、``quality``、
    ``tmp_prefix``。返回 ``[{"page", "temp", "final"}]``；最终文件由父进程
    按块顺序 ``os.replace`` 提交，子进程不直接写最终文件。
    """
    pdf_path = str(payload["pdf_path"])
    pages = [int(index) for index in payload["pages"]]
    out_dir = str(payload["out_dir"])
    stem = str(payload["stem"])
    image_format = str(payload["image_format"]).lower()
    dpi = int(payload["dpi"])
    quality = int(payload["quality"])
    tmp_prefix = str(payload["tmp_prefix"])

    if image_format in {"jpeg", "jpg"}:
        ext = "jpg"
        output = "jpeg"
    elif image_format == "png":
        ext = "png"
        output = "png"
    else:
        raise RuntimeError(f"不支持的图片格式：{image_format}")

    document = _open_document(pdf_path)
    try:
        results: list[dict] = []
        for index in pages:
            _check_page_index(document, index)
            final_name = f"{stem}_page_{index + 1:03d}.{ext}"
            final_path = os.path.join(out_dir, final_name)
            temp_path = os.path.join(
                out_dir, f".{tmp_prefix}_{final_name}.tmp"
            )
            try:
                pixmap = document.load_page(index).get_pixmap(
                    dpi=dpi, alpha=False
                )
                # 临时文件以 .tmp 结尾，必须显式指定输出格式。
                if output == "jpeg":
                    pixmap.save(
                        temp_path, output="jpeg", jpg_quality=quality
                    )
                else:
                    pixmap.save(temp_path, output="png")
            except Exception as exc:  # noqa: BLE001 - 带上页码便于定位
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"渲染第 {index + 1} 页失败：{exc}"
                ) from exc
            results.append(
                {"page": index, "temp": temp_path, "final": final_path}
            )
        return results
    finally:
        document.close()


def render_pages_jpeg_chunk(payload: dict) -> list[dict]:
    """把一个分块内的页面渲染为 JPEG 字节，供父进程重建 PDF。

    payload 字段：``pdf_path``、``pages``（0 基页码）、``dpi``、``quality``。
    返回 ``[{"page", "width", "height", "jpeg"}]``；document 在子进程内
    独立打开并关闭，不跨进程共享。
    """
    pdf_path = str(payload["pdf_path"])
    pages = [int(index) for index in payload["pages"]]
    dpi = int(payload["dpi"])
    quality = int(payload["quality"])

    document = _open_document(pdf_path)
    try:
        results: list[dict] = []
        for index in pages:
            _check_page_index(document, index)
            try:
                page = document.load_page(index)
                pixmap = page.get_pixmap(
                    dpi=dpi,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                jpeg = pixmap.tobytes("jpeg", jpg_quality=quality)
            except Exception as exc:  # noqa: BLE001 - 带上页码便于定位
                raise RuntimeError(
                    f"渲染第 {index + 1} 页失败：{exc}"
                ) from exc
            results.append(
                {
                    "page": index,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "jpeg": jpeg,
                }
            )
        return results
    finally:
        document.close()
