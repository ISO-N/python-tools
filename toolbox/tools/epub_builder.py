"""EPUB 2 电子书生成工具页（PyQt6 界面）。

把「图片按页码归档」后的目录（一级子目录 = 章节）或单个文件夹中的图片
打包为 EPUB 2 电子书。仅使用标准库 ``zipfile`` 与 Pillow，不依赖 ebooklib。

本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import io
import os
import re
import threading
import uuid
import warnings
import zipfile
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from PIL import Image, ImageOps
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import natural_path_key, open_directory
from toolbox.images import scan_image_files
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import (
    MAX_IMAGE_WORKERS,
    Cancelled,
    SingleTaskWorker,
)

EPUB_MIMETYPE = "application/epub+zip"
JPEG_QUALITY = 92
XHTML_MEDIA_TYPE = "application/xhtml+xml"
NCX_MEDIA_TYPE = "application/x-dtbncx+xml"

#: 预览树的图片节点上限；超过后按章节懒加载。
PREVIEW_IMAGE_NODE_LIMIT = 2000
#: 超过该大小的图片单独提交 normalize，避免内存峰值失控。
LARGE_IMAGE_BYTES = 16 * 1024 * 1024

# (manifest id, href, media-type)
ManifestItem = tuple[str, str, str]

# XML 1.0 允许的字符：#x9、#xA、#xD、#x20-#xD7FF、#xE000-#xFFFD、
# #x10000-#x10FFFF。其余字符（含孤立代理项）在写入 XML 前移除。
_INVALID_XML_CHARS = re.compile(
    "[^\\u0009\\u000A\\u000D\\u0020-\\uD7FF"
    "\\uE000-\\uFFFD\\U00010000-\\U0010FFFF]"
)


# --------------------------------------------------------------------- 数据模型


@dataclass
class BookMetadata:
    """电子书元数据。"""

    title: str
    author: str = ""
    language: str = "zh-CN"
    cover_path: Path | None = None


@dataclass
class Chapter:
    """一个章节及其包含的图片（按阅读顺序）。"""

    title: str
    images: list[Path]


@dataclass
class BookPlan:
    """一次 EPUB 构建的图片组织方案。"""

    input_root: Path
    chapters: list[Chapter]
    cover: Path | None
    skipped: list[tuple[Path, str]]
    total_images: int
    #: (目录路径, st_mtime_ns) 快照，供界面判断计划是否仍然有效。
    fingerprint: tuple[tuple[str, int], ...] = ()


@dataclass
class EpubResult:
    """一次 EPUB 构建的结果。"""

    output: Path
    chapters: int
    images: int
    skipped: list[tuple[Path, str]]


# --------------------------------------------------------------------- 纯逻辑


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _find_cover_in(images: Sequence[Path]) -> Path | None:
    for path in images:
        if path.stem.casefold() == "cover":
            return path
    return None


def _find_cover_image(root: Path) -> Path | None:
    """查找根目录下文件名主干为 cover 的图片（大小写不敏感）。"""
    return _find_cover_in(scan_image_files(root, recursive=False))


def discover_book(
    input_root: Path | str,
    *,
    mode: str = "chapters",
    include_root_images: bool = True,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> BookPlan:
    """扫描输入目录并生成书籍组织方案。

    ``chapters`` 模式：第一层非隐藏、非符号链接子目录各为一章，章内图片
    递归扫描并按自然顺序排列；根目录图片可作为最前面的「正文」章节，根目录
    下名为 ``cover`` 的图片自动作为封面并从正文中排除。若没有子目录，根目录
    图片退化为单章。

    ``single`` 模式：根目录（含子目录）的全部图片作为一章，章标题为目录名；
    根目录下名为 ``cover`` 的图片同样自动识别为封面并从正文中排除。

    空章节会被忽略；目录中没有任何图片时抛出 ``ValueError``。
    ``cancel_event`` 在每次目录扫描前检查，取消时抛出
    :class:`toolbox.errors.Cancelled`；``on_progress`` 接收累计发现的图片数。
    """
    root = Path(input_root)
    if mode not in {"chapters", "single"}:
        raise ValueError(f"不支持的目录模式：{mode}")
    if not root.is_dir():
        raise ValueError(f"输入目录不存在：{root}")

    fingerprint: list[tuple[str, int]] = []
    found_total = 0

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()

    def record_directory(directory: Path) -> None:
        try:
            info = directory.stat()
        except OSError:
            return
        fingerprint.append((str(directory), info.st_mtime_ns))

    def scan_directory(directory: Path, *, recursive: bool) -> list[Path]:
        record_directory(directory)
        if on_progress is None:
            return scan_image_files(
                directory,
                recursive=recursive,
                cancel_event=cancel_event,
            )

        last = 0

        def report(found: int) -> None:
            nonlocal found_total, last
            found_total += found - last
            last = found
            on_progress(found_total)

        return scan_image_files(
            directory,
            recursive=recursive,
            cancel_event=cancel_event,
            on_progress=report,
        )

    if mode == "single":
        check_cancelled()
        all_images = scan_directory(root, recursive=True)
        cover = next(
            (
                path
                for path in all_images
                if path.parent == root
                and path.stem.casefold() == "cover"
            ),
            None,
        )
        images = [
            path for path in all_images if cover is None or path != cover
        ]
        if not images and cover is None:
            raise ValueError("目录中没有找到图片文件")
        title = root.name or str(root)
        chapters = [Chapter(title=title, images=images)] if images else []
        return BookPlan(
            input_root=root,
            chapters=chapters,
            cover=cover,
            skipped=[],
            total_images=len(images) + (1 if cover is not None else 0),
            fingerprint=tuple(fingerprint),
        )

    subdirs = [
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.is_symlink()
        and not _is_hidden(entry.name)
    ]
    subdirs.sort(key=lambda path: natural_path_key(path.name))

    check_cancelled()
    root_scan = scan_directory(root, recursive=False)
    cover = _find_cover_in(root_scan)
    root_images: list[Path] = []
    if include_root_images:
        root_images = [
            path for path in root_scan if cover is None or path != cover
        ]

    chapters: list[Chapter] = []
    if root_images:
        chapters.append(Chapter(title="正文", images=root_images))

    for directory in subdirs:
        check_cancelled()
        images = scan_directory(directory, recursive=True)
        if images:
            chapters.append(Chapter(title=directory.name, images=images))

    total_images = sum(len(chapter.images) for chapter in chapters)
    if cover is not None:
        total_images += 1
    if total_images == 0:
        raise ValueError("目录中没有找到可制作 EPUB 的图片")

    return BookPlan(
        input_root=root,
        chapters=chapters,
        cover=cover,
        skipped=[],
        total_images=total_images,
        fingerprint=tuple(fingerprint),
    )


def book_plan_is_current(plan: BookPlan) -> bool:
    """检查计划记录的各目录 mtime 是否仍与磁盘一致。"""
    if not plan.fingerprint:
        return False
    for directory, mtime_ns in plan.fingerprint:
        try:
            if Path(directory).stat().st_mtime_ns != mtime_ns:
                return False
        except OSError:
            return False
    return True


def _has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_orientation(raw: bytes) -> int:
    """从 PNG 的 eXIf 块读取方向，避免 ``getexif()`` 触发完整解码。"""
    if not raw.startswith(_PNG_SIGNATURE):
        return 1
    position = len(_PNG_SIGNATURE)
    limit = len(raw) - 4
    while position + 8 <= limit:
        length = int.from_bytes(raw[position : position + 4], "big")
        chunk_type = raw[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + length
        if data_end + 4 > len(raw):
            break
        if chunk_type == b"eXIf":
            try:
                exif = Image.Exif()
                exif.load(raw[data_start:data_end])
                return int(exif.get(0x0112, 1) or 1)
            except Exception:  # noqa: BLE001 - EXIF 损坏时按正常方向处理
                return 1
        if chunk_type == b"IEND":
            break
        position = data_end + 4
    return 1


def _encode_loaded_image(
    image: Image.Image,
    image_format: str,
    frames: int,
    orientation: int,
    name: str,
) -> tuple[bytes, str, str]:
    """在已打开的图像上完成方向转正与重新编码。"""
    try:
        image.load()
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(f"无法读取图片「{name}」：{exc}") from exc

    source_image = image
    transposed: Image.Image | None = None
    if orientation != 1:
        try:
            ImageOps.exif_transpose(image, in_place=True)
        except TypeError:  # 旧版 Pillow 没有 in_place 参数
            transposed = ImageOps.exif_transpose(image)
            source_image = transposed

    try:
        try:
            animated = image_format in {"GIF", "WEBP"} and frames > 1
            alpha = _has_alpha(source_image)

            if animated or alpha:
                target = (
                    source_image.convert("RGBA")
                    if alpha
                    else source_image.convert("RGB")
                )
                try:
                    buffer = io.BytesIO()
                    target.save(buffer, format="PNG", optimize=True)
                    return buffer.getvalue(), "image/png", "png"
                finally:
                    target.close()

            source = source_image
            converted: Image.Image | None = None
            if source_image.mode not in {"RGB", "L"}:
                converted = source_image.convert("RGB")
                source = converted
            try:
                buffer = io.BytesIO()
                source.save(
                    buffer,
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True,
                )
                return buffer.getvalue(), "image/jpeg", "jpg"
            finally:
                if converted is not None:
                    converted.close()
        except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
            raise ValueError(f"无法编码图片「{name}」：{exc}") from exc
    finally:
        if transposed is not None:
            transposed.close()


def normalize_epub_image(path: Path | str) -> tuple[bytes, str, str]:
    """把图片转换为 EPUB2 可用的数据。

    返回 ``(数据, media_type, 扩展名)``，扩展名不带点，只会是 ``jpg`` 或
    ``png``。方向为 1 且模式兼容的 JPEG / PNG 复用原文件字节，并用
    ``verify()`` 做完整性检查；其余图片在内存流上解码、转正方向后重新编码：
    有透明通道输出 PNG RGBA，否则输出 JPEG（quality=92）。动画 GIF / WebP
    只取第一帧并输出 PNG。无法读取或编码时抛出中文 ``ValueError``。
    """
    path = Path(path)

    try:
        raw = path.read_bytes()
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(f"无法读取图片「{path.name}」：{exc}") from exc

    image_format = ""
    frames = 1
    orientation = 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", Image.DecompressionBombWarning
            )
            with Image.open(io.BytesIO(raw)) as opened:
                image_format = (opened.format or "").upper()
                image_mode = opened.mode
                frames = int(getattr(opened, "n_frames", 1) or 1)
                if image_format == "PNG":
                    # PNG 的 getexif() 在缺少 eXIf 时会 load()，破坏
                    # verify() 且失去复用原字节的意义。
                    orientation = _png_orientation(raw)
                else:
                    try:
                        orientation = int(
                            opened.getexif().get(0x0112, 1) or 1
                        )
                    except Exception:  # noqa: BLE001 - EXIF 损坏按正常方向
                        orientation = 1

                reusable = (
                    orientation == 1
                    and frames <= 1
                    and (
                        (
                            image_format == "JPEG"
                            and image_mode in {"RGB", "L"}
                        )
                        or (
                            image_format == "PNG"
                            and image_mode in {"RGB", "RGBA", "L", "LA"}
                        )
                    )
                )
                verify_failed = False
                if reusable:
                    try:
                        # verify() 会消耗图像对象，但不需要完整解码。
                        opened.verify()
                    except Exception:  # noqa: BLE001 - 回退到重新编码
                        verify_failed = True
                    else:
                        if image_format == "JPEG":
                            return raw, "image/jpeg", "jpg"
                        return raw, "image/png", "png"

                if not verify_failed:
                    return _encode_loaded_image(
                        opened,
                        image_format,
                        frames,
                        orientation,
                        path.name,
                    )

        # verify() 失败后对象不可再用，从同一份内存字节重新打开。
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", Image.DecompressionBombWarning
            )
            with Image.open(io.BytesIO(raw)) as reopened:
                return _encode_loaded_image(
                    reopened,
                    image_format,
                    frames,
                    orientation,
                    path.name,
                )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(f"无法读取图片「{path.name}」：{exc}") from exc


# ----------------------------------------------------------------- XML 构造


def _clean_xml_text(value: object) -> str:
    return _INVALID_XML_CHARS.sub("", str(value))


def _xml_text(value: object) -> str:
    return escape(_clean_xml_text(value))


def _xml_attr(value: object) -> str:
    return quoteattr(_clean_xml_text(value))


def _image_src(href: str) -> str:
    """把 ``images/xxx`` 或 ``xxx`` 统一为 XHTML 中的 ``images/xxx``。

    XHTML 与图片位于同一个 ``OEBPS/`` 根目录下（``chapter_001.xhtml`` 与
    ``images/img_0001.jpg``），因此相对路径是 ``images/xxx``。
    """
    name = str(href).replace("\\", "/").rsplit("/", 1)[-1]
    return f"images/{name}"


def build_container_xml() -> str:
    """构造 ``META-INF/container.xml``。"""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        "  <rootfiles>\n"
        '    <rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/>\n'
        "  </rootfiles>\n"
        "</container>\n"
    )


def build_opf(
    metadata: BookMetadata,
    manifest_items: Sequence[ManifestItem],
    spine_ids: Sequence[str],
    book_id: str,
    cover_id: str | None = None,
) -> str:
    """构造 EPUB2 的 ``OEBPS/content.opf``。

    ``manifest_items`` 为 ``(id, href, media-type)`` 序列；``cover_id`` 为封面
    图片在 manifest 中的 id，传入时同时写入 EPUB2 的 ``meta name="cover"``
    与 guide reference。
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
        'unique-identifier="BookId">',
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">',
        f"    <dc:title>{_xml_text(metadata.title)}</dc:title>",
        f"    <dc:creator>{_xml_text(metadata.author)}</dc:creator>",
        f"    <dc:language>{_xml_text(metadata.language)}</dc:language>",
        f'    <dc:identifier id="BookId">{_xml_text(book_id)}</dc:identifier>',
    ]
    if cover_id:
        lines.append(
            '    <meta name="cover" '
            f'content={_xml_attr(cover_id)}/>'
        )
    lines.append("  </metadata>")

    lines.append("  <manifest>")
    for item_id, href, media_type in manifest_items:
        normalized_href = str(href).replace("\\", "/")
        lines.append(
            f"    <item id={_xml_attr(item_id)} "
            f"href={_xml_attr(normalized_href)} "
            f"media-type={_xml_attr(media_type)}/>"
        )
    lines.append("  </manifest>")

    lines.append('  <spine toc="ncx">')
    for idref in spine_ids:
        lines.append(f"    <itemref idref={_xml_attr(idref)}/>")
    lines.append("  </spine>")

    if cover_id:
        lines.extend(
            [
                "  <guide>",
                '    <reference type="cover" title="封面" '
                'href="cover.xhtml"/>',
                "  </guide>",
            ]
        )
    lines.append("</package>\n")
    return "\n".join(lines)


def build_ncx(
    chapters: Sequence[tuple[str, str]], book_id: str, title: str
) -> str:
    """构造 ``OEBPS/toc.ncx``；``chapters`` 为 ``(标题, xhtml 文件)`` 序列。"""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" '
        'version="2005-1">',
        "  <head>",
        f'    <meta name="dtb:uid" content={_xml_attr(book_id)}/>',
        '    <meta name="dtb:depth" content="1"/>',
        '    <meta name="dtb:totalPageCount" content="0"/>',
        '    <meta name="dtb:maxPageNumber" content="0"/>',
        "  </head>",
        f"  <docTitle><text>{_xml_text(title)}</text></docTitle>",
        "  <navMap>",
    ]
    for index, (chapter_title, href) in enumerate(chapters, start=1):
        normalized_href = str(href).replace("\\", "/")
        lines.extend(
            [
                f'    <navPoint id="navPoint-{index}" '
                f'playOrder="{index}">',
                f"      <navLabel><text>{_xml_text(chapter_title)}</text>"
                "</navLabel>",
                f"      <content src={_xml_attr(normalized_href)}/>",
                "    </navPoint>",
            ]
        )
    lines.extend(["  </navMap>", "</ncx>\n"])
    return "\n".join(lines)


def build_chapter_xhtml(
    title: str, images: Sequence[str], language: str
) -> str:
    """构造一个章节的 XHTML 1.1 文档。

    ``images`` 为 ``OEBPS/images`` 下的文件名（也可传入 ``images/xxx``），
    输出路径统一为 ``images/xxx``；每张图片一个 ``div.page``，最后一页
    额外带 ``last`` 类。
    """
    language_attr = _xml_attr(language)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" '
        '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        f"xml:lang={language_attr}>",
        "  <head>",
        '    <meta http-equiv="Content-Type" '
        'content="text/html; charset=UTF-8"/>',
        f"    <title>{_xml_text(title)}</title>",
        '    <link rel="stylesheet" type="text/css" href="style.css"/>',
        "  </head>",
        "  <body>",
    ]
    last_index = len(images) - 1
    for index, image in enumerate(images):
        classes = "page last" if index == last_index else "page"
        src = _xml_attr(_image_src(image))
        lines.append(
            f'    <div class="{classes}"><img src={src} alt=""/></div>'
        )
    lines.extend(["  </body>", "</html>\n"])
    return "\n".join(lines)


def build_cover_xhtml(cover_href: str) -> str:
    """构造封面页 XHTML；``cover_href`` 可传文件名或 ``images/xxx``。"""
    src = _xml_attr(_image_src(cover_href))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" '
        '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "  <head>\n"
        '    <meta http-equiv="Content-Type" '
        'content="text/html; charset=UTF-8"/>\n'
        "    <title>封面</title>\n"
        '    <link rel="stylesheet" type="text/css" href="style.css"/>\n'
        "  </head>\n"
        "  <body>\n"
        f'    <div class="cover"><img src={src} alt="封面"/></div>\n'
        "  </body>\n"
        "</html>\n"
    )


def build_css() -> str:
    """构造电子书公共样式表。"""
    return (
        "html, body {\n"
        "  margin: 0;\n"
        "  padding: 0;\n"
        "}\n"
        "div.page, div.cover {\n"
        "  margin: 0;\n"
        "  padding: 0;\n"
        "  text-align: center;\n"
        "}\n"
        "div.page {\n"
        "  page-break-after: always;\n"
        "}\n"
        "div.page.last {\n"
        "  page-break-after: auto;\n"
        "}\n"
        "div.page img, div.cover img {\n"
        "  max-width: 100%;\n"
        "  max-height: 100%;\n"
        "}\n"
    )


# ------------------------------------------------------------------- 构建逻辑


def _write_stored_image(
    archive: zipfile.ZipFile, name: str, data: bytes
) -> None:
    """图片以不压缩方式写入（JPEG/PNG 本身已压缩）。"""
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    archive.writestr(info, data)


def _is_large_image(path: Path) -> bool:
    """图片是否超过并行 normalize 的单文件内存预算。"""
    try:
        return path.stat().st_size > LARGE_IMAGE_BYTES
    except OSError:
        return False


def build_epub(
    plan: BookPlan,
    metadata: BookMetadata,
    output: Path | str,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> EpubResult:
    """按照 ``plan`` 生成 EPUB 2 文件并返回结果。

    输出先写入同目录的 ``.epub.tmp`` 临时文件，完成后原子替换；取消或异常
    时删除临时文件。单张图片损坏会跳过并记录，某章图片全部失败则省略该章；
    全部图片都失败时抛出 ``RuntimeError``。每张图片处理前检查取消状态，
    取消时抛出 ``toolbox.workers.Cancelled``。
    """
    output = Path(output)
    if not str(metadata.title).strip():
        raise ValueError("书名不能为空")
    if output.exists() and output.is_dir():
        raise ValueError(f"输出路径是一个目录：{output}")
    if not output.parent.is_dir():
        raise ValueError(f"输出目录不存在：{output.parent}")

    cover_path = (
        metadata.cover_path
        if metadata.cover_path is not None
        else plan.cover
    )
    if cover_path is not None:
        cover_path = Path(cover_path)
    body_total = sum(len(chapter.images) for chapter in plan.chapters)
    total_images = body_total + (1 if cover_path is not None else 0)
    if total_images == 0:
        raise ValueError("没有可写入 EPUB 的图片")

    book_id = f"urn:uuid:{uuid.uuid4()}"
    tmp_path = output.with_name(f"{output.name}.tmp")
    skipped: list[tuple[Path, str]] = list(plan.skipped)
    processed = 0
    images_written = 0
    chapter_count = 0

    try:
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            # mimetype 必须是第一个条目且不压缩（EPUB OCF 要求）。
            mimetype_info = zipfile.ZipInfo("mimetype")
            mimetype_info.compress_type = zipfile.ZIP_STORED
            archive.writestr(mimetype_info, EPUB_MIMETYPE.encode("ascii"))
            archive.writestr(
                "META-INF/container.xml",
                build_container_xml().encode("utf-8"),
            )
            archive.writestr("OEBPS/style.css", build_css().encode("utf-8"))

            manifest: list[ManifestItem] = []
            spine_ids: list[str] = []
            nav_entries: list[tuple[str, str]] = []
            cover_image_id: str | None = None
            cover_file_name: str | None = None

            def check_cancelled() -> None:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled(
                        EpubResult(
                            output=output,
                            chapters=chapter_count,
                            images=images_written,
                            skipped=list(skipped),
                        )
                    )

            # 封面先处理，固定命名为 cover.<ext>。
            if cover_path is not None:
                check_cancelled()
                try:
                    data, media_type, extension = normalize_epub_image(
                        cover_path
                    )
                except Exception as exc:  # noqa: BLE001 - 跳过损坏封面
                    skipped.append((cover_path, str(exc)))
                else:
                    cover_file_name = f"cover.{extension}"
                    _write_stored_image(
                        archive,
                        f"OEBPS/images/{cover_file_name}",
                        data,
                    )
                    manifest.append(
                        (
                            "cover-image",
                            f"images/{cover_file_name}",
                            media_type,
                        )
                    )
                    cover_image_id = "cover-image"
                    images_written += 1
                processed += 1
                if on_progress is not None:
                    on_progress(processed, total_images)

                if cover_file_name is not None:
                    archive.writestr(
                        "OEBPS/cover.xhtml",
                        build_cover_xhtml(cover_file_name).encode("utf-8"),
                    )
                    manifest.append(
                        ("cover", "cover.xhtml", XHTML_MEDIA_TYPE)
                    )
                    spine_ids.append("cover")

            # 正文图片只并行 normalize；zip 写入、manifest 与编号全部留在
            # 当前线程，并按章 / 图的输入顺序消费 future，保证输出与串行
            # 实现逐字节一致。某章图片全部失败时不生成该章 XHTML。
            entries: list[tuple[int, Path]] = []
            chapter_images_by_index: list[list[str]] = [
                [] for _chapter in plan.chapters
            ]
            chapter_ends: list[int] = []
            for chapter_index, chapter in enumerate(plan.chapters):
                entries.extend(
                    (chapter_index, image)
                    for image in chapter.images
                )
                chapter_ends.append(len(entries))

            image_serial = 0
            next_index = 0
            consumed = 0
            chapter_cursor = 0
            window = max(1, MAX_IMAGE_WORKERS * 2)
            pending: deque[
                Future[tuple[bytes, str, str]]
            ] = deque()
            executor = ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS)
            try:
                while consumed < len(entries):
                    check_cancelled()
                    # 预取窗口：大文件前先排空已有任务，单独提交大文件。
                    while (
                        next_index < len(entries)
                        and len(pending) < window
                    ):
                        check_cancelled()
                        image_path = entries[next_index][1]
                        if _is_large_image(image_path):
                            if pending:
                                break
                            pending.append(
                                executor.submit(
                                    normalize_epub_image, image_path
                                )
                            )
                            next_index += 1
                            break
                        pending.append(
                            executor.submit(
                                normalize_epub_image, image_path
                            )
                        )
                        next_index += 1

                    if not pending:
                        continue

                    chapter_index, image_path = entries[consumed]
                    future = pending.popleft()
                    try:
                        data, media_type, extension = future.result()
                    except Exception as exc:  # noqa: BLE001 - 跳过损坏图片
                        skipped.append((image_path, str(exc)))
                    else:
                        image_serial += 1
                        file_name = f"img_{image_serial:04d}.{extension}"
                        _write_stored_image(
                            archive,
                            f"OEBPS/images/{file_name}",
                            data,
                        )
                        manifest.append(
                            (
                                f"img_{image_serial:04d}",
                                f"images/{file_name}",
                                media_type,
                            )
                        )
                        chapter_images_by_index[
                            chapter_index
                        ].append(file_name)
                        images_written += 1

                    consumed += 1
                    processed += 1
                    if on_progress is not None:
                        on_progress(processed, total_images)

                    while (
                        chapter_cursor < len(chapter_ends)
                        and chapter_ends[chapter_cursor] <= consumed
                    ):
                        chapter_images = chapter_images_by_index[
                            chapter_cursor
                        ]
                        if chapter_images:
                            chapter_count += 1
                            xhtml_name = (
                                f"chapter_{chapter_count:03d}.xhtml"
                            )
                            archive.writestr(
                                f"OEBPS/{xhtml_name}",
                                build_chapter_xhtml(
                                    plan.chapters[
                                        chapter_cursor
                                    ].title,
                                    chapter_images,
                                    metadata.language,
                                ).encode("utf-8"),
                            )
                            chapter_id = (
                                f"chapter_{chapter_count:03d}"
                            )
                            manifest.append(
                                (
                                    chapter_id,
                                    xhtml_name,
                                    XHTML_MEDIA_TYPE,
                                )
                            )
                            spine_ids.append(chapter_id)
                            nav_entries.append(
                                (
                                    plan.chapters[
                                        chapter_cursor
                                    ].title,
                                    xhtml_name,
                                )
                            )
                        chapter_cursor += 1
                check_cancelled()
            finally:
                # 取消或异常时停止提交并等待运行中的 normalize 结束。
                executor.shutdown(wait=True, cancel_futures=True)

            if images_written == 0:
                sample = ""
                if skipped:
                    first_path, first_reason = skipped[0]
                    sample = f"\n示例：{first_path.name}：{first_reason}"
                raise RuntimeError(
                    "所有图片都无法读取或编码，未生成 EPUB。" + sample
                )

            check_cancelled()

            manifest.append(("ncx", "toc.ncx", NCX_MEDIA_TYPE))
            manifest.append(("css", "style.css", "text/css"))
            archive.writestr(
                "OEBPS/content.opf",
                build_opf(
                    metadata,
                    manifest,
                    spine_ids,
                    book_id,
                    cover_image_id,
                ).encode("utf-8"),
            )
            archive.writestr(
                "OEBPS/toc.ncx",
                build_ncx(
                    nav_entries, book_id, metadata.title
                ).encode("utf-8"),
            )

        os.replace(tmp_path, output)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return EpubResult(
        output=output,
        chapters=chapter_count,
        images=images_written,
        skipped=skipped,
    )


# ------------------------------------------------------------------- 界面部分


class EpubBuilderPage(ToolPage):
    """EPUB 2 电子书生成工具页。"""

    shutdown_label = "生成"

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._last_output: Path | None = None
        self._auto_output: str | None = None
        self._auto_cover: str | None = None
        self._input_widgets: list[QWidget] = []
        self._browse_buttons: list[QWidget] = []
        self._plan: BookPlan | None = None
        self._plan_key: tuple[str, str, bool] | None = None
        self._preview_metadata: BookMetadata | None = None
        self._preview_key: tuple[str, str, bool] | None = None
        self._preview_plan: BookPlan | None = None
        self._build_key: tuple[str, str, bool] | None = None
        self._build_ui()
        self._sync_mode_controls()

    def current_worker(self) -> QThread | None:
        return self._worker

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
            placeholder="选择图片目录（章节模式下子文件夹即章节）",
            dialog_title="选择图片目录",
            parent=self,
        )

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            "章节子文件夹（一级子目录 = 章节）", "chapters"
        )
        self.mode_combo.addItem("单文件夹（全部图片作为一章）", "single")
        form.addWidget(QLabel("目录模式："), 1, 0)
        form.addWidget(self.mode_combo, 1, 1, 1, 2)

        self.include_root_check = QCheckBox(
            "把根目录图片作为最前面的「正文」章节"
        )
        self.include_root_check.setChecked(True)
        form.addWidget(QLabel("根目录图片："), 2, 0)
        form.addWidget(self.include_root_check, 2, 1, 1, 2)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("必填，例如：我的电子书")
        form.addWidget(QLabel("书名："), 3, 0)
        form.addWidget(self.title_edit, 3, 1, 1, 2)

        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("可选")
        form.addWidget(QLabel("作者："), 4, 0)
        form.addWidget(self.author_edit, 4, 1, 1, 2)

        self.language_edit = QLineEdit("zh-CN")
        form.addWidget(QLabel("语言："), 5, 0)
        form.addWidget(self.language_edit, 5, 1, 1, 2)

        self.cover_edit = add_path_row(
            form,
            6,
            "封面文件：",
            mode="open_file",
            placeholder="可留空；章节模式会自动识别根目录的 cover.*",
            dialog_title="选择封面图片",
            file_filter=(
                "图片文件 (*.jpg *.jpeg *.png *.webp *.gif *.bmp "
                "*.tif *.tiff);;所有文件 (*)"
            ),
            parent=self,
        )

        self.output_edit = add_path_row(
            form,
            7,
            "输出 EPUB：",
            mode="save_file",
            placeholder="选择 EPUB 保存位置",
            dialog_title="保存 EPUB",
            file_filter="EPUB 文件 (*.epub)",
            parent=self,
        )

        for row in (0, 6, 7):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                self._browse_buttons.append(item.widget())

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 8, 1, 1, 2)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("预览")
        self.preview_btn.clicked.connect(self._preview)
        self.start_btn = QPushButton("开始生成")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_build)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_build)
        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output_dir)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 9, 1, 1, 2)

        layout.addLayout(form)

        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["章节 / 图片", "图片数"])
        self.preview_tree.setRootIsDecorated(True)
        self.preview_tree.setAlternatingRowColors(True)
        layout.addWidget(self.preview_tree, 1)

        self.status_label = QLabel("请选择图片目录并填写书名")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.input_edit,
            self.mode_combo,
            self.include_root_check,
            self.title_edit,
            self.author_edit,
            self.language_edit,
            self.cover_edit,
            self.output_edit,
            *self._browse_buttons,
            self.preview_btn,
            self.start_btn,
        ]

        self._cover_timer = QTimer(self)
        self._cover_timer.setSingleShot(True)
        self._cover_timer.setInterval(300)
        self._cover_timer.timeout.connect(self._autodetect_cover)

        self.input_edit.textChanged.connect(self._on_input_changed)
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.cover_edit.textEdited.connect(self._on_cover_edited)
        self.mode_combo.currentIndexChanged.connect(
            self._sync_mode_controls
        )
        self.include_root_check.toggled.connect(
            self._on_plan_options_changed
        )
        self.preview_tree.itemExpanded.connect(
            self._on_preview_item_expanded
        )

    def _on_output_edited(self, _text: str) -> None:
        self._auto_output = None

    def _on_cover_edited(self, _text: str) -> None:
        self._auto_cover = None

    def _on_input_changed(self, _text: str) -> None:
        # 输出路径立即更新；封面探测延后 300 ms，避免逐字符扫描目录。
        self._update_default_output()
        self._cover_timer.start()

    def _on_plan_options_changed(self, *_args: object) -> None:
        # 目录模式变化会使已缓存的计划失效。
        self._cover_timer.start()

    def _update_default_output(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return
        root = Path(text).expanduser()
        if not root.name:
            return
        default_output = str(root.parent / f"{root.name}.epub")
        current = self.output_edit.text().strip()
        if not current or current == self._auto_output:
            self._auto_output = default_output
            self.output_edit.setText(default_output)

    @staticmethod
    def _plan_cache_key(
        input_root: Path, mode: str, include_root: bool
    ) -> tuple[str, str, bool]:
        return (str(input_root), mode, bool(include_root))

    def _current_plan(self) -> BookPlan | None:
        """返回与当前输入及磁盘状态一致的计划，否则返回 None。"""
        plan = self._plan
        if plan is None:
            return None
        text = self.input_edit.text().strip()
        if not text:
            return None
        try:
            root = Path(text).expanduser()
        except (OSError, RuntimeError):
            return None
        mode = str(self.mode_combo.currentData() or "chapters")
        key = self._plan_cache_key(
            root, mode, self.include_root_check.isChecked()
        )
        if key != self._plan_key:
            return None
        if not book_plan_is_current(plan):
            return None
        return plan

    def _autodetect_cover(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return
        root = Path(text).expanduser()
        if not root.is_dir():
            return
        current = self.cover_edit.text().strip()
        if current and current != self._auto_cover:
            return
        plan = self._current_plan()
        if plan is not None:
            found = plan.cover
        else:
            try:
                found = _find_cover_image(root)
            except OSError:
                return
        if found is not None:
            self._auto_cover = str(found)
            if self.cover_edit.text().strip() != self._auto_cover:
                self.cover_edit.setText(self._auto_cover)
        elif current == self._auto_cover:
            self._auto_cover = None
            self.cover_edit.clear()

    def _sync_mode_controls(self, *_args: object) -> None:
        is_chapters = self.mode_combo.currentData() == "chapters"
        self.include_root_check.setEnabled(is_chapters and not self._busy)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            # 避免已排队的封面探测在后台任务期间又扫描目录。
            self._cover_timer.stop()
        for widget in self._input_widgets:
            widget.setEnabled(not busy)
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if busy:
            self.open_btn.setEnabled(False)
        else:
            self._sync_mode_controls()

    # --------------------------------------------------------------- 校验流程

    def _collect_common(
        self,
    ) -> tuple[Path, str, bool, BookMetadata] | None:
        """校验输入目录、书名与封面；失败时弹窗并返回 None。"""
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

        title = self.title_edit.text().strip()
        if not title:
            QMessageBox.warning(
                self, "缺少书名", "请填写电子书书名。"
            )
            return None

        cover_text = self.cover_edit.text().strip()
        cover_path: Path | None = None
        if cover_text:
            cover_path = Path(cover_text).expanduser()
            if not cover_path.exists():
                QMessageBox.warning(
                    self, "封面文件不存在", f"找不到文件：\n{cover_path}"
                )
                return None
            if not cover_path.is_file():
                QMessageBox.warning(
                    self,
                    "封面路径无效",
                    f"封面路径不是文件：\n{cover_path}",
                )
                return None
            if not os.access(cover_path, os.R_OK):
                QMessageBox.warning(
                    self,
                    "封面文件不可读",
                    f"没有读取权限：\n{cover_path}",
                )
                return None

        mode = str(self.mode_combo.currentData() or "chapters")
        metadata = BookMetadata(
            title=title,
            author=self.author_edit.text().strip(),
            language=self.language_edit.text().strip() or "zh-CN",
            cover_path=cover_path,
        )
        return (
            input_root,
            mode,
            self.include_root_check.isChecked(),
            metadata,
        )

    def _prepare_output(self) -> Path | None:
        """校验输出路径、补齐 .epub 后缀并确认覆盖。"""
        text = self.output_edit.text().strip()
        if not text:
            QMessageBox.warning(
                self, "缺少输出文件", "请选择 EPUB 的保存位置。"
            )
            return None
        output = Path(text).expanduser()
        if output.suffix.lower() != ".epub":
            output = Path(f"{output}.epub")
            self.output_edit.setText(str(output))
        if output.exists() and output.is_dir():
            QMessageBox.warning(
                self,
                "输出路径无效",
                f"输出路径是一个目录，请选择 EPUB 文件路径：\n{output}",
            )
            return None
        if not output.parent.is_dir():
            QMessageBox.warning(
                self,
                "输出目录不存在",
                f"目录不存在：\n{output.parent}",
            )
            return None
        if not os.access(output.parent, os.W_OK):
            QMessageBox.warning(
                self,
                "输出目录不可写",
                f"没有写入权限：\n{output.parent}",
            )
            return None
        if output.exists():
            reply = QMessageBox.question(
                self,
                "文件已存在",
                f"输出文件已存在：\n{output}\n\n继续将覆盖该文件，是否继续？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None
        return output

    # --------------------------------------------------------------- 预览流程

    def _preview(self) -> None:
        if self._busy:
            return
        collected = self._collect_common()
        if collected is None:
            return
        input_root, mode, include_root, metadata = collected
        cached = self._current_plan()
        if cached is not None:
            self._populate_preview(cached, metadata)
            return

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> BookPlan:
            return discover_book(
                input_root,
                mode=mode,
                include_root_images=include_root,
                cancel_event=cancel_event,
                on_progress=lambda found: on_progress(found, 0),
            )

        self._preview_key = self._plan_cache_key(
            input_root, mode, include_root
        )
        self._preview_metadata = metadata
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status("正在扫描图片目录…")
        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_progress)
        worker.task_finished.connect(self._on_preview_finished)
        worker.task_cancelled.connect(self._on_preview_cancelled)
        worker.task_failed.connect(self._on_preview_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_preview_finished(self, result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        if not isinstance(result, BookPlan):
            self._on_preview_failed("后台任务返回了未知结果")
            return
        metadata = self._preview_metadata
        if metadata is None:
            self._on_preview_failed("预览缺少书籍信息")
            return
        self._plan = result
        self._plan_key = self._preview_key
        self._populate_preview(result, metadata)

    def _on_preview_cancelled(self, _result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("已取消预览")

    def _on_preview_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("预览失败")
        QMessageBox.warning(self, "无法读取图片目录", message)

    def _populate_preview(
        self, plan: BookPlan, metadata: BookMetadata
    ) -> None:
        body_count = sum(len(chapter.images) for chapter in plan.chapters)
        cover = (
            metadata.cover_path
            if metadata.cover_path is not None
            else plan.cover
        )
        lazy = plan.total_images > PREVIEW_IMAGE_NODE_LIMIT

        self._preview_plan = None
        self.preview_tree.setUpdatesEnabled(False)
        try:
            self.preview_tree.clear()

            if cover is not None:
                cover_item = QTreeWidgetItem(
                    ["封面（不进入正文）", "1"]
                )
                cover_item.setToolTip(0, str(cover))
                QTreeWidgetItem(cover_item, [cover.name, ""])
                self.preview_tree.addTopLevelItem(cover_item)

            for index, chapter in enumerate(plan.chapters):
                item = QTreeWidgetItem(
                    [chapter.title, str(len(chapter.images))]
                )
                if lazy:
                    item.setData(
                        0, Qt.ItemDataRole.UserRole, index
                    )
                    placeholder = QTreeWidgetItem(
                        ["展开章节查看图片", ""]
                    )
                    placeholder.setData(
                        0, Qt.ItemDataRole.UserRole, -1
                    )
                    item.addChild(placeholder)
                else:
                    for image in chapter.images:
                        child = QTreeWidgetItem([image.name, ""])
                        child.setToolTip(0, str(image))
                        item.addChild(child)
                self.preview_tree.addTopLevelItem(item)

            if not lazy:
                for column in (0, 1):
                    self.preview_tree.resizeColumnToContents(column)
                self.preview_tree.expandToDepth(0)
        finally:
            self.preview_tree.setUpdatesEnabled(True)
        self._preview_plan = plan

        cover_text = cover.name if cover is not None else "无"
        suffix = "；图片较多，展开章节查看" if lazy else ""
        self._set_status(
            f"预览：{len(plan.chapters)} 章 / {body_count} 张正文图片，"
            f"封面：{cover_text}，共 {plan.total_images} 张图片"
            f"{suffix}"
        )

    def _on_preview_item_expanded(self, item: QTreeWidgetItem) -> None:
        """大目录预览：首次展开章节时再创建该章的图片节点。"""
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or index < 0:
            return
        plan = self._preview_plan
        if plan is None or index >= len(plan.chapters):
            return
        if item.childCount() != 1:
            return
        placeholder = item.child(0)
        if placeholder.data(0, Qt.ItemDataRole.UserRole) != -1:
            return

        item.takeChildren()
        self.preview_tree.setUpdatesEnabled(False)
        try:
            for image in plan.chapters[index].images:
                child = QTreeWidgetItem([image.name, ""])
                child.setToolTip(0, str(image))
                item.addChild(child)
        finally:
            self.preview_tree.setUpdatesEnabled(True)

    # --------------------------------------------------------------- 生成流程

    def _start_build(self) -> None:
        if self._busy:
            return
        collected = self._collect_common()
        if collected is None:
            return
        input_root, mode, include_root, metadata = collected
        output = self._prepare_output()
        if output is None:
            return

        cached = self._current_plan()
        self._last_output = output
        self._build_key = self._plan_cache_key(
            input_root, mode, include_root
        )
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        if cached is None:
            self.progress_bar.setRange(0, 0)
            self._set_status("正在扫描图片目录…")
        else:
            self._set_status(
                f"正在生成 EPUB：{len(cached.chapters)} 章，"
                f"共 {cached.total_images} 张图片…"
            )
        self._set_busy(True)

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> tuple[EpubResult, BookPlan]:
            plan = cached
            if plan is None:
                plan = discover_book(
                    input_root,
                    mode=mode,
                    include_root_images=include_root,
                    cancel_event=cancel_event,
                    on_progress=lambda found: on_progress(found, 0),
                )
            result = build_epub(
                plan,
                metadata,
                output,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )
            return result, plan

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
        if total_count <= 0:
            self.progress_bar.setRange(0, 0)
            self._set_status(
                f"正在扫描图片目录：已发现 {done_count} 张…"
            )
            return
        self.progress_bar.setRange(0, 100)
        percent = int(done_count / total_count * 100)
        self.progress_bar.setValue(min(100, percent))
        self._set_status(
            f"正在处理第 {done_count}/{total_count} 张图片…"
        )

    def _on_finished(self, result: object) -> None:
        plan: BookPlan | None = None
        if isinstance(result, tuple) and len(result) == 2:
            result, plan = result
        if not isinstance(result, EpubResult):
            self._on_failed("任务返回了无法识别的结果。")
            return
        if isinstance(plan, BookPlan):
            self._plan = plan
            self._plan_key = self._build_key

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.open_btn.setEnabled(True)
        self._last_output = result.output
        skipped_count = len(result.skipped)
        self._set_status(
            f"完成：{result.chapters} 章 / {result.images} 张图片，"
            f"跳过 {skipped_count} 张 → {result.output}"
        )
        summary = (
            f"章节数：{result.chapters}\n"
            f"图片数：{result.images}\n"
            f"跳过：{skipped_count}\n"
            f"输出：{result.output}"
        )
        if not skipped_count:
            QMessageBox.information(self, "生成完成", summary)
            return

        preview = "\n".join(
            f"{path.name}：{reason}"
            for path, reason in result.skipped[:5]
        )
        if skipped_count > 5:
            preview += f"\n…另有 {skipped_count - 5} 张"
        QMessageBox.warning(
            self,
            "生成完成（有跳过项）",
            f"{summary}\n\n跳过项：\n{preview}",
        )

    def _on_cancelled(self, _result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("已取消，未生成 EPUB 文件")

    def _on_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
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
            QMessageBox.critical(
                self, "无法打开目录", f"{folder}\n{exc}"
            )
