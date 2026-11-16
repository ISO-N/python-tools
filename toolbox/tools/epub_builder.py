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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from PIL import Image
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
from toolbox.images import load_oriented, scan_image_files
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import Cancelled, SingleTaskWorker, shutdown_worker

EPUB_MIMETYPE = "application/epub+zip"
JPEG_QUALITY = 92
XHTML_MEDIA_TYPE = "application/xhtml+xml"
NCX_MEDIA_TYPE = "application/x-dtbncx+xml"

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


def _find_cover_image(root: Path) -> Path | None:
    """查找根目录下文件名主干为 cover 的图片（大小写不敏感）。"""
    for path in scan_image_files(root, recursive=False):
        if path.stem.casefold() == "cover":
            return path
    return None


def discover_book(
    input_root: Path | str,
    *,
    mode: str = "chapters",
    include_root_images: bool = True,
) -> BookPlan:
    """扫描输入目录并生成书籍组织方案。

    ``chapters`` 模式：第一层非隐藏、非符号链接子目录各为一章，章内图片
    递归扫描并按自然顺序排列；根目录图片可作为最前面的「正文」章节，根目录
    下名为 ``cover`` 的图片自动作为封面并从正文中排除。若没有子目录，根目录
    图片退化为单章。

    ``single`` 模式：根目录（含子目录）的全部图片作为一章，章标题为目录名；
    根目录下名为 ``cover`` 的图片同样自动识别为封面并从正文中排除。

    空章节会被忽略；目录中没有任何图片时抛出 ``ValueError``。
    """
    root = Path(input_root)
    if mode not in {"chapters", "single"}:
        raise ValueError(f"不支持的目录模式：{mode}")
    if not root.is_dir():
        raise ValueError(f"输入目录不存在：{root}")

    if mode == "single":
        cover = _find_cover_image(root)
        images = [
            path
            for path in scan_image_files(root, recursive=True)
            if cover is None or path != cover
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
        )

    subdirs = [
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.is_symlink()
        and not _is_hidden(entry.name)
    ]
    subdirs.sort(key=lambda path: natural_path_key(path.name))

    cover = _find_cover_image(root)
    root_images: list[Path] = []
    if include_root_images:
        root_images = [
            path
            for path in scan_image_files(root, recursive=False)
            if cover is None or path != cover
        ]

    chapters: list[Chapter] = []
    if root_images:
        chapters.append(Chapter(title="正文", images=root_images))

    for directory in subdirs:
        images = scan_image_files(directory, recursive=True)
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
    )


def _has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def normalize_epub_image(path: Path | str) -> tuple[bytes, str, str]:
    """把图片转换为 EPUB2 可用的数据。

    返回 ``(数据, media_type, 扩展名)``，扩展名不带点，只会是 ``jpg`` 或
    ``png``。方向为 1 且模式兼容的 JPEG / PNG 直接复用原文件字节；其余图片
    转正方向后重新编码：有透明通道输出 PNG RGBA，否则输出 JPEG（quality=92）。
    动画 GIF / WebP 只取第一帧并输出 PNG。无法读取或编码时抛出中文
    ``ValueError``。
    """
    path = Path(path)

    # 先探测格式/EXIF 方向/动画帧数；对可直接复用的图片完整解码一次，
    # 以便发现被截断或损坏的文件。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", Image.DecompressionBombWarning
            )
            with Image.open(path) as probe:
                image_format = (probe.format or "").upper()
                image_mode = probe.mode
                frames = getattr(probe, "n_frames", 1)
                try:
                    orientation = probe.getexif().get(0x0112, 1)
                except Exception:  # noqa: BLE001 - EXIF 损坏时按正常方向处理
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
                if reusable:
                    probe.load()
                    data = path.read_bytes()
                    if image_format == "JPEG":
                        return data, "image/jpeg", "jpg"
                    return data, "image/png", "png"
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(f"无法读取图片「{path.name}」：{exc}") from exc

    try:
        image = load_oriented(path)
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(f"无法读取图片「{path.name}」：{exc}") from exc

    try:
        animated = image_format in {"GIF", "WEBP"} and frames > 1
        alpha = _has_alpha(image)

        if animated or alpha:
            if alpha:
                target = image.convert("RGBA")
            else:
                target = image.convert("RGB")
            try:
                buffer = io.BytesIO()
                target.save(buffer, format="PNG", optimize=True)
                return buffer.getvalue(), "image/png", "png"
            finally:
                target.close()

        source = image
        converted: Image.Image | None = None
        if image.mode not in {"RGB", "L"}:
            converted = image.convert("RGB")
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
        raise ValueError(f"无法编码图片「{path.name}」：{exc}") from exc
    finally:
        image.close()


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
                    archive.writestr(
                        f"OEBPS/images/{cover_file_name}", data
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

            # 正文图片按成功顺序全局编号；某章全部失败时不生成该章 XHTML。
            image_serial = 0
            for chapter in plan.chapters:
                chapter_images: list[str] = []
                for image_path in chapter.images:
                    check_cancelled()
                    try:
                        data, media_type, extension = normalize_epub_image(
                            image_path
                        )
                    except Exception as exc:  # noqa: BLE001 - 跳过损坏图片
                        skipped.append((image_path, str(exc)))
                    else:
                        image_serial += 1
                        file_name = f"img_{image_serial:04d}.{extension}"
                        archive.writestr(
                            f"OEBPS/images/{file_name}", data
                        )
                        manifest.append(
                            (
                                f"img_{image_serial:04d}",
                                f"images/{file_name}",
                                media_type,
                            )
                        )
                        chapter_images.append(file_name)
                        images_written += 1

                    processed += 1
                    if on_progress is not None:
                        on_progress(processed, total_images)

                if not chapter_images:
                    continue

                chapter_count += 1
                xhtml_name = f"chapter_{chapter_count:03d}.xhtml"
                archive.writestr(
                    f"OEBPS/{xhtml_name}",
                    build_chapter_xhtml(
                        chapter.title,
                        chapter_images,
                        metadata.language,
                    ).encode("utf-8"),
                )
                chapter_id = f"chapter_{chapter_count:03d}"
                manifest.append(
                    (chapter_id, xhtml_name, XHTML_MEDIA_TYPE)
                )
                spine_ids.append(chapter_id)
                nav_entries.append((chapter.title, xhtml_name))

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

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._last_output: Path | None = None
        self._auto_output: str | None = None
        self._auto_cover: str | None = None
        self._input_widgets: list[QWidget] = []
        self._browse_buttons: list[QWidget] = []
        self._build_ui()
        self._sync_mode_controls()

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

        self.input_edit.textChanged.connect(self._on_input_changed)
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.cover_edit.textEdited.connect(self._on_cover_edited)
        self.mode_combo.currentIndexChanged.connect(
            self._sync_mode_controls
        )

    def _on_output_edited(self, _text: str) -> None:
        self._auto_output = None

    def _on_cover_edited(self, _text: str) -> None:
        self._auto_cover = None

    def _on_input_changed(self, _text: str) -> None:
        self._update_default_output()
        self._autodetect_cover()

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

    def _collect_plan(self) -> tuple[BookPlan, BookMetadata] | None:
        collected = self._collect_common()
        if collected is None:
            return None
        input_root, mode, include_root, metadata = collected
        try:
            plan = discover_book(
                input_root,
                mode=mode,
                include_root_images=include_root,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "无法读取图片目录", str(exc))
            return None
        except OSError as exc:
            QMessageBox.critical(
                self, "无法读取图片目录", f"{input_root}\n{exc}"
            )
            return None
        return plan, metadata

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
        collected = self._collect_plan()
        if collected is None:
            return
        plan, metadata = collected
        self._populate_preview(plan, metadata)

    def _populate_preview(
        self, plan: BookPlan, metadata: BookMetadata
    ) -> None:
        self.preview_tree.clear()
        body_count = sum(len(chapter.images) for chapter in plan.chapters)

        cover = (
            metadata.cover_path
            if metadata.cover_path is not None
            else plan.cover
        )
        if cover is not None:
            cover_item = QTreeWidgetItem(["封面（不进入正文）", "1"])
            cover_item.setToolTip(0, str(cover))
            QTreeWidgetItem(cover_item, [cover.name, ""])
            self.preview_tree.addTopLevelItem(cover_item)

        for chapter in plan.chapters:
            item = QTreeWidgetItem(
                [chapter.title, str(len(chapter.images))]
            )
            for image in chapter.images:
                child = QTreeWidgetItem([image.name, ""])
                child.setToolTip(0, str(image))
                item.addChild(child)
            self.preview_tree.addTopLevelItem(item)

        for column in (0, 1):
            self.preview_tree.resizeColumnToContents(column)
        self.preview_tree.expandToDepth(0)

        cover_text = cover.name if cover is not None else "无"
        self._set_status(
            f"预览：{len(plan.chapters)} 章 / {body_count} 张正文图片，"
            f"封面：{cover_text}，共 {plan.total_images} 张图片"
        )

    # --------------------------------------------------------------- 生成流程

    def _start_build(self) -> None:
        if self._busy:
            return
        collected = self._collect_plan()
        if collected is None:
            return
        plan, metadata = collected
        output = self._prepare_output()
        if output is None:
            return

        self._last_output = output
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._set_status(
            f"正在生成 EPUB：{len(plan.chapters)} 章，"
            f"共 {plan.total_images} 张图片…"
        )

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> EpubResult:
            return build_epub(
                plan,
                metadata,
                output,
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
        self._set_status(
            f"正在处理第 {done_count}/{total_count} 张图片…"
        )

    def _on_finished(self, result: object) -> None:
        if not isinstance(result, EpubResult):
            self._on_failed("任务返回了无法识别的结果。")
            return

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
        self.progress_bar.setValue(0)
        self._set_status("已取消，未生成 EPUB 文件")

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
            QMessageBox.critical(
                self, "无法打开目录", f"{folder}\n{exc}"
            )

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；生成进行中时确认并等待线程结束。"""
        return shutdown_worker(self._worker, self.window(), "生成 EPUB")
