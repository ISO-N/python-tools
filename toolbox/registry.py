"""工具箱内各工具的注册表。

新增工具时，在 ``TOOLS`` 的构造列表中添加一个 ``ToolSpec`` 即可；列表会
按显示名的字面字典序（Unicode 码点）自动升序排列。工厂函数内部再导入具体
页面，实现按需加载，避免启动时加载 pymupdf 等较重的依赖。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from toolbox.page import ToolPage


@dataclass(frozen=True)
class ToolSpec:
    """一个工具在工具箱中的注册信息。"""

    key: str
    title: str
    description: str
    factory: Callable[[], ToolPage]


def _create_pdf_to_images() -> ToolPage:
    from toolbox.tools.pdf_to_images import PdfToImagesPage

    return PdfToImagesPage()


def _create_move_images_to_folders() -> ToolPage:
    from toolbox.tools.move_images_to_folders import MoveImagesPage

    return MoveImagesPage()


def _create_images_to_pdf() -> ToolPage:
    from toolbox.tools.images_to_pdf import ImagesToPdfPage

    return ImagesToPdfPage()


def _create_pdf_merge_split() -> ToolPage:
    from toolbox.tools.pdf_merge_split import PdfMergeSplitPage

    return PdfMergeSplitPage()


def _create_pdf_compress() -> ToolPage:
    from toolbox.tools.pdf_compress import PdfCompressPage

    return PdfCompressPage()


def _create_pdf_page_tools() -> ToolPage:
    from toolbox.tools.pdf_page_tools import PdfPageToolsPage

    return PdfPageToolsPage()


def _create_batch_image_convert() -> ToolPage:
    from toolbox.tools.batch_image_convert import BatchImageConvertPage

    return BatchImageConvertPage()


def _create_batch_rename() -> ToolPage:
    from toolbox.tools.batch_rename import BatchRenamePage

    return BatchRenamePage()


def _create_image_collage() -> ToolPage:
    from toolbox.tools.image_collage import ImageCollagePage

    return ImageCollagePage()


def _create_image_watermark() -> ToolPage:
    from toolbox.tools.image_watermark import ImageWatermarkPage

    return ImageWatermarkPage()


def _create_duplicate_finder() -> ToolPage:
    from toolbox.tools.duplicate_finder import DuplicateFinderPage

    return DuplicateFinderPage()


def _create_file_organizer() -> ToolPage:
    from toolbox.tools.file_organizer import FileOrganizerPage

    return FileOrganizerPage()


def _create_epub_builder() -> ToolPage:
    from toolbox.tools.epub_builder import EpubBuilderPage

    return EpubBuilderPage()


TOOLS: tuple[ToolSpec, ...] = tuple(
    sorted(
        (
            ToolSpec(
                key="pdf_to_images",
                title="PDF 转图片",
                description="把 PDF 逐页渲染为 PNG / JPEG，支持页码范围、DPI 与取消。",
                factory=_create_pdf_to_images,
            ),
            ToolSpec(
                key="move_images_to_folders",
                title="图片按页码归档",
                description=(
                    "按 JSON 配置的章节起始页码，把图片移动到对应子文件夹，"
                    "支持预览与撤销。"
                ),
                factory=_create_move_images_to_folders,
            ),
            ToolSpec(
                key="images_to_pdf",
                title="图片合成 PDF",
                description=(
                    "把图片文件夹按自然页码顺序合成为 PDF，"
                    "支持递归子文件夹、A4 适配与原图尺寸。"
                ),
                factory=_create_images_to_pdf,
            ),
            ToolSpec(
                key="pdf_merge_split",
                title="PDF 合并 / 拆分",
                description=(
                    "按顺序合并多个 PDF；或按每页、每 N 页、自定义范围拆分。"
                ),
                factory=_create_pdf_merge_split,
            ),
            ToolSpec(
                key="pdf_compress",
                title="PDF 压缩",
                description=(
                    "无损清理冗余数据，或把页面重编码为图片以大幅压缩扫描件。"
                ),
                factory=_create_pdf_compress,
            ),
            ToolSpec(
                key="pdf_page_tools",
                title="PDF 页码 / 书签 / 水印",
                description=(
                    "批量添加页脚页码、按章节 JSON 生成目录书签、"
                    "绘制中文文字水印。"
                ),
                factory=_create_pdf_page_tools,
            ),
            ToolSpec(
                key="batch_image_convert",
                title="图片批量转换",
                description=(
                    "批量转换为 JPEG / PNG / WebP，支持长边或百分比缩放，"
                    "保留目录结构。"
                ),
                factory=_create_batch_image_convert,
            ),
            ToolSpec(
                key="batch_rename",
                title="文件批量重命名",
                description=(
                    "查找替换、序号重排、加前后缀；表格预览、冲突检测、"
                    "两阶段执行并支持撤销。"
                ),
                factory=_create_batch_rename,
            ),
            ToolSpec(
                key="image_collage",
                title="图片长图 / 拼图",
                description=(
                    "把图片按自然顺序拼接为纵向长图或网格拼图，"
                    "可设宽度、列数、间距与背景色。"
                ),
                factory=_create_image_collage,
            ),
            ToolSpec(
                key="image_watermark",
                title="图片批量水印",
                description=(
                    "批量添加中文文字水印，支持九宫格位置或平铺、"
                    "透明度、颜色与角度。"
                ),
                factory=_create_image_watermark,
            ),
            ToolSpec(
                key="duplicate_finder",
                title="重复文件查找",
                description=(
                    "按大小、前缀哈希与完整哈希查找内容相同的文件；"
                    "移入隔离区而非删除，并支持撤销移回。"
                ),
                factory=_create_duplicate_finder,
            ),
            ToolSpec(
                key="file_organizer",
                title="文件批量整理",
                description=(
                    "按扩展名、修改日期或文件大类整理文件，"
                    "支持递归、冲突处理与撤销。"
                ),
                factory=_create_file_organizer,
            ),
            ToolSpec(
                key="epub_builder",
                title="EPUB 电子书制作",
                description=(
                    "把章节子文件夹中的图片打包为 EPUB 2 电子书，"
                    "支持封面、元数据与目录。"
                ),
                factory=_create_epub_builder,
            ),
        ),
        key=lambda spec: spec.title,
    )
)
