"""图片扫描、方向校正与安全保存的共享逻辑（依赖 Pillow）。"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from toolbox.common import IMAGE_EXTENSIONS, natural_path_key
from toolbox.errors import Cancelled

# 与旧实现一致：把“解压炸弹”警告升级为异常。警告过滤器是进程级全局
# 状态，因此只在这里设置一次，不再在线程池里使用 catch_warnings。
warnings.filterwarnings("error", category=Image.DecompressionBombWarning)

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@dataclass(frozen=True)
class ImageProbe:
    """只读文件头得到的图片元数据。"""

    path: Path
    width: int
    height: int
    mode: str
    format: str
    orientation: int
    frames: int
    has_alpha: bool


@dataclass(frozen=True)
class ImagePayload:
    """可直接嵌入 PDF / EPUB 的图片数据。"""

    data: bytes
    width: int
    height: int
    has_alpha: bool
    extension: str
    reusable_original: bool


def is_image_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def media_type_for(path: Path | str) -> str:
    return _MEDIA_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def image_has_alpha(image: Image.Image) -> bool:
    """判断图片是否带透明通道（包括 P 模式调色板透明）。"""
    return any(band.lower() == "a" for band in image.getbands()) or (
        image.mode == "P" and "transparency" in image.info
    )


def _orientation_of(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(0x0112, 1) or 1)
    except Exception:  # noqa: BLE001 - EXIF 损坏时按正常方向处理
        return 1


def probe_image(path: Path | str) -> ImageProbe:
    """打开一次文件头，返回尺寸 / 方向 / 动画 / alpha 等信息。"""
    path = Path(path)
    with Image.open(path) as image:
        width, height = image.size
        orientation = _orientation_of(image)
        if orientation in {5, 6, 7, 8}:
            width, height = height, width
        return ImageProbe(
            path=path,
            width=width,
            height=height,
            mode=image.mode,
            format=(image.format or "").upper(),
            orientation=orientation,
            frames=int(getattr(image, "n_frames", 1) or 1),
            has_alpha=image_has_alpha(image),
        )


def scan_image_files(
    root: Path | str,
    *,
    recursive: bool = False,
    exclude_roots: tuple[Path | str, ...] = (),
    include_hidden: bool = False,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> list[Path]:
    """扫描图片文件，按相对路径自然排序返回。

    排除目录为空时完全跳过 ``resolve()``；递归遍历使用 ``os.scandir`` 的
    ``DirEntry`` 缓存，避免对每个文件额外 stat。
    """
    root = Path(root)
    if not root.is_dir():
        return []

    root_resolved = root.resolve()
    excluded: list[Path] = []
    for item in exclude_roots:
        try:
            resolved = Path(item).resolve()
        except OSError:
            continue
        # 排除目录是扫描根本身或祖先时忽略，否则会把整个输入目录排除。
        if resolved == root_resolved or root_resolved.is_relative_to(resolved):
            continue
        try:
            excluded.append(resolved.relative_to(root_resolved))
        except ValueError:
            continue

    def is_excluded(relative: Path) -> bool:
        if not excluded:
            return False
        return any(
            relative == item or relative.is_relative_to(item)
            for item in excluded
        )

    found: list[tuple[Path, Path]] = []

    def consider(entry: os.DirEntry[str], relative: Path) -> bool:
        if not include_hidden and _is_hidden(entry.name):
            return False
        if entry.is_symlink():
            return False
        if is_excluded(relative):
            return False
        if entry.is_file(follow_symlinks=False) and is_image_file(
            entry.name
        ):
            found.append((relative, Path(entry.path)))
            if on_progress is not None:
                on_progress(len(found))
            return False
        return entry.is_dir(follow_symlinks=False)

    if recursive:
        stack: list[tuple[Path, Path]] = [(root, Path())]
        while stack:
            current, current_rel = stack.pop()
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        relative = current_rel / entry.name
                        try:
                            if consider(entry, relative):
                                stack.append((Path(entry.path), relative))
                        except OSError:
                            continue
            except OSError:
                continue
    else:
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if cancel_event is not None and cancel_event.is_set():
                        raise Cancelled()
                    try:
                        consider(entry, Path(entry.name))
                    except OSError:
                        continue
        except OSError:
            return []

    found.sort(key=lambda item: natural_path_key(item[0]))
    return [path for _relative, path in found]


def load_oriented(
    path: Path | str,
    *,
    max_size: tuple[int, int] | None = None,
) -> Image.Image:
    """读取图片并按 EXIF 方向转正；调用方负责关闭返回的图片。

    ``max_size`` 用于 JPEG 的 ``draft`` 解码降采样；方向为 1 时不再复制
    整幅图。Pillow 版本不支持 ``in_place`` 时自动回退。
    """
    with Image.open(path) as opened:
        orientation = _orientation_of(opened)
        if max_size is not None and opened.format in {"JPEG", "MPO"}:
            hint = max_size
            if orientation in {5, 6, 7, 8}:
                hint = (max_size[1], max_size[0])
            try:
                opened.draft(opened.mode, hint)
            except Exception:  # noqa: BLE001 - draft 仅是可选的加速
                pass
        opened.load()
        if orientation == 1:
            return opened
        try:
            ImageOps.exif_transpose(opened, in_place=True)
            return opened
        except TypeError:  # 旧版 Pillow 没有 in_place 参数
            return ImageOps.exif_transpose(opened)


def oriented_size(path: Path | str) -> tuple[int, int]:
    """只读文件头获取考虑 EXIF 方向后的宽高。"""
    probe = probe_image(path)
    return probe.width, probe.height


def is_animated(image: Image.Image) -> bool:
    return getattr(image, "n_frames", 1) > 1


def flatten_rgb(
    image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """把带透明通道或非 RGB 模式的图片铺白底转为 RGB。"""
    if image.mode == "RGB":
        return image
    has_alpha = image_has_alpha(image)
    if image.mode == "RGBA":
        canvas = Image.new("RGB", image.size, background)
        alpha = image.getchannel("A")
        try:
            canvas.paste(image, mask=alpha)
        finally:
            alpha.close()
        return canvas
    if has_alpha:
        rgba = image.convert("RGBA")
        try:
            alpha = rgba.getchannel("A")
            try:
                canvas = Image.new("RGB", rgba.size, background)
                canvas.paste(rgba, mask=alpha)
            finally:
                alpha.close()
        finally:
            rgba.close()
        return canvas
    return image.convert("RGB")


def fit_size(
    src_width: int,
    src_height: int,
    box_width: float,
    box_height: float,
    *,
    upscale: bool = True,
) -> tuple[int, int]:
    """等比缩放到指定盒子内，返回整数尺寸。"""
    if src_width <= 0 or src_height <= 0:
        return 1, 1
    scale = min(box_width / src_width, box_height / src_height)
    if not upscale:
        scale = min(scale, 1.0)
    return max(1, round(src_width * scale)), max(1, round(src_height * scale))


def _trailer_looks_complete(raw: bytes, image_format: str) -> bool:
    """用文件尾标记做低成本的完整性检查。"""
    stripped = raw.rstrip()
    if image_format == "JPEG":
        return stripped.endswith(b"\xff\xd9")
    if image_format == "PNG":
        return stripped.endswith(b"IEND\xaeB`\x82")
    return True


def prepare_image_payload(
    path: Path | str,
    *,
    jpeg_quality: int = 92,
    prefer_original: bool = True,
    optimize: bool = False,
    verify: str = "trailer",
) -> ImagePayload:
    """准备图片字节：方向正常且格式兼容时直接复用原文件字节。

    ``verify="trailer"`` 只检查文件尾标记；``"full"`` 强制完整解码验证。
    需要方向校正或重新编码时只读取文件一次，并在内存流上解码。
    """
    if verify not in {"trailer", "full"}:
        raise ValueError(f"不支持的校验方式：{verify}")
    path = Path(path)
    raw = path.read_bytes()

    with Image.open(io.BytesIO(raw)) as opened:
        image_format = (opened.format or "").upper()
        mode = opened.mode
        frames = int(getattr(opened, "n_frames", 1) or 1)
        orientation = _orientation_of(opened)
        has_alpha = image_has_alpha(opened)
        width, height = opened.size
        if orientation in {5, 6, 7, 8}:
            width, height = height, width

        reusable = (
            prefer_original
            and orientation == 1
            and frames <= 1
            and (
                (image_format == "JPEG" and mode in {"RGB", "L"})
                or (
                    image_format == "PNG"
                    and mode in {"RGB", "RGBA", "L", "LA"}
                    and not (mode == "P" and "transparency" in opened.info)
                )
            )
        )
        if reusable:
            if verify == "full":
                try:
                    opened.load()
                except Exception:  # noqa: BLE001 - 回退到重新编码路径
                    reusable = False
            elif not _trailer_looks_complete(raw, image_format):
                reusable = False

        if reusable:
            extension = "jpg" if image_format == "JPEG" else "png"
            return ImagePayload(
                raw, width, height, has_alpha, extension, True
            )

        opened.load()
        owns_image = False
        image: Image.Image = opened
        if orientation != 1:
            try:
                ImageOps.exif_transpose(opened, in_place=True)
            except TypeError:  # 旧版 Pillow 没有 in_place 参数
                image = ImageOps.exif_transpose(opened)
                owns_image = True
        try:
            width, height = image.width, image.height
            has_alpha = image_has_alpha(image)
            buffer = io.BytesIO()
            if has_alpha:
                target = (
                    image
                    if image.mode == "RGBA"
                    else image.convert("RGBA")
                )
                try:
                    target.save(buffer, format="PNG", optimize=optimize)
                finally:
                    if target is not image:
                        target.close()
                return ImagePayload(
                    buffer.getvalue(),
                    width,
                    height,
                    True,
                    "png",
                    False,
                )

            target = flatten_rgb(image)
            try:
                target.save(
                    buffer,
                    format="JPEG",
                    quality=jpeg_quality,
                    optimize=optimize,
                )
            finally:
                if target is not image:
                    target.close()
            return ImagePayload(
                buffer.getvalue(), width, height, False, "jpg", False
            )
        finally:
            if owns_image:
                image.close()


def image_to_bytes(
    image: Image.Image,
    image_format: str = "jpeg",
    quality: int = 92,
    *,
    optimize: bool = True,
) -> tuple[bytes, str]:
    """编码为 JPEG / PNG / WebP 字节，返回（数据, 扩展名）。"""
    fmt = image_format.lower()
    buffer = io.BytesIO()
    if fmt in {"jpg", "jpeg"}:
        flatten_rgb(image).save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=optimize,
        )
        return buffer.getvalue(), "jpg"
    if fmt == "png":
        source = image
        if image.mode not in {"RGB", "RGBA", "L", "LA"}:
            source = image.convert(
                "RGBA" if "transparency" in image.info else "RGB"
            )
        try:
            source.save(buffer, format="PNG", optimize=optimize)
        finally:
            if source is not image:
                source.close()
        return buffer.getvalue(), "png"
    if fmt == "webp":
        source = image
        if image.mode not in {"RGB", "RGBA"}:
            source = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        try:
            source.save(buffer, format="WEBP", quality=quality, method=4)
        finally:
            if source is not image:
                source.close()
        return buffer.getvalue(), "webp"
    raise ValueError(f"不支持的图片格式：{image_format}")


def save_image_atomic(
    image: Image.Image,
    target: Path | str,
    *,
    image_format: str,
    **save_kwargs: object,
) -> Path:
    """先写临时文件再原子替换，避免中途失败留下损坏的目标文件。"""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".img", dir=target.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    fmt = image_format.upper()
    source = image
    if fmt in {"JPEG", "JPG"}:
        source = flatten_rgb(image)
    elif fmt == "WEBP" and image.mode not in {"RGB", "RGBA"}:
        source = image.convert(
            "RGBA"
            if "A" in image.getbands() or "transparency" in image.info
            else "RGB"
        )
    try:
        source.save(tmp_path, format=image_format, **save_kwargs)
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if source is not image:
            source.close()
    return target
