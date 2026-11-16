"""图片扫描、方向校正与安全保存的共享逻辑（依赖 Pillow）。"""

from __future__ import annotations

import io
import os
import tempfile
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from toolbox.common import IMAGE_EXTENSIONS, natural_path_key

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


def is_image_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def media_type_for(path: Path | str) -> str:
    return _MEDIA_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _is_excluded(path: Path, excluded: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return True
    return any(
        resolved == item or resolved.is_relative_to(item) for item in excluded
    )


def scan_image_files(
    root: Path | str,
    *,
    recursive: bool = False,
    exclude_roots: tuple[Path | str, ...] = (),
    include_hidden: bool = False,
) -> list[Path]:
    """扫描图片文件，按相对路径自然排序返回。"""
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
        excluded.append(resolved)

    results: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(
            root, followlinks=False
        ):
            current = Path(dirpath)
            kept: list[str] = []
            for name in dirnames:
                child = current / name
                if child.is_symlink():
                    continue
                if not include_hidden and _is_hidden(name):
                    continue
                if _is_excluded(child, excluded):
                    continue
                kept.append(name)
            dirnames[:] = kept

            for name in filenames:
                if not include_hidden and _is_hidden(name):
                    continue
                path = current / name
                if path.is_symlink() or not is_image_file(path):
                    continue
                if _is_excluded(path, excluded):
                    continue
                results.append(path)
    else:
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_file():
                continue
            if not include_hidden and _is_hidden(entry.name):
                continue
            if is_image_file(entry):
                results.append(entry)

    results.sort(key=lambda path: natural_path_key(path.relative_to(root)))
    return results


def load_oriented(path: Path | str) -> Image.Image:
    """读取图片并按 EXIF 方向转正；调用方负责关闭返回的图片。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as opened:
            opened.load()
            return ImageOps.exif_transpose(opened)


def oriented_size(path: Path | str) -> tuple[int, int]:
    """只读文件头获取考虑 EXIF 方向后的宽高。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            width, height = image.size
            try:
                orientation = image.getexif().get(0x0112, 1)
            except Exception:  # noqa: BLE001 - 无 EXIF 时按正常方向
                orientation = 1
    if orientation in {5, 6, 7, 8}:
        return height, width
    return width, height


def is_animated(image: Image.Image) -> bool:
    return getattr(image, "n_frames", 1) > 1


def flatten_rgb(
    image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """把带透明通道或非 RGB 模式的图片铺白底转为 RGB。"""
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, background)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


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


def image_to_bytes(
    image: Image.Image, image_format: str = "jpeg", quality: int = 92
) -> tuple[bytes, str]:
    """编码为 JPEG / PNG / WebP 字节，返回（数据, 扩展名）。"""
    fmt = image_format.lower()
    buffer = io.BytesIO()
    if fmt in {"jpg", "jpeg"}:
        flatten_rgb(image).save(
            buffer, format="JPEG", quality=quality, optimize=True
        )
        return buffer.getvalue(), "jpg"
    if fmt == "png":
        source = image
        if image.mode not in {"RGB", "RGBA", "L", "LA"}:
            source = image.convert(
                "RGBA" if "transparency" in image.info else "RGB"
            )
        source.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "png"
    if fmt == "webp":
        source = image
        if image.mode not in {"RGB", "RGBA"}:
            source = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        source.save(buffer, format="WEBP", quality=quality, method=4)
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
    return target
