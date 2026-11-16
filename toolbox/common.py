"""工具箱公共 UI 辅助函数。"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase, QGuiApplication
from PyQt6.QtWidgets import QWidget

# 支持的中文字体，按优先级排列（需由系统 fontconfig 识别）
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "SimHei",
)


def apply_ui_font(app: QGuiApplication) -> str | None:
    """把可用的中文字体设为应用默认字体；找不到时返回 None。"""
    available = set(QFontDatabase.families())
    for family in CJK_FONT_CANDIDATES:
        if family in available:
            app.setFont(QFont(family, 10))
            return family
    return None


def center_on_screen(widget: QWidget) -> None:
    """把窗口移动到主屏幕可用区域中央。"""
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return
    frame = widget.frameGeometry()
    frame.moveCenter(screen.availableGeometry().center())
    widget.move(frame.topLeft())


def open_directory(path: Path) -> None:
    """在系统文件管理器中打开目录，兼容 Windows / WSL / Linux / macOS。"""
    path = Path(path)
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    if _is_wsl():
        windows_path = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        subprocess.Popen(["explorer.exe", windows_path])
        return
    subprocess.Popen(["xdg-open", str(path)])


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "microsoft" in version.lower()


# --------------------------------------------------------------------- 通用工具

#: 工具箱识别的图片扩展名（全小写）。
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

_NATURAL_TOKEN = re.compile(r"(\d+)")


def natural_sort_key(text: str) -> tuple[tuple[int, int, str], ...]:
    """按自然顺序比较字符串：数字段按数值，其余按归一化文本。

    ``page_2`` 会排在 ``page_10`` 之前；对大小写、全半角差异做归一化。
    """
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    key: list[tuple[int, int, str]] = []
    for token in _NATURAL_TOKEN.split(normalized):
        if not token:
            continue
        if token.isdigit():
            key.append((0, int(token), ""))
        else:
            key.append((1, 0, token))
    # 末尾附完整字符串，保证数字相同（如 01 与 1）时排序稳定。
    key.append((1, 0, normalized))
    return tuple(key)


def natural_path_key(path: Path | str) -> tuple:
    """按自然顺序比较路径（逐级比较各路径片段）。"""
    return tuple(natural_sort_key(part) for part in Path(path).parts)


def human_size(size: int | float) -> str:
    """把字节数格式化为易读字符串。"""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def unique_path(path: Path | str) -> Path:
    """返回一个尚不存在的路径；已存在时追加 `` (1)``、`` (2)`` 等序号。"""
    path = Path(path)
    if not path.exists():
        return path
    for index in range(1, 100000):
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为 {path} 生成不重复的文件名")


def filesystem_key(name: str) -> str:
    """生成用于跨平台文件冲突比较的键（大小写不敏感 + Unicode NFC）。"""
    return unicodedata.normalize("NFC", os.path.normcase(str(name))).casefold()


_FONT_FILE_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttf",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)


@lru_cache(maxsize=1)
def find_cjk_font_file() -> Path | None:
    """查找可用于嵌入 PDF / Pillow 绘制的中文字体文件。"""
    for candidate in _FONT_FILE_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path

    fc_list = shutil.which("fc-list")
    if fc_list:
        try:
            result = subprocess.run(
                [fc_list, "-f", "%{file}\n", ":lang=zh"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in result.stdout.splitlines():
            path = Path(line.strip())
            if path.is_file():
                return path
    return None
