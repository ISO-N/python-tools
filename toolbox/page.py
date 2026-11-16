"""工具页基类。"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget


class ToolPage(QWidget):
    """所有工具箱页面的基类。"""

    def shutdown(self) -> bool:
        """主窗口关闭前调用；返回 False 表示暂不能关闭。"""
        return True
