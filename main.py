"""Python 工具箱启动入口。

用法：
    uv run main.py
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from toolbox.app import ToolboxWindow
from toolbox.common import apply_ui_font


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Python 工具箱")
    font_family = apply_ui_font(app)
    window = ToolboxWindow()
    window.show()
    if font_family is None:
        QMessageBox.warning(
            window,
            "字体提示",
            "未找到中文字体，界面文字可能显示为方框。",
        )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
