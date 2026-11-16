"""Python 工具箱主窗口。"""

from __future__ import annotations

import time

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from toolbox.common import center_on_screen
from toolbox.page import ToolPage
from toolbox.registry import TOOLS, ToolSpec

SIDEBAR_WIDTH = 220
SIDEBAR_STYLE = """
QListWidget#toolList {
    background: #f5f6f8;
    border: none;
    outline: none;
    padding: 6px 0;
}
QListWidget#toolList::item {
    padding: 10px 14px;
    margin: 2px 10px;
    border-radius: 6px;
    color: #333333;
}
QListWidget#toolList::item:hover {
    background: #e9edf3;
}
QListWidget#toolList::item:selected {
    background: #dce8fb;
    color: #1a5fb4;
}
"""


class ToolboxWindow(QWidget):
    """左侧工具导航、右侧工具页面的主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Python 工具箱")
        self.resize(1040, 720)
        self.setMinimumSize(900, 620)
        self._pages: dict[str, ToolPage] = {}
        self._build_ui()
        self._populate_tools()
        center_on_screen(self)
        self.tool_list.setCurrentRow(0)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setStyleSheet(
            "QWidget#sidebar {"
            "  background: #f5f6f8;"
            "  border-right: 1px solid #e0e0e0;"
            "}"
        )
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 12)
        sidebar_layout.setSpacing(4)

        app_title = QLabel("Python 工具箱")
        app_title.setStyleSheet(
            "font-size: 17px; font-weight: 600; padding: 0 20px;"
        )
        sidebar_layout.addWidget(app_title)

        app_subtitle = QLabel(f"共 {len(TOOLS)} 个工具")
        app_subtitle.setStyleSheet("color: #777777; padding: 0 20px 8px 20px;")
        sidebar_layout.addWidget(app_subtitle)

        self.tool_list = QListWidget()
        self.tool_list.setObjectName("toolList")
        self.tool_list.setStyleSheet(SIDEBAR_STYLE)
        self.tool_list.setFrameShape(QFrame.Shape.NoFrame)
        self.tool_list.currentRowChanged.connect(self._on_tool_changed)
        sidebar_layout.addWidget(self.tool_list, 1)

        layout.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 16)
        content_layout.setSpacing(8)

        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            "font-size: 18px; font-weight: 600; color: #222222;"
        )
        content_layout.addWidget(self.title_label)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #666666;")
        content_layout.addWidget(self.description_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("color: #e0e0e0;")
        content_layout.addWidget(divider)

        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack, 1)

        layout.addWidget(content, 1)

    def _populate_tools(self) -> None:
        for spec in TOOLS:
            item = QListWidgetItem(spec.title)
            item.setData(Qt.ItemDataRole.UserRole, spec.key)
            item.setToolTip(spec.description)
            item.setSizeHint(QSize(SIDEBAR_WIDTH - 20, 40))
            self.tool_list.addItem(item)

    @property
    def current_spec(self) -> ToolSpec | None:
        """当前选中的工具；未选中时返回 None。"""
        row = self.tool_list.currentRow()
        if 0 <= row < len(TOOLS):
            return TOOLS[row]
        return None

    def _on_tool_changed(self, row: int) -> None:
        if not 0 <= row < len(TOOLS):
            return
        spec = TOOLS[row]
        self.title_label.setText(spec.title)
        self.description_label.setText(spec.description)

        page = self._pages.get(spec.key)
        if page is None:
            page = spec.factory()
            self._pages[spec.key] = page
            self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    # ----------------------------------------------------------------- 关闭

    def closeEvent(self, event: QCloseEvent) -> None:
        # 阶段 1：逐个询问各页面是否取消正在运行的任务。
        pages = list(self._pages.values())
        for page in pages:
            if not page.confirm_shutdown():
                event.ignore()
                return

        # 阶段 2：所有页面共享同一个 5 秒截止时间等待后台线程结束。
        deadline = time.monotonic() + 5.0
        for page in pages:
            if not page.wait_shutdown(deadline):
                QMessageBox.information(
                    self,
                    "正在停止",
                    "后台任务仍在结束中，请稍候再关闭窗口。",
                )
                event.ignore()
                return
        event.accept()
