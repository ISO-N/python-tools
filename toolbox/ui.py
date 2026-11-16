"""新工具页共用的轻量 UI 辅助。"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

PAGE_MARGINS = (16, 16, 16, 16)
FORM_H_SPACING = 10
FORM_V_SPACING = 8


def make_form() -> QGridLayout:
    """创建与现有工具页一致的表单布局。"""
    form = QGridLayout()
    form.setHorizontalSpacing(FORM_H_SPACING)
    form.setVerticalSpacing(FORM_V_SPACING)
    form.setColumnStretch(1, 1)
    return form


def add_path_row(
    form: QGridLayout,
    row: int,
    label: str,
    *,
    mode: str = "open_file",
    placeholder: str = "",
    dialog_title: str = "选择",
    file_filter: str = "所有文件 (*)",
    parent: QWidget | None = None,
) -> QLineEdit:
    """添加「标签 + 路径输入框 + 浏览…」一行，返回输入框。

    ``mode``：``open_file`` / ``open_dir`` / ``save_file``。
    """
    if mode not in {"open_file", "open_dir", "save_file"}:
        raise ValueError(f"不支持的路径选择模式：{mode}")

    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    button = QPushButton("浏览…")

    def browse() -> None:
        current = edit.text().strip()
        base = current or str(Path.home())
        if mode == "open_dir":
            selected = QFileDialog.getExistingDirectory(
                parent, dialog_title, base
            )
        elif mode == "save_file":
            selected, _ = QFileDialog.getSaveFileName(
                parent, dialog_title, base, file_filter
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                parent, dialog_title, base, file_filter
            )
        if selected:
            edit.setText(selected)

    button.clicked.connect(browse)
    form.addWidget(QLabel(label), row, 0)
    form.addWidget(edit, row, 1)
    form.addWidget(button, row, 2)
    return edit
