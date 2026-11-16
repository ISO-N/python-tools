"""图片归档工具页（PyQt6 界面）。

针对「PDF 转图片」产出的图片文件夹，把 ``{PDF文件名}_page_{页码}.{png|jpg}``
按 JSON 配置的「文件夹名 → 起始页码」移动到对应子文件夹。

换算关系：图片文件名页码 = JSON 起始页码 + 索引差值。
界面还提供「复制 AI 提示词」按钮：复制一段提示词模板到剪贴板，粘贴给 AI
并附上书籍目录图片后，可让 AI 返回本工具需要的 JSON 配置。

本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from toolbox.chapters import (
    AI_PROMPT,
    ConfigError,
    parse_chapter_config,
)
from toolbox.common import open_directory
from toolbox.fileops import (
    execute_moves,
    find_conflicts,
    remove_empty_dirs,
)
from toolbox.page import ToolPage

# 兼容旧名称：本工具原先把解析函数命名为 parse_folder_config。
parse_folder_config = parse_chapter_config

# 本工具只处理 PDF 转图片产出的三种格式，保持原有行为。
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PAGE_PATTERN = re.compile(r"_page_(\d+)$")
DIFF_MIN, DIFF_MAX = -99999, 99999


@dataclass
class FolderGroup:
    """一个目标文件夹及其包含的图片（按页码排序）。"""

    name: str
    start_page: int
    images: list[tuple[int, Path]] = field(default_factory=list)

    @property
    def page_range(self) -> tuple[int, int] | None:
        if not self.images:
            return None
        return self.images[0][0], self.images[-1][0]


@dataclass
class ArchivePlan:
    """一次归档操作的完整计划。"""

    folder: Path
    diff: int
    images: list[tuple[int, Path]]
    groups: list[FolderGroup]
    unassigned: list[tuple[int, Path]]
    unrecognized: list[Path]

    @property
    def move_items(self) -> list[tuple[Path, Path]]:
        return build_move_items(self.groups, self.folder)


# --------------------------------------------------------------------- 纯逻辑


def extract_page_number(path: Path) -> int | None:
    """从 ``xxx_page_012.png`` 形式的文件名中取出页码；不匹配时返回 None。"""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    match = PAGE_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def scan_images(folder: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
    """扫描目录第一层，返回（按页码排序的图片, 无法识别页码的图片）。"""
    folder = Path(folder)
    matched: list[tuple[int, Path]] = []
    unrecognized: list[Path] = []
    for entry in sorted(folder.iterdir(), key=lambda item: item.name):
        if not entry.is_file():
            continue
        page = extract_page_number(entry)
        if page is None:
            if entry.suffix.lower() in IMAGE_EXTENSIONS:
                unrecognized.append(entry)
            continue
        matched.append((page, entry))
    matched.sort(key=lambda item: (item[0], item[1].name))
    return matched, unrecognized


def build_move_plan(
    images: Sequence[tuple[int, Path]],
    diff: int,
    config: Sequence[tuple[str, int]],
) -> tuple[list[FolderGroup], list[tuple[int, Path]]]:
    """按「图片页码 − 索引差值」把图片划入各文件夹的区间。

    图片归入起始页码不晚于它的最后一个文件夹，即每个文件夹覆盖
    ``[起始页码, 下一个文件夹起始页码)``。第一个文件夹之前的图片视为未归类。
    """
    groups = [
        FolderGroup(name=name, start_page=start_page)
        for name, start_page in config
    ]
    starts = [start_page for _name, start_page in config]
    unassigned: list[tuple[int, Path]] = []
    for page, path in images:
        index = bisect.bisect_right(starts, page - diff) - 1
        if index < 0:
            unassigned.append((page, path))
        else:
            groups[index].images.append((page, path))
    return groups, unassigned


def build_move_items(
    groups: Sequence[FolderGroup], base_dir: Path
) -> list[tuple[Path, Path]]:
    """把分组结果展开为（源路径, 目标路径）列表。"""
    base_dir = Path(base_dir)
    items: list[tuple[Path, Path]] = []
    for group in groups:
        for _page, src in group.images:
            items.append((src, base_dir / group.name / src.name))
    return items


def detect_stems(images: Sequence[tuple[int, Path]]) -> list[str]:
    """返回识别到的 PDF 文件名前缀（正常情况下只有一个）。"""
    stems = {PAGE_PATTERN.sub("", path.stem) for _page, path in images}
    return sorted(stems)


def format_preview(plan: ArchivePlan) -> str:
    """生成预览/结果文本。"""
    lines: list[str] = [
        f"图片文件夹：{plan.folder}",
        f"索引差值：{plan.diff}（图片页码 = 起始页码 + {plan.diff}）",
        "",
    ]

    stems = detect_stems(plan.images)
    if len(stems) > 1:
        shown = "、".join(stem or "(空)" for stem in stems)
        lines.append(f"注意：检测到多个文件前缀（{shown}），仍按页码统一处理。")
        lines.append("")

    for group in plan.groups:
        page_range = group.page_range
        if page_range is None:
            lines.append(
                f"{group.name}：起始页码 {group.start_page} → "
                "无匹配图片（不会创建该文件夹）"
            )
            continue
        first_page, last_page = page_range
        lines.append(
            f"{group.name}：起始页码 {group.start_page} → "
            f"图片页码 {first_page}–{last_page}（{len(group.images)} 张）"
        )

    if plan.unassigned:
        first_page = plan.unassigned[0][0]
        last_page = plan.unassigned[-1][0]
        names = "、".join(path.name for _page, path in plan.unassigned[:5])
        suffix = " …" if len(plan.unassigned) > 5 else ""
        lines.append("")
        lines.append(
            f"未归类（留在原处）：{len(plan.unassigned)} 张，"
            f"页码 {first_page}–{last_page}：{names}{suffix}"
        )

    if plan.unrecognized:
        names = "、".join(path.name for path in plan.unrecognized[:5])
        suffix = " …" if len(plan.unrecognized) > 5 else ""
        lines.append(
            f"文件名无法识别页码（保持不动）：{len(plan.unrecognized)} 张："
            f"{names}{suffix}"
        )

    move_count = sum(len(group.images) for group in plan.groups)
    folder_count = sum(1 for group in plan.groups if group.images)
    lines.append("")
    if move_count:
        lines.append(
            f"共识别 {len(plan.images)} 张图片，"
            f"将移动 {move_count} 张到 {folder_count} 个文件夹。"
        )
    else:
        lines.append(
            f"共识别 {len(plan.images)} 张图片，当前配置没有可移动的图片。"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------- 界面部分


class MoveImagesPage(ToolPage):
    """图片按页码归档工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._undo_items: list[tuple[Path, Path]] = []
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setColumnStretch(1, 1)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(
            "选择「PDF 转图片」产出的图片文件夹"
        )
        self.folder_edit.textChanged.connect(self._update_open_button)
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self._browse_folder)
        form.addWidget(QLabel("图片文件夹："), 0, 0)
        form.addWidget(self.folder_edit, 0, 1)
        form.addWidget(browse_btn, 0, 2)

        self.diff_spin = QSpinBox()
        self.diff_spin.setRange(DIFF_MIN, DIFF_MAX)
        self.diff_spin.setValue(0)
        self.diff_spin.setFixedWidth(120)
        form.addWidget(QLabel("索引差值："), 1, 0)
        form.addWidget(self.diff_spin, 1, 1)

        diff_hint = QLabel(
            "图片文件名页码 = JSON 起始页码 + 索引差值"
            "（例：book_page_011 是「第一章」第 1 页时，起始页码填 1，差值填 10）"
        )
        diff_hint.setStyleSheet("color: #666;")
        diff_hint.setWordWrap(True)
        form.addWidget(diff_hint, 2, 1, 1, 2)

        config_label = QLabel('文件夹配置（JSON，格式：{"文件夹名": 起始页码}）：')
        form.addWidget(config_label, 3, 0, 1, 3)

        self.config_edit = QPlainTextEdit()
        self.config_edit.setPlaceholderText(
            '{\n  "第一章": 1,\n  "第二章": 15,\n  "第三章": 31\n}'
        )
        self.config_edit.setMinimumHeight(110)
        self.config_edit.setTabChangesFocus(True)
        form.addWidget(self.config_edit, 4, 0, 1, 3)

        buttons = QHBoxLayout()
        self.copy_prompt_btn = QPushButton("复制 AI 提示词")
        self.copy_prompt_btn.setToolTip(
            "复制一段提示词模板到剪贴板，粘贴给 AI 并附上书籍目录图片，"
            "让 AI 返回本工具需要的 JSON 配置"
        )
        self.copy_prompt_btn.clicked.connect(self._copy_ai_prompt)
        self.preview_btn = QPushButton("预览")
        self.preview_btn.clicked.connect(self._preview)
        self.move_btn = QPushButton("开始移动")
        self.move_btn.setDefault(True)
        self.move_btn.clicked.connect(self._start_move)
        self.undo_btn = QPushButton("撤销移动")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("撤销上一次移动（暂无可撤销的操作）")
        self.undo_btn.clicked.connect(self._undo_move)
        self.open_btn = QPushButton("打开文件夹")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        buttons.addWidget(self.copy_prompt_btn)
        buttons.addSpacing(16)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(self.move_btn)
        buttons.addWidget(self.undo_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 5, 0, 1, 3)

        layout.addLayout(form)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("点击「预览」查看归档结果")
        layout.addWidget(self.output, 1)

        self.status_label = QLabel("请选择图片文件夹并填写 JSON 配置")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _browse_folder(self) -> None:
        current = self.folder_edit.text().strip()
        start_dir = current if current else str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹", start_dir
        )
        if not selected:
            return
        self.folder_edit.setText(selected)
        self._set_status(f"已选择：{selected}")

    def _update_open_button(self, *_args: object) -> None:
        text = self.folder_edit.text().strip()
        self.open_btn.setEnabled(bool(text) and Path(text).expanduser().is_dir())

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.copy_prompt_btn,
            self.preview_btn,
            self.move_btn,
            self.folder_edit,
            self.config_edit,
            self.diff_spin,
        ):
            widget.setEnabled(not busy)
        if busy:
            self.open_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)
        else:
            self._update_open_button()
            self._refresh_undo_button()

    def _refresh_undo_button(self) -> None:
        count = len(self._undo_items)
        self.undo_btn.setEnabled(count > 0)
        if count:
            self.undo_btn.setToolTip(f"撤销上一次移动（{count} 张图片）")
        else:
            self.undo_btn.setToolTip("撤销上一次移动（暂无可撤销的操作）")

    # --------------------------------------------------------------- 归档流程

    def _load_folder_images(
        self,
    ) -> tuple[Path, list[tuple[int, Path]], list[Path]] | None:
        """校验图片文件夹并扫描图片；失败时弹窗并返回 None。"""
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            QMessageBox.warning(
                self, "缺少图片文件夹", "请先选择要整理的图片文件夹。"
            )
            return None
        folder = Path(folder_text).expanduser()
        if not folder.is_dir():
            QMessageBox.warning(
                self, "文件夹不存在", f"目录不存在：\n{folder}"
            )
            return None

        try:
            images, unrecognized = scan_images(folder)
        except OSError as exc:
            QMessageBox.critical(
                self, "无法读取文件夹", f"{folder}\n{exc}"
            )
            return None

        if not images:
            detail = ""
            if unrecognized:
                detail = (
                    f"\n\n检测到 {len(unrecognized)} 个图片文件，"
                    "但文件名不符合 `..._page_数字.扩展名` 格式。"
                )
            QMessageBox.warning(
                self,
                "没有可处理的图片",
                "该文件夹中未找到符合 `*_page_数字.png/jpg` 命名的图片。"
                + detail,
            )
            return None
        return folder, images, unrecognized

    def _collect_plan(self) -> ArchivePlan | None:
        """校验输入并生成归档计划；失败时弹窗并返回 None。"""
        loaded = self._load_folder_images()
        if loaded is None:
            return None
        folder, images, unrecognized = loaded

        try:
            config = parse_folder_config(self.config_edit.toPlainText())
        except ConfigError as exc:
            QMessageBox.warning(self, "JSON 配置错误", str(exc))
            return None

        diff = self.diff_spin.value()
        groups, unassigned = build_move_plan(images, diff, config)
        return ArchivePlan(folder, diff, images, groups, unassigned, unrecognized)

    def _copy_ai_prompt(self) -> None:
        QApplication.clipboard().setText(AI_PROMPT)
        self.output.setPlainText(AI_PROMPT)
        self._set_status(
            "已复制 AI 提示词。请粘贴到 AI 对话并上传书籍目录图片，"
            "再把返回的 JSON 粘贴到文件夹配置框，"
            "并按目录页码与图片页码的差值设置索引差值。"
        )

    def _preview(self) -> None:
        plan = self._collect_plan()
        if plan is None:
            return
        self.output.setPlainText(format_preview(plan))
        move_count = len(plan.move_items)
        if move_count:
            self._set_status(
                f"预览完成：共识别 {len(plan.images)} 张，将移动 {move_count} 张"
            )
        else:
            self._set_status("预览完成：当前配置没有可移动的图片")

    def _ask_conflict_policy(
        self, count: int, location: str = "目标文件夹"
    ) -> bool | None:
        """询问同名文件处理方式：返回 True=覆盖，False=跳过，None=取消。"""
        box = QMessageBox(self)
        box.setWindowTitle("存在同名文件")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"有 {count} 张图片在{location}中已存在同名文件，如何处理？")
        overwrite_btn = box.addButton("覆盖", QMessageBox.ButtonRole.AcceptRole)
        skip_btn = box.addButton("跳过", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(skip_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite_btn:
            return True
        if clicked is skip_btn:
            return False
        return None

    def _start_move(self) -> None:
        plan = self._collect_plan()
        if plan is None:
            return

        self.output.setPlainText(format_preview(plan))
        items = plan.move_items
        if not items:
            QMessageBox.information(
                self,
                "没有可移动的图片",
                "当前配置没有匹配到任何图片，未做任何移动。",
            )
            self._set_status("没有可移动的图片")
            return

        overwrite = False
        conflicts = find_conflicts(items)
        if conflicts:
            policy = self._ask_conflict_policy(len(conflicts))
            if policy is None:
                self._set_status("已取消移动")
                return
            overwrite = policy

        folder_count = sum(1 for group in plan.groups if group.images)
        summary = (
            f"将把 {len(items)} 张图片移动到 {folder_count} 个子文件夹。\n\n"
            "是否继续？"
        )
        if conflicts:
            action = "覆盖" if overwrite else "跳过"
            summary += f"\n\n其中 {len(conflicts)} 张的目标位置已存在同名文件（{action}）。"
        reply = QMessageBox.question(
            self,
            "确认移动",
            summary,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消移动")
            return

        self._set_busy(True)
        try:
            moved, skipped, errors, moved_items = execute_moves(
                items, overwrite=overwrite
            )
            if moved_items:
                self._undo_items = list(moved_items)
        finally:
            self._set_busy(False)

        result_lines = [f"完成：已移动 {moved} 张图片，跳过 {skipped} 张。"]
        if errors:
            result_lines.append("")
            result_lines.append(f"失败 {len(errors)} 项：")
            result_lines.extend(errors)
            self.output.setPlainText("\n".join(result_lines))
            self._set_status(f"完成：移动 {moved} 张，失败 {len(errors)} 张")
            QMessageBox.warning(
                self, "部分图片移动失败", "\n".join(result_lines)
            )
        else:
            self.output.setPlainText("\n".join(result_lines))
            self._set_status(f"完成：已移动 {moved} 张图片（跳过 {skipped} 张）")
            QMessageBox.information(
                self, "移动完成", f"已移动 {moved} 张图片。"
            )

    def _undo_move(self) -> None:
        if not self._undo_items:
            return

        record = list(self._undo_items)
        available = [(dst, src) for src, dst in record if dst.exists()]
        missing = len(record) - len(available)
        if not available:
            remove_empty_dirs([dst.parent for _src, dst in record])
            self._undo_items = []
            self._refresh_undo_button()
            self.output.setPlainText(
                "撤销未执行：上次移动的图片已不在对应的子文件夹中。"
            )
            QMessageBox.information(
                self,
                "没有可撤销的文件",
                "上次移动的图片已不在对应的子文件夹中，无法撤销。",
            )
            self._set_status("没有可撤销的文件")
            return

        conflicts = find_conflicts(available)
        overwrite = False
        if conflicts:
            policy = self._ask_conflict_policy(len(conflicts), "原文件夹")
            if policy is None:
                self._set_status("已取消撤销")
                return
            overwrite = policy

        message = f"将把上次移动的 {len(available)} 张图片移回原文件夹。"
        if missing:
            message += f"\n另有 {missing} 张图片已不在子文件夹中，无法移回。"
        if conflicts:
            action = "覆盖" if overwrite else "跳过"
            message += f"\n原文件夹中有 {len(conflicts)} 个同名文件（{action}）。"
        message += "\n\n是否继续？"
        reply = QMessageBox.question(
            self,
            "确认撤销",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消撤销")
            return

        subdirs = {dst.parent for _src, dst in record}
        self._set_busy(True)
        try:
            moved, skipped, errors, _ = execute_moves(
                available, overwrite=overwrite
            )
        finally:
            self._set_busy(False)

        remove_empty_dirs(list(subdirs))

        result_lines = [f"撤销完成：已移回 {moved} 张图片，跳过 {skipped} 张。"]
        if missing:
            result_lines.append(
                f"另有 {missing} 张图片已不在原来的子文件夹中，未处理。"
            )
        if errors:
            result_lines.append("")
            result_lines.append(f"失败 {len(errors)} 项：")
            result_lines.extend(errors)

        self._undo_items = [(src, dst) for src, dst in record if dst.exists()]
        self._refresh_undo_button()
        self.output.setPlainText("\n".join(result_lines))

        if errors:
            self._set_status(f"撤销完成：移回 {moved} 张，失败 {len(errors)} 张")
            QMessageBox.warning(
                self, "部分图片撤销失败", "\n".join(result_lines)
            )
        else:
            remaining = len(self._undo_items)
            extra = f"，仍有 {remaining} 张可继续撤销" if remaining else ""
            self._set_status(
                f"撤销完成：已移回 {moved} 张图片（跳过 {skipped} 张）{extra}"
            )
            QMessageBox.information(
                self, "撤销完成", f"已移回 {moved} 张图片。"
            )

    def _open_folder(self) -> None:
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            return
        folder = Path(folder_text).expanduser()
        if not folder.is_dir():
            QMessageBox.critical(
                self, "目录不存在", f"目录不存在：\n{folder}"
            )
            return
        try:
            open_directory(folder)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(self, "无法打开目录", f"{folder}\n{exc}")
