"""文件移动、冲突检测与空目录清理的共享逻辑。"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path


def find_conflicts(
    items: Sequence[tuple[Path, Path]],
) -> list[tuple[Path, Path]]:
    """返回目标位置已存在同名文件的移动项。"""
    return [(src, dst) for src, dst in items if dst.exists()]


def execute_moves(
    items: Sequence[tuple[Path, Path]], overwrite: bool
) -> tuple[int, int, list[str], list[tuple[Path, Path]]]:
    """执行移动，返回（成功数, 跳过数, 错误列表, 成功的移动项）。

    失败项不中断后续移动；成功的移动项可用于撤销。
    """
    moved = 0
    skipped = 0
    errors: list[str] = []
    moved_items: list[tuple[Path, Path]] = []
    for src, dst in items:
        try:
            if dst.exists():
                if not overwrite:
                    skipped += 1
                    continue
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
            moved_items.append((src, dst))
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其余移动
            errors.append(f"{src.name} → {dst.parent.name}/{dst.name}：{exc}")
    return moved, skipped, errors, moved_items


def remove_empty_dirs(directories: Sequence[Path]) -> None:
    """删除已空的子文件夹；非空或删除失败时静默忽略。"""
    for directory in {Path(item) for item in directories}:
        try:
            directory.rmdir()
        except OSError:
            pass


def prune_empty_dirs(
    root: Path, exclude: Sequence[Path] = ()
) -> list[Path]:
    """递归删除 ``root`` 下（不含 root 自身）的空目录，返回已删除目录。

    ``exclude`` 中的目录及其所有子目录不会被删除。
    """
    root = Path(root)
    removed: list[Path] = []
    if not root.is_dir():
        return removed

    excluded: list[Path] = []
    for item in exclude:
        try:
            excluded.append(Path(item).resolve())
        except OSError:
            continue

    for dirpath, _dirnames, _filenames in os.walk(
        root, topdown=False, followlinks=False
    ):
        directory = Path(dirpath)
        if directory == root or directory.is_symlink():
            continue
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if any(
            resolved == item or resolved.is_relative_to(item)
            for item in excluded
        ):
            continue
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                removed.append(directory)
        except OSError:
            pass
    return removed
