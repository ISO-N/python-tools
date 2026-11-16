"""文件移动、冲突检测与空目录清理的共享逻辑。"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path

from toolbox.errors import Cancelled


def find_conflicts(
    items: Sequence[tuple[Path, Path]],
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[Path, Path]]:
    """返回目标位置已存在同名文件的移动项。"""
    total = len(items)
    conflicts: list[tuple[Path, Path]] = []
    for index, (src, dst) in enumerate(items, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(list(conflicts))
        if dst.exists():
            conflicts.append((src, dst))
        if on_progress is not None:
            on_progress(index, total)
    return conflicts


@dataclass
class MoveExecutionResult:
    """批量移动的结果。"""

    moved: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    moved_items: list[tuple[Path, Path]] = field(default_factory=list)
    cancelled: bool = False

    def __iter__(self):
        """兼容旧的 ``(moved, skipped, errors, moved_items)`` 解包用法。"""
        yield self.moved
        yield self.skipped
        yield self.errors
        yield self.moved_items


def execute_moves(
    items: Sequence[tuple[Path, Path]],
    overwrite: bool,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    known_conflicts: Set[Path] | None = None,
    error_formatter: Callable[[Path, Path, BaseException], str] | None = None,
) -> MoveExecutionResult:
    """执行移动，返回移动 / 跳过 / 错误与成功的移动项。

    失败项不中断后续移动；成功的移动项可用于撤销。``known_conflicts``
    是 ``find_conflicts`` 已确认的目标集合，可省去一次重复 stat。
    """
    result = MoveExecutionResult()
    known = known_conflicts or frozenset()
    created_parents: set[Path] = set()
    total = len(items)

    def format_error(
        src: Path, dst: Path, exc: BaseException
    ) -> str:
        if error_formatter is not None:
            return error_formatter(src, dst, exc)
        return f"{src.name} → {dst.parent.name}/{dst.name}：{exc}"

    for processed, (src, dst) in enumerate(items, start=1):
        if cancel_event is not None and cancel_event.is_set():
            result.cancelled = True
            raise Cancelled(result)

        try:
            if not overwrite:
                if dst in known or dst.exists():
                    result.skipped += 1
                    if on_progress is not None:
                        on_progress(processed, total)
                    continue
            else:
                dst.unlink(missing_ok=True)

            parent = dst.parent
            if parent not in created_parents:
                parent.mkdir(parents=True, exist_ok=True)
                created_parents.add(parent)
            shutil.move(str(src), str(dst))
            result.moved += 1
            result.moved_items.append((src, dst))
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其余移动
            result.errors.append(format_error(src, dst, exc))

        if on_progress is not None:
            on_progress(processed, total)

    return result


def copy_file_atomic(src: Path | str, dst: Path | str) -> Path:
    """先复制到同目录临时文件再原子替换，保留源文件元数据。"""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=dst.suffix or ".copy", dir=dst.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def remove_empty_dirs(directories: Sequence[Path]) -> None:
    """删除已空的子文件夹；非空或删除失败时静默忽略。"""
    for directory in {Path(item) for item in directories}:
        try:
            directory.rmdir()
        except OSError:
            pass


def prune_empty_dirs(
    root: Path,
    exclude: Sequence[Path] = (),
    *,
    cancel_event: threading.Event | None = None,
) -> list[Path]:
    """递归删除 ``root`` 下（不含 root 自身）的空目录，返回已删除目录。

    ``exclude`` 中的目录及其所有子目录不会被删除。排除目录为空时完全
    跳过 ``resolve()``；删除判断直接交给 ``rmdir``，不再额外列举目录。
    """
    root = Path(root)
    removed: list[Path] = []
    if not root.is_dir():
        return removed

    excluded: list[Path] = []
    if exclude:
        root_resolved = root.resolve()
        for item in exclude:
            try:
                resolved = Path(item).resolve()
                excluded.append(resolved.relative_to(root_resolved))
            except (OSError, ValueError):
                continue

    for dirpath, _dirnames, _filenames in os.walk(
        root, topdown=False, followlinks=False
    ):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled(removed)
        directory = Path(dirpath)
        if directory == root or directory.is_symlink():
            continue
        if excluded:
            relative = directory.relative_to(root)
            if any(
                relative == item or relative.is_relative_to(item)
                for item in excluded
            ):
                continue
        try:
            directory.rmdir()
        except OSError:
            continue
        removed.append(directory)
    return removed
