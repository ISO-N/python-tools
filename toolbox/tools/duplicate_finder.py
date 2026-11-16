"""重复文件查找工具页（PyQt6 界面）。

在指定目录（可递归）中查找内容完全相同的文件，先按文件大小分组，再对候选
文件做快速前缀哈希和完整 BLAKE2b 哈希。重复文件只会被移入同级的隔离目录，
绝不直接删除，并可通过 manifest 记录撤销移回。

本模块作为「Python 工具箱」的一个工具页使用。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from toolbox.common import (
    UniquePathAllocator,
    human_size,
    open_directory,
)
from toolbox.fileops import execute_moves, prune_empty_dirs
from toolbox.page import ToolPage
from toolbox.ui import PAGE_MARGINS, add_path_row, make_form
from toolbox.workers import (
    Cancelled,
    MAX_HASH_WORKERS,
    SingleTaskWorker,
    shutdown_worker,
)

HASH_CHUNK_SIZE = 1024 * 1024
PREFIX_HASH_SIZE = 64 * 1024
QUARANTINE_DIR_PATTERN = re.compile(r"_quarantine_\d{8}_\d{6}$")


@dataclass
class DuplicateGroup:
    """一组内容完全相同的文件。"""

    size: int
    digest: str
    files: list[Path]


@dataclass
class DuplicateScanResult:
    """一次重复文件扫描的结果。"""

    root: Path
    groups: list[DuplicateGroup]
    errors: list[str]
    scanned_files: int
    scanned_bytes: int


@dataclass
class QuarantineMove:
    """一个已移入隔离区的文件记录。"""

    original: Path
    quarantined: Path
    size: int
    digest: str


@dataclass
class _FileRecord:
    """扫描阶段收集到的普通文件。"""

    path: Path
    size: int
    #: 扫描时的大小与 mtime，用作前缀哈希阶段的“读取前”签名。
    signature: tuple[int, int]


@dataclass
class _PrefixRecord:
    """前缀哈希阶段的结果，供完整哈希阶段复用。"""

    record: _FileRecord
    digest_hex: str
    digest: object
    bytes_read: int
    signature: tuple[int, int]


@dataclass
class _PrefixOutcome:
    """一个文件的前缀哈希结果。"""

    prefix: str
    digest: object
    bytes_read: int
    signature: tuple[int, int]


@dataclass
class _FullHashOutcome:
    """一个文件的完整哈希结果与读取前后签名。"""

    before: tuple[int, int]
    digest: str
    after: tuple[int, int]


# --------------------------------------------------------------------- 纯逻辑


def _run_ordered(
    items: Sequence[object],
    task: Callable[[object], object],
    *,
    max_workers: int,
    cancel_event: threading.Event | None,
    handle_result: Callable[[int, object], None],
    handle_error: Callable[[int, BaseException], None],
) -> None:
    """在线程池中执行 ``task``，严格按输入顺序交回结果。

    提交窗口限制为 ``workers * 2``，避免一次性排满大量文件读取任务。
    取消时停止提交、等待运行中的任务结束并抛出 :class:`Cancelled`；
    普通异常交给 ``handle_error`` 按顺序记录。
    """
    items = list(items)
    if not items:
        return
    workers = max(1, int(max_workers))
    window = max(1, min(workers * 2, len(items)))
    executor = ThreadPoolExecutor(max_workers=workers)
    pending: dict[int, Future[object]] = {}
    next_submit = 0
    try:
        for index in range(len(items)):
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            while next_submit < len(items) and len(pending) < window:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                pending[next_submit] = executor.submit(
                    task, items[next_submit]
                )
                next_submit += 1

            future = pending.pop(index)
            while not future.done():
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                wait((future,), timeout=0.1)
            try:
                value = future.result()
            except Cancelled:
                raise
            except CancelledError:
                raise Cancelled() from None
            except Exception as exc:  # noqa: BLE001 - 交给调用方分类
                handle_error(index, exc)
            else:
                handle_result(index, value)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _new_digest() -> "hashlib._Hash":
    return hashlib.blake2b(digest_size=32)


def _hash_suffix(
    path: Path,
    digest: "hashlib._Hash",
    offset: int,
    cancel_event: threading.Event | None = None,
    on_chunk: Callable[[int], None] | None = None,
) -> str:
    """从 ``offset`` 处续算摘要，返回十六进制字符串。"""
    with Path(path).open("rb") as handle:
        if offset:
            handle.seek(offset)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    return digest.hexdigest()


def hash_file(
    path: Path,
    cancel_event: threading.Event | None = None,
    on_chunk: Callable[[int], None] | None = None,
) -> str:
    """计算文件的 BLAKE2b 摘要（32 字节，返回十六进制字符串）。

    每次读取 1 MiB；每块读取前检查 ``cancel_event``，已取消时抛出
    ``toolbox.workers.Cancelled``。``on_chunk`` 在每块读取后回调本次字节数。
    """
    return _hash_suffix(
        Path(path),
        _new_digest(),
        0,
        cancel_event=cancel_event,
        on_chunk=on_chunk,
    )


def _hash_prefix(
    path: Path, cancel_event: threading.Event | None = None
) -> tuple[str, "hashlib._Hash", int]:
    """计算文件前 64 KiB 的摘要。

    返回 ``(十六进制摘要, 可续算的摘要副本, 实际读取字节数)``，供完整
    哈希阶段从 ``seek(bytes_read)`` 处继续，避免重复读取前缀。
    """
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    digest = _new_digest()
    with Path(path).open("rb") as handle:
        data = handle.read(PREFIX_HASH_SIZE)
    digest.update(data)
    return digest.hexdigest(), digest.copy(), len(data)


def _is_quarantine_dir_name(name: str) -> bool:
    """判断目录名是否匹配默认隔离目录命名模式。"""
    return QUARANTINE_DIR_PATTERN.search(name) is not None


def _signature(path: Path) -> tuple[int, int]:
    """返回普通文件的大小与 mtime（纳秒）；非普通文件视为错误。"""
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("不是普通文件")
    return info.st_size, info.st_mtime_ns


def find_duplicates(
    root: Path,
    recursive: bool = True,
    exclude_names: Sequence[str] = (),
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> DuplicateScanResult:
    """查找 ``root`` 下内容完全相同的文件。

    ``exclude_names`` 是额外跳过的目录名集合；默认还会跳过隐藏目录、
    符号链接目录以及匹配 ``*_quarantine_<时间戳>`` 的隔离目录。隐藏文件和
    符号链接文件同样跳过。扫描错误不会中断流程。

    ``on_progress`` 回调（已处理文件数, 待处理文件总数），待处理文件指按
    大小分组后需要参与哈希的候选文件。
    """
    root = Path(root)
    errors: list[str] = []
    scanned_files = 0
    scanned_bytes = 0

    if not root.is_dir():
        return DuplicateScanResult(
            root=root,
            groups=[],
            errors=[f"目录不存在或不是文件夹：{root}"],
            scanned_files=0,
            scanned_bytes=0,
        )

    excluded = set(exclude_names)

    def on_walk_error(error: OSError) -> None:
        location = error.filename or str(root)
        errors.append(f"{location}：{error}")

    by_size: dict[int, list[_FileRecord]] = {}
    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=on_walk_error
    ):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()

        current = Path(dirpath)
        kept_dirs: list[str] = []
        for name in dirnames:
            if (
                name in excluded
                or name.startswith(".")
                or _is_quarantine_dir_name(name)
            ):
                continue
            try:
                if (current / name).is_symlink():
                    continue
            except OSError:
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs if recursive else []

        for name in filenames:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            if name.startswith("."):
                continue
            path = current / name
            try:
                info = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                errors.append(f"{path}：{exc}")
                continue
            if stat.S_ISLNK(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            scanned_files += 1
            scanned_bytes += info.st_size
            by_size.setdefault(info.st_size, []).append(
                _FileRecord(
                    path=path,
                    size=info.st_size,
                    signature=(info.st_size, info.st_mtime_ns),
                )
            )

    candidates = [
        records for records in by_size.values() if len(records) >= 2
    ]
    total = sum(len(records) for records in candidates)
    processed = 0

    def advance() -> None:
        nonlocal processed
        processed += 1
        if on_progress is not None:
            on_progress(processed, total)

    if on_progress is not None:
        on_progress(0, total)

    # 第一轮：只读前 64 KiB，把前缀不同的候选文件排除。扫描阶段已经
    # 做过一次 stat，因此这里只需在读取后再取一次签名。
    workers = max(1, int(MAX_HASH_WORKERS))
    flat_candidates: list[tuple[int, _FileRecord]] = []
    for group_index, records in enumerate(candidates):
        for record in records:
            flat_candidates.append((group_index, record))

    prefix_groups_by_candidate: dict[int, dict[str, list[_PrefixRecord]]] = {
        index: {} for index in range(len(candidates))
    }

    def hash_prefix_task(record: _FileRecord) -> _PrefixOutcome:
        prefix, digest, bytes_read = _hash_prefix(
            record.path, cancel_event
        )
        after = _signature(record.path)
        return _PrefixOutcome(prefix, digest, bytes_read, after)

    def handle_prefix(index: int, value: object) -> None:
        group_index, record = flat_candidates[index]
        outcome = value
        if record.signature != outcome.signature:
            errors.append(
                f"{record.path}：读取期间文件发生变化，已跳过"
            )
            advance()
            return
        prefix_groups_by_candidate[group_index].setdefault(
            outcome.prefix, []
        ).append(
            _PrefixRecord(
                record=record,
                digest_hex=outcome.prefix,
                digest=outcome.digest,
                bytes_read=outcome.bytes_read,
                signature=outcome.signature,
            )
        )

    def handle_prefix_error(index: int, exc: BaseException) -> None:
        if not isinstance(exc, OSError):
            raise exc
        _group_index, record = flat_candidates[index]
        errors.append(f"{record.path}：{exc}")
        advance()

    _run_ordered(
        [record for _group_index, record in flat_candidates],
        hash_prefix_task,
        max_workers=workers,
        cancel_event=cancel_event,
        handle_result=handle_prefix,
        handle_error=handle_prefix_error,
    )

    full_hash_batches: list[list[_PrefixRecord]] = []
    for group_index, _records in enumerate(candidates):
        for batch in prefix_groups_by_candidate[group_index].values():
            if len(batch) >= 2:
                full_hash_batches.append(batch)
            else:
                # 前缀唯一，不可能与其他文件内容相同。
                for _entry in batch:
                    advance()

    # 第二轮：完整哈希，确认内容完全相同。前缀阶段已读过的部分不再
    # 读取；文件大小不超过前缀长度时直接复用前缀摘要。
    flat_entries: list[tuple[int, _PrefixRecord]] = []
    for batch_index, batch in enumerate(full_hash_batches):
        for entry in batch:
            flat_entries.append((batch_index, entry))

    digest_groups_by_batch: dict[int, dict[str, list[_FileRecord]]] = {
        index: {} for index in range(len(full_hash_batches))
    }

    def hash_full_task(entry: _PrefixRecord) -> _FullHashOutcome:
        record = entry.record
        before = _signature(record.path)
        if before == entry.signature:
            if record.size <= PREFIX_HASH_SIZE:
                digest = entry.digest_hex
            else:
                # 复制前缀摘要后在当前线程继续计算，摘要对象不跨线程并发使用。
                work = entry.digest.copy()  # type: ignore[attr-defined]
                digest = _hash_suffix(
                    record.path,
                    work,
                    entry.bytes_read,
                    cancel_event=cancel_event,
                )
        else:
            # 前缀阶段之后文件已变化，丢弃旧前缀并从 0 重新哈希。
            digest = hash_file(record.path, cancel_event=cancel_event)
        after = _signature(record.path)
        return _FullHashOutcome(before, digest, after)

    def handle_full(index: int, value: object) -> None:
        batch_index, entry = flat_entries[index]
        record = entry.record
        outcome = value
        if outcome.before != outcome.after:
            errors.append(
                f"{record.path}：读取期间文件发生变化，已跳过"
            )
            advance()
            return
        digest_groups_by_batch[batch_index].setdefault(
            outcome.digest, []
        ).append(record)
        advance()

    def handle_full_error(index: int, exc: BaseException) -> None:
        if not isinstance(exc, OSError):
            raise exc
        _batch_index, entry = flat_entries[index]
        errors.append(f"{entry.record.path}：{exc}")
        advance()

    _run_ordered(
        [entry for _batch_index, entry in flat_entries],
        hash_full_task,
        max_workers=workers,
        cancel_event=cancel_event,
        handle_result=handle_full,
        handle_error=handle_full_error,
    )

    groups: list[DuplicateGroup] = []
    for batch_index, _batch in enumerate(full_hash_batches):
        for digest, matched in digest_groups_by_batch[batch_index].items():
            if len(matched) < 2:
                continue
            files = sorted(
                (record.path for record in matched), key=lambda item: str(item)
            )
            groups.append(
                DuplicateGroup(
                    size=matched[0].size, digest=digest, files=files
                )
            )

    # 浪费空间越大越靠前：size * (文件数 - 1)。
    groups.sort(
        key=lambda group: (
            -(group.size * (len(group.files) - 1)),
            str(group.files[0]),
        )
    )
    return DuplicateScanResult(
        root=root,
        groups=groups,
        errors=errors,
        scanned_files=scanned_files,
        scanned_bytes=scanned_bytes,
    )


def quarantine_root_for(root: Path) -> Path:
    """返回扫描目录同级的隔离目录路径（带当前时间戳）。"""
    root = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root.parent / f"{root.name}_quarantine_{timestamp}"


def build_quarantine_items(
    root: Path,
    quarantine_root: Path,
    selected: list[tuple[Path, DuplicateGroup]],
) -> list[tuple[Path, Path]]:
    """把勾选的文件映射为（原路径, 隔离路径），保留相对扫描根目录的结构。

    使用 :class:`toolbox.common.UniquePathAllocator`（``reserve_planned``
    模式）一次建立目录名字快照，避免对同一隔离目录做 O(k²) 的重复探测。
    """
    root = Path(root)
    quarantine_root = Path(quarantine_root)
    allocator = UniquePathAllocator(reserve_planned=True)
    items: list[tuple[Path, Path]] = []
    for path, _group in selected:
        path = Path(path)
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        target = allocator.reserve(quarantine_root / relative)
        items.append((path, target))
    return items


def _manifest_path(quarantine_root: Path) -> Path:
    return Path(quarantine_root) / "manifest.json"


def _read_manifest_records(
    quarantine_root: Path,
) -> list[dict[str, object]]:
    """读取现有隔离清单；文件不存在时返回空列表，损坏时抛出 ValueError。"""
    target = _manifest_path(quarantine_root)
    if not target.is_file():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 统一转换为中文提示
        raise ValueError(f"无法读取隔离清单 {target}：{exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"隔离清单格式不正确：{target}")
    return [item for item in data if isinstance(item, dict)]


def _write_manifest_records(
    quarantine_root: Path, records: Sequence[dict[str, object]]
) -> None:
    """把记录原子写入隔离清单（临时文件 + ``os.replace``）。"""
    target = _manifest_path(quarantine_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(list(records), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, target)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_manifest(
    quarantine_root: Path, moves: Sequence[QuarantineMove]
) -> None:
    """把本批移动记录合并进隔离清单（同一隔离路径覆盖旧记录）。"""
    moved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    new_records = [
        {
            "original": str(move.original),
            "quarantined": str(move.quarantined),
            "size": move.size,
            "digest": move.digest,
            "moved_at": moved_at,
        }
        for move in moves
    ]
    new_paths = {str(move.quarantined) for move in moves}
    # 保留此前批次的记录，避免多次移入同一隔离目录时互相覆盖。
    existing = [
        record
        for record in _read_manifest_records(quarantine_root)
        if str(record.get("quarantined", "")) not in new_paths
    ]
    _write_manifest_records(quarantine_root, [*existing, *new_records])


def _build_quarantine_moves(
    moved_items: Sequence[tuple[Path, Path]],
    known_digests: Mapping[Path, str] | None,
    errors: list[str],
) -> list[QuarantineMove]:
    """把成功的移动项转换为隔离记录，必要时补算摘要。"""
    moves: list[QuarantineMove] = []
    for original, quarantined in moved_items:
        original = Path(original)
        quarantined = Path(quarantined)
        try:
            size = quarantined.stat().st_size
        except OSError as exc:
            size = 0
            errors.append(f"{quarantined}：移动后读取大小失败：{exc}")

        digest = ""
        if known_digests is not None:
            digest = known_digests.get(original, "")
        if not digest:
            try:
                digest = hash_file(quarantined)
            except Exception as exc:  # noqa: BLE001 - 保留可撤销记录
                errors.append(f"{quarantined}：移动后哈希校验失败：{exc}")
        moves.append(
            QuarantineMove(
                original=original,
                quarantined=quarantined,
                size=size,
                digest=digest,
            )
        )
    return moves


def _write_quarantine_manifest(
    quarantine_root: Path,
    moves: Sequence[QuarantineMove],
    errors: list[str],
) -> None:
    if not moves:
        return
    try:
        _write_manifest(quarantine_root, moves)
    except Exception as exc:  # noqa: BLE001 - 清单失败不回滚
        errors.append(
            f"无法写入隔离清单 {quarantine_root / 'manifest.json'}：{exc}"
        )


def move_to_quarantine(
    items: Sequence[tuple[Path, Path]],
    quarantine_root: Path,
    *,
    known_digests: Mapping[Path, str] | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[QuarantineMove], list[str]]:
    """把文件移动到隔离区，成功后写 manifest，返回（成功记录, 错误）。

    移动使用 ``execute_moves(..., overwrite=False)``，绝不删除已有文件。
    ``known_digests`` 可传入扫描阶段已计算的摘要，避免移动后重复读取大文件；
    缺失时再回退到完整哈希。manifest 写入失败只记录错误，不回滚已经完成的
    移动。取消时先把已完成项的记录写入 manifest，再抛出
    :class:`toolbox.errors.Cancelled`，其 ``result`` 为 ``(moves, errors)``。
    """
    quarantine_root = Path(quarantine_root)

    try:
        result = execute_moves(
            list(items),
            overwrite=False,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )
    except Cancelled as exc:
        partial = exc.result
        errors = list(getattr(partial, "errors", ()))
        moves = _build_quarantine_moves(
            getattr(partial, "moved_items", ()), known_digests, errors
        )
        _write_quarantine_manifest(quarantine_root, moves, errors)
        raise Cancelled((moves, errors)) from exc

    errors = list(result.errors)
    moves = _build_quarantine_moves(
        result.moved_items, known_digests, errors
    )
    _write_quarantine_manifest(quarantine_root, moves, errors)
    return moves, errors


def _remove_manifest_entries(
    quarantine_root: Path, restored_paths: set[str]
) -> None:
    """撤销后从隔离清单移除已移回的条目；清单不存在时什么也不做。"""
    target = _manifest_path(quarantine_root)
    if not target.is_file():
        return
    records = _read_manifest_records(quarantine_root)
    remaining = [
        record
        for record in records
        if str(record.get("quarantined", "")) not in restored_paths
    ]
    if len(remaining) != len(records):
        _write_manifest_records(quarantine_root, remaining)


def _manifest_roots(moves: Sequence[QuarantineMove]) -> set[Path]:
    """根据已移动文件的位置推断隔离根目录（用于清理空目录）。"""
    roots: set[Path] = set()
    for move in moves:
        parent = Path(move.quarantined).parent
        chain = [parent, *parent.parents]
        for candidate in chain:
            if (candidate / "manifest.json").is_file():
                roots.add(candidate)
                break
        else:
            for candidate in chain:
                if _is_quarantine_dir_name(candidate.name):
                    roots.add(candidate)
                    break
    return roots


def _finalize_quarantine_undo(
    moves: Sequence[QuarantineMove],
    moved: Sequence[tuple[Path, Path]],
    errors: list[str],
) -> None:
    """移回成功后更新 manifest 并清理隔离区空目录。"""
    if not moved:
        return
    successful = {source for source, _target in moved}
    roots = _manifest_roots(
        [move for move in moves if Path(move.quarantined) in successful]
    )
    restored_paths = {str(path) for path in successful}
    for root in roots:
        try:
            _remove_manifest_entries(root, restored_paths)
        except Exception as exc:  # noqa: BLE001 - 清单失败不影响移回
            errors.append(f"{_manifest_path(root)}：更新隔离清单失败：{exc}")
        prune_empty_dirs(root)


def undo_quarantine(
    moves: Sequence[QuarantineMove],
    *,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """把隔离文件逐个移回原位置。

    返回（成功移回的（隔离路径, 原路径）列表, 错误列表）。原位置已存在
    文件时不覆盖，对应文件继续留在隔离区。成功移动后清理隔离区空目录。
    取消时先完成已移回文件的清单更新与目录清理，再抛出
    :class:`toolbox.errors.Cancelled`，其 ``result`` 为 ``(moved, errors)``。
    """
    errors: list[str] = []
    candidates: list[tuple[Path, Path]] = []
    for move in moves:
        source = Path(move.quarantined)
        target = Path(move.original)
        try:
            if not source.exists():
                errors.append(f"{source}：隔离区中的文件已不存在")
                continue
            if os.path.lexists(target):
                errors.append(
                    f"{target}：原位置已存在同名文件，未覆盖，"
                    "文件继续保留在隔离区"
                )
                continue
        except OSError as exc:
            errors.append(f"{source} → {target}：{exc}")
            continue
        candidates.append((source, target))

    moved: list[tuple[Path, Path]] = []
    if candidates:
        try:
            result = execute_moves(
                candidates,
                overwrite=False,
                cancel_event=cancel_event,
                on_progress=on_progress,
                error_formatter=lambda src, dst, exc: (
                    f"{src} → {dst}：{exc}"
                ),
            )
        except Cancelled as exc:
            partial = exc.result
            moved = list(getattr(partial, "moved_items", ()))
            errors.extend(getattr(partial, "errors", ()))
            _finalize_quarantine_undo(moves, moved, errors)
            raise Cancelled((moved, errors)) from exc
        moved = list(result.moved_items)
        errors.extend(result.errors)

    _finalize_quarantine_undo(moves, moved, errors)
    return moved, errors


# ------------------------------------------------------------------- 界面部分


class DuplicateFinderPage(ToolPage):
    shutdown_label = "重复文件处理"
    """重复文件查找工具页。"""

    def __init__(self) -> None:
        super().__init__()
        self._worker: SingleTaskWorker | None = None
        self._busy = False
        self._operation = ""
        self._scan_result: DuplicateScanResult | None = None
        self._quarantine_path: Path | None = None
        self._moves: list[QuarantineMove] = []
        self._checked: dict[Path, DuplicateGroup] = {}
        self._checked_savings = 0
        self._input_widgets: list[object] = []
        self._build_ui()
        self._refresh_actions()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_MARGINS)
        layout.setSpacing(10)

        form = make_form()

        self.root_edit = add_path_row(
            form,
            0,
            "扫描目录：",
            mode="open_dir",
            placeholder="选择要查找重复文件的目录",
            dialog_title="选择扫描目录",
            parent=self,
        )
        self.root_edit.textChanged.connect(self._refresh_actions)

        self.recursive_check = QCheckBox("包含子目录（递归扫描）")
        self.recursive_check.setChecked(True)
        form.addWidget(self.recursive_check, 1, 1)

        self.quarantine_edit = add_path_row(
            form,
            2,
            "隔离目录：",
            mode="open_dir",
            placeholder="留空自动创建同级隔离目录（不会删除文件）",
            dialog_title="选择隔离目录",
            parent=self,
        )
        self.quarantine_edit.textChanged.connect(self._refresh_actions)

        hint = QLabel(
            "重复文件只会移入隔离目录，绝不直接删除；移动后可随时撤销移回。"
        )
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        form.addWidget(hint, 3, 1, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        form.addWidget(self.progress_bar, 4, 1, 1, 2)

        buttons = QHBoxLayout()
        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setDefault(True)
        self.scan_btn.clicked.connect(self._start_scan)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_operation)
        self.move_btn = QPushButton("移入隔离区")
        self.move_btn.clicked.connect(self._start_move)
        self.undo_btn = QPushButton("撤销移回")
        self.undo_btn.clicked.connect(self._start_undo)
        self.open_quarantine_btn = QPushButton("打开隔离目录")
        self.open_quarantine_btn.clicked.connect(self._open_quarantine)
        buttons.addWidget(self.scan_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addSpacing(16)
        buttons.addWidget(self.move_btn)
        buttons.addWidget(self.undo_btn)
        buttons.addWidget(self.open_quarantine_btn)
        buttons.addStretch(1)
        form.addLayout(buttons, 5, 0, 1, 3)

        layout.addLayout(form)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["名称", "大小", "路径"])
        self.tree.setColumnWidth(0, 260)
        self.tree.setColumnWidth(1, 100)
        self.tree.header().setStretchLastSection(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree, 1)

        self.status_label = QLabel(
            "请选择扫描目录后点击「扫描」，默认扫描会跳过隔离目录。"
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._input_widgets = [
            self.root_edit,
            self.recursive_check,
            self.quarantine_edit,
        ]
        for row in (0, 2):
            item = form.itemAtPosition(row, 2)
            if item is not None and item.widget() is not None:
                self._input_widgets.append(item.widget())

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool, operation: str = "") -> None:
        self._busy = busy
        self._operation = operation if busy else ""
        for widget in self._input_widgets:
            widget.setEnabled(not busy)  # type: ignore[attr-defined]
        self.tree.setEnabled(not busy)
        if busy:
            self.scan_btn.setEnabled(False)
            self.cancel_btn.setEnabled(operation == "scan")
            self.move_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)
            self.open_quarantine_btn.setEnabled(False)
        else:
            self.cancel_btn.setEnabled(False)
            self._refresh_actions()

    def _loose_quarantine_path(self) -> Path | None:
        """返回界面当前指向的隔离目录（不校验，可能不存在）。"""
        text = self.quarantine_edit.text().strip()
        if text:
            return Path(text).expanduser()
        return self._quarantine_path

    def _refresh_actions(self, *_args: object) -> None:
        if self._busy:
            return
        self.scan_btn.setEnabled(True)
        self.move_btn.setEnabled(self.tree.topLevelItemCount() > 0)
        self.undo_btn.setEnabled(bool(self._moves))
        path = self._loose_quarantine_path()
        try:
            self.open_quarantine_btn.setEnabled(
                path is not None and path.is_dir()
            )
        except OSError:
            self.open_quarantine_btn.setEnabled(False)

    def _resolve_quarantine_root(
        self, root: Path, *, reuse_scanned: bool
    ) -> Path | None:
        """校验并返回隔离目录；失败时弹窗并返回 None。"""
        text = self.quarantine_edit.text().strip()
        if text:
            quarantine = Path(text).expanduser()
        elif reuse_scanned and self._quarantine_path is not None:
            quarantine = self._quarantine_path
        else:
            quarantine = quarantine_root_for(root)
        try:
            quarantine = quarantine.resolve()
            resolved_root = root.resolve()
        except OSError as exc:
            QMessageBox.warning(
                self, "隔离目录无效", f"无法解析隔离目录：\n{exc}"
            )
            return None

        if quarantine == resolved_root or resolved_root.is_relative_to(
            quarantine
        ):
            QMessageBox.warning(
                self,
                "隔离目录无效",
                "隔离目录不能是扫描目录本身或其上级目录，请换一个位置。",
            )
            return None
        if quarantine.exists() and not quarantine.is_dir():
            QMessageBox.warning(
                self,
                "隔离目录无效",
                f"该路径已存在且不是目录：\n{quarantine}",
            )
            return None
        return quarantine

    def _validate_scan_root(self) -> Path | None:
        text = self.root_edit.text().strip()
        if not text:
            QMessageBox.warning(
                self, "缺少扫描目录", "请先选择要扫描的目录。"
            )
            return None
        root = Path(text).expanduser()
        if not root.is_dir():
            QMessageBox.warning(
                self, "扫描目录不存在", f"目录不存在：\n{root}"
            )
            return None
        try:
            return root.resolve()
        except OSError as exc:
            QMessageBox.warning(
                self, "扫描目录无效", f"无法解析目录：\n{root}\n{exc}"
            )
            return None

    def _refresh_checked_status(self) -> None:
        self._set_status(
            f"已勾选 {len(self._checked)} 个文件，"
            f"可节省 {human_size(self._checked_savings)}。"
        )

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or item.parent() is None:
            return
        path_text = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_text or self._scan_result is None:
            return
        parent = item.parent()
        group_index = parent.data(0, Qt.ItemDataRole.UserRole)
        groups = self._scan_result.groups
        if not isinstance(group_index, int) or not (
            0 <= group_index < len(groups)
        ):
            return
        path = Path(path_text)
        if item.checkState(0) == Qt.CheckState.Checked:
            if path not in self._checked:
                group = groups[group_index]
                self._checked[path] = group
                self._checked_savings += group.size
        else:
            previous = self._checked.pop(path, None)
            if previous is not None:
                self._checked_savings -= previous.size
        self._refresh_checked_status()

    # --------------------------------------------------------------- 扫描流程

    def _start_scan(self) -> None:
        if self._busy:
            return
        root = self._validate_scan_root()
        if root is None:
            return
        quarantine = self._resolve_quarantine_root(
            root, reuse_scanned=False
        )
        if quarantine is None:
            return

        exclude_names: set[str] = set()
        try:
            if quarantine != root and quarantine.is_relative_to(root):
                exclude_names.add(quarantine.name)
        except OSError:
            pass

        recursive = self.recursive_check.isChecked()
        self._quarantine_path = quarantine
        self.progress_bar.setRange(0, 0)
        self._set_busy(True, "scan")
        self._set_status("正在扫描并统计文件大小…")

        def task(
            on_progress: Callable[[object, object], None],
            cancel_event: threading.Event,
        ) -> DuplicateScanResult:
            return find_duplicates(
                root,
                recursive=recursive,
                exclude_names=tuple(exclude_names),
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        worker = SingleTaskWorker(task, self)
        worker.progress_changed.connect(self._on_scan_progress)
        worker.task_finished.connect(self._on_scan_finished)
        worker.task_cancelled.connect(self._on_scan_cancelled)
        worker.task_failed.connect(self._on_task_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_scan_progress(self, done: object, total: object) -> None:
        done_count = int(done)
        total_count = int(total)
        if total_count > 0:
            self.progress_bar.setRange(0, total_count)
            self.progress_bar.setValue(done_count)
            self._set_status(
                f"正在哈希候选文件：{done_count}/{total_count}…"
            )
        else:
            self.progress_bar.setRange(0, 0)

    def _on_scan_finished(self, result: object) -> None:
        if not isinstance(result, DuplicateScanResult):
            self._on_task_failed("扫描返回了无法识别的结果。")
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._scan_result = result
        self._populate_tree()

        default_savings = sum(
            group.size * (len(group.files) - 1)
            for group in result.groups
        )
        message = (
            f"扫描完成：共 {result.scanned_files} 个文件"
            f"（{human_size(result.scanned_bytes)}），"
            f"找到 {len(result.groups)} 组重复文件，"
            f"默认勾选可节省 {human_size(default_savings)}。"
        )
        if result.errors:
            message += f" 另有 {len(result.errors)} 项读取错误。"
        self._set_status(message)
        self._refresh_actions()

        if result.errors:
            preview = self._format_errors(result.errors)
            QMessageBox.warning(
                self,
                "扫描完成（有错误）",
                f"扫描已完成，但部分文件无法读取：\n\n{preview}",
            )

    def _on_scan_cancelled(self, _result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("已取消扫描。")

    def _on_task_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status(f"操作失败：{message}")
        QMessageBox.critical(self, "操作失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel_operation(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._set_status("正在取消…")

    # --------------------------------------------------------------- 结果展示

    def _populate_tree(self) -> None:
        """按扫描结果重建树；每组第一个文件默认保留（不勾选）。"""
        groups = self._scan_result.groups if self._scan_result else []
        self._checked = {}
        self._checked_savings = 0
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            for index, group in enumerate(groups):
                parent = QTreeWidgetItem(self.tree)
                parent.setText(
                    0,
                    f"{human_size(group.size)}，"
                    f"{len(group.files)} 个重复文件",
                )
                parent.setData(0, Qt.ItemDataRole.UserRole, index)
                parent.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                parent.setFirstColumnSpanned(True)
                font = parent.font(0)
                font.setBold(True)
                parent.setFont(0, font)

                for file_index, path in enumerate(group.files):
                    child = QTreeWidgetItem(parent)
                    child.setText(0, path.name)
                    child.setText(1, human_size(group.size))
                    child.setText(2, str(path))
                    child.setToolTip(0, str(path))
                    child.setData(
                        0, Qt.ItemDataRole.UserRole, str(path)
                    )
                    child.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    checked = file_index != 0
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if checked
                        else Qt.CheckState.Unchecked,
                    )
                    if checked:
                        self._checked[path] = group
                        self._checked_savings += group.size
                parent.setExpanded(True)
        finally:
            self.tree.blockSignals(False)

    def _checked_entries(self) -> list[tuple[Path, DuplicateGroup]]:
        """返回当前勾选的（文件路径, 所属组）。"""
        if self._scan_result is None:
            return []
        return list(self._checked.items())

    def _auto_uncheck_full_groups(
        self,
    ) -> list[Path]:
        """若某组全部勾选，取消第一个文件的勾选并返回这些文件。"""
        kept: list[Path] = []
        self.tree.blockSignals(True)
        try:
            for parent_index in range(self.tree.topLevelItemCount()):
                parent = self.tree.topLevelItem(parent_index)
                if parent.childCount() == 0:
                    continue
                states = [
                    parent.child(index).checkState(0)
                    for index in range(parent.childCount())
                ]
                if all(
                    state == Qt.CheckState.Checked for state in states
                ):
                    first = parent.child(0)
                    path_text = first.data(0, Qt.ItemDataRole.UserRole)
                    if path_text:
                        path = Path(path_text)
                        kept.append(path)
                        previous = self._checked.pop(path, None)
                        if previous is not None:
                            self._checked_savings -= previous.size
                    first.setCheckState(0, Qt.CheckState.Unchecked)
        finally:
            self.tree.blockSignals(False)
        return kept

    def _format_errors(
        self, errors: Sequence[str], limit: int = 5
    ) -> str:
        lines = list(errors[:limit])
        if len(errors) > limit:
            lines.append(f"…另有 {len(errors) - limit} 项")
        return "\n".join(lines)

    # --------------------------------------------------------------- 移动流程

    def _start_move(self) -> None:
        if self._busy or self._scan_result is None:
            return

        kept = self._auto_uncheck_full_groups()
        entries = self._checked_entries()
        if kept:
            shown = "\n".join(
                str(path) for path in kept[:5]
            )
            if len(kept) > 5:
                shown += f"\n…另有 {len(kept) - 5} 个"
            QMessageBox.information(
                self,
                "已自动保留文件",
                "以下分组原本全部勾选，已自动取消每组第一个文件"
                "（保留在原处）：\n\n"
                f"{shown}",
            )
        if not entries:
            QMessageBox.information(
                self,
                "没有勾选文件",
                "请先勾选要移入隔离区的重复文件。",
            )
            self._set_status("没有勾选任何文件，未执行移动。")
            self._refresh_actions()
            return

        assert self._scan_result is not None
        root = self._scan_result.root
        quarantine = self._resolve_quarantine_root(
            root, reuse_scanned=True
        )
        if quarantine is None:
            return

        for path, _group in entries:
            try:
                if path.resolve().is_relative_to(quarantine):
                    QMessageBox.warning(
                        self,
                        "勾选文件位于隔离区",
                        f"不能把隔离目录中的文件再次隔离：\n{path}",
                    )
                    return
            except OSError:
                continue

        items = build_quarantine_items(root, quarantine, entries)
        total_size = sum(group.size for _path, group in entries)
        reply = QMessageBox.question(
            self,
            "确认移入隔离区",
            f"将把 {len(items)} 个文件移入隔离目录：\n"
            f"{quarantine}\n\n"
            f"可节省约 {human_size(total_size)}，"
            "不会删除任何文件，之后可撤销移回。\n\n"
            "是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消移动。")
            return

        self._quarantine_path = quarantine
        self.progress_bar.setRange(0, 0)
        self._set_busy(True, "move")
        self._set_status(f"正在移动 {len(items)} 个文件到隔离区…")

        known_digests = {
            path: group.digest for path, group in entries
        }

        def task(
            on_progress: Callable[[object, object], None],
            _cancel_event: threading.Event,
        ) -> tuple[list[QuarantineMove], list[str]]:
            return move_to_quarantine(
                items, quarantine, known_digests=known_digests
            )

        worker = SingleTaskWorker(task, self)
        worker.task_finished.connect(self._on_move_finished)
        worker.task_cancelled.connect(self._on_move_cancelled)
        worker.task_failed.connect(self._on_task_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_move_finished(self, result: object) -> None:
        if not (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], list)
            and isinstance(result[1], list)
        ):
            self._on_task_failed("移动返回了无法识别的结果。")
            return

        moves, errors = result
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._moves.extend(
            move for move in moves if isinstance(move, QuarantineMove)
        )

        moved_paths = {
            Path(move.original)
            for move in moves
            if isinstance(move, QuarantineMove)
        }
        self._remove_moved_files(moved_paths)

        message = (
            f"移动完成：成功 {len(moves)} 个，失败 {len(errors)} 项。"
        )
        if errors:
            message += "\n失败项（前 5 条）：\n" + self._format_errors(errors)
            self._set_status(message)
            QMessageBox.warning(
                self,
                "部分文件移动失败",
                message,
            )
        else:
            self._set_status(message)
            QMessageBox.information(
                self,
                "移动完成",
                f"已将 {len(moves)} 个文件移入隔离目录：\n"
                f"{self._quarantine_path}",
            )
        self._refresh_actions()

    def _on_move_cancelled(self, _result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("已取消移动。")

    def _remove_moved_files(self, moved_paths: set[Path]) -> None:
        """从扫描结果中移除已移动文件，并刷新树。"""
        if self._scan_result is None or not moved_paths:
            return
        groups: list[DuplicateGroup] = []
        for group in self._scan_result.groups:
            remaining = [
                path for path in group.files if path not in moved_paths
            ]
            if len(remaining) >= 2:
                groups.append(
                    DuplicateGroup(
                        size=group.size,
                        digest=group.digest,
                        files=remaining,
                    )
                )
        self._scan_result = DuplicateScanResult(
            root=self._scan_result.root,
            groups=groups,
            errors=self._scan_result.errors,
            scanned_files=self._scan_result.scanned_files,
            scanned_bytes=self._scan_result.scanned_bytes,
        )
        self._populate_tree()

    # --------------------------------------------------------------- 撤销流程

    def _start_undo(self) -> None:
        if self._busy or not self._moves:
            return
        snapshot = list(self._moves)
        reply = QMessageBox.question(
            self,
            "确认撤销",
            f"将把隔离区中的 {len(snapshot)} 个文件移回原位置。\n"
            "若原位置已有同名文件，将跳过并继续保留在隔离区。\n\n"
            "是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("已取消撤销。")
            return

        self.progress_bar.setRange(0, 0)
        self._set_busy(True, "undo")
        self._set_status(f"正在撤销移回 {len(snapshot)} 个文件…")

        def task(
            on_progress: Callable[[object, object], None],
            _cancel_event: threading.Event,
        ) -> tuple[list[tuple[Path, Path]], list[str]]:
            return undo_quarantine(snapshot)

        worker = SingleTaskWorker(task, self)
        worker.task_finished.connect(self._on_undo_finished)
        worker.task_cancelled.connect(self._on_undo_cancelled)
        worker.task_failed.connect(self._on_task_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_undo_finished(self, result: object) -> None:
        if not (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], list)
            and isinstance(result[1], list)
        ):
            self._on_task_failed("撤销返回了无法识别的结果。")
            return

        restored, errors = result
        restored_sources = {
            Path(source) for source, _target in restored
        }
        self._moves = [
            move
            for move in self._moves
            if Path(move.quarantined) not in restored_sources
        ]

        # 原位置可能已恢复出新的重复文件，清空旧扫描结果避免误操作。
        self._scan_result = None
        self._checked = {}
        self._checked_savings = 0
        self.tree.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        message = f"撤销完成：成功移回 {len(restored)} 个文件"
        if errors:
            message += f"，失败 {len(errors)} 项。"
            message += "\n失败项（前 5 条）：\n" + self._format_errors(errors)
            self._set_status(message)
            QMessageBox.warning(
                self,
                "部分文件撤销失败",
                message,
            )
        else:
            message += "，隔离记录已清空。"
            self._set_status(message)
            QMessageBox.information(
                self, "撤销完成", f"已移回 {len(restored)} 个文件。"
            )
        self._refresh_actions()

    def _on_undo_cancelled(self, _result: object) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("已取消撤销。")

    # --------------------------------------------------------------- 目录操作

    def _open_quarantine(self) -> None:
        path = self._loose_quarantine_path()
        if path is None:
            QMessageBox.information(
                self,
                "尚未指定隔离目录",
                "请先选择隔离目录，或先执行一次扫描/移动。",
            )
            return
        if not path.is_dir():
            QMessageBox.information(
                self,
                "隔离目录尚未创建",
                f"目录尚不存在（移动文件后会自动创建）：\n{path}",
            )
            return
        try:
            open_directory(path)
        except Exception as exc:  # noqa: BLE001 - 打开失败时提示用户
            QMessageBox.critical(
                self, "无法打开目录", f"{path}\n{exc}"
            )

    # ----------------------------------------------------------------- 关闭

    def shutdown(self) -> bool:
        """主窗口关闭前调用；后台任务进行中时确认并等待结束。"""
        labels = {"scan": "扫描", "move": "移动", "undo": "撤销"}
        label = labels.get(self._operation, "文件操作")
        return shutdown_worker(self._worker, self.window(), label)
