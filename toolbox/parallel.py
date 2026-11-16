"""与 GUI 无关的分块进程池执行器。

只依赖标准库，供 PDF 逐页渲染等可独立拆分的任务使用。任务函数必须是
模块顶层可 pickle 的函数，payload 只使用 dict / tuple / str / int 等
原始类型；不要传入 Qt 对象、PIL 字体或 PyMuPDF Document。
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """一个可独立执行的工作块。"""

    index: int
    units: int
    payload: object


class ChunkFailure(RuntimeError):
    """某个工作块执行失败。"""

    def __init__(self, chunk_index: int, message: str) -> None:
        super().__init__(f"第 {chunk_index + 1} 个工作块失败：{message}")
        self.chunk_index = chunk_index
        self.message = message


class ChunkedCancelled(Exception):
    """任务被取消；``completed_chunks`` 为已按顺序提交的块索引。"""

    def __init__(self, completed_chunks: list[int]) -> None:
        super().__init__("任务已取消")
        self.completed_chunks = completed_chunks


def _validate_chunks(chunks: Sequence[Chunk]) -> None:
    for index, chunk in enumerate(chunks):
        if chunk.index != index:
            raise ValueError("chunk 必须按 index 0..n-1 的顺序排列")


def run_chunked_pool(
    chunks: Sequence[Chunk],
    chunk_fn: Callable[[object], object],
    *,
    cancel_event: threading.Event | None,
    on_committed: Callable[[Chunk, object], None],
    max_workers: int,
    max_pending: int | None = None,
    poll_interval: float = 0.1,
    start_method: str = "spawn",
) -> None:
    """按索引顺序提交分块，并按索引顺序把完成结果交给 ``on_committed``。

    取消或失败时立即终止进程池；已提交但未 commit 的结果丢弃，由调用方
    根据临时文件前缀清理残留。``max_workers <= 1`` 时在当前线程串行执行。
    """
    chunks = list(chunks)
    if not chunks:
        return
    _validate_chunks(chunks)
    workers = max(1, int(max_workers))

    if workers <= 1:
        for position, chunk in enumerate(chunks):
            if cancel_event is not None and cancel_event.is_set():
                raise ChunkedCancelled(list(range(position)))
            try:
                result = chunk_fn(chunk.payload)
            except Exception as exc:  # noqa: BLE001 - 统一包装工作块错误
                raise ChunkFailure(chunk.index, str(exc)) from exc
            on_committed(chunk, result)
        return

    pending_limit = max(1, int(max_pending or workers * 2))
    context = multiprocessing.get_context(start_method)
    pool = context.Pool(processes=workers)
    try:
        pending: dict[int, multiprocessing.pool.AsyncResult] = {}
        next_submit = 0
        committed = 0

        while committed < len(chunks):
            if cancel_event is not None and cancel_event.is_set():
                raise ChunkedCancelled(list(range(committed)))

            while (
                next_submit < len(chunks)
                and len(pending) < pending_limit
            ):
                chunk = chunks[next_submit]
                pending[chunk.index] = pool.apply_async(
                    chunk_fn, (chunk.payload,)
                )
                next_submit += 1

            # 任意块报告异常时尽快终止，而不是等前面的块完成。
            for index, result in list(pending.items()):
                if result.ready() and not result.successful():
                    try:
                        result.get()
                    except Exception as exc:  # noqa: BLE001 - 统一包装
                        raise ChunkFailure(index, str(exc)) from exc

            progressed = False
            while committed in pending:
                result = pending[committed]
                if not result.ready():
                    break
                try:
                    value = result.get()
                except Exception as exc:  # noqa: BLE001 - 统一包装
                    raise ChunkFailure(committed, str(exc)) from exc
                del pending[committed]
                on_committed(chunks[committed], value)
                committed += 1
                progressed = True

            if committed >= len(chunks):
                break
            if not progressed:
                time.sleep(poll_interval)
    finally:
        pool.terminate()
        pool.join()
