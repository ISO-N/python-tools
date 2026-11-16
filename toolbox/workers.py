"""可取消后台任务的通用 QThread 封装。

约定：worker 内不操作任何 QWidget / QMessageBox，只通过信号回主线程；
所有耗时逻辑在取消点之间检查 ``threading.Event``。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from toolbox.errors import Cancelled

# 图片批处理与哈希的默认并发度。图片受内存带宽限制，哈希受磁盘限制，
# 4 个线程通常已能覆盖多数桌面场景，同时避免内存峰值失控。
MAX_IMAGE_WORKERS = min(4, os.cpu_count() or 1)
MAX_HASH_WORKERS = min(4, os.cpu_count() or 1)

__all__ = [
    "BatchResult",
    "BatchWorker",
    "Cancelled",
    "MAX_HASH_WORKERS",
    "MAX_IMAGE_WORKERS",
    "SingleTaskWorker",
    "confirm_cancel_worker",
    "shutdown_worker",
    "wait_worker",
]


class _ProgressGate:
    """按时间间隔节流进度回调，并保证最终值一定会发出。"""

    def __init__(
        self,
        emit: Callable[[object, object], None],
        interval_ms: int,
    ) -> None:
        self._emit = emit
        self._interval = max(0.0, interval_ms / 1000.0)
        self._last_time = 0.0
        self._last_value: tuple[object, object] | None = None

    def update(self, done: object, total: object, *, force: bool = False) -> None:
        value = (done, total)
        now = time.monotonic()
        if not force and self._interval > 0:
            if value == self._last_value:
                return
            if now - self._last_time < self._interval:
                return
        self._last_time = now
        self._last_value = value
        self._emit(done, total)

    def force_last(self) -> None:
        if self._last_value is not None:
            self._emit(*self._last_value)


@dataclass
class BatchResult:
    """批量任务的结果。"""

    total: int
    done: int
    errors: list[tuple[str, str]] = field(default_factory=list)
    values: list[object] = field(default_factory=list)
    cancelled: bool = False


class BatchWorker(QThread):
    """逐项执行批量任务；单项失败记录错误后继续。

    ``max_workers > 1`` 时使用线程池并发执行，但结果值与错误仍按输入
    顺序排列；状态栏的“当前项”顺序可能随完成顺序变化。
    """

    item_started = pyqtSignal(str)
    progress_changed = pyqtSignal(object, object)
    batch_finished = pyqtSignal(object)
    batch_cancelled = pyqtSignal(object)
    batch_failed = pyqtSignal(str)

    def __init__(
        self,
        items: Iterable[object],
        process: Callable[[object, threading.Event], object],
        label_fn: Callable[[object], str] | None = None,
        parent: QWidget | None = None,
        *,
        max_workers: int = 1,
        progress_interval_ms: int = 0,
        max_in_flight: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._items = list(items)
        self._process = process
        self._label_fn = label_fn or (lambda item: str(item))
        self._cancel_event = threading.Event()
        self._max_workers = max(1, int(max_workers))
        self._progress_interval_ms = max(0, int(progress_interval_ms))
        self._max_in_flight = max_in_flight
        self._gate: _ProgressGate | None = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def _progress(self, done: object, total: object, *, force: bool = False) -> None:
        if self._gate is not None:
            self._gate.update(done, total, force=force)

    def _run_serial(self) -> None:
        total = len(self._items)
        done = 0
        errors: list[tuple[str, str]] = []
        values: list[object] = []
        for item in self._items:
            if self._cancel_event.is_set():
                self._progress(done, total, force=True)
                self.batch_cancelled.emit(
                    BatchResult(total, done, errors, values, True)
                )
                return
            label = self._label_fn(item)
            self.item_started.emit(label)
            self._progress(done, total)
            try:
                value = self._process(item, self._cancel_event)
            except Cancelled:
                self._progress(done, total, force=True)
                self.batch_cancelled.emit(
                    BatchResult(total, done, errors, values, True)
                )
                return
            except Exception as exc:  # noqa: BLE001 - 单项失败继续
                errors.append((label, str(exc)))
            else:
                values.append(value)
                done += 1
            self._progress(done, total)
        self._progress(done, total, force=True)
        self.batch_finished.emit(
            BatchResult(total, done, errors, values, False)
        )

    def _run_parallel(self) -> None:
        total = len(self._items)
        started: queue.Queue[tuple[int, str]] = queue.Queue()

        def task(index: int, item: object, label: str) -> tuple[int, object]:
            started.put((index, label))
            return index, self._process(item, self._cancel_event)

        executor = ThreadPoolExecutor(max_workers=self._max_workers)
        pending: dict[Future[tuple[int, object]], tuple[int, str]] = {}
        values_by_index: dict[int, object] = {}
        errors_by_index: dict[int, tuple[str, str]] = {}
        done = 0
        next_index = 0
        cancelled = False
        window = self._max_in_flight or (self._max_workers * 2)
        window = max(1, min(window, total or 1))

        def submit_one() -> bool:
            nonlocal next_index
            if next_index >= total:
                return False
            index = next_index
            item = self._items[index]
            next_index += 1
            label = self._label_fn(item)
            future = executor.submit(task, index, item, label)
            pending[future] = (index, label)
            return True

        def drain_started() -> None:
            while True:
                try:
                    _index, label = started.get_nowait()
                except queue.Empty:
                    return
                self.item_started.emit(label)

        try:
            self._progress(0, total)
            while len(pending) < window and submit_one():
                pass
            while pending:
                drain_started()
                if self._cancel_event.is_set() and not cancelled:
                    cancelled = True
                    for future in pending:
                        future.cancel()
                finished, _ = wait(
                    pending.keys(),
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                for future in finished:
                    index, label = pending.pop(future)
                    try:
                        _index, value = future.result()
                    except (Cancelled, CancelledError):
                        cancelled = True
                    except Exception as exc:  # noqa: BLE001 - 单项失败继续
                        errors_by_index[index] = (label, str(exc))
                    else:
                        values_by_index[index] = value
                        done += 1
                    self._progress(done, total, force=not pending)
                if not cancelled:
                    while len(pending) < window and submit_one():
                        pass
            drain_started()
            self._progress(done, total, force=True)
            errors = [
                errors_by_index[index]
                for index in sorted(errors_by_index)
            ]
            values = [
                values_by_index[index]
                for index in sorted(values_by_index)
            ]
            if cancelled:
                self.batch_cancelled.emit(
                    BatchResult(total, done, errors, values, True)
                )
            else:
                self.batch_finished.emit(
                    BatchResult(total, done, errors, values, False)
                )
        except Exception as exc:  # noqa: BLE001 - 未预期异常回传界面
            self.batch_failed.emit(str(exc))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def run(self) -> None:
        self._gate = _ProgressGate(
            self.progress_changed.emit, self._progress_interval_ms
        )
        if self._max_workers <= 1:
            self._run_serial()
        else:
            self._run_parallel()


class SingleTaskWorker(QThread):
    """执行单个长任务；进度由任务通过回调报告。"""

    progress_changed = pyqtSignal(object, object)
    task_finished = pyqtSignal(object)
    task_cancelled = pyqtSignal(object)
    task_failed = pyqtSignal(str)

    def __init__(
        self,
        task: Callable[
            [Callable[[object, object], None], threading.Event], object
        ],
        parent: QWidget | None = None,
        *,
        progress_interval_ms: int = 0,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._cancel_event = threading.Event()
        self._progress_interval_ms = max(0, int(progress_interval_ms))

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        gate = _ProgressGate(
            self.progress_changed.emit, self._progress_interval_ms
        )

        def on_progress(done: object, total: object) -> None:
            gate.update(done, total)

        try:
            result = self._task(on_progress, self._cancel_event)
        except Cancelled as exc:
            gate.force_last()
            self.task_cancelled.emit(exc.result)
        except Exception as exc:  # noqa: BLE001 - 未预期异常回传界面
            gate.force_last()
            self.task_failed.emit(str(exc))
        else:
            gate.force_last()
            self.task_finished.emit(result)


def confirm_cancel_worker(
    worker: QThread | None,
    parent: QWidget | None,
    action_label: str = "任务",
) -> bool:
    """询问用户是否取消正在运行的任务；确认时调用 ``cancel()``。"""
    if worker is None or not worker.isRunning():
        return True
    reply = QMessageBox.question(
        parent,
        "退出",
        f"{action_label}正在进行，确定要退出吗？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return False
    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        cancel()
    return True


def wait_worker(
    worker: QThread | None,
    parent: QWidget | None,
    deadline: float,
) -> bool:
    """在共享截止时间前等待 worker 结束；超时返回 False。"""
    if worker is None or not worker.isRunning():
        return True
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    return worker.wait(max(0, int(remaining * 1000)))


def shutdown_worker(
    worker: QThread | None,
    parent: QWidget | None,
    action_label: str = "任务",
) -> bool:
    """关闭窗口前的通用处理：询问、取消并等待后台线程结束。"""
    if not confirm_cancel_worker(worker, parent, action_label):
        return False
    if not wait_worker(worker, parent, time.monotonic() + 5.0):
        QMessageBox.information(
            parent,
            "正在停止",
            "后台任务仍在结束中，请稍候再关闭窗口。",
        )
        return False
    return True
