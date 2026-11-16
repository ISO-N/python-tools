"""可取消后台任务的通用 QThread 封装。

约定：worker 内不操作任何 QWidget / QMessageBox，只通过信号回主线程；
所有耗时逻辑在取消点之间检查 ``threading.Event``。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget


class Cancelled(Exception):
    """任务被用户取消；``result`` 可携带已完成工作的部分结果。"""

    def __init__(self, result: object = None) -> None:
        super().__init__("任务已取消")
        self.result = result


@dataclass
class BatchResult:
    """批量任务的结果。"""

    total: int
    done: int
    errors: list[tuple[str, str]] = field(default_factory=list)
    values: list[object] = field(default_factory=list)
    cancelled: bool = False


class BatchWorker(QThread):
    """逐项执行批量任务；单项失败记录错误后继续。"""

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
    ) -> None:
        super().__init__(parent)
        self._items = list(items)
        self._process = process
        self._label_fn = label_fn or (lambda item: str(item))
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        total = len(self._items)
        done = 0
        errors: list[tuple[str, str]] = []
        values: list[object] = []
        try:
            for item in self._items:
                if self._cancel_event.is_set():
                    self.batch_cancelled.emit(
                        BatchResult(total, done, errors, values, True)
                    )
                    return
                label = self._label_fn(item)
                self.item_started.emit(label)
                self.progress_changed.emit(done, total)
                try:
                    value = self._process(item, self._cancel_event)
                except Cancelled:
                    self.batch_cancelled.emit(
                        BatchResult(total, done, errors, values, True)
                    )
                    return
                except Exception as exc:  # noqa: BLE001 - 单项失败继续
                    errors.append((label, str(exc)))
                else:
                    values.append(value)
                    done += 1
                self.progress_changed.emit(done, total)
            self.batch_finished.emit(
                BatchResult(total, done, errors, values, False)
            )
        except Exception as exc:  # noqa: BLE001 - 未预期异常回传界面
            self.batch_failed.emit(str(exc))


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
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        def on_progress(done: object, total: object) -> None:
            self.progress_changed.emit(done, total)

        try:
            result = self._task(on_progress, self._cancel_event)
        except Cancelled as exc:
            self.task_cancelled.emit(exc.result)
        except Exception as exc:  # noqa: BLE001 - 未预期异常回传界面
            self.task_failed.emit(str(exc))
        else:
            self.task_finished.emit(result)


def shutdown_worker(
    worker: QThread | None, parent: QWidget | None, action_label: str = "任务"
) -> bool:
    """关闭窗口前的通用处理：询问、取消并等待后台线程结束。"""
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
    if not worker.wait(5000):
        QMessageBox.information(
            parent,
            "正在停止",
            "后台任务仍在结束中，请稍候再关闭窗口。",
        )
        return False
    return True
