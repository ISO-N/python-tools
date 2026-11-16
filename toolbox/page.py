"""工具页基类。"""

from __future__ import annotations

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QWidget

from toolbox.workers import (
    confirm_cancel_worker,
    shutdown_worker,
    wait_worker,
)


class ToolPage(QWidget):
    """所有工具箱页面的基类。"""

    #: 关闭询问 / 正在停止提示中使用的任务名称。
    shutdown_label = "任务"

    def current_worker(self) -> QThread | None:
        """返回页面当前运行的后台任务；没有时返回 None。

        默认自动识别常见的 ``self._worker`` 字段，旧工具页即使只覆盖了
        ``shutdown()`` 也能被主窗口的两阶段关闭流程感知。
        """
        worker = getattr(self, "_worker", None)
        if isinstance(worker, QThread) and worker.isRunning():
            return worker
        return None

    def confirm_shutdown(self) -> bool:
        """关闭窗口前询问是否取消当前任务；返回 False 表示暂不能关闭。"""
        return confirm_cancel_worker(
            self.current_worker(), self, self.shutdown_label
        )

    def wait_shutdown(self, deadline: float) -> bool:
        """在共享截止时间前等待当前任务结束。"""
        return wait_worker(self.current_worker(), self, deadline)

    def shutdown(self) -> bool:
        """兼容旧调用：询问、取消并在 5 秒内等待后台任务结束。"""
        return shutdown_worker(
            self.current_worker(), self, self.shutdown_label
        )
