"""与图形界面无关的通用异常。

放在独立模块中，便于图片 / PDF / 文件操作等共享层引用取消异常，
也避免进程池子进程被迫导入 PyQt6。
"""

from __future__ import annotations


class Cancelled(Exception):
    """任务被用户取消；``result`` 可携带已完成工作的部分结果。"""

    def __init__(self, result: object = None) -> None:
        super().__init__("任务已取消")
        self.result = result
