"""重复文件查找（线程池并行）单元测试。"""

from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from toolbox.errors import Cancelled
from toolbox.tools import duplicate_finder
from toolbox.tools.duplicate_finder import (
    PREFIX_HASH_SIZE,
    find_duplicates,
)


class DuplicateFinderParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = self.root / "data"
        self.folder.mkdir()

        self.small = b"same-small-content"
        self.prefix = bytes(range(256)) * (PREFIX_HASH_SIZE // 256)
        self.large = self.prefix + b"tail" * 400
        self.collision_left = self.prefix + b"A" * 100
        self.collision_right = self.prefix + b"B" * 100

        for name in ("small_a.bin", "small_b.bin"):
            (self.folder / name).write_bytes(self.small)
        for name in ("prefix_a.bin", "prefix_b.bin"):
            (self.folder / name).write_bytes(self.prefix)
        for name in ("large_a.bin", "large_b.bin"):
            (self.folder / name).write_bytes(self.large)
        (self.folder / "collision_a.bin").write_bytes(
            self.collision_left
        )
        (self.folder / "collision_b.bin").write_bytes(
            self.collision_right
        )
        # 同尺寸但前缀不同，不应组成重复组。
        (self.folder / "unique_a.bin").write_bytes(b"U" * 5000)
        (self.folder / "unique_b.bin").write_bytes(b"V" * 5000)
        # 其中一个文件模拟读取失败（与 peer 同尺寸，进入候选列表）。
        (self.folder / "broken.bin").write_bytes(b"x" * 50)
        (self.folder / "broken_peer.bin").write_bytes(b"y" * 50)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _flaky_prefix(self):
        real = duplicate_finder._hash_prefix

        def wrapper(path, cancel_event=None):
            if Path(path).name == "broken.bin":
                raise OSError("模拟读取失败")
            return real(path, cancel_event)

        return wrapper

    def _scan(
        self,
        workers: int,
        *,
        cancel_event: threading.Event | None = None,
        on_progress=None,
    ):
        progress: list[tuple[int, int]] = []

        def report(done: int, total: int) -> None:
            progress.append((done, total))
            if on_progress is not None:
                on_progress(done, total)

        with (
            mock.patch.object(
                duplicate_finder, "MAX_HASH_WORKERS", workers
            ),
            mock.patch.object(
                duplicate_finder,
                "_hash_prefix",
                side_effect=self._flaky_prefix(),
            ),
        ):
            result = find_duplicates(
                self.folder,
                cancel_event=cancel_event,
                on_progress=report,
            )
        return result, progress

    def test_parallel_matches_serial_groups_errors_and_progress(self) -> None:
        serial, serial_progress = self._scan(1)
        parallel, parallel_progress = self._scan(4)

        self.assertEqual(serial.groups, parallel.groups)
        self.assertEqual(serial.errors, parallel.errors)
        self.assertEqual(serial.scanned_files, parallel.scanned_files)
        self.assertEqual(serial.scanned_bytes, parallel.scanned_bytes)
        self.assertEqual(serial_progress, parallel_progress)

        expected_digests = {
            hashlib.blake2b(self.small, digest_size=32).hexdigest(),
            hashlib.blake2b(self.prefix, digest_size=32).hexdigest(),
            hashlib.blake2b(self.large, digest_size=32).hexdigest(),
        }
        self.assertEqual(
            {group.digest for group in parallel.groups}, expected_digests
        )
        self.assertEqual(
            {
                tuple(sorted(path.name for path in group.files))
                for group in parallel.groups
            },
            {
                ("small_a.bin", "small_b.bin"),
                ("prefix_a.bin", "prefix_b.bin"),
                ("large_a.bin", "large_b.bin"),
            },
        )
        self.assertEqual(len(parallel.errors), 1)
        self.assertIn("broken.bin", parallel.errors[0])
        self.assertIn("模拟读取失败", parallel.errors[0])
        # 12 个候选文件都会推进一次进度，最终应到达 (12, 12)。
        self.assertEqual(parallel_progress[0], (0, 12))
        self.assertEqual(parallel_progress[-1], (12, 12))
        self.assertEqual(
            [done for done, _total in parallel_progress],
            list(range(13)),
        )

    def test_cancel_during_hash_raises_cancelled(self) -> None:
        cancel_event = threading.Event()

        def stop_after_first(done: int, _total: int) -> None:
            if done >= 1:
                cancel_event.set()

        with self.assertRaises(Cancelled):
            self._scan(
                4,
                cancel_event=cancel_event,
                on_progress=stop_after_first,
            )

    def test_cancel_before_scan_raises_cancelled(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        with self.assertRaises(Cancelled):
            self._scan(4, cancel_event=cancel_event)


if __name__ == "__main__":
    unittest.main()
