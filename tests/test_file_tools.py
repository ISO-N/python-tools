"""文件整理 / 重复文件 / 图片归档计划的单元测试。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from toolbox.errors import Cancelled
from toolbox.tools import duplicate_finder as duplicate
from toolbox.tools import file_organizer as organizer
from toolbox.tools import move_images_to_folders as move_images


class FileOrganizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        self.source.mkdir()
        self.target.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_plan_and_execute_basic(self) -> None:
        (self.source / "a.txt").write_text("a", encoding="utf-8")
        (self.source / "b.jpg").write_text("b", encoding="utf-8")
        (self.source / ".hidden.txt").write_text("h", encoding="utf-8")
        options = organizer.OrganizeOptions(
            mode=organizer.MODE_EXTENSION, recursive=False
        )
        plan = organizer.build_organize_plan(
            self.source, self.target, options
        )

        self.assertEqual(
            [(item.src.name, item.bucket) for item in plan.items],
            [("b.jpg", "jpg"), ("a.txt", "txt")],
        )
        self.assertIs(plan.move_items, plan.move_items)
        self.assertEqual(plan.skipped_count, 1)
        self.assertIn(
            str(self.source), [entry[0] for entry in plan.fingerprint]
        )

        result = organizer.execute_organize(plan)
        self.assertEqual((len(result.moved), result.skipped), (2, 0))
        self.assertEqual(result.errors, [])
        self.assertTrue((self.target / "txt" / "a.txt").is_file())
        self.assertTrue((self.target / "jpg" / "b.jpg").is_file())
        self.assertFalse((self.source / "a.txt").exists())

    def test_auto_rename_uses_next_available_name(self) -> None:
        (self.target / "txt").mkdir()
        (self.target / "txt" / "x.txt").write_text(
            "old", encoding="utf-8"
        )
        (self.source / "x.txt").write_text("new", encoding="utf-8")
        options = organizer.OrganizeOptions(
            mode=organizer.MODE_EXTENSION,
            recursive=False,
            conflict_policy=organizer.CONFLICT_RENAME,
        )
        plan = organizer.build_organize_plan(
            self.source, self.target, options
        )
        self.assertEqual(plan.items[0].dst.name, "x (1).txt")

        result = organizer.execute_organize(plan)
        self.assertEqual(len(result.moved), 1)
        self.assertEqual(
            (self.target / "txt" / "x.txt").read_text(encoding="utf-8"),
            "old",
        )
        self.assertEqual(
            (self.target / "txt" / "x (1).txt").read_text(
                encoding="utf-8"
            ),
            "new",
        )

    def test_duplicate_planned_destination_skips_second(self) -> None:
        first = self.source / "a.txt"
        second = self.source / "b.txt"
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")
        destination = self.target / "txt" / "same.txt"
        plan = organizer.OrganizePlan(
            root=self.source,
            target_root=self.target,
            items=[
                organizer.OrganizeItem(first, destination, "txt"),
                organizer.OrganizeItem(second, destination, "txt"),
            ],
        )

        result = organizer.execute_organize(plan)
        self.assertEqual((len(result.moved), result.skipped), (1, 1))
        self.assertEqual(
            destination.read_text(encoding="utf-8"), "first"
        )
        self.assertTrue(second.is_file())

    def test_cancel_returns_partial_result(self) -> None:
        source = self.source / "a.txt"
        source.write_text("a", encoding="utf-8")
        plan = organizer.OrganizePlan(
            root=self.source,
            target_root=self.target,
            items=[
                organizer.OrganizeItem(
                    source, self.target / "txt" / "a.txt", "txt"
                )
            ],
            skipped=[(self.source / ".hidden", "隐藏文件")],
        )
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled) as ctx:
            organizer.execute_organize(plan, cancel_event=cancel)
        partial = ctx.exception.result
        self.assertIsInstance(partial, organizer.OrganizeResult)
        self.assertEqual(partial.moved, [])
        self.assertEqual(partial.skipped, 0)
        self.assertTrue(source.is_file())

    def test_classify_mtime_accepts_cached_stat(self) -> None:
        source = self.source / "a.txt"
        source.write_text("a", encoding="utf-8")
        info = os.stat(source)
        expected = organizer.classify_file(
            source, organizer.MODE_MTIME
        )
        self.assertEqual(
            organizer.classify_file(
                source, organizer.MODE_MTIME, stat_result=info
            ),
            expected,
        )
        self.assertEqual(
            organizer.classify_file(
                source, organizer.MODE_MTIME, mtime=info.st_mtime
            ),
            expected,
        )


class DuplicateFinderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_small_files_are_not_read_twice(self) -> None:
        (self.root / "a.bin").write_bytes(b"x" * 128)
        (self.root / "b.bin").write_bytes(b"x" * 128)

        hash_calls: list[tuple[object, ...]] = []
        original_hash = duplicate.hash_file

        def spy(*args: object, **kwargs: object) -> str:
            hash_calls.append(args)
            return original_hash(*args, **kwargs)

        opened: list[Path] = []
        original_open = Path.open

        def counting_open(
            path: Path, *args: object, **kwargs: object
        ):
            opened.append(path)
            return original_open(path, *args, **kwargs)

        with mock.patch.object(duplicate, "hash_file", side_effect=spy):
            with mock.patch.object(Path, "open", counting_open):
                result = duplicate.find_duplicates(self.root)

        self.assertEqual(len(result.groups), 1)
        self.assertEqual(
            sorted(path.name for path in result.groups[0].files),
            ["a.bin", "b.bin"],
        )
        # 小文件的前缀摘要即全量摘要，不再调用完整哈希或重新读文件。
        self.assertEqual(hash_calls, [])
        self.assertEqual(sorted(path.name for path in opened), ["a.bin", "b.bin"])

    def test_planned_quarantine_paths_are_unique(self) -> None:
        source = self.root / "a.bin"
        source.write_bytes(b"x")
        group = duplicate.DuplicateGroup(
            size=1, digest="digest", files=[source, source]
        )
        items = duplicate.build_quarantine_items(
            self.root,
            self.root / "quarantine",
            [(source, group), (source, group)],
        )
        self.assertEqual(
            [target.name for _source, target in items],
            ["a.bin", "a (1).bin"],
        )

    def test_move_and_undo_quarantine_manifest(self) -> None:
        source = self.root / "a.bin"
        source.write_bytes(b"payload")
        quarantine_root = self.root / "quarantine"
        items = [(source, quarantine_root / "a.bin")]
        moves, errors = duplicate.move_to_quarantine(
            items, quarantine_root, known_digests={source: "digest"}
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(moves), 1)
        self.assertTrue((quarantine_root / "a.bin").is_file())
        self.assertTrue((quarantine_root / "manifest.json").is_file())

        restored, undo_errors = duplicate.undo_quarantine(moves)
        self.assertEqual(undo_errors, [])
        self.assertEqual(restored, [(quarantine_root / "a.bin", source)])
        self.assertTrue(source.is_file())


class MoveImagesPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_plan_cache_hit_and_invalidation(self) -> None:
        (self.root / "book_page_001.png").write_bytes(b"")
        (self.root / "book_page_002.png").write_bytes(b"")
        config = '{"第一章": 1}'
        cache = move_images.ArchivePlanCache()

        scan_count = 0
        original_scan = move_images.scan_images

        def scanning(
            folder: Path,
            *,
            cancel_event: threading.Event | None = None,
        ):
            nonlocal scan_count
            scan_count += 1
            return original_scan(folder, cancel_event=cancel_event)

        with mock.patch.object(
            move_images, "scan_images", side_effect=scanning
        ):
            first = move_images.build_archive_plan(
                self.root, 0, config, cache
            )
            second = move_images.build_archive_plan(
                self.root, 0, config, cache
            )
            self.assertIs(first, second)
            self.assertEqual(scan_count, 1)

            move_images.build_archive_plan(self.root, 1, config, cache)
            self.assertEqual(scan_count, 2)

            move_images.build_archive_plan(
                self.root, 1, config + " ", cache
            )
            self.assertEqual(scan_count, 3)

            current = self.root.stat().st_mtime_ns
            os.utime(
                self.root,
                ns=(current + 1_000_000, current + 1_000_000),
            )
            move_images.build_archive_plan(
                self.root, 1, config + " ", cache
            )
            self.assertEqual(scan_count, 4)

    def test_scan_images_sorts_by_page_once(self) -> None:
        (self.root / "book_page_10.png").write_bytes(b"")
        (self.root / "book_page_2.png").write_bytes(b"")
        (self.root / "book_page_2_extra.png").write_bytes(b"")
        images, _unrecognized = move_images.scan_images(self.root)
        self.assertEqual(
            [page for page, _path in images], [2, 10]
        )


class ScanCancellationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_file_organizer_scan_cancel(self) -> None:
        (self.root / "a.txt").write_text("a", encoding="utf-8")
        target = self.root / "target"
        target.mkdir()
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            organizer.build_organize_plan(
                self.root,
                target,
                organizer.OrganizeOptions(
                    mode=organizer.MODE_EXTENSION, recursive=False
                ),
                cancel_event=event,
            )

    def test_move_images_scan_cancel(self) -> None:
        (self.root / "book_page_1.png").write_bytes(b"")
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            move_images.build_archive_plan(
                self.root, 0, '{"第一章": 1}', cancel_event=event
            )


if __name__ == "__main__":
    unittest.main()
