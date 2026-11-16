"""EPUB 构建、图片复用与 ZIP 压缩方式的单元测试。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

from toolbox.errors import Cancelled
from toolbox.tools import epub_builder
from toolbox.tools.epub_builder import (
    BookMetadata,
    book_plan_is_current,
    build_epub,
    discover_book,
)


class EpubBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input_root = self.root / "book"
        self.chapter = self.input_root / "chapter1"
        self.chapter.mkdir(parents=True)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir()

        self.cover = self.input_root / "cover.png"
        Image.new("RGB", (40, 30), (200, 10, 10)).save(self.cover)
        self.jpg = self.chapter / "p1.jpg"
        Image.new("RGB", (30, 20), (10, 200, 10)).save(self.jpg)
        self.png = self.chapter / "p2.png"
        Image.new("RGB", (20, 30), (10, 10, 200)).save(self.png)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_zip_entries_and_reused_image_bytes(self) -> None:
        plan = discover_book(self.input_root)
        self.assertEqual(plan.total_images, 3)
        self.assertEqual([chapter.title for chapter in plan.chapters], ["chapter1"])
        self.assertEqual(plan.cover, self.cover)
        self.assertTrue(book_plan_is_current(plan))

        output = self.output_dir / "book.epub"
        result = build_epub(plan, BookMetadata(title="测试书"), output)
        self.assertEqual((result.chapters, result.images), (1, 3))
        self.assertEqual(result.skipped, [])

        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertEqual(names[0], "mimetype")
            self.assertEqual(
                archive.getinfo("mimetype").compress_type,
                zipfile.ZIP_STORED,
            )

            image_names = [
                name
                for name in names
                if name.startswith("OEBPS/images/")
            ]
            self.assertEqual(
                image_names,
                [
                    "OEBPS/images/cover.png",
                    "OEBPS/images/img_0001.jpg",
                    "OEBPS/images/img_0002.png",
                ],
            )
            for name in image_names:
                self.assertEqual(
                    archive.getinfo(name).compress_type,
                    zipfile.ZIP_STORED,
                )

            for name in (
                "META-INF/container.xml",
                "OEBPS/style.css",
                "OEBPS/cover.xhtml",
                "OEBPS/chapter_001.xhtml",
                "OEBPS/content.opf",
                "OEBPS/toc.ncx",
            ):
                self.assertEqual(
                    archive.getinfo(name).compress_type,
                    zipfile.ZIP_DEFLATED,
                )

            # 可复用的 JPEG / PNG 在 zip 内与源文件字节完全一致。
            self.assertEqual(
                archive.read("OEBPS/images/cover.png"),
                self.cover.read_bytes(),
            )
            self.assertEqual(
                archive.read("OEBPS/images/img_0001.jpg"),
                self.jpg.read_bytes(),
            )
            self.assertEqual(
                archive.read("OEBPS/images/img_0002.png"),
                self.png.read_bytes(),
            )

    def test_plan_fingerprint_detects_directory_changes(self) -> None:
        plan = discover_book(self.input_root)
        self.assertTrue(book_plan_is_current(plan))

        current = os.stat(self.chapter).st_mtime_ns
        os.utime(
            self.chapter,
            ns=(current + 1_000_000, current + 1_000_000),
        )
        self.assertFalse(book_plan_is_current(plan))

    def test_cancel_removes_temporary_file(self) -> None:
        plan = discover_book(self.input_root)
        output = self.output_dir / "cancelled.epub"
        cancel = threading.Event()
        cancel.set()

        with self.assertRaises(Cancelled):
            build_epub(
                plan,
                BookMetadata(title="取消测试"),
                output,
                cancel_event=cancel,
            )

        self.assertFalse(output.exists())
        self.assertFalse(
            output.with_name(f"{output.name}.tmp").exists()
        )

    def test_discover_book_honours_cancel_event(self) -> None:
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            discover_book(self.input_root, cancel_event=cancel)

    def test_plan_invalidated_when_image_added(self) -> None:
        plan = discover_book(self.input_root)
        self.assertTrue(book_plan_is_current(plan))

        Image.new("RGB", (12, 12), (1, 2, 3)).save(
            self.chapter / "p3.jpg"
        )
        self.assertFalse(book_plan_is_current(plan))

    def test_parallel_normalize_matches_serial_output(self) -> None:
        second = self.input_root / "chapter2"
        second.mkdir()
        for index in range(1, 7):
            Image.new(
                "RGB",
                (24 + index, 16),
                (index * 20 % 255, 40, 60),
            ).save(second / f"p{index}.jpg")

        plan = discover_book(self.input_root)
        metadata = BookMetadata(title="并行一致性")
        serial = self.output_dir / "serial.epub"
        parallel = self.output_dir / "parallel.epub"
        large = self.output_dir / "large.epub"

        with mock.patch.object(
            epub_builder.uuid,
            "uuid4",
            return_value=uuid.UUID(int=0),
        ):
            with mock.patch.object(
                epub_builder, "MAX_IMAGE_WORKERS", 1
            ):
                serial_result = build_epub(plan, metadata, serial)
            with mock.patch.object(
                epub_builder, "MAX_IMAGE_WORKERS", 4
            ):
                parallel_result = build_epub(
                    plan, metadata, parallel
                )
            with mock.patch.object(
                epub_builder, "MAX_IMAGE_WORKERS", 4
            ):
                with mock.patch.object(
                    epub_builder, "LARGE_IMAGE_BYTES", 1
                ):
                    large_result = build_epub(
                        plan, metadata, large
                    )

        self.assertEqual(serial.read_bytes(), parallel.read_bytes())
        self.assertEqual(serial.read_bytes(), large.read_bytes())
        self.assertEqual(
            serial_result.skipped, parallel_result.skipped
        )
        self.assertEqual(
            serial_result.skipped, large_result.skipped
        )
        self.assertEqual(
            (serial_result.chapters, serial_result.images),
            (parallel_result.chapters, parallel_result.images),
        )
        self.assertEqual(
            (serial_result.chapters, serial_result.images),
            (large_result.chapters, large_result.images),
        )
        self.assertEqual(parallel_result.images, 9)

    def test_parallel_skipped_follows_input_order(self) -> None:
        second = self.input_root / "chapter2"
        second.mkdir()
        (second / "a_broken.png").write_bytes(b"not an image")
        Image.new("RGB", (16, 16), (9, 9, 9)).save(
            second / "b_good.jpg"
        )
        (second / "c_broken.jpg").write_bytes(b"also bad")

        plan = discover_book(self.input_root)
        result = build_epub(
            plan,
            BookMetadata(title="跳过顺序"),
            self.output_dir / "skipped.epub",
        )

        self.assertEqual(
            [path.name for path, _reason in result.skipped],
            ["a_broken.png", "c_broken.jpg"],
        )
        self.assertEqual(result.images, 4)
        self.assertEqual(result.chapters, 2)

    def test_parallel_cancel_removes_temporary_file(self) -> None:
        chapter = self.input_root / "chapter3"
        chapter.mkdir()
        for index in range(12):
            Image.new(
                "RGB", (32, 24), (index * 5, 10, 20)
            ).save(chapter / f"p{index:02d}.jpg")

        plan = discover_book(self.input_root)
        output = self.output_dir / "midway.epub"
        cancel = threading.Event()
        progress_events: list[tuple[int, int]] = []

        def progress(done: int, total: int) -> None:
            progress_events.append((done, total))
            if done >= 2:
                cancel.set()

        with mock.patch.object(
            epub_builder, "MAX_IMAGE_WORKERS", 2
        ):
            with self.assertRaises(Cancelled) as ctx:
                build_epub(
                    plan,
                    BookMetadata(title="中途取消"),
                    output,
                    cancel_event=cancel,
                    on_progress=progress,
                )

        self.assertIsInstance(ctx.exception.result, epub_builder.EpubResult)
        self.assertFalse(output.exists())
        self.assertFalse(
            output.with_name(f"{output.name}.tmp").exists()
        )
        self.assertGreaterEqual(progress_events[0][0], 1)


if __name__ == "__main__":
    unittest.main()
