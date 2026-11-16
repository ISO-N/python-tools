"""共享层（图片 / 文件 / PDF / 排序）单元测试。"""

from __future__ import annotations

import io
import queue
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image

import pymupdf

from toolbox.common import UniquePathAllocator, natural_sort_key
from toolbox.fileops import (
    MoveExecutionResult,
    execute_moves,
    find_conflicts,
    prune_empty_dirs,
)
from toolbox.images import (
    flatten_rgb,
    load_oriented,
    prepare_image_payload,
    probe_image,
    scan_image_files,
)
from toolbox.pdf_utils import (
    SAVE_PRESETS,
    TextStamp,
    add_text_stamps,
    format_page_number,
    load_pdf_font_resource,
    open_pdf,
    parse_page_number_template,
    save_pdf_atomic,
)


class ImageHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _jpeg(self, name: str = "a.jpg") -> Path:
        path = self.root / name
        Image.new("RGB", (80, 40), (200, 30, 40)).save(path)
        return path

    def _rgba_png(self, name: str = "a.png") -> Path:
        path = self.root / name
        Image.new("RGBA", (20, 30), (1, 2, 3, 128)).save(path)
        return path

    def test_probe_and_payload_reuse_original(self) -> None:
        jpeg = self._jpeg()
        probe = probe_image(jpeg)
        self.assertEqual((probe.width, probe.height), (80, 40))
        self.assertEqual(probe.format, "JPEG")
        payload = prepare_image_payload(jpeg)
        self.assertTrue(payload.reusable_original)
        self.assertEqual(payload.extension, "jpg")
        self.assertEqual(payload.data, jpeg.read_bytes())

        png = self._rgba_png()
        payload = prepare_image_payload(png)
        self.assertTrue(payload.reusable_original)
        self.assertTrue(payload.has_alpha)
        self.assertEqual(payload.extension, "png")
        self.assertEqual(payload.data, png.read_bytes())

    def test_orientation_is_applied(self) -> None:
        path = self.root / "rotated.jpg"
        image = Image.new("RGB", (80, 40), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6
        image.save(path, exif=exif)
        probe = probe_image(path)
        self.assertEqual(probe.orientation, 6)
        self.assertEqual((probe.width, probe.height), (40, 80))
        payload = prepare_image_payload(path)
        self.assertFalse(payload.reusable_original)
        self.assertEqual((payload.width, payload.height), (40, 80))
        loaded = load_oriented(path)
        try:
            self.assertEqual(loaded.size, (40, 80))
        finally:
            loaded.close()

    def test_flatten_rgb_preserves_rgb_object(self) -> None:
        image = Image.new("RGB", (4, 4), (1, 2, 3))
        self.assertIs(flatten_rgb(image), image)
        rgba = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
        flattened = flatten_rgb(rgba)
        self.assertEqual(flattened.mode, "RGB")
        self.assertEqual(flattened.getpixel((0, 0)), (255, 255, 255))

    def test_scan_images_sort_hidden_and_exclude(self) -> None:
        (self.root / "sub").mkdir()
        self._jpeg("img10.jpg")
        self._jpeg("img2.jpg")
        self._jpeg(".hidden.jpg")
        self._jpeg("sub/img1.jpg")
        excluded = self.root / "sub"
        files = scan_image_files(self.root, recursive=True)
        self.assertEqual([p.name for p in files], ["img2.jpg", "img10.jpg", "img1.jpg"])
        files = scan_image_files(
            self.root, recursive=True, exclude_roots=(excluded,)
        )
        self.assertEqual([p.name for p in files], ["img2.jpg", "img10.jpg"])


class FileOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_execute_moves_and_compat_unpack(self) -> None:
        src = self.root / "src.txt"
        src.write_text("hello", encoding="utf-8")
        dst = self.root / "out" / "src.txt"
        progress: list[tuple[int, int]] = []
        result = execute_moves(
            [(src, dst)], False, on_progress=lambda d, t: progress.append((d, t))
        )
        self.assertEqual((result.moved, result.skipped), (1, 0))
        self.assertTrue(dst.is_file())
        self.assertEqual(progress[-1], (1, 1))
        moved, skipped, errors, moved_items = MoveExecutionResult(1, 2, [], [])
        self.assertEqual((moved, skipped, errors, moved_items), (1, 2, [], []))

    def test_known_conflicts_and_overwrite(self) -> None:
        src = self.root / "src" / "a.txt"
        src.parent.mkdir()
        src.write_text("new", encoding="utf-8")
        dst = self.root / "dst" / "a.txt"
        dst.parent.mkdir()
        dst.write_text("old", encoding="utf-8")
        self.assertEqual(find_conflicts([(src, dst)]), [(src, dst)])
        result = execute_moves(
            [(src, dst)], False, known_conflicts={dst}
        )
        self.assertEqual((result.moved, result.skipped), (0, 1))

        result = execute_moves([(src, dst)], True)
        self.assertEqual(result.moved, 1)
        self.assertEqual(dst.read_text(encoding="utf-8"), "new")

    def test_cancel_carries_partial_result(self) -> None:
        items = []
        for index in range(5):
            src = self.root / f"f{index}.txt"
            src.write_text("x", encoding="utf-8")
            items.append((src, self.root / "out" / src.name))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Exception) as ctx:
            execute_moves(items, False, cancel_event=cancel)
        partial = getattr(ctx.exception, "result", None)
        self.assertIsInstance(partial, MoveExecutionResult)
        self.assertEqual(partial.moved, 0)

    def test_prune_empty_dirs(self) -> None:
        (self.root / "a" / "b").mkdir(parents=True)
        keep = self.root / "keep"
        keep.mkdir()
        (keep / "file.txt").write_text("x", encoding="utf-8")
        removed = prune_empty_dirs(self.root, exclude=(keep,))
        self.assertIn(self.root / "a" / "b", removed)
        self.assertIn(self.root / "a", removed)
        self.assertTrue(keep.is_dir())
        self.assertTrue(self.root.is_dir())

    def test_unique_path_allocator_modes(self) -> None:
        target = self.root / "x.jpg"
        target.write_text("x", encoding="utf-8")
        allocator = UniquePathAllocator(reserve_planned=True)
        first = allocator.reserve(target)
        second = allocator.reserve(target)
        self.assertEqual(first.name, "x (1).jpg")
        self.assertEqual(second.name, "x (2).jpg")

        legacy = UniquePathAllocator(reserve_planned=False)
        a = legacy.reserve(target)
        b = legacy.reserve(target)
        self.assertEqual(a, b)
        self.assertEqual(a.name, "x (1).jpg")

    def test_natural_sort_cache(self) -> None:
        self.assertEqual(natural_sort_key("a2"), natural_sort_key("a2"))
        self.assertLess(natural_sort_key("a2"), natural_sort_key("a10"))


class PdfHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_text_stamps_and_presets(self) -> None:
        document = pymupdf.open()
        page = document.new_page(width=300, height=200)
        resource = load_pdf_font_resource(None)
        font = resource.font
        stamps = [
            TextStamp((150, 100), "Hello 123", 18),
            TextStamp((80, 50), "Top", 12, angle=30, fill_opacity=0.5),
        ]
        add_text_stamps(
            page, stamps, fontname=resource.fontname, font=font
        )
        text = page.get_text()
        self.assertIn("Hello 123", text)
        self.assertIn("Top", text)
        out = self.root / "out.pdf"
        save_pdf_atomic(document, out, preset="fast")
        document.close()
        reopened = open_pdf(out)
        try:
            self.assertEqual(reopened.page_count, 1)
            self.assertIn("Hello 123", reopened[0].get_text())
        finally:
            reopened.close()
        self.assertEqual(SAVE_PRESETS["fast"].clean, False)

    def test_page_number_template_errors(self) -> None:
        self.assertEqual(format_page_number("{page:03d}", 3, 10), "003")
        with self.assertRaises(ValueError):
            format_page_number("", 1, 1)
        with self.assertRaises(ValueError):
            format_page_number("{other}", 1, 1)
        self.assertIs(
            parse_page_number_template("{page}/{total}"),
            parse_page_number_template("{page}/{total}"),
        )


class BatchWorkerTest(unittest.TestCase):
    def test_progress_gate(self) -> None:
        from toolbox.workers import _ProgressGate

        values: list[tuple[object, object]] = []
        gate = _ProgressGate(lambda d, t: values.append((d, t)), 1000)
        gate.update(1, 10)
        gate.update(2, 10)
        gate.force_last()
        self.assertEqual(values[-1], (1, 10))

    def test_parallel_worker_order(self) -> None:
        from PyQt6.QtCore import QCoreApplication

        app = QCoreApplication.instance() or QCoreApplication([])
        from toolbox.workers import BatchWorker

        items = list(range(8))
        results: queue.Queue[object] = queue.Queue()

        def process(item: int, cancel: threading.Event) -> int:
            if item % 2:
                raise ValueError(f"bad {item}")
            return item

        worker = BatchWorker(
            items, process, max_workers=4, progress_interval_ms=0
        )
        worker.batch_finished.connect(results.put)
        worker.run()
        result = results.get(timeout=5)
        self.assertEqual(result.values, [0, 2, 4, 6])
        self.assertEqual([label for label, _ in result.errors], ["1", "3", "5", "7"])
        self.assertEqual(result.done, 4)
        self.assertEqual(app.thread().isRunning(), True)

    def test_parallel_cancel(self) -> None:
        from toolbox.workers import BatchWorker

        results: queue.Queue[object] = queue.Queue()
        cancel_started = threading.Event()

        def process(item: int, cancel: threading.Event) -> int:
            if item == 0:
                cancel_started.set()
                cancel.set()
            return item

        worker = BatchWorker(
            list(range(20)), process, max_workers=2
        )
        worker.batch_cancelled.connect(results.put)
        worker.batch_finished.connect(results.put)
        worker.run()
        result = results.get(timeout=5)
        self.assertTrue(result.cancelled)
        self.assertLess(result.done, 20)


if __name__ == "__main__":
    unittest.main()
