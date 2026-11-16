"""PDF 工具（阶段 2A 串行优化）单元测试。"""

from __future__ import annotations

import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import pymupdf
from PIL import Image

from toolbox.errors import Cancelled
from toolbox.images import prepare_image_payload
from toolbox.parallel import run_chunked_pool
from toolbox.pdf_utils import PdfError, open_pdf
from toolbox import pdf_parallel as pdf_parallel_module
from toolbox.tools import images_to_pdf as images_to_pdf_module
from toolbox.tools.images_to_pdf import (
    PageOptions,
    PdfBuildResult,
    build_pdf,
)
from toolbox.tools.pdf_compress import CompressionOptions, compress_pdf
from toolbox.tools.pdf_to_images import (
    RenderResult,
    convert_pdf,
    convert_pdf_parallel,
)
from toolbox.tools.pdf_merge_split import (
    SplitChunk,
    _RANGE_TRANSLATION,
    merge_pdfs,
    parse_range_groups,
)
from toolbox.tools.pdf_page_tools import (
    PageNumberOptions,
    WatermarkOptions,
    process_pdf,
)


# --------------------------------------------------------------- 参考实现


def _legacy_parse_range_groups(
    text: str, total_pages: int
) -> list[SplitChunk]:
    """旧版 O(k²) 重叠检查，仅用于随机对照测试。"""
    if total_pages < 1:
        raise ValueError("PDF 页数必须大于 0")

    normalized = (text or "").translate(_RANGE_TRANSLATION)
    normalized = "".join(normalized.split())
    if not normalized:
        raise ValueError("拆分范围不能为空")

    chunks: list[SplitChunk] = []
    for token in normalized.split(","):
        if not token:
            continue
        pieces = token.split("-")
        if len(pieces) == 1:
            if not pieces[0].isdigit():
                raise ValueError(f"无法识别的页码范围片段：{token}")
            start = end = int(pieces[0])
        elif len(pieces) == 2:
            start_text, end_text = pieces
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"无法识别的页码范围片段：{token}")
            start, end = int(start_text), int(end_text)
        else:
            raise ValueError(f"无法识别的页码范围片段：{token}")

        if start > end:
            raise ValueError(f"页码范围起点大于终点：{token}")
        if start < 1 or end > total_pages:
            raise ValueError(f"页码超出范围（1-{total_pages}）：{token}")
        for previous in chunks:
            if start <= previous.end and end >= previous.start:
                raise ValueError(f"页码范围重叠：{token}")
        chunks.append(SplitChunk(start, end))

    if not chunks:
        raise ValueError("拆分范围不能为空")
    return chunks


_SEPARATORS = [",", "，", "、", "；", ";", ", ", " , "]
_DASHES = ["-", "～", "~", "—", "–", "－", "‐", "‑", "−", "至"]
_MALFORMED = ["abc", "1-2-3", "-3", "3-", "1..3", "1-2a"]


def _random_page(rng: random.Random, total_pages: int) -> int:
    if rng.random() < 0.15:
        return rng.randint(-2, total_pages + 2)
    return rng.randint(1, max(1, total_pages))


def _random_range_text(
    rng: random.Random, total_pages: int
) -> str:
    parts: list[str] = []
    for _ in range(rng.randint(0, 8)):
        kind = rng.random()
        if kind < 0.04:
            parts.append("")
        elif kind < 0.12:
            parts.append(rng.choice(_MALFORMED))
        elif kind < 0.4:
            parts.append(str(_random_page(rng, total_pages)))
        else:
            start, end = sorted(
                (
                    _random_page(rng, total_pages),
                    _random_page(rng, total_pages),
                )
            )
            if rng.random() < 0.1:
                start, end = end, start
            parts.append(f"{start}{rng.choice(_DASHES)}{end}")
    if not parts:
        parts.append(str(rng.randint(1, max(1, total_pages))))
    text = rng.choice(_SEPARATORS).join(parts)
    if rng.random() < 0.3:
        text = f" {text} "
    return text


def _outcome(func, text: str, total_pages: int) -> tuple[object, object]:
    try:
        chunks = func(text, total_pages)
    except ValueError as exc:
        return "error", str(exc)
    return "ok", [(chunk.start, chunk.end) for chunk in chunks]


class ParseRangeGroupsTest(unittest.TestCase):
    def test_matches_legacy_implementation_on_random_inputs(self) -> None:
        rng = random.Random(20261116)
        for _case in range(500):
            total_pages = rng.randint(1, 40)
            text = _random_range_text(rng, total_pages)
            expected = _outcome(
                _legacy_parse_range_groups, text, total_pages
            )
            actual = _outcome(parse_range_groups, text, total_pages)
            self.assertEqual(
                actual,
                expected,
                msg=f"输入={text!r}，总页数={total_pages}",
            )

    def test_overlap_and_fullwidth_forms(self) -> None:
        chunks = parse_range_groups("1-3, 5，8～10", 10)
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in chunks],
            [(1, 3), (5, 5), (8, 10)],
        )
        with self.assertRaisesRegex(ValueError, "页码范围重叠"):
            parse_range_groups("1-5,5-6", 10)
        with self.assertRaisesRegex(ValueError, "页码范围重叠"):
            parse_range_groups("5-6,1-5", 10)


class ImagesToPdfTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_jpeg_reuses_original_bytes(self) -> None:
        path = self.root / "photo.jpg"
        Image.new("RGB", (80, 40), (10, 20, 30)).save(path)
        output = self.root / "out.pdf"

        calls: list[tuple[object, dict[str, object]]] = []
        original = images_to_pdf_module.prepare_image_payload

        def spy(payload_path, **kwargs):
            payload = original(payload_path, **kwargs)
            calls.append((payload, kwargs))
            return payload

        with (
            mock.patch.object(
                images_to_pdf_module,
                "prepare_image_payload",
                side_effect=spy,
            ),
            mock.patch(
                "toolbox.images.load_oriented",
                side_effect=AssertionError("不应调用 load_oriented"),
            ),
        ):
            result = build_pdf([path], output, PageOptions())

        self.assertEqual(result.count, 1)
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(calls), 1)
        payload, kwargs = calls[0]
        self.assertTrue(payload.reusable_original)
        self.assertTrue(kwargs["prefer_original"])

        document = pymupdf.open(output)
        try:
            self.assertEqual(document.page_count, 1)
            self.assertEqual(
                len(document.load_page(0).get_images(full=True)), 1
            )
        finally:
            document.close()

    def test_transparent_png_builds_pdf(self) -> None:
        path = self.root / "alpha.png"
        Image.new("RGBA", (30, 20), (200, 0, 0, 0)).save(path)
        output = self.root / "out.pdf"

        self.assertTrue(prepare_image_payload(path).has_alpha)
        result = build_pdf([path], output, PageOptions())

        self.assertEqual(result.count, 1)
        self.assertEqual(result.skipped, [])
        document = pymupdf.open(output)
        try:
            self.assertEqual(document.page_count, 1)
            self.assertEqual(
                len(document.load_page(0).get_images(full=True)), 1
            )
        finally:
            document.close()


class PdfPageToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_source(self, pages: int = 2) -> Path:
        path = self.root / "source.pdf"
        document = pymupdf.open()
        try:
            for index in range(pages):
                page = document.new_page(width=300, height=400)
                page.insert_text(
                    (40, 60), f"SRC{index + 1}", fontsize=12
                )
            document.save(path)
        finally:
            document.close()
        return path

    def test_process_pdf_text_is_extractable(self) -> None:
        source = self._make_source(2)
        output = self.root / "annotated.pdf"
        page_options = PageNumberOptions(
            template="Page {page} of {total}",
            start_number=1,
            first_page=1,
            position="底居中",
            font_size=12.0,
            margin_mm=10.0,
        )
        watermark_options = WatermarkOptions(
            text="CONFIDENTIAL",
            font_size=18.0,
            opacity=0.3,
            angle=30.0,
            tiled=False,
        )

        result = process_pdf(
            source,
            output,
            page_options,
            None,
            0,
            watermark_options,
            font_path="",
        )

        self.assertEqual(result.pages, 2)
        self.assertEqual(result.page_numbers, 2)
        self.assertEqual(result.watermarks, 2)
        self.assertFalse(result.font_embedded)

        document = open_pdf(output)
        try:
            self.assertEqual(document.page_count, 2)
            for index in range(2):
                text = document.load_page(index).get_text()
                self.assertIn(f"Page {index + 1} of 2", text)
                self.assertIn("CONFIDENTIAL", text)
        finally:
            document.close()


class MergeSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_text_pdf(self, name: str, labels: list[str]) -> Path:
        path = self.root / name
        document = pymupdf.open()
        try:
            for label in labels:
                page = document.new_page(width=200, height=200)
                page.insert_text((20, 50), label, fontsize=12)
            document.save(path)
        finally:
            document.close()
        return path

    def test_merge_two_pass_preserves_order(self) -> None:
        first = self._make_text_pdf("a.pdf", ["A1", "A2"])
        second = self._make_text_pdf("b.pdf", ["B1"])
        third = self._make_text_pdf("c.pdf", ["C1", "C2"])
        output = self.root / "merged.pdf"
        progress: list[tuple[int, int]] = []

        result = merge_pdfs(
            [first, second, third],
            output,
            on_progress=lambda done, total: progress.append((done, total)),
        )

        self.assertEqual(result.files, 3)
        self.assertEqual(result.pages, 5)
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])

        document = open_pdf(output)
        try:
            texts = [
                document.load_page(index).get_text().strip()
                for index in range(document.page_count)
            ]
        finally:
            document.close()
        self.assertEqual(len(texts), 5)
        for label, text in zip(
            ["A1", "A2", "B1", "C1", "C2"], texts
        ):
            self.assertIn(label, text)

    def test_merge_collects_all_errors_without_output(self) -> None:
        good = self._make_text_pdf("good.pdf", ["OK"])
        missing = self.root / "missing.pdf"
        corrupt = self.root / "corrupt.pdf"
        corrupt.write_bytes(b"not a pdf")
        output = self.root / "merged.pdf"

        with self.assertRaises(PdfError) as ctx:
            merge_pdfs([good, missing, corrupt], output)

        message = str(ctx.exception)
        self.assertIn("missing.pdf", message)
        self.assertIn("corrupt.pdf", message)
        self.assertFalse(output.exists())


class PdfProcessPoolTest(unittest.TestCase):
    """PDF 进程池（spawn）路径与串行路径的一致性测试。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_pdf(
        self, name: str = "source.pdf", pages: int = 12, *, with_toc: bool = False
    ) -> Path:
        path = self.root / name
        document = pymupdf.open()
        try:
            for index in range(pages):
                width = 200 + index * 3
                height = 300 - index * 2
                page = document.new_page(width=width, height=height)
                page.draw_rect(
                    pymupdf.Rect(10, 10, width - 10, height - 10),
                    color=(0.2, 0.4, 0.6),
                    width=1,
                )
                page.draw_circle(
                    pymupdf.Point(width / 2, height / 2),
                    20,
                    color=(1, 0, 0),
                    fill=(1, 1, 0),
                )
                page.insert_text(
                    (20, 40), f"PAGE {index + 1}", fontsize=11
                )
            if with_toc:
                document.set_toc(
                    [[1, "First", 1], [1, "Second", 5], [2, "Sub", 7]]
                )
            document.save(path)
        finally:
            document.close()
        return path

    def test_parallel_png_matches_serial_pixel_for_pixel(self) -> None:
        source = self._make_pdf()
        pages = list(range(12))
        serial_dir = self.root / "serial"
        parallel_dir = self.root / "parallel"

        serial = convert_pdf(source, serial_dir, "png", 96, pages)
        progress: list[tuple[int, int]] = []
        with (
            mock.patch(
                "toolbox.tools.pdf_to_images.os.cpu_count",
                return_value=4,
            ),
            mock.patch(
                "toolbox.tools.pdf_to_images.MIN_PARALLEL_BYTES",
                0,
            ),
            mock.patch(
                "toolbox.tools.pdf_to_images.run_chunked_pool",
                wraps=run_chunked_pool,
            ) as pool_spy,
        ):
            parallel = convert_pdf_parallel(
                source,
                parallel_dir,
                "png",
                96,
                pages,
                on_progress=lambda done, total: progress.append(
                    (done, total)
                ),
                max_workers=2,
            )

        self.assertGreaterEqual(pool_spy.call_count, 1)
        self.assertEqual(pool_spy.call_args.kwargs["max_workers"], 2)
        self.assertEqual(serial, (serial_dir, 12))
        self.assertEqual(parallel, (parallel_dir, 12))

        serial_names = sorted(
            path.name for path in serial_dir.glob("*.png")
        )
        parallel_names = sorted(
            path.name for path in parallel_dir.glob("*.png")
        )
        self.assertEqual(serial_names, parallel_names)
        self.assertEqual(len(parallel_names), 12)

        for name in parallel_names:
            with (
                Image.open(serial_dir / name) as expected,
                Image.open(parallel_dir / name) as actual,
            ):
                self.assertEqual(actual.size, expected.size, msg=name)
                self.assertEqual(
                    actual.convert("RGB").tobytes(),
                    expected.convert("RGB").tobytes(),
                    msg=name,
                )
        self.assertEqual(progress[-1], (12, 12))
        self.assertEqual(list(parallel_dir.glob("*.tmp")), [])

    def test_parallel_jpeg_matches_serial_names_and_sizes(self) -> None:
        source = self._make_pdf("jpeg.pdf")
        pages = list(range(12))
        serial_dir = self.root / "jpeg_serial"
        parallel_dir = self.root / "jpeg_parallel"

        convert_pdf(source, serial_dir, "jpeg", 72, pages)
        with mock.patch(
            "toolbox.tools.pdf_to_images.os.cpu_count", return_value=4
        ):
            convert_pdf_parallel(
                source,
                parallel_dir,
                "jpeg",
                72,
                pages,
                max_workers=2,
            )

        serial_names = sorted(
            path.name for path in serial_dir.glob("*.jpg")
        )
        parallel_names = sorted(
            path.name for path in parallel_dir.glob("*.jpg")
        )
        self.assertEqual(serial_names, parallel_names)
        self.assertEqual(len(parallel_names), 12)
        for name in parallel_names:
            with (
                Image.open(serial_dir / name) as expected,
                Image.open(parallel_dir / name) as actual,
            ):
                self.assertEqual(actual.size, expected.size, msg=name)

    def test_cancel_keeps_committed_prefix_without_tmp_files(self) -> None:
        source = self._make_pdf()
        out_dir = self.root / "cancelled_serial"
        cancel_event = threading.Event()

        def on_progress(done: int, _total: int) -> None:
            if done >= 3:
                cancel_event.set()

        with self.assertRaises(Cancelled) as ctx:
            convert_pdf_parallel(
                source,
                out_dir,
                "png",
                72,
                list(range(12)),
                cancel_event=cancel_event,
                on_progress=on_progress,
                max_workers=1,
            )

        result = ctx.exception.result
        self.assertIsInstance(result, RenderResult)
        self.assertEqual(result.done, 3)
        self.assertEqual(result.out_dir, out_dir)
        generated = list(out_dir.glob("*.png"))
        self.assertEqual(len(generated), result.done)
        self.assertEqual(
            {path.name for path in generated},
            {f"source_page_{index:03d}.png" for index in (1, 2, 3)},
        )
        self.assertEqual(list(out_dir.glob("*.tmp")), [])

    def test_parallel_cancel_leaves_no_temp_files(self) -> None:
        source = self._make_pdf()
        out_dir = self.root / "cancelled_parallel"
        cancel_event = threading.Event()
        cancel_event.set()

        with (
            mock.patch(
                "toolbox.tools.pdf_to_images.os.cpu_count",
                return_value=4,
            ),
            self.assertRaises(Cancelled) as ctx,
        ):
            convert_pdf_parallel(
                source,
                out_dir,
                "png",
                72,
                list(range(12)),
                cancel_event=cancel_event,
                max_workers=2,
            )

        result = ctx.exception.result
        self.assertIsInstance(result, RenderResult)
        self.assertEqual(result.done, 0)
        self.assertEqual(list(out_dir.glob("*.png")), [])
        self.assertEqual(list(out_dir.glob("*.tmp")), [])

    def test_lossy_compress_parallel_matches_serial(self) -> None:
        source = self._make_pdf("lossy.pdf", with_toc=True)
        options = CompressionOptions(mode="lossy", dpi=72, quality=60)
        serial_out = self.root / "lossy_serial.pdf"
        parallel_out = self.root / "lossy_parallel.pdf"

        with mock.patch(
            "toolbox.tools.pdf_compress.os.cpu_count", return_value=1
        ):
            serial_result = compress_pdf(source, serial_out, options)

        with (
            mock.patch(
                "toolbox.tools.pdf_compress.os.cpu_count",
                return_value=4,
            ),
            mock.patch(
                "toolbox.tools.pdf_compress.run_chunked_pool",
                wraps=run_chunked_pool,
            ) as pool_spy,
        ):
            parallel_result = compress_pdf(source, parallel_out, options)

        self.assertGreaterEqual(pool_spy.call_count, 1)
        self.assertEqual(serial_result.pages, 12)
        self.assertEqual(parallel_result.pages, 12)
        self.assertTrue(serial_out.is_file())
        self.assertTrue(parallel_out.is_file())

        source_document = open_pdf(source)
        serial_document = open_pdf(serial_out)
        parallel_document = open_pdf(parallel_out)
        try:
            self.assertEqual(serial_document.page_count, 12)
            self.assertEqual(parallel_document.page_count, 12)
            self.assertEqual(
                serial_document.get_toc(), parallel_document.get_toc()
            )
            self.assertEqual(
                parallel_document.get_toc(), source_document.get_toc()
            )
            for index in range(12):
                serial_rect = serial_document.load_page(index).rect
                parallel_rect = parallel_document.load_page(index).rect
                self.assertAlmostEqual(
                    parallel_rect.width, serial_rect.width, places=3
                )
                self.assertAlmostEqual(
                    parallel_rect.height, serial_rect.height, places=3
                )
                self.assertGreater(
                    parallel_rect.width * parallel_rect.height, 0
                )
        finally:
            source_document.close()
            serial_document.close()
            parallel_document.close()

    def test_pdf_parallel_module_is_gui_free(self) -> None:
        source = Path(pdf_parallel_module.__file__).read_text(
            encoding="utf-8"
        )
        # 只检查真正的 import 语句，避免文档字符串中的说明造成误报。
        self.assertNotIn("\nimport PyQt", source)
        self.assertNotIn("\nfrom PyQt", source)
        self.assertNotIn("\nimport toolbox.workers", source)
        self.assertNotIn("\nfrom toolbox.workers", source)


class ImagesToPdfParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_images(self) -> list[Path]:
        specs = [
            ("01.png", (20, 30), (255, 0, 0)),
            ("02.png", (40, 10), (0, 255, 0)),
            ("04.png", (15, 25), (0, 0, 255)),
            ("05.png", (50, 20), (9, 9, 9)),
        ]
        paths: list[Path] = []
        for name, size, color in specs:
            path = self.root / name
            Image.new("RGB", size, color).save(path)
            paths.append(path)
        bad = self.root / "03_bad.png"
        bad.write_bytes(b"not an image")
        paths.insert(2, bad)
        return paths

    def _build(
        self,
        workers: int,
        output: Path,
        paths: list[Path],
        *,
        cancel_event: threading.Event | None = None,
        on_progress=None,
    ) -> PdfBuildResult:
        options = PageOptions(
            mode="original", margin_mm=1.0, dpi=96
        )
        with mock.patch.object(
            images_to_pdf_module, "MAX_IMAGE_WORKERS", workers
        ):
            return build_pdf(
                paths,
                output,
                options,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

    def _fingerprint(self, path: Path) -> list[tuple[object, ...]]:
        document = pymupdf.open(path)
        try:
            pages: list[tuple[object, ...]] = []
            for page in document:
                references = page.get_images(full=True)
                streams = tuple(
                    document.extract_image(reference[0])["image"]
                    for reference in references
                )
                pages.append(
                    (
                        round(page.rect.width, 2),
                        round(page.rect.height, 2),
                        streams,
                    )
                )
            return pages
        finally:
            document.close()

    def test_parallel_matches_serial_page_order_and_sizes(self) -> None:
        paths = self._make_images()
        serial_output = self.root / "serial.pdf"
        parallel_output = self.root / "parallel.pdf"
        serial_progress: list[tuple[int, int]] = []
        parallel_progress: list[tuple[int, int]] = []

        serial = self._build(
            1,
            serial_output,
            paths,
            on_progress=lambda done, total: serial_progress.append(
                (done, total)
            ),
        )
        parallel = self._build(
            4,
            parallel_output,
            paths,
            on_progress=lambda done, total: parallel_progress.append(
                (done, total)
            ),
        )

        self.assertEqual(serial.count, parallel.count)
        self.assertEqual(
            [path for path, _reason in serial.skipped],
            [path for path, _reason in parallel.skipped],
        )
        self.assertEqual(serial_progress, parallel_progress)
        self.assertEqual(
            serial_progress,
            [(index, len(paths)) for index in range(1, len(paths) + 1)],
        )
        self.assertEqual(len(parallel.skipped), 1)
        self.assertIn("03_bad.png", parallel.skipped[0][0].name)
        self.assertIn("identify image", parallel.skipped[0][1])

        serial_pages = self._fingerprint(serial_output)
        parallel_pages = self._fingerprint(parallel_output)
        self.assertEqual(parallel_pages, serial_pages)
        self.assertEqual(len(parallel_pages), parallel.count)

        # 原图模式页面宽度由输入图片宽度决定，可验证页顺序未被并行打乱。
        expected_widths = [
            20 * 72 / 96 + 2 * 72 / 25.4,
            40 * 72 / 96 + 2 * 72 / 25.4,
            15 * 72 / 96 + 2 * 72 / 25.4,
            50 * 72 / 96 + 2 * 72 / 25.4,
        ]
        for page, width in zip(parallel_pages, expected_widths):
            self.assertAlmostEqual(page[0], width, places=2)

    def test_original_insert_failure_retries_reencoded_payload(self) -> None:
        source = self.root / "photo.jpg"
        Image.new("RGB", (80, 40), (10, 20, 30)).save(source)
        output = self.root / "retry.pdf"

        prepared: list[bool] = []
        payloads: list[object] = []
        real_prepare = images_to_pdf_module._prepare_payload
        real_insert = images_to_pdf_module._insert_payload

        def spy_prepare(path: Path, prefer_original: bool):
            prepared.append(prefer_original)
            return real_prepare(path, prefer_original)

        def flaky_insert(document, layout, payload) -> None:
            payloads.append(payload)
            if len(payloads) == 1:
                raise RuntimeError("模拟原字节插入失败")
            real_insert(document, layout, payload)

        with (
            mock.patch.object(
                images_to_pdf_module,
                "_prepare_payload",
                side_effect=spy_prepare,
            ),
            mock.patch.object(
                images_to_pdf_module,
                "_insert_payload",
                side_effect=flaky_insert,
            ),
            mock.patch.object(
                images_to_pdf_module, "MAX_IMAGE_WORKERS", 4
            ),
        ):
            result = build_pdf([source], output, PageOptions())

        self.assertEqual(prepared, [True, False])
        self.assertTrue(payloads[0].reusable_original)
        self.assertFalse(payloads[1].reusable_original)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.skipped, [])

        document = pymupdf.open(output)
        try:
            self.assertEqual(document.page_count, 1)
        finally:
            document.close()

    def test_cancel_leaves_no_output_pdf(self) -> None:
        paths = self._make_images()
        output = self.root / "cancelled.pdf"
        cancel_event = threading.Event()
        progress: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress.append((done, total))
            if done == 2:
                cancel_event.set()

        with self.assertRaises(Cancelled) as ctx:
            self._build(
                4,
                output,
                paths,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        partial = ctx.exception.result
        self.assertIsInstance(partial, PdfBuildResult)
        self.assertEqual(partial.count, 2)
        self.assertEqual(partial.skipped, [])
        self.assertEqual(progress, [(1, len(paths)), (2, len(paths))])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
