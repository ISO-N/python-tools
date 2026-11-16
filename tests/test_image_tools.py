"""图片工具页（水印 / 拼接 / 批量转换）的单元测试。"""

from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest import mock

from PIL import Image

from toolbox.common import find_cjk_font_file
from toolbox.images import probe_image
from toolbox.tools import batch_image_convert, image_collage
from toolbox.tools.batch_image_convert import (
    ConvertJob,
    ConvertOptions,
    convert_image,
)
from toolbox.tools.image_collage import (
    CollageOptions,
    CollageResult,
    _paste_fitted,
    build_collage,
)
from toolbox.tools.image_watermark import (
    WatermarkOptions,
    WatermarkRenderer,
    apply_watermark,
    apply_watermark_rgb,
    load_watermark_font,
    render_text_layer,
    watermark_one,
)
from toolbox.workers import BatchWorker, Cancelled


def _max_channel_diff(left: Image.Image, right: Image.Image) -> int:
    return max(
        abs(first - second)
        for first, second in zip(left.tobytes(), right.tobytes())
    )


class WatermarkToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        font_path = find_cjk_font_file()
        if font_path is None:
            self.skipTest("未找到可用于测试的字体文件")
        self.font_path = str(font_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _options(self, **overrides: object) -> WatermarkOptions:
        values: dict[str, object] = {
            "text": "测试水印",
            "position": "左上",
            "font_path": self.font_path,
            "font_size": 18,
            "color": (255, 0, 0),
            "opacity": 0.6,
            "angle": 0,
            "margin": 4,
            "gap": 10,
        }
        values.update(overrides)
        return WatermarkOptions(**values)  # type: ignore[arg-type]

    def test_jpeg_output_is_rgb(self) -> None:
        src = self.root / "in.jpg"
        Image.new("RGB", (64, 48), (12, 34, 56)).save(src, quality=95)
        original_bytes = src.read_bytes()
        options = self._options(position="右下")
        dst = self.root / "out.jpg"

        renderer = WatermarkRenderer(options)
        try:
            result = watermark_one(src, dst, options, renderer=renderer)
        finally:
            renderer.close()

        self.assertEqual((result.width, result.height), (64, 48))
        self.assertEqual(src.read_bytes(), original_bytes)
        with Image.open(dst) as output:
            self.assertEqual(output.format, "JPEG")
            self.assertEqual(output.mode, "RGB")
            self.assertEqual(output.size, (64, 48))

    def test_png_output_keeps_rgba(self) -> None:
        src = self.root / "in.png"
        Image.new("RGBA", (40, 30), (10, 20, 30, 128)).save(src)
        options = self._options(position="平铺", angle=15)
        dst = self.root / "out.png"

        result = watermark_one(src, dst, options)

        self.assertEqual((result.width, result.height), (40, 30))
        with Image.open(dst) as output:
            self.assertEqual(output.format, "PNG")
            self.assertEqual(output.mode, "RGBA")
            self.assertEqual(output.size, (40, 30))

    def test_rgb_fast_path_matches_alpha_composite(self) -> None:
        canvas = Image.new("RGB", (160, 120), (30, 60, 90))
        cases = (
            ("single", self._options(angle=0)),
            ("tile", self._options(position="平铺", angle=15)),
        )
        for name, options in cases:
            with self.subTest(name=name):
                font = load_watermark_font(options)
                layer = render_text_layer(
                    options.text,
                    font,
                    options.color,
                    options.angle,
                    opacity=options.opacity,
                )
                expected_rgba = apply_watermark(canvas, options, font)
                actual = apply_watermark_rgb(canvas.copy(), options, layer)
                try:
                    self.assertEqual(actual.mode, "RGB")
                    self.assertEqual(actual.size, canvas.size)
                    expected = expected_rgba.convert("RGB")
                    try:
                        self.assertLessEqual(
                            _max_channel_diff(actual, expected), 1
                        )
                    finally:
                        expected.close()
                    # 旧路径输出仍是 RGBA，且背景保持不透明。
                    self.assertEqual(expected_rgba.mode, "RGBA")
                    alpha = expected_rgba.getchannel("A")
                    try:
                        self.assertEqual(alpha.getextrema(), (255, 255))
                    finally:
                        alpha.close()
                finally:
                    expected_rgba.close()
                    actual.close()
                    layer.close()

    def test_renderer_reuses_layer_per_options(self) -> None:
        options = self._options()
        renderer = WatermarkRenderer(options)
        try:
            first = renderer.layer_for(options)
            second = renderer.layer_for(options)
            self.assertIs(first, second)
            other = renderer.layer_for(replace(options, opacity=0.3))
            self.assertIsNot(first, other)
        finally:
            renderer.close()
            renderer.close()

    def test_renderer_uses_thread_local_font_and_layer(self) -> None:
        options = self._options()
        renderer = WatermarkRenderer(options)
        barrier = threading.Barrier(2)

        def work() -> tuple[int, int, int, int]:
            barrier.wait(timeout=5)
            first_font = renderer.font_for(options)
            second_font = renderer.font_for(options)
            first_layer = renderer.layer_for(options)
            second_layer = renderer.layer_for(options)
            return (
                id(first_font),
                id(second_font),
                id(first_layer),
                id(second_layer),
            )

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first, second = [
                    future.result()
                    for future in (
                        executor.submit(work),
                        executor.submit(work),
                    )
                ]
        finally:
            renderer.close()

        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[3])
        self.assertEqual(second[0], second[1])
        self.assertEqual(second[2], second[3])
        # 每个线程持有独立字体与图层，避免跨线程共享可变 PIL 对象。
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[2], second[2])

    def test_parallel_watermark_matches_serial_output(self) -> None:
        sources = [
            ("a.jpg", "RGB"),
            ("b.png", "RGBA"),
            ("c.png", "RGB"),
            ("d.jpg", "RGB"),
        ]
        entries: list[tuple[Path, Path, Path]] = []
        for name, mode in sources:
            source = self.root / name
            color = (10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30)
            Image.new(mode, (60, 40), color).save(source)
            entries.append(
                (
                    source,
                    self.root / "serial" / name,
                    self.root / "parallel" / name,
                )
            )
        options = self._options(position="平铺", angle=15)

        serial_renderer = WatermarkRenderer(options)
        try:
            for source, serial_output, _parallel_output in entries:
                watermark_one(
                    source, serial_output, options, renderer=serial_renderer
                )
        finally:
            serial_renderer.close()

        parallel_renderer = WatermarkRenderer(options)
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(
                    executor.map(
                        lambda entry: watermark_one(
                            entry[0],
                            entry[2],
                            options,
                            renderer=parallel_renderer,
                        ),
                        entries,
                    )
                )
        finally:
            parallel_renderer.close()

        for _source, serial_output, parallel_output in entries:
            with (
                Image.open(serial_output) as serial_image,
                Image.open(parallel_output) as parallel_image,
            ):
                self.assertEqual(
                    serial_image.size, parallel_image.size
                )
                self.assertEqual(
                    serial_image.tobytes(), parallel_image.tobytes()
                )


class CollagePasteTest(unittest.TestCase):
    def test_paste_fitted_keeps_rgb_source_unchanged(self) -> None:
        source = Image.new("RGB", (20, 10), (11, 22, 33))
        source.putpixel((0, 0), (1, 2, 3))
        before = source.tobytes()

        for box, canvas_size in (
            ((0, 0, 6, 3), (6, 3)),
            ((2, 1, 20, 10), (24, 12)),
        ):
            with self.subTest(box=box):
                canvas = Image.new("RGB", canvas_size, (255, 255, 255))
                try:
                    _paste_fitted(canvas, source, box)
                finally:
                    canvas.close()
                self.assertEqual(source.size, (20, 10))
                self.assertEqual(source.mode, "RGB")
                self.assertEqual(source.tobytes(), before)

    def test_paste_fitted_keeps_transparent_source_unchanged(self) -> None:
        source = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        source.putpixel((0, 0), (1, 2, 3, 4))
        before = source.tobytes()

        for box, canvas_size in (
            ((0, 0, 8, 8), (8, 8)),
            ((0, 0, 4, 4), (4, 4)),
        ):
            with self.subTest(box=box):
                canvas = Image.new("RGB", canvas_size, (0, 0, 255))
                try:
                    _paste_fitted(canvas, source, box)
                    center = canvas.getpixel(
                        (canvas.width // 2, canvas.height // 2)
                    )
                    self.assertEqual(center[0], 128)
                finally:
                    canvas.close()
                self.assertEqual(source.size, (8, 8))
                self.assertEqual(source.mode, "RGBA")
                self.assertEqual(source.tobytes(), before)


class BatchConvertTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _source(self) -> Path:
        src = self.root / "big.jpg"
        Image.new("RGB", (400, 300), (10, 20, 30)).save(src, quality=95)
        return src

    def _save_spy(self, recorded: dict[str, object]):
        real = batch_image_convert.save_image_atomic

        def spy(image, target, *, image_format, **kwargs):
            recorded["image_format"] = image_format
            recorded.update(kwargs)
            return real(image, target, image_format=image_format, **kwargs)

        return spy

    def test_shrink_jpeg_dimensions_and_optimize(self) -> None:
        src = self._source()
        dst = self.root / "small.jpg"
        options = ConvertOptions(
            target_format="jpeg",
            quality=85,
            resize_mode="long_edge",
            long_edge=200,
        )
        recorded: dict[str, object] = {}

        with mock.patch.object(
            batch_image_convert,
            "save_image_atomic",
            side_effect=self._save_spy(recorded),
        ):
            result = convert_image(src, dst, options)

        self.assertEqual((result.width, result.height), (200, 150))
        self.assertEqual(recorded.get("image_format"), "JPEG")
        self.assertEqual(recorded.get("quality"), 85)
        self.assertIs(recorded.get("optimize"), True)
        probe = probe_image(dst)
        self.assertEqual((probe.width, probe.height), (200, 150))
        self.assertEqual(probe.format, "JPEG")

    def test_optimize_can_be_disabled(self) -> None:
        src = self._source()
        dst = self.root / "plain.png"
        options = ConvertOptions(
            target_format="png",
            resize_mode="percent",
            percent=50,
            optimize=False,
        )
        recorded: dict[str, object] = {}

        with mock.patch.object(
            batch_image_convert,
            "save_image_atomic",
            side_effect=self._save_spy(recorded),
        ):
            result = convert_image(src, dst, options)

        self.assertEqual((result.width, result.height), (200, 150))
        self.assertEqual(recorded.get("image_format"), "PNG")
        self.assertIs(recorded.get("optimize"), False)
        self.assertTrue(dst.is_file())
        probe = probe_image(dst)
        self.assertEqual((probe.width, probe.height), (200, 150))


class BatchConvertParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _sources(self) -> list[Path]:
        specs: list[tuple[str, str, tuple[int, int], object]] = [
            ("a.png", "RGBA", (31, 23), (10, 20, 30, 120)),
            ("b.jpg", "RGB", (48, 32), (200, 10, 10)),
            ("c.png", "RGB", (25, 40), (0, 100, 200)),
            ("d.png", "L", (64, 16), 128),
        ]
        sources: list[Path] = []
        folder = self.root / "input"
        folder.mkdir()
        for name, mode, size, color in specs:
            path = folder / name
            Image.new(mode, size, color).save(path)
            sources.append(path)
        return sources

    def _jobs(
        self, output_root: Path, sources: list[Path]
    ) -> list[ConvertJob]:
        jobs = [
            ConvertJob(
                src=source,
                dst=output_root / f"{index:02d}_{source.name}",
            )
            for index, source in enumerate(sources)
        ]
        jobs.append(
            ConvertJob(
                src=self.root / "input" / "missing.png",
                dst=output_root / "missing.png",
            )
        )
        return jobs

    def _run(
        self, jobs: list[ConvertJob], workers: int
    ) -> tuple[object, list[tuple[object, object]]]:
        options = ConvertOptions(
            target_format="png",
            resize_mode="long_edge",
            long_edge=20,
            optimize=True,
        )
        results: queue.Queue[object] = queue.Queue()
        progress: list[tuple[object, object]] = []

        def process(job: ConvertJob, cancel_event: threading.Event):
            return convert_image(job.src, job.dst, options, cancel_event)

        worker = BatchWorker(
            jobs,
            process,
            label_fn=lambda job: job.src.name,
            max_workers=workers,
            progress_interval_ms=100,
        )
        worker.batch_finished.connect(results.put)
        worker.batch_cancelled.connect(results.put)
        worker.progress_changed.connect(
            lambda done, total: progress.append((done, total))
        )
        worker.run()
        result = results.get(timeout=10)
        self.assertFalse(result.cancelled)
        return result, progress

    def test_parallel_matches_serial_order_and_pixels(self) -> None:
        sources = self._sources()
        serial_jobs = self._jobs(self.root / "serial", sources)
        parallel_jobs = self._jobs(self.root / "parallel", sources)

        serial, serial_progress = self._run(serial_jobs, 1)
        parallel, parallel_progress = self._run(parallel_jobs, 4)

        self.assertEqual(serial.total, parallel.total)
        self.assertEqual(serial.done, parallel.done)
        self.assertEqual(serial.errors, parallel.errors)
        self.assertEqual(
            [
                (value.before, value.width, value.height)
                for value in serial.values
            ],
            [
                (value.before, value.width, value.height)
                for value in parallel.values
            ],
        )
        self.assertEqual(
            serial_progress[-1], (serial.done, serial.total)
        )
        self.assertEqual(
            parallel_progress[-1], (parallel.done, parallel.total)
        )

        for serial_job, parallel_job in zip(serial_jobs, parallel_jobs):
            self.assertEqual(
                serial_job.dst.exists(), parallel_job.dst.exists()
            )
            if not serial_job.dst.exists():
                continue
            with (
                Image.open(serial_job.dst) as serial_image,
                Image.open(parallel_job.dst) as parallel_image,
            ):
                self.assertEqual(serial_image.size, parallel_image.size)
                self.assertEqual(
                    serial_image.tobytes(), parallel_image.tobytes()
                )


class CollageParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _paths(self) -> list[Path]:
        folder = self.root / "images"
        folder.mkdir()
        specs: list[tuple[str, str, tuple[int, int]]] = [
            ("a.png", "RGBA", (31, 23)),
            ("b.jpg", "RGB", (48, 32)),
            ("c.png", "P", (25, 40)),
            ("d.png", "RGB", (64, 16)),
        ]
        paths: list[Path] = []
        for name, mode, size in specs:
            path = folder / name
            if mode == "P":
                image = Image.new("RGB", size, (0, 100, 200)).convert("P")
            elif mode == "RGBA":
                image = Image.new(mode, size, (10, 20, 30, 120))
            else:
                image = Image.new(mode, size, (200, 10, 10))
            image.save(path)
            image.close()
            paths.append(path)

        bad = folder / "bad.png"
        bad.write_bytes(b"not an image")
        paths.insert(1, bad)
        return paths

    def _options(self) -> CollageOptions:
        return CollageOptions(
            mode="long",
            target_width=80,
            spacing=3,
            margin=5,
            format="png",
        )

    def _run(
        self, paths: list[Path], output: Path, workers: int
    ) -> CollageResult:
        with mock.patch.object(image_collage, "MAX_IMAGE_WORKERS", workers):
            return build_collage(paths, output, self._options())

    def test_parallel_matches_serial_pixel_for_pixel(self) -> None:
        paths = self._paths()
        serial_output = self.root / "serial.png"
        parallel_output = self.root / "parallel.png"

        serial = self._run(paths, serial_output, 1)
        parallel = self._run(paths, parallel_output, 4)

        self.assertEqual(serial.count, parallel.count)
        self.assertEqual(serial.width, parallel.width)
        self.assertEqual(serial.height, parallel.height)
        self.assertEqual(serial.skipped, parallel.skipped)
        self.assertEqual(
            [path.name for path, _reason in serial.skipped],
            [path.name for path, _reason in parallel.skipped],
        )

        with (
            Image.open(serial_output) as serial_image,
            Image.open(parallel_output) as parallel_image,
        ):
            self.assertEqual(serial_image.size, parallel_image.size)
            self.assertEqual(
                serial_image.tobytes(), parallel_image.tobytes()
            )

    def test_cancel_closes_unpasted_tiles_and_writes_nothing(self) -> None:
        paths = self._paths()
        output = self.root / "cancelled.png"
        cancel_event = threading.Event()

        def on_progress(done: int, total: int) -> None:
            # 第一遍探针占前 total_files 步，进入第二遍后立即取消。
            if done > len(paths):
                cancel_event.set()

        with (
            mock.patch.object(
                image_collage, "MAX_IMAGE_WORKERS", 4
            ),
            self.assertRaises(Cancelled) as ctx,
        ):
            build_collage(
                paths,
                output,
                self._options(),
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

        partial = ctx.exception.result
        self.assertIsInstance(partial, CollageResult)
        self.assertGreaterEqual(partial.count, 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
