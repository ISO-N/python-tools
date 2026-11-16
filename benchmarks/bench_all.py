"""性能基准：用合成数据对比优化前后的关键路径。

用法：
    uv run python benchmarks/bench_all.py
    uv run python benchmarks/bench_all.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

import pymupdf

from toolbox.common import find_cjk_font_file
from toolbox.images import prepare_image_payload
from toolbox.pdf_utils import (
    TextStamp,
    add_text_stamps,
    insert_rotated_text,
    load_pdf_font_resource,
    register_cjk_font,
    save_pdf_atomic,
)
from toolbox.tools.pdf_to_images import convert_pdf, convert_pdf_parallel


def timed(func: Callable[[], object], repeat: int = 3) -> float:
    """返回多次运行的中位数秒数。"""
    values: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        func()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def make_jpegs(root: Path, count: int, size=(800, 600)) -> list[Path]:
    paths = []
    for index in range(count):
        path = root / f"img_{index:03d}.jpg"
        Image.new(
            "RGB",
            size,
            ((index * 37) % 255, (index * 71) % 255, (index * 113) % 255),
        ).save(path, quality=90)
        paths.append(path)
    return paths


def make_pdf(
    path: Path,
    pages: int,
    *,
    width: float = 595,
    height: float = 842,
) -> None:
    document = pymupdf.open()
    try:
        for index in range(pages):
            page = document.new_page(width=width, height=height)
            page.insert_text(
                (72, 72), f"Benchmark page {index + 1}", fontsize=14
            )
        document.save(path)
    finally:
        document.close()


def bench_font(pages: int, repeat: int) -> dict[str, float]:
    font_path = find_cjk_font_file()
    if font_path is None:
        return {}
    resource = load_pdf_font_resource(font_path)

    def fontfile() -> None:
        document = pymupdf.open()
        try:
            for _ in range(pages):
                page = document.new_page()
                register_cjk_font(page, font_path)
        finally:
            document.close()

    def fontbuffer() -> None:
        document = pymupdf.open()
        try:
            for _ in range(pages):
                page = document.new_page()
                register_cjk_font(page, font_path, resource=resource)
        finally:
            document.close()

    return {
        "pdf_font_fontfile": timed(fontfile, repeat),
        "pdf_font_fontbuffer": timed(fontbuffer, repeat),
    }


def bench_stamps(pages: int, stamps_per_page: int, repeat: int) -> dict[str, float]:
    resource = load_pdf_font_resource(None)
    font = resource.font
    stamps = [
        TextStamp((100, 100), "Watermark", 18, angle=30, fill_opacity=0.4)
        for _ in range(stamps_per_page)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def per_stamp() -> None:
            document = pymupdf.open()
            try:
                for _ in range(pages):
                    page = document.new_page()
                    fontname = register_cjk_font(
                        page, None, resource=resource
                    )
                    for stamp in stamps:
                        insert_rotated_text(
                            page,
                            stamp.center,
                            stamp.text,
                            fontname=fontname,
                            font=font,
                            fontsize=stamp.fontsize,
                            fill_opacity=stamp.fill_opacity,
                            angle=stamp.angle,
                        )
                save_pdf_atomic(
                    document, root / "per.pdf", preset="fast"
                )
            finally:
                document.close()

        def one_shape() -> None:
            document = pymupdf.open()
            try:
                for _ in range(pages):
                    page = document.new_page()
                    fontname = register_cjk_font(
                        page, None, resource=resource
                    )
                    add_text_stamps(
                        page, stamps, fontname=fontname, font=font
                    )
                save_pdf_atomic(
                    document, root / "shape.pdf", preset="fast"
                )
            finally:
                document.close()

        return {
            "pdf_stamps_per_stamp": timed(per_stamp, repeat),
            "pdf_stamps_one_shape": timed(one_shape, repeat),
        }


def bench_payload(count: int, repeat: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = make_jpegs(root, count, size=(1600, 1200))

        def serial() -> None:
            for path in paths:
                prepare_image_payload(path, optimize=False)

        def parallel() -> None:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(
                    executor.map(
                        lambda path: prepare_image_payload(
                            path, optimize=False
                        ),
                        paths,
                    )
                )

        return {
            "image_payload_serial": timed(serial, repeat),
            "image_payload_parallel": timed(parallel, repeat),
        }


def bench_pdf_to_images(pages: int, repeat: int) -> dict[str, float]:
    # 页面取 1200×1600pt、100DPI，确保总像素字节数超过并行阈值，
    # 否则 convert_pdf_parallel 会按设计回退到串行路径。
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "bench.pdf"
        make_pdf(source, pages, width=1200, height=1600)
        page_list = list(range(pages))

        def serial() -> None:
            convert_pdf(
                source,
                root / "serial",
                "png",
                100,
                page_list,
            )

        def parallel() -> None:
            convert_pdf_parallel(
                source,
                root / "parallel",
                "png",
                100,
                page_list,
                max_workers=4,
            )

        return {
            "pdf_to_images_serial": timed(serial, repeat),
            "pdf_to_images_parallel": timed(parallel, repeat),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repeat = 1 if args.quick else 3
    results: dict[str, float] = {
        "cpu_count": float(os.cpu_count() or 1),
    }
    results.update(bench_font(12 if args.quick else 20, repeat))
    results.update(bench_stamps(6, 20, repeat))
    results.update(bench_payload(8 if args.quick else 12, repeat))
    results.update(bench_pdf_to_images(8 if args.quick else 24, repeat))
    for name, value in results.items():
        print(f"{name:34s} {value:8.3f}s" if name != "cpu_count" else f"{name:34s} {value:8.0f}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
