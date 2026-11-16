"""批量重命名计划与两阶段执行的单元测试。"""

from __future__ import annotations

import random
import tempfile
import threading
import unittest
from pathlib import Path

from toolbox.common import filesystem_key
from toolbox.errors import Cancelled
from toolbox.tools.batch_rename import (
    STATUS_CONFLICT,
    STATUS_DUPLICATE,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_UNCHANGED,
    RenameItem,
    RenameOptions,
    RenamePlan,
    _mark_conflicts,
    build_rename_plan,
    execute_rename_plan,
    sanitize_stem,
)


def _reference_mark_conflicts(
    entries: list[Path], items: list[RenameItem]
) -> None:
    """优化前的多轮传播实现，用作随机对照。"""
    entries_by_key: dict[str, list[str]] = {}
    for entry in entries:
        key = filesystem_key(entry.name)
        entries_by_key.setdefault(key, []).append(entry.name)
    source_by_name = {item.src.name: item for item in items}

    changed = True
    while changed:
        changed = False
        for item in items:
            if item.status != STATUS_OK:
                continue
            target_names = entries_by_key.get(
                filesystem_key(item.dst.name), ()
            )
            blocked = False
            for name in target_names:
                if name == item.src.name:
                    continue
                other = source_by_name.get(name)
                if other is not None and other.status == STATUS_OK:
                    continue
                blocked = True
                break
            if blocked:
                item.status = STATUS_CONFLICT
                item.reason = f"目标文件名已被占用：{item.dst.name}"
                changed = True


def _copy_items(items: list[RenameItem]) -> list[RenameItem]:
    return [
        RenameItem(
            src=item.src,
            dst=item.dst,
            status=item.status,
            reason=item.reason,
        )
        for item in items
    ]


_BASE_NAMES = [
    "a.txt",
    "A.TXT",
    "b.txt",
    "B.txt",
    "c.jpg",
    "d",
    "e.txt",
    "f.txt",
    "g.txt",
    "h.txt",
    "01.txt",
    "02.txt",
    "cover.jpg",
]
_EXTRA_NAMES = ["n1.txt", "n2.txt", "N2.TXT", "n3.jpg", "new", "z.txt"]


def _random_case(
    rng: random.Random,
) -> tuple[list[Path], list[RenameItem]]:
    count = rng.randint(2, 7)
    source_names = rng.sample(_BASE_NAMES, count)
    remaining = [
        name for name in _BASE_NAMES + _EXTRA_NAMES
        if name not in source_names
    ]
    extras = rng.sample(remaining, rng.randint(0, 3))
    entries = [Path(name) for name in source_names + extras]

    target_pool = source_names + extras + _EXTRA_NAMES
    used_target_keys: set[str] = set()
    items: list[RenameItem] = []
    for source_name in source_names:
        roll = rng.random()
        if roll < 0.6:
            target_name = rng.choice(source_names)
        elif roll < 0.85:
            target_name = rng.choice(target_pool)
        else:
            target_name = source_name

        status = rng.choices(
            [STATUS_OK, STATUS_UNCHANGED, STATUS_INVALID, STATUS_DUPLICATE],
            weights=[65, 15, 10, 10],
        )[0]
        key = filesystem_key(target_name)
        if status == STATUS_OK and key in used_target_keys:
            # 模拟 _mark_duplicates 的保证：每个目标最多一个 ok 项。
            status = STATUS_DUPLICATE
        if status == STATUS_OK:
            used_target_keys.add(key)
            reason = ""
        else:
            reason = f"初始状态：{status}"
        items.append(
            RenameItem(
                src=Path(source_name),
                dst=Path(target_name),
                status=status,
                reason=reason,
            )
        )
    return entries, items


class MarkConflictsTest(unittest.TestCase):
    def test_random_cases_match_reference(self) -> None:
        rng = random.Random(20261116)
        for _ in range(200):
            entries, items = _random_case(rng)
            expected = _copy_items(items)
            actual = _copy_items(items)
            _reference_mark_conflicts(entries, expected)
            _mark_conflicts(entries, actual)
            self.assertEqual(
                [(item.status, item.reason) for item in actual],
                [(item.status, item.reason) for item in expected],
            )

    def test_chain_propagates_from_initial_conflict(self) -> None:
        entries = [Path(name) for name in ("a.txt", "b.txt", "c.txt", "d.txt")]
        items = [
            RenameItem(Path("a.txt"), Path("b.txt"), STATUS_OK),
            RenameItem(Path("b.txt"), Path("c.txt"), STATUS_OK),
            RenameItem(Path("c.txt"), Path("d.txt"), STATUS_OK),
        ]
        _mark_conflicts(entries, items)
        self.assertEqual(
            [item.status for item in items],
            [STATUS_CONFLICT, STATUS_CONFLICT, STATUS_CONFLICT],
        )
        self.assertEqual(
            [item.reason for item in items],
            [
                "目标文件名已被占用：b.txt",
                "目标文件名已被占用：c.txt",
                "目标文件名已被占用：d.txt",
            ],
        )

    def test_all_ok_cycle_stays_ok(self) -> None:
        entries = [Path("a.txt"), Path("b.txt")]
        items = [
            RenameItem(Path("a.txt"), Path("b.txt"), STATUS_OK),
            RenameItem(Path("b.txt"), Path("a.txt"), STATUS_OK),
        ]
        _mark_conflicts(entries, items)
        self.assertEqual([item.status for item in items], [STATUS_OK, STATUS_OK])

    def test_case_insensitive_cycle_stays_ok(self) -> None:
        entries = [Path("a.txt"), Path("B.TXT")]
        items = [
            RenameItem(Path("a.txt"), Path("B.TXT"), STATUS_OK),
            RenameItem(Path("B.TXT"), Path("A.TXT"), STATUS_OK),
        ]
        _mark_conflicts(entries, items)
        self.assertEqual([item.status for item in items], [STATUS_OK, STATUS_OK])

    def test_sanitize_stem_public_behavior(self) -> None:
        self.assertEqual(sanitize_stem("CON"), "")
        self.assertEqual(sanitize_stem("con.txt"), "")
        self.assertEqual(sanitize_stem("  name<>:?  "), "  name____")
        self.assertEqual(sanitize_stem("a."), "a")


class ExecuteRenamePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _plan(self) -> RenamePlan:
        source_a = self.root / "a.txt"
        source_b = self.root / "b.txt"
        source_a.write_text("a", encoding="utf-8")
        source_b.write_text("b", encoding="utf-8")
        return RenamePlan(
            folder=self.root,
            items=[
                RenameItem(source_a, self.root / "a2.txt", STATUS_OK),
                RenameItem(source_b, self.root / "b2.txt", STATUS_OK),
            ],
        )

    def test_progress_reaches_final_value(self) -> None:
        plan = self._plan()
        events: list[tuple[int, int]] = []
        result = execute_rename_plan(
            plan, on_progress=lambda done, total: events.append((done, total))
        )
        self.assertFalse(result.errors)
        self.assertEqual(len(result.renamed), 2)
        self.assertEqual(events[-1], (4, 4))
        self.assertTrue((self.root / "a2.txt").is_file())
        self.assertTrue((self.root / "b2.txt").is_file())

    def test_cancel_restores_staged_names(self) -> None:
        plan = self._plan()
        cancel = threading.Event()
        events: list[tuple[int, int]] = []

        def progress(done: int, total: int) -> None:
            events.append((done, total))
            if done == 1:
                cancel.set()

        with self.assertRaises(Cancelled) as ctx:
            execute_rename_plan(
                plan, cancel_event=cancel, on_progress=progress
            )
        partial = ctx.exception.result
        self.assertEqual(partial.errors, [])
        self.assertTrue((self.root / "a.txt").is_file())
        self.assertTrue((self.root / "b.txt").is_file())
        self.assertFalse((self.root / "a2.txt").exists())
        self.assertFalse((self.root / "b2.txt").exists())
        self.assertEqual(events, [(1, 4)])
        leftovers = [
            path.name for path in self.root.iterdir()
            if path.name.startswith(".pyrename")
        ]
        self.assertEqual(leftovers, [])


class BuildRenamePlanCancellationTest(unittest.TestCase):
    def test_scan_cancel_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(Cancelled):
                build_rename_plan(
                    root, RenameOptions(), cancel_event=cancel
                )


if __name__ == "__main__":
    unittest.main()
