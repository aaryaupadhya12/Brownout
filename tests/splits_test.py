"""src/splits.py: fixed train/val/test splits. Done when all pass.

They fail with NotImplementedError until you fill in make_split and select.
No dataset needed: the tasks are made up here (brownout for the run_benchmark test).
"""

import os
import tempfile
import unittest
from collections import Counter

from src.fake_model import FakeModel, call, scripted
from src.interface_v1 import Task
from src.run import run_benchmark
from src.splits import PARTS, load_split, make_split, save_split, select, spec_key

# 4 strata of different sizes, like root causes: 20, 7, 2 and 1 tasks
SIZES = {"common": 20, "medium": 7, "pair": 2, "single": 1}
TASKS = [Task(f"case-{cause}-{i}", 0, {"answer": {"root_cause": cause}})
         for cause, n in SIZES.items() for i in range(n)]
BY_CAUSE = spec_key("answer.root_cause")


def part_of(split):
    return {i: p for p in PARTS for i in split[p]}


class TestMakeSplit(unittest.TestCase):
    def test_every_task_in_exactly_one_part(self):
        split = make_split(TASKS, BY_CAUSE)
        ids = [i for p in PARTS for i in split[p]]
        self.assertEqual(sorted(ids), sorted(t.id for t in TASKS))     # complete, no duplicates

    def test_deterministic_and_order_independent(self):
        self.assertEqual(make_split(TASKS, BY_CAUSE, seed=0),
                         make_split(list(reversed(TASKS)), BY_CAUSE, seed=0))
        self.assertNotEqual(make_split(TASKS, BY_CAUSE, seed=0),
                            make_split(TASKS, BY_CAUSE, seed=1))

    def test_stratified(self):
        where = part_of(make_split(TASKS, BY_CAUSE))
        per = {c: Counter(where[t.id] for t in TASKS if BY_CAUSE(t) == c) for c in SIZES}
        for cause in ("common", "medium", "pair"):                    # 2+ tasks: trained AND tested
            self.assertGreaterEqual(per[cause]["train"], 1, cause)
            self.assertGreaterEqual(per[cause]["test"], 1, cause)
        self.assertEqual(per["single"]["train"], 1)                   # 1 task: train only
        self.assertEqual((per["common"]["train"], per["common"]["val"], per["common"]["test"]),
                         (14, 2, 4))                                  # 70 / 10 / 20 of 20

    def test_adding_a_stratum_does_not_move_the_others(self):
        before = part_of(make_split(TASKS, BY_CAUSE))
        extra = [Task(f"case-new-{i}", 0, {"answer": {"root_cause": "new"}}) for i in range(5)]
        after = part_of(make_split(TASKS + extra, BY_CAUSE))
        self.assertEqual({i: after[i] for i in before}, before)

    def test_without_stratify(self):
        split = make_split(TASKS, None)
        self.assertEqual(sum(map(len, split.values())), len(TASKS))
        self.assertEqual(len(split["test"]), round(len(TASKS) * 0.2))

    def test_missing_spec_field_names_the_task(self):
        with self.assertRaises(KeyError) as ctx:
            make_split(TASKS, spec_key("answer.fault_object"))
        self.assertIn("case-", str(ctx.exception))


class TestSelect(unittest.TestCase):
    def saved(self, d, split):
        path = os.path.join(d, "split.json")
        save_split(path, split, {"envs": ["x"], "stratify": None, "fractions": {}, "seed": 0})
        return path

    def test_round_trip_and_select(self):
        with tempfile.TemporaryDirectory() as d:
            split = make_split(TASKS, BY_CAUSE)
            path = self.saved(d, split)
            self.assertEqual(load_split(path)["splits"], split)
            test = [t.id for t in select(TASKS, path, "test")]
            self.assertEqual(sorted(test), split["test"])
            self.assertEqual(test, [t.id for t in TASKS if t.id in split["test"]])   # original order

    def test_unknown_part(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.saved(d, make_split(TASKS, BY_CAUSE))
            with self.assertRaises(ValueError):
                list(select(TASKS, path, "dev"))

    def test_task_not_in_the_split_is_an_error(self):
        """A case added after the split was made must not be silently skipped."""
        with tempfile.TemporaryDirectory() as d:
            path = self.saved(d, make_split(TASKS, BY_CAUSE))
            new = TASKS + [Task("case-added-later", 0, {"answer": {"root_cause": "common"}})]
            with self.assertRaises(ValueError) as ctx:
                list(select(new, path, "train"))
            self.assertIn("case-added-later", str(ctx.exception))


class TestRunWithSplit(unittest.TestCase):
    def test_run_benchmark_runs_only_the_chosen_part(self):
        good = lambda: FakeModel([scripted(call("c1", "submit", {}))])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "split.json")
            save_split(path, {"train": ["control"], "val": [], "test": ["brownout_B"]},
                       {"envs": ["brownout"], "stratify": None, "fractions": {}, "seed": 0})
            ran = run_benchmark("brownout", good, out_dir=None, split=(path, "test"))
            self.assertEqual([t.task_id for t in ran], ["brownout_B"])


if __name__ == "__main__":
    unittest.main()
