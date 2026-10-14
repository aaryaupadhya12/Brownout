"""src/report.py: the terminal summary of saved runs."""

import json
import os
import tempfile
import unittest

from src.report import category, load, report, summarise


def traj(task_id, reward, ended_by="submit", ok=True, errors=(), cost=0.01):
    return {"task_id": task_id, "model_id": "m", "ended_by": ended_by,
            "grader": {"ok": ok, "reward": reward, "signals": {"ca": 1.0}},
            "errors": list(errors), "totals": {"api_cost_usd": cost},
            "timing": {"started_at": 0, "finished_at": 60},
            "tracking": {"verifier": {"name": "v"}}}


class TestReport(unittest.TestCase):
    def test_category_drops_the_case_number(self):
        self.assertEqual(category("boutique-admission-17"), "boutique-admission")
        self.assertEqual(category("control"), "control")

    def test_infra_failures_are_not_averaged_as_zero(self):
        rows = [traj("a-1", 1.0), traj("a-2", 0.0),
                traj("a-3", 0.0, ended_by="error", errors=[{"where": "model"}])]
        row = summarise(rows, ["ca"])
        self.assertEqual((row["n"], row["scored"], row["infra"]), (3, 2, 1))
        self.assertAlmostEqual(row["reward"], 0.5)
        self.assertAlmostEqual(row["cost"], 0.03)

    def test_reads_a_folder_and_skips_torn_files(self):
        with tempfile.TemporaryDirectory() as d:
            for i, r in enumerate([1.0, 0.0]):
                with open(os.path.join(d, f"x-{i}.json"), "w", encoding="utf-8") as f:
                    json.dump(traj(f"x-{i}", r), f)
            with open(os.path.join(d, "torn.json"), "w", encoding="utf-8") as f:
                f.write('{"task_id": "x-')
            self.assertEqual(len(load(d)), 2)
            text = report(d, show_failures=True)
            self.assertIn("x ", text)
            self.assertIn("0.50", text)
            self.assertIn("x-1", text.split("failed episodes")[1])


if __name__ == "__main__":
    unittest.main()
