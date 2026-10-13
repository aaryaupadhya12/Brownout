"""Append + resume (src/store.py, src/run.py). Done when all pass."""

import json
import os
import tempfile
import unittest

from src.fake_model import FakeModel, call, scripted
from src.run import run_benchmark
from src.store import is_done, write_json

SUBMIT = [scripted(call("c1", "submit", {}))]


def make_model():
    return FakeModel(SUBMIT)


class TestWriteJson(unittest.TestCase):
    def test_failed_write_keeps_the_old_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            write_json(path, {"v": 1})
            with self.assertRaises(TypeError):
                write_json(path, {"v": object()})      # can't be serialized: fails mid-write
            with open(path) as f:
                self.assertEqual(json.load(f), {"v": 1})
            self.assertEqual(os.listdir(d), ["x.json"])   # no .tmp left behind


class TestIsDone(unittest.TestCase):
    def saved(self, d, data, name):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        return path

    def test_rules(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_done(os.path.join(d, "missing.json")))
            torn = self.saved(d, '{"ended_by": "subm', "torn.json")
            self.assertFalse(is_done(torn))
            api_down = self.saved(d, {"ended_by": "error", "grader": {"ok": True},
                                      "errors": [{"where": "model"}]}, "api.json")
            self.assertFalse(is_done(api_down))
            model_failed = self.saved(d, {"ended_by": "max_turns", "grader": {"ok": True},
                                          "errors": []}, "failed.json")
            self.assertTrue(is_done(model_failed))          # a real result: keep
            scorer_crashed = self.saved(d, {"ended_by": "submit", "grader": {"ok": False},
                                            "errors": [{"where": "scorer"}]}, "scorer.json")
            self.assertTrue(is_done(scorer_crashed))        # re-grade it, don't re-run it


class TestResume(unittest.TestCase):
    def test_resume_runs_only_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            first = run_benchmark("brownout", make_model, trials=2, out_dir=d)
            self.assertEqual(len(first), 4)                    # 2 tasks x 2 trials
            self.assertEqual(len(os.listdir(d)), 4)

            os.remove(os.path.join(d, sorted(os.listdir(d))[0]))
            second = run_benchmark("brownout", make_model, trials=2, out_dir=d, resume=True)
            self.assertEqual(len(second), 1)                   # only the deleted one ran again
            self.assertEqual(len(os.listdir(d)), 4)

    def test_without_resume_everything_runs(self):
        with tempfile.TemporaryDirectory() as d:
            run_benchmark("brownout", make_model, trials=1, out_dir=d)
            again = run_benchmark("brownout", make_model, trials=1, out_dir=d)
            self.assertEqual(len(again), 2)


if __name__ == "__main__":
    unittest.main()
