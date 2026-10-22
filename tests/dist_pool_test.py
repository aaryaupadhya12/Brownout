"""Spec for the whole runner (queue + worker + faults + coordinator). Skips until each piece
exists. Uses fake episodes only: free, about 10 seconds.

    python -m unittest tests.dist_pool_test -v
"""

import json
import os
import tempfile
import unittest

from src.dist import coordinator, faults, worker


def todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        raise unittest.SkipTest(str(e))


class PiecesTest(unittest.TestCase):
    def test_job_id_is_stable(self):
        self.assertEqual(todo(coordinator.job_id, "g1", 0), todo(coordinator.job_id, "g1", 0))
        self.assertNotEqual(todo(coordinator.job_id, "g1", 0), todo(coordinator.job_id, "g1", 1))

    def test_plan_jobs_unique(self):
        jobs = todo(coordinator.plan_jobs, 50)
        self.assertEqual(len({j for j, _ in jobs}), 50)

    def test_fake_episode_is_deterministic(self):
        plan = faults.FaultPlan(seed=1, episode_ms=1, infra_error_prob=0.0)
        a = todo(faults.fake_episode, {"job_id": "j1"}, plan)
        b = todo(faults.fake_episode, {"job_id": "j1"}, plan)
        self.assertEqual(a, b)

    def test_save_is_atomic_idempotent_and_fenced(self):
        with tempfile.TemporaryDirectory() as d:
            w = worker.Worker("unused.sqlite", "w1", None, d)
            p1 = todo(w.save, "j1", 1, {"reward": 1})
            p2 = todo(w.save, "j1", 1, {"reward": 1})
            p3 = todo(w.save, "j1", 2, {"reward": 0})       # a later attempt: its own file
            self.assertEqual(p1, p2)
            self.assertNotEqual(p1, p3)
            self.assertEqual(sorted(os.listdir(d)), ["j1.attempt1.json", "j1.attempt2.json"])
            with open(p1, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"reward": 1})


class PoolTest(unittest.TestCase):
    def test_killing_a_worker_loses_nothing_and_duplicates_nothing(self):
        plan = faults.FaultPlan(seed=0, episode_ms=20, tail_prob=0.05, infra_error_prob=0.05,
                                kill_after_s=0.3)
        with tempfile.TemporaryDirectory() as d:
            report = todo(coordinator.run_pool, os.path.join(d, "q.sqlite"), os.path.join(d, "out"),
                          4, plan, 60)
            self.assertEqual(report.killed, 1)
            self.assertEqual(report.lost, 0)
            self.assertEqual(report.mismatched, 0)
            self.assertEqual(report.done + report.dead, 60)
            files = [f for f in os.listdir(os.path.join(d, "out")) if f.endswith(".json")]
            self.assertEqual(len(files), report.done + report.stale_files)


if __name__ == "__main__":
    unittest.main()
