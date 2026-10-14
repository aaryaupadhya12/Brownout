"""Phase 9: run_benchmark(..., concurrency=N). Done when all pass.

They fail with NotImplementedError until you fill in the TODO in src/run.py.
No API calls: a FakeModel that sleeps stands in for a slow model.
"""

import os
import tempfile
import threading
import time
import unittest

from src.fake_model import FakeModel, call, scripted
from src.run import run_benchmark

DELAY = 0.1          # seconds per model call; a brownout episode below makes 4 calls


def good_script():
    return [scripted(call("c1", "get_metrics", {"zone": "B"})),
            scripted(call("c2", "shift_traffic", {"zone_source": "B", "zone_destination": "A",
                                                  "split_percentage": 25})),
            scripted(call("c3", "shift_traffic", {"zone_source": "B", "zone_destination": "C",
                                                  "split_percentage": 20})),
            scripted(call("c4", "submit", {}))]


class Gauge:
    """Counts how many model calls are in flight at once, across threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc):
        with self.lock:
            self.now -= 1


class SlowModel(FakeModel):
    def __init__(self, gauge=None, fail=False):
        super().__init__(good_script())
        self.gauge = gauge or Gauge()
        self.fail = fail

    def complete(self, messages, tools, temperature=0.0):
        with self.gauge:
            time.sleep(DELAY)
            if self.fail:
                raise ConnectionError("simulated API outage")
            return super().complete(messages, tools, temperature)


def run(concurrency, trials=4, out_dir=None, gauge=None, resume=False, make_model=None):
    """brownout has 2 tasks, so trials=4 -> 8 episodes."""
    make_model = make_model or (lambda: SlowModel(gauge))
    return run_benchmark("brownout", make_model, trials=trials, out_dir=out_dir,
                         resume=resume, concurrency=concurrency)


def summary(trajectories):
    return [(t.task_id, t.run_index, t.ended_by, t.grader.reward) for t in trajectories]


class TestConcurrency(unittest.TestCase):
    def test_same_results_as_sequential_in_plan_order(self):
        self.assertEqual(summary(run(concurrency=4)), summary(run(concurrency=1)))

    def test_faster_than_sequential(self):
        start = time.perf_counter()
        trajectories = run(concurrency=8)
        took = time.perf_counter() - start
        self.assertEqual(len(trajectories), 8)
        sequential = 8 * 4 * DELAY                     # 3.2 s
        self.assertLess(took, sequential / 2, f"took {took:.2f} s, sequential is ~{sequential} s")

    def test_never_more_than_concurrency_at_once(self):
        gauge = Gauge()
        run(concurrency=3, gauge=gauge)
        self.assertGreater(gauge.peak, 1)              # it really ran in parallel
        self.assertLessEqual(gauge.peak, 3)            # and respected the limit

    def test_one_failing_episode_does_not_stop_the_rest(self):
        made = []

        def make_model():
            made.append(1)
            return SlowModel(fail=len(made) == 3)      # the 3rd episode's API is down

        trajectories = run(concurrency=4, make_model=make_model)
        self.assertEqual(len(trajectories), 8)
        self.assertEqual(sum(t.ended_by == "error" for t in trajectories), 1)

    def test_saves_every_episode_and_resume_skips_them(self):
        with tempfile.TemporaryDirectory() as d:
            run(concurrency=4, out_dir=d)
            files = os.listdir(d)
            self.assertEqual(len(files), 8)
            self.assertFalse(any(f.endswith(".tmp") for f in files))

            os.remove(os.path.join(d, sorted(files)[0]))
            again = run(concurrency=4, out_dir=d, resume=True)
            self.assertEqual(len(again), 1)            # only the deleted one ran again
            self.assertEqual(len(os.listdir(d)), 8)

    def test_concurrency_1_is_the_default(self):
        trajectories = run_benchmark("brownout", lambda: SlowModel(), trials=1, out_dir=None)
        self.assertEqual(len(trajectories), 2)


if __name__ == "__main__":
    unittest.main()
