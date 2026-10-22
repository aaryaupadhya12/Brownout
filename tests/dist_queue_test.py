"""Spec for src/dist/queue.py. Skips until JobQueue is implemented.

    python -m unittest tests.dist_queue_test -v

A fake clock moves time instantly, so lease expiry is tested without sleeping.
"""

import multiprocessing as mp
import os
import tempfile
import unittest

from src.dist.queue import JobQueue


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def open_queue(path, clock=None, max_attempts=3):
    try:
        return JobQueue(path, clock=clock or Clock(), max_attempts=max_attempts)
    except NotImplementedError as e:
        raise unittest.SkipTest(str(e))


def todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        raise unittest.SkipTest(str(e))


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "q.sqlite")
        self.clock = Clock()

    def tearDown(self):
        q = getattr(self, "q", None)
        if q is not None:
            try:
                q.close()
            except NotImplementedError:
                pass
        self.dir.cleanup()

    def queue(self, max_attempts=3):
        self.q = open_queue(self.path, self.clock, max_attempts)
        return self.q

    def test_enqueue_is_idempotent(self):
        q = self.queue()
        self.assertTrue(todo(q.enqueue, "j1", {"x": 1}))
        self.assertFalse(todo(q.enqueue, "j1", {"x": 2}))
        self.assertEqual(todo(q.stats)["pending"], 1)

    def test_lease_returns_payload_and_first_attempt(self):
        q = self.queue()
        todo(q.enqueue, "j1", {"x": 1})
        lease = todo(q.lease, "w1", 30)
        self.assertEqual((lease.job_id, lease.worker_id, lease.attempt, lease.payload),
                         ("j1", "w1", 1, {"x": 1}))
        self.assertAlmostEqual(lease.expires_at, self.clock.now + 30)

    def test_a_job_is_leased_to_one_worker_at_a_time(self):
        q = self.queue()
        todo(q.enqueue, "j1", {})
        self.assertIsNotNone(todo(q.lease, "w1", 30))
        self.assertIsNone(todo(q.lease, "w2", 30))

    def test_expired_lease_is_released_and_fenced(self):
        q = self.queue()
        todo(q.enqueue, "j1", {})
        old = todo(q.lease, "w1", 30)
        self.clock.now += 31                         # w1 froze; its lease expired
        new = todo(q.lease, "w2", 30)
        self.assertEqual(new.attempt, 2)
        self.assertFalse(todo(q.heartbeat, old, 30))  # w1 learns it lost the job
        self.assertFalse(todo(q.complete, old, {"r": "stale"}))
        self.assertTrue(todo(q.complete, new, {"r": "fresh"}))
        self.assertEqual(todo(q.stats)["done"], 1)

    def test_heartbeat_keeps_a_slow_worker_alive(self):
        q = self.queue()
        todo(q.enqueue, "j1", {})
        lease = todo(q.lease, "w1", 30)
        self.clock.now += 25
        self.assertTrue(todo(q.heartbeat, lease, 30))
        self.clock.now += 25                         # 50 s after the lease, 25 after heartbeat
        self.assertIsNone(todo(q.lease, "w2", 30))

    def test_attempts_are_capped(self):
        q = self.queue(max_attempts=2)
        todo(q.enqueue, "j1", {})
        todo(q.lease, "w1", 10)
        self.clock.now += 11
        todo(q.lease, "w2", 10)
        self.clock.now += 11
        self.assertIsNone(todo(q.lease, "w3", 10))
        self.assertEqual(todo(q.stats)["dead"], 1)

    def test_fail_retryable_and_not(self):
        q = self.queue()
        todo(q.enqueue, "a", {})
        todo(q.enqueue, "b", {})
        la = todo(q.lease, "w1", 30)
        lb = todo(q.lease, "w1", 30)
        todo(q.fail, la, "timeout", True)
        todo(q.fail, lb, "bad input", False)
        stats = todo(q.stats)
        self.assertEqual((stats["pending"], stats["dead"]), (1, 1))
        self.assertEqual(todo(q.lease, "w2", 30).attempt, 2)

    def test_state_survives_reopen(self):
        q = self.queue()
        todo(q.enqueue, "j1", {})
        todo(q.close)
        self.q = None
        q2 = open_queue(self.path, self.clock)
        self.q = q2
        self.assertEqual(todo(q2.stats)["pending"], 1)

    def test_stats_has_every_status(self):
        q = self.queue()
        self.assertEqual(set(todo(q.stats)), {"pending", "leased", "done", "dead"})


def _lease_all(path, worker_id, out):
    q = JobQueue(path)
    got = []
    while True:
        lease = q.lease(worker_id, 300)
        if lease is None:
            break
        got.append(lease.job_id)
    q.close()
    out.put(got)


class ProcessRaceTest(unittest.TestCase):
    """Four real processes lease from the same file at once: no job may go to two of them."""

    def test_no_job_is_leased_twice(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.sqlite")
            q = open_queue(path, clock=None)
            for i in range(200):
                todo(q.enqueue, f"j{i}", {})
            todo(q.close)
            out = mp.Queue()
            procs = [mp.Process(target=_lease_all, args=(path, f"w{k}", out)) for k in range(4)]
            for p in procs:
                p.start()
            leased = [job for _ in procs for job in out.get(timeout=60)]
            for p in procs:
                p.join(timeout=60)
            self.assertEqual(len(leased), 200)
            self.assertEqual(len(set(leased)), 200)


if __name__ == "__main__":
    unittest.main()
