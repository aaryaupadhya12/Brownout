import os
import tempfile
import threading
import unittest

from src.audit.swarm import ledger


def make_row(i):
    row = {f: "" for f in ledger.ROW_FIELDS}
    row.update(exploit_id=f"x{i}", time=None)
    return row


def write_together(path, rows):
    """Start every write at the same instant, so any race shows up."""
    barrier = threading.Barrier(len(rows))
    results = []

    def go(row):
        barrier.wait()
        results.append(ledger.append_once(path, row))

    threads = [threading.Thread(target=go, args=(r,)) for r in rows]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ledger.jsonl")

    def tearDown(self):
        self.dir.cleanup()

    def test_same_row_from_16_threads_is_written_once(self):
        results = write_together(self.path, [make_row(0) for _ in range(16)])
        self.assertEqual(len(ledger.read(self.path)), 1)
        self.assertEqual(results.count(True), 1)

    def test_16_different_rows_are_all_written(self):
        write_together(self.path, [make_row(i) for i in range(16)])
        rows = ledger.read(self.path)
        self.assertEqual(len(rows), 16)
        self.assertEqual({r["exploit_id"] for r in rows}, {f"x{i}" for i in range(16)})

    def test_resume_writes_nothing_new(self):
        self.assertTrue(ledger.append_once(self.path, make_row(0)))
        self.assertFalse(ledger.append_once(self.path, make_row(0)))
        self.assertEqual(len(ledger.read(self.path)), 1)


if __name__ == "__main__":
    unittest.main()