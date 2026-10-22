"""Fault-tolerant distributed episode runner (you write it; guide: interview/11_distributed_runner.md).

Today `src/run.py --concurrency N` runs episodes on threads in one process: if the process
dies, everything in flight is lost and `--resume` cleans up afterwards. This package runs
episodes on N worker PROCESSES that pull jobs from a shared queue, survive workers dying
mid-episode, and never save an episode twice.

    queue.py        JobQueue on SQLite: enqueue / lease / heartbeat / complete / fail / stats
    worker.py       Worker: lease -> heartbeat while running -> save -> complete
    coordinator.py  plan jobs (idempotent ids), start workers, measure, report
    faults.py       FaultPlan + a fake episode: slow calls, crashes, kills (for experiments)

Standard library only (sqlite3, multiprocessing, threading). Tests: tests/dist_*_test.py.
"""
