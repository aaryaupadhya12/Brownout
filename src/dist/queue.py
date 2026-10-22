"""JobQueue: a durable work queue with leases, on one SQLite file.

Why leases, not locks: a worker that dies cannot release a lock. A lease expires on its own,
so a dead worker's job becomes available again after `lease_s` seconds.

Why a fencing token (`attempt`): worker A leases job 7 (attempt 1), freezes for 2 minutes, its
lease expires, worker B leases job 7 (attempt 2) and finishes. Then A wakes up and calls
complete(). Without a token A's stale result would overwrite B's. With it, complete() only
succeeds if the caller's attempt is the job's CURRENT attempt.

Delivery semantics: at-least-once (a job can RUN twice: A and B above). Saving is made
idempotent by keying results on job_id, so the effect is exactly-once.

Suggested table (one row per job):
    jobs(job_id TEXT PRIMARY KEY, payload TEXT,            -- JSON
         status TEXT,                                      -- pending | leased | done | dead
         attempt INTEGER, worker_id TEXT, lease_expires REAL,
         result TEXT, last_error TEXT, enqueued_at REAL, finished_at REAL)

Concurrency: several PROCESSES open the same file. Make lease() atomic with one
`BEGIN IMMEDIATE` transaction (select a free job and claim it before anyone else can), set
`PRAGMA journal_mode=WAL` and a busy timeout so writers wait instead of failing with
"database is locked". Every time read goes through `self.clock()`, so tests can move time
without sleeping.
"""

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Lease:
    job_id: str
    worker_id: str
    attempt: int          # the fencing token
    payload: dict
    expires_at: float


class JobQueue:
    def __init__(self, path: str, clock=time.time, max_attempts: int = 3):
        """Open (and create if needed) the queue at `path`.
        clock: a function returning seconds; tests pass a fake one.
        max_attempts: after this many leases without completion a job becomes 'dead'."""
        self.path = path
        self.clock = clock
        self.max_attempts = max_attempts
        raise NotImplementedError("TODO(you): JobQueue.__init__ (connect, WAL, busy timeout, create table)")

    def enqueue(self, job_id: str, payload: dict) -> bool:
        """Add a pending job. Idempotent: enqueueing an existing job_id changes nothing and
        returns False (so re-planning after a crash never duplicates work). True if added."""
        raise NotImplementedError("TODO(you): enqueue")

    def lease(self, worker_id: str, lease_s: float) -> Lease | None:
        """Atomically claim ONE job: a pending one, or a leased one whose lease has expired.
        Increment attempt, set worker_id and lease_expires = clock() + lease_s.
        A job whose attempts are used up (attempt == max_attempts) and expired becomes 'dead'
        instead of being leased. Return None if nothing is available."""
        raise NotImplementedError("TODO(you): lease")

    def heartbeat(self, lease: Lease, lease_s: float) -> bool:
        """Extend the lease to clock() + lease_s, only if this lease is still the current one
        (same job, same attempt, status leased). False = you lost it: stop working on it."""
        raise NotImplementedError("TODO(you): heartbeat")

    def complete(self, lease: Lease, result: dict) -> bool:
        """Mark done with `result`, only if `lease.attempt` is still the job's current attempt
        (the fencing check). False = a newer attempt owns the job; discard your result."""
        raise NotImplementedError("TODO(you): complete")

    def fail(self, lease: Lease, error: str, retryable: bool) -> None:
        """Record the error. retryable -> back to pending (if attempts remain, else dead);
        not retryable -> dead now. Same fencing check as complete()."""
        raise NotImplementedError("TODO(you): fail")

    def stats(self) -> dict:
        """Counts by status, e.g. {"pending": 3, "leased": 2, "done": 40, "dead": 1}
        (every status key present, 0 if none)."""
        raise NotImplementedError("TODO(you): stats")

    def close(self) -> None:
        raise NotImplementedError("TODO(you): close")
