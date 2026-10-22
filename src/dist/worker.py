"""Worker: one process that pulls jobs until told to stop.

Loop:
    lease = queue.lease(worker_id, lease_s)        None -> sleep briefly (backoff), try again
    start a heartbeat thread: every heartbeat_s, queue.heartbeat(lease, lease_s);
        if it returns False, set a "lost" flag (another attempt owns the job now)
    result = run_fn(lease.payload)                 the episode; may raise, may be slow
    stop the heartbeat thread
    if lost: drop the result (do not save, do not complete)
    else: save(job_id, attempt, result) (atomic write), then queue.complete(lease, result)
    on exception: queue.fail(lease, repr(e), retryable=<infra error?>)

Rules worth defending in an interview:
- heartbeat_s must be well below lease_s (e.g. lease 30 s, heartbeat 10 s), or a healthy but
  slow worker loses its job.
- Save BEFORE complete. If the worker dies between the two, the job is re-run: no result lost.
- The fencing token must reach STORAGE too. With one file per job_id, a frozen worker A can
  wake after B completed and overwrite B's file with its stale result (A's complete() is
  refused, but the file is already wrong). So the file name carries the attempt:
  <job_id>.attempt<k>.json. The queue's done row says which attempt counts; files of stale
  attempts are harmless leftovers (count them, delete them later).
- Model errors are data (reward 0, retryable False); infra errors (timeout, 5xx) are retryable.
"""

import threading


class Worker:
    def __init__(self, queue_path: str, worker_id: str, run_fn, out_dir: str,
                 lease_s: float = 30.0, heartbeat_s: float = 10.0):
        """run_fn(payload: dict) -> dict. out_dir: where results are saved, one file per job_id.
        Open your own JobQueue inside run() (a sqlite connection must not cross processes)."""
        self.queue_path = queue_path
        self.worker_id = worker_id
        self.run_fn = run_fn
        self.out_dir = out_dir
        self.lease_s = lease_s
        self.heartbeat_s = heartbeat_s

    def run(self, stop: threading.Event | None = None, max_jobs: int | None = None) -> int:
        """Process jobs until `stop` is set, max_jobs are done, or the queue is empty for a
        while. Return how many jobs this worker completed."""
        raise NotImplementedError("TODO(you): Worker.run")

    def save(self, job_id: str, attempt: int, result: dict) -> str:
        """Atomic, idempotent write of result to <out_dir>/<job_id>.attempt<attempt>.json
        (write a temp file, then os.replace). Return the path. Same idea as src/store.py."""
        raise NotImplementedError("TODO(you): Worker.save")
