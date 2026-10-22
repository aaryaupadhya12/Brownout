"""Coordinator: plan the jobs, start the workers, watch, measure.

    python -m src.dist.coordinator --jobs 2000 --workers 16 --kill-after 2 --out runs/dist/exp1

(The CLI is yours to write once run_pool works.) The report is the experiment's result:

    {"jobs": 2000, "done": 2000, "dead": 0, "lost": 0, "mismatched": 0, "stale_files": 3,
     "workers": 16, "killed": 1, "wall_s": 12.3, "throughput_per_s": 162.6,
     "episode_p50_ms": 51, "episode_p99_ms": 497, "retries": 17}
"""

from dataclasses import dataclass


def job_id(group_id: str, run_index: int) -> str:
    """Stable id for one episode: the same task + config + trial always gives the same id,
    so planning twice (e.g. after a coordinator crash) enqueues nothing new."""
    raise NotImplementedError("TODO(you): job_id")


def plan_jobs(n_jobs: int) -> list[tuple[str, dict]]:
    """For experiments: n synthetic (job_id, payload) pairs, payload {"job_id": ...}.
    Later: build them from real Tasks with src/tracking.py group ids."""
    raise NotImplementedError("TODO(you): plan_jobs")


@dataclass
class PoolReport:
    jobs: int
    done: int
    dead: int
    lost: int                 # jobs neither done nor dead at the end
    mismatched: int           # done jobs whose file for the completing attempt is missing or
                              # differs from the result the queue recorded (must be 0)
    stale_files: int          # result files of attempts that did not complete (harmless)
    workers: int
    killed: int
    wall_s: float
    throughput_per_s: float
    episode_p50_ms: float
    episode_p99_ms: float
    retries: int


def run_pool(queue_path: str, out_dir: str, n_workers: int, plan, n_jobs: int) -> PoolReport:
    """Enqueue n_jobs, start n_workers multiprocessing.Process workers running fake_episode
    with `plan`, kill one with Process.kill() after plan.kill_after_s if set, wait until no
    job is pending or leased, join everything, and compute the report.

    Think about: how do you KNOW nothing was lost or corrupted? For every done row, read the
    file <job_id>.attempt<k>.json for the attempt that completed and compare it with the
    result stored in the queue (mismatched); count the other result files (stale_files).
    You will need a way to read the done rows: add a small JobQueue method for it."""
    raise NotImplementedError("TODO(you): run_pool")
