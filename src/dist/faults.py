"""Fault injection for experiments: make failure happen on purpose, then measure.

A fake episode stands in for run_episode so experiments are free (no API): it sleeps like a
model call, sometimes raises an infra error, and returns a small result.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultPlan:
    seed: int = 0
    episode_ms: float = 50.0        # mean simulated episode time
    tail_prob: float = 0.05         # chance an episode is 10x slower (tail latency)
    infra_error_prob: float = 0.0   # chance the episode raises a retryable error
    kill_after_s: float | None = None   # coordinator kills one worker after this many seconds


class InfraError(RuntimeError):
    """A retryable failure (timeout, 5xx, lost connection), as opposed to a model failure."""


def fake_episode(payload: dict, plan: FaultPlan) -> dict:
    """Sleep about plan.episode_ms (10x with tail_prob), raise InfraError with
    infra_error_prob, else return {"job_id": payload["job_id"], "reward": <0 or 1>}.
    Use random.Random seeded from (plan.seed, payload["job_id"]) so a rerun of the same job
    behaves the same: deterministic experiments."""
    raise NotImplementedError("TODO(you): fake_episode")
