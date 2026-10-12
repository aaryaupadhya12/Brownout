"""The three pieces every environment plugs into the harness with.

    TaskSource.load()  -> Tasks          where problems come from
    Env                -> tools + state  what the agent acts on
    Verifier.score()   -> reward         reads ONLY (task, trajectory), never the env

The harness (src/harness.py:run_episode) is the one loop that drives all of them.
The env hands its final state to the trajectory through snapshot(), so a verifier
never needs to reach into the env. That is what lets Cloud-OpsBench, whose ground
truth lives in the task and not in the env, use the same harness.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import ErrorRecord, GraderOutput, Trajectory


@dataclass(frozen=True)
class Task:
    """Task data: everything needed to rebuild the problem, nothing runtime.
    frozen stops reassigning fields, but spec is still a dict: never mutate it."""
    id: str
    seed: int
    spec: Mapping[str, Any]   # Brownout: capacity, base_delay, arrivals, degraded, slo_ms, cost_budget, max_steps

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for hashing (task_hash)."""
        return {"id": self.id, "seed": self.seed, "spec": dict(self.spec)}


class TaskSource(Protocol):
    def load(self) -> Iterable[Task]:
        """In: nothing (config/seed live on the source). Out: Tasks."""
        ...


class Env(Protocol):
    def reset(self, task: Task) -> str:
        """In: Task. Out: the opening prompt for the agent."""
        ...

    def step(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """In: tool name + parsed args. Out: tool result dict."""
        ...

    def tool_schema(self) -> list[dict[str, Any]]:
        """Out: schemas the agent is shown."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Out: facts about the final state (no pass/fail), copied into Trajectory.final_state."""
        ...

    def close(self) -> None:
        """Free anything reset() opened. Nothing for Brownout; files or sandboxes later."""
        ...

    @property
    def done(self) -> bool: ...


class Verifier(Protocol):
    """src/scoring.py:Verifier implements this. Metrics are facts (@metric), rewards
    are policy on top of them (@reward, weighted); see src/envs/brownout_verifierv1.py."""
    name: str
    version: str

    def score(self, task: Task, trace: Trajectory) -> tuple[GraderOutput, list[ErrorRecord]]: ...
