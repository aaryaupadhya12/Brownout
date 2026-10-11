from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Task:
    """Task data: everything needed to rebuild the problem, nothing runtime."""
    id: str
    seed: int
    spec: Mapping[str, Any]   # Brownout: capacity, base_delay, arrivals, degraded, slo_ms, cost_budget, max_steps


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Trajectory:
    """The ONLY thing the verifier sees besides Task."""
    steps: list[Step] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)  # env.snapshot() taken at episode end
    submitted: bool = False
    info: dict[str, Any] = field(default_factory=dict)         # scratch, judge calls go here later
    extra_usage: list[Any] = field(default_factory=list)       # judge token usage, kept out of agent usage


@dataclass(frozen=True)
class Reward:
    score: float
    weight: float = 1.0


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
        """Out: read-only copy of final state for the Trajectory."""
        ...

    @property
    def done(self) -> bool: ...


class Judge(Protocol):
    name: str
    weight: float

    def score(self, task: Task, trajectory: Trajectory) -> float | Mapping[str, float]: ...


class Verifier(Protocol):
    judges: list[Judge]

    def metrics(self, task: Task, trajectory: Trajectory) -> dict[str, float]:
        """Facts only: worst_p99_ms, cost, overloaded, submitted, steps_used."""
        ...

    def rewards(self, task: Task, trajectory: Trajectory) -> dict[str, Reward]:
        """Policy on top of metrics: slo_ok, cost_ok, efficiency, plus the gate."""
        ...


def run_episode(task: Task,env: Env,agent: Callable[[str, list[dict[str, Any]], list[Step]], tuple[str, dict[str, Any]]],
    verifier: Verifier,
    hard_cap : int = 100,
) -> tuple[Trajectory, dict[str, float], dict[str, Reward]]:
    prompt = env.reset(taks)
    schema = env.tool_schema()
    traj = Trajectory()
    while not env.done and len(traj.steps) > hard_cap:
        tool , args = agent(prompt,schema,traj.steps)
        result = env.step(tool,args)
        traj.steps.append(Step(tool,args,result))
    traj.final_state = env.snapshot()
    traj.submitted = any(s.tool == "submit" and "error" not in s.result for s in traj.steps)
    metrics = Verifier.metrics(task,traj)
    rewards = Verifier.rewards(task,traj)
    return traj , metrics, rewards