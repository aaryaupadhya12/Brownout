"""When an episode ends early: limits (Phase 4 + 5) and @stop conditions (Phase 4).

The harness asks once before every model call (src/harness.py:run_episode):

    reason = limits.reached(progress) or first_stop(verifier, task, steps)

The first answer that isn't None ends the episode and becomes trajectory.ended_by.
Limits are checked before stops: a framework budget outranks a task's own rule.

    RolloutLimits   framework budgets, same for every env: turns, tokens, wall-clock seconds
    @stop           env-specific rules, written on the verifier next to @reward / @metric

Both are "soft by one turn": they're checked between turns, so the turn that crosses a
limit still finishes. A hung model call is bounded by the client's own timeout, not here.

Modelled on verifiers' RolloutLimits.reached() and RolloutSession.refused()
"""

import time
from dataclasses import dataclass, field

from .contracts import Usage


@dataclass
class Progress:
    """Running totals for one episode, updated after every model call."""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    started: float = field(default_factory=time.monotonic)   # monotonic: immune to clock changes

    def add(self, usage: Usage) -> None:
        """Count one model call and its tokens."""
        self.turns += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.reasoning_tokens += usage.reasoning_tokens

    @property
    def total_tokens(self) -> int:
        total = 0
        total = self.input_tokens + self.output_tokens + self.reasoning_tokens
        return total

    def elapsed(self) -> float:
        return time.monotonic() - self.started


@dataclass(frozen=True)
class RolloutLimits:
    """Per-episode budgets. None = no cap (max_turns always has one, so a loop always ends)."""
    max_turns: int = 8
    max_total_tokens: int | None = None    # Phase 4: protects the API budget
    max_seconds: float | None = None       # Phase 5: whole-episode wall clock

    def reached(self, progress: Progress) -> str | None:
        """The name of the first limit reached ("max_turns", "max_total_tokens",
        "max_seconds"), or None while within every limit. Check them in that order."""
        if progress.turns >= self.max_turns:
            return "max_turns"
        if(self.max_total_tokens is not None and  progress.total_tokens >= self.max_total_tokens): return "max_total_tokens"
        if(self.max_seconds is not None and progress.elapsed() >= self.max_seconds): return "max_seconds"
        
        return None

    def as_config(self) -> dict:
        """The limits that change behaviour beyond max_turns, for config_hash. Empty when
        only max_turns is set, so runs without new limits keep their old group_id."""
        extra = {"max_total_tokens": self.max_total_tokens, "max_seconds": self.max_seconds}
        return {k: v for k, v in extra.items() if v is not None}


def first_stop(verifier, task, steps) -> str | None:
    """Run the verifier's @stop methods in order: fn(task, steps) -> bool.
    Return the name of the first one that returns True, else None.

    A stop that returns something other than a bool is a bug in the stop: raise
    TypeError(f"@stop {name} must return bool, got {type(result).__name__}").
    The harness records it as ErrorRecord("scorer", ...) and ends the episode.

    Hint: verifier.scorers("stop") gives (name, method) pairs, like for "reward".
    """
    for name, method in verifier.scorers("stop"):
        result = method(task,steps)

        if not isinstance(result,bool):
            raise TypeError(
                "Type Mismaatch in @stop"
            )
        if result:
            return name
    return None
