"""Brownout's grade, read from the trajectory instead of the env.

Same numbers as IncidentEnv.grade() (0.6 * slo + 0.3 * cost + 0.1 * efficiency, and 0
if any zone is overloaded or the episode was never submitted), but every input comes
from trace.final_state (facts the env reported) and task.spec (the thresholds).
No env is passed in, so this verifier can re-grade a saved trajectory on its own.
"""

from ..scoring import Verifier, metric, reward


class BrownoutVerifierV1(Verifier):
    name = "brownout"
    version = "1.0.0"
    needs_env = False

    def _gate_passed(self, trace):
        s = trace.final_state
        return s["submitted"] and not s["overloaded"]

    @reward(weight=0.6)
    def slo(self, task, trace):
        if not self._gate_passed(trace):
            return 0.0
        return 1.0 if trace.final_state["worst_p99_ms"] <= task.spec["slo_ms"] else 0.0

    @reward(weight=0.3)
    def cost(self, task, trace):
        if not self._gate_passed(trace):
            return 0.0
        return 1.0 if trace.final_state["traffic_moved"] <= task.spec["cost_budget"] + 1e-9 else 0.0

    @reward(weight=0.1)
    def efficiency(self, task, trace):
        if not self._gate_passed(trace):
            return 0.0
        return max(0, 1 - trace.final_state["steps_used"] / task.spec["max_steps"])

    @metric
    def state(self, task, trace):
        s = trace.final_state
        return {"worst_p99_ms": s["worst_p99_ms"], "traffic_moved": s["traffic_moved"],
                "overloaded": s["overloaded"], "submitted": s["submitted"]}

    @metric
    def investigated_before_acting(self, task, trace):
        """Process check, read from the trace: did get_metrics come before the first
        shift_traffic? 1.0 yes, 0.0 no, None if traffic was never shifted."""
        looked = False
        for s in trace.steps:
            if s.tool == "get_metrics":
                looked = True
            if s.tool == "shift_traffic":
                return 1.0 if looked else 0.0
        return None
