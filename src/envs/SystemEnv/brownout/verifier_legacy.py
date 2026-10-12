"""Brownout's grade, split into weighted rewards plus one process metric.

The total equals IncidentEnv.grade() (0.6 * slo + 0.3 * cost + 0.1 * efficiency, and 0 if
any zone is overloaded or the episode was never submitted), but each part is now its own
named reward, so a saved trajectory shows WHY a run scored what it did.
"""

from ....scoring import Verifier, metric, reward


class BrownoutVerifier(Verifier):
    name = "brownout"
    version = "0.3.0"

    def _gate_passed(self, env):
        _, s = env.grade()
        # "submitted" is the key IncidentEnv.grade() writes (env.py)
        return s["submitted"] and not s["overloaded"]

    @reward(weight=0.6)
    def slo(self, task, trace, env):
        _, s = env.grade()
        return s["slo_ok"] if self._gate_passed(env) else 0.0

    @reward(weight=0.3)
    def cost(self, task, trace, env):
        _, s = env.grade()
        return s["cost_ok"] if self._gate_passed(env) else 0.0

    @reward(weight=0.1)
    def efficiency(self, task, trace, env):
        _, s = env.grade()
        return s["efficiency"] if self._gate_passed(env) else 0.0

    @metric
    def state(self, task, trace, env):
        _, s = env.grade()
        return {"worst_p99_ms": s["worst_p99_ms"], "traffic_moved": s["cost"],
                "overloaded": s["overloaded"], "submitted": s["submitted"],
                "slo_ok": s["slo_ok"], "cost_ok": s["cost_ok"]}

    @metric
    def investigated_before_acting(self, task, trace, env):
        """Process check, read from the trace: did get_metrics come before the first
        shift_traffic? 1.0 yes, 0.0 no, None if traffic was never shifted."""
        looked = False
        for s in trace.steps:
            if s.tool == "get_metrics":
                looked = True
            if s.tool == "shift_traffic":
                return 1.0 if looked else 0.0
        return None
