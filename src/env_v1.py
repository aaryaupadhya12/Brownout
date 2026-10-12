"""Brownout on the v1 interfaces (src/interface_v1.py).

BrownoutEnv is IncidentEnv plus the two things the interface needs: reset() takes a
Task, and snapshot() reports the final state so the verifier never calls grade().
BrownoutTasks is the TaskSource: the scenarios, with every threshold written out, so
the env and the verifier read the same numbers from task.spec.
"""

from .env import IncidentEnv
from .interface_v1 import Task


BASE = {"arrivals": 900, "capacity": 500, "base_delay": 20,
        "slo_ms": 300, "cost_budget": 0.4, "max_steps": 8}

SCENARIOS = [
    {"id": "brownout_B", "degraded": {"B": 0.5}},
    {"id": "control", "degraded": {}},
]


class BrownoutTasks:
    def __init__(self, scenarios=None, seed=0):
        self.scenarios = scenarios or SCENARIOS
        self.seed = seed

    def load(self):
        for s in self.scenarios:
            spec = dict(BASE)
            for key, value in s.items():
                if key != "id":
                    spec[key] = value
            yield Task(id=s["id"], seed=self.seed, spec=spec)


class BrownoutEnv(IncidentEnv):
    def reset(self, task):
        return super().reset(task.spec)

    def snapshot(self):
        """Facts only. Pass/fail (slo_ok, cost_ok) is the verifier's job."""
        zones = {}
        for z in self.ZONES:
            zones[z] = self.zone_latency(z)

        worst_p99 = 0.0
        overloaded = False
        traffic_moved = 0.0
        for z in self.ZONES:
            worst_p99 = max(worst_p99, zones[z]["p99_ms"])
            if zones[z]["overloaded"]:
                overloaded = True
            traffic_moved += max(0, self.split[z] - self.start_split[z])

        return {"zones": zones,
                "split": dict(self.split),
                "worst_p99_ms": worst_p99,
                "overloaded": overloaded,
                "traffic_moved": traffic_moved,
                "submitted": self.done,
                "steps_used": self.step_used}

    def close(self):
        pass



from .envs.brownout_verifierv1 import BrownoutVerifierV1  # noqa: E402
from .registry import EnvSpec, register  # noqa: E402

register(EnvSpec("brownout", BrownoutTasks, BrownoutEnv, BrownoutVerifierV1))
