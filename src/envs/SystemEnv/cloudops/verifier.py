"""Cloud-OpsBench scoring, read from the trajectory only.

Outcome (upstream agents/cloudops_agent/evaluation.py:380, prediction_scores):
only rank 1 counts; compare after lowercasing and collapsing whitespace.

    CA   component right   (fault_object)
    FA   fault type right  (root_cause)
    JRA  both right        <- the reward

Why JRA is the reward and not FA: guessing the most common root cause for each of the
6 query texts, with no tool calls, already gets FA ~21% (JRA ~6.6%). RL would learn that
prior. CA and FA are recorded as metrics.

Process metrics (MC, EOC, ECR, EE from upstream evaluation_utils/ + process-label/)
come later: port them as @metric once the outcome scores match upstream.
"""

import json
import re

from ....scoring import Verifier, metric, reward


def normalize(value) -> str:
    if value is None:
        return ""
    
    value = str(value).lower().strip()
    return re.sub(r"\s+", " ", value)


class CloudOpsVerifier(Verifier):
    name = "cloudops"
    version = "0.1.0"
    needs_env = False

    def rank1(self, trace) -> dict | None:
        submission = trace.final_state.get("submission")
        if submission is None:
            return None
        return submission["top_3_predictions"][0]

    def matches(self, task, trace, field: str) -> bool:
        first = self.rank1(trace)
        if first is None:
            return False

        predicted = first[field]
        expected = task.spec["answer"][field]
        return normalize(predicted) == normalize(expected)

 
    @reward(weight=1.0)
    def jra(self, task, trace):
        return 1.0 if self.matches(task, trace, "fault_object") and \
                      self.matches(task, trace, "root_cause") else 0.0

    @metric
    def ca(self, task, trace):
        return 1.0 if self.matches(task, trace, "fault_object") else 0.0

    @metric
    def fa(self, task, trace):
        return 1.0 if self.matches(task, trace, "root_cause") else 0.0

    @metric
    def redundant_action_rate(self, task, trace) -> float:
        seen = set()
        repeated = 0
        total = 0
        for s in trace.steps:
            if s.tool is None or s.tool == "Submit":
                continue
            
            total += 1
            key = (s.tool, json.dumps(s.args, sort_keys=True)) # Any order of the dictionary hyas to be correct 

            if key in seen:
                repeated += 1
            else:
                seen.add(key)
        if total == 0:
            return 0.0
        return repeated / total

    @metric
    def invalid_submits(self, task, trace):
        return trace.final_state.get("invalid_submits", 0)
