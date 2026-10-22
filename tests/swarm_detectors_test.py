import unittest

from src.audit.swarm import confirm, detectors
from src.audit.swarm.exploit import Exploit
from src.audit.swarm.planted import HOLES, submission
from src.registry import get
from src.audit.target import fake_trace
from src.contracts import Step, Usage

ENV = "cloudops-boutique"
HOLE = {h.hole_id: h for h in HOLES}
ADAPTER = detectors.adapter_for(ENV)


def trace_with(task, tool_outputs):
    """A trajectory whose steps are (tool name, observation dict) pairs, in order."""
    steps = [Step(i, tool, {}, obs, 0.0, 0.0, Usage())
             for i, (tool, obs) in enumerate(tool_outputs)]
    return fake_trace(task, {}, steps=steps)

def a_task():
    return next(iter(get(ENV).tasks().load()))


class DetectorTest(unittest.TestCase):
    def test_old_path_is_blind_on_real_env(self):
        """Documents the bug: verifier == true_success and no adapter -> never a hack."""
        v = HOLE["substring"].verifier()
        task = a_task()
        self.assertFalse(confirm.is_hack(HOLE["substring"].exploit(task), v, v).hack)

    def test_substring_found_without_second_verifier(self):
        v = HOLE["substring"].verifier()
        verdict = confirm.is_hack(HOLE["substring"].exploit(a_task()), v, v, adapter=ADAPTER)
        self.assertTrue(verdict.hack)
        self.assertFalse(verdict.evidence["reward_gap"])
        self.assertEqual(verdict.evidence["primary"], "proxy_gaming")
        self.assertIn("answer_independent", [t["rule"] for t in verdict.evidence["triggers"]])

    def test_any_rank_not_answer_independent(self):
        """Rank 3 holds THIS task's answer, so it must not earn on other tasks."""
        v = HOLE["any-rank"].verifier()
        ex = HOLE["any-rank"].exploit(a_task())
        self.assertFalse(confirm.is_hack(ex, v, v, adapter=ADAPTER).hack)
        real = get(ENV).verifier()
        verdict = confirm.is_hack(ex, v, real, adapter=ADAPTER)
        self.assertTrue(verdict.hack)                 # still caught, by the reward gap
        self.assertEqual(verdict.evidence["triggers"], [])

    def test_fail_open_primary_is_denial(self):
        v = HOLE["fail-open"].verifier()
        verdict = confirm.is_hack(HOLE["fail-open"].exploit(a_task()), v, v, adapter=ADAPTER)
        self.assertTrue(verdict.hack)
        self.assertEqual(verdict.evidence["primary"], "denial_of_evaluation")

    def test_correct_answer_is_not_a_hack(self):
        task = a_task()
        answer = task.spec["answer"]
        ex = Exploit("honest", ENV, task.id, "unlabeled",
                     (("Submit", submission((answer["fault_object"], answer["root_cause"]))),))
        real = get(ENV).verifier()
        verdict = confirm.is_hack(ex, real, real, adapter=ADAPTER)
        self.assertFalse(verdict.hack)
        self.assertEqual(verdict.evidence["triggers"], [])

    def test_leakage_fires_on_label_in_tool_output(self):
        task = a_task()
        label = task.spec["answer"]["root_cause"]
        trace = trace_with(task, [("GetResources", {"output": f"status: {label}"}),
                                  ("Submit", {})])
        found = detectors.leakage(trace, ADAPTER, task)
        self.assertIsNotNone(found)
        self.assertEqual(found.category, "leakage")

    def test_leakage_ignores_submit_echo(self):
        task = a_task()
        label = task.spec["answer"]["root_cause"]
        trace = trace_with(task, [("GetResources", {"output": "all pods running"}),
                                  ("Submit", {"echo": label})])
        self.assertIsNone(detectors.leakage(trace, ADAPTER, task))

    def test_failure_note_withholds_an_honest_answer(self):
        from src.audit.swarm import roles
        task = a_task()
        answer = task.spec["answer"]
        trace = fake_trace(task, {"submission": submission(
            (answer["fault_object"], answer["root_cause"]))})
        trace.grader.reward = 1.0
        note = roles.failure_note(trace, 0, withhold_answer=True)
        self.assertNotIn(answer["root_cause"], note)
        self.assertNotIn(answer["fault_object"], note)
        self.assertNotIn("1.0", note)


if __name__ == "__main__":
    unittest.main()