"""The audit itself. A check is only trustworthy once it has caught a bug planted on purpose:
each test here builds a tiny Target whose grader is broken in one known way and asserts
the matching check FAILs (and that a correct grader PASSes).

No dataset, no API calls.
"""

import unittest

from src.audit.report import Report
from src.audit.target import Target, fake_trace
from src.audit.universal import UniversalAudit
from src.contracts import GraderOutput
from src.interface_v1 import Task


class ToyTarget(Target):
    """One task, answer "42". The grader is whatever function the test passes in."""

    name = "toy"

    def __init__(self, grade_fn):
        self.grade_fn = grade_fn
        self.task = Task(id="toy-1", seed=0, spec={"answer": "42"})

    def tasks(self):
        return [self.task]

    def grade(self, task, trace):
        reward = self.grade_fn(task, trace.final_state.get("answer"))
        return GraderOutput(reward, {}, "toy"), []

    def gold(self, task):
        return fake_trace(task, {"answer": "42"})

    def floor(self, task):
        return fake_trace(task, {})


def correct(task, answer):
    return 1.0 if answer == task.spec["answer"] else 0.0


def statuses(target, check_id):
    report = Report(env=target.name, layers=["1"], grader="toy")
    UniversalAudit().run(target, target.tasks(), report)
    return [f.status for f in report.findings if f.check_id == check_id]


class OracleTest(unittest.TestCase):
    def test_correct_grader_passes(self):
        self.assertEqual(statuses(ToyTarget(correct), "universal.oracle"), ["pass"])

    def test_grader_rejecting_the_answer_fails(self):
        self.assertEqual(statuses(ToyTarget(lambda task, answer: 0.0), "universal.oracle"), ["fail"])


# TODO(you): one class per check, same shape:
#   FloorTest          a grader returning 1.0 for an empty answer must FAIL floor
#   PerturbationTest   a grader comparing case-sensitively must FAIL on a re-cased variant
#   DeterminismTest    a grader returning random.random() must FAIL determinism
#   WellFormedTest     a grader returning True / None / float("nan") must FAIL well_formed
#   SkipTest           a Target with no gold() must give "skip", never "pass"


if __name__ == "__main__":
    unittest.main()
