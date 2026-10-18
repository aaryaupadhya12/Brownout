import unittest
from types import SimpleNamespace

from src.audit.universal import UniversalAudit
from src.interface_v1 import Task


class UniversalCheckTest(unittest.TestCase):
    def setUp(self):
        self.task = Task(id="task-1", seed=0, spec={})
        self.trace = object()

    def target(self, grade, **methods):
        target = SimpleNamespace(
            reward_range=(0.0, 1.0),
            grade=grade,
            gold=lambda task: self.trace,
            floor=lambda task: self.trace,
            **methods,
        )
        return target

    def grader(self, reward=1.0, ok=True):
        return SimpleNamespace(reward=reward, ok=ok)

    def test_determinism_passes_for_identical_rewards(self):
        target = self.target(lambda task, trace: (self.grader(), []))

        finding = UniversalAudit().determinism(target, self.task)

        self.assertEqual(finding.status, "pass")
        self.assertEqual(finding.measured["scores"], [1.0, 1.0, 1.0])

    def test_determinism_fails_for_different_rewards(self):
        rewards = iter((0.0, 1.0, 0.0))
        target = self.target(
            lambda task, trace: (self.grader(next(rewards)), [])
        )

        finding = UniversalAudit().determinism(target, self.task)

        self.assertEqual(finding.status, "fail")

    def test_well_formed_rejects_boolean_and_non_finite_rewards(self):
        rewards = iter((True, float("nan")))
        target = self.target(
            lambda task, trace: (self.grader(next(rewards)), [])
        )

        finding = UniversalAudit().well_formed(target, self.task)

        self.assertEqual(finding.status, "fail")
        self.assertFalse(finding.measured["gold"]["finite_number"])
        self.assertFalse(finding.measured["floor"]["finite_number"])

    def test_well_formed_passes_rewards_inside_range(self):
        target = self.target(lambda task, trace: (self.grader(0.5), []))

        finding = UniversalAudit().well_formed(target, self.task)

        self.assertEqual(finding.status, "pass")

    def test_malformed_passes_when_grader_reports_an_error(self):
        target = self.target(
            lambda task, trace: (self.grader(0.0, ok=False), []),
            malformed=lambda task: self.trace,
        )

        finding = UniversalAudit().malformed(target, self.task)

        self.assertEqual(finding.status, "pass")

    def test_malformed_fails_when_grader_scores_it_normally(self):
        target = self.target(
            lambda task, trace: (self.grader(0.0), []),
            malformed=lambda task: self.trace,
        )

        finding = UniversalAudit().malformed(target, self.task)

        self.assertEqual(finding.status, "fail")


if __name__ == "__main__":
    unittest.main()
