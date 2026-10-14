"""Cloud-OpsBench env (src/envs/SystemEnv/cloudops/). Done when all pass.

Needs the dataset: clone https://github.com/LLM4Ops/Cloud-OpsBench and set CLOUDOPSBENCH_ROOT
(in .env or the environment). Without it these tests are skipped, not failed.
No API calls: the "model" is a FakeModel.
"""

import json
import os
import unittest

from src.envs.SystemEnv.cloudops.env import BoutiqueEnv, TrainTicketEnv, validate_submission
from src.envs.SystemEnv.cloudops.tasks import (BoutiqueCodeTasks, BoutiqueTasks,
                                               TrainTicketTasks, data_root)
from src.envs.SystemEnv.cloudops.verifier import CloudOpsVerifier, normalize
from src.fake_model import FakeModel, call, scripted
from src.harness import run_episode
from src.registry import names

try:
    ROOT = data_root()
except FileNotFoundError:
    ROOT = None
needs_data = unittest.skipIf(ROOT is None, "set CLOUDOPSBENCH_ROOT to a Cloud-OpsBench checkout")

CASE_ID = "boutique-admission-1"       # answer: app/adservice + missing_service_account


def predictions(first, second=("app/frontend", "pod_failure"), third=("node/worker-01", "node_cordon")):
    return {"key_evidence_summary": "evidence",
            "top_3_predictions": [{"rank": i + 1, "fault_object": o, "root_cause": c}
                                  for i, (o, c) in enumerate([first, second, third])]}


RIGHT = predictions(("app/adservice", "missing_service_account"))


def task(source, task_id):
    return next(t for t in source.load() if t.id == task_id)


def play(*tool_calls, task_id=CASE_ID):
    """Run one episode where the 'model' makes these calls in order, one per turn."""
    model = FakeModel([scripted(call(f"c{i}", name, args)) for i, (name, args) in enumerate(tool_calls)])
    return run_episode(BoutiqueEnv(), task(BoutiqueTasks(), task_id), model,
                       max_turns=20, verifier=CloudOpsVerifier())


class TestNoDataNeeded(unittest.TestCase):
    def test_three_envs_registered(self):
        for name in ("cloudops-boutique", "cloudops-boutique-code", "cloudops-trainticket"):
            self.assertIn(name, names())

    def test_normalize(self):
        self.assertEqual(normalize("  App/AdService "), "app/adservice")
        self.assertEqual(normalize("missing \n  service_account"), "missing service_account")
        self.assertEqual(normalize(None), "")

    def test_validate_submission(self):
        self.assertIsNone(validate_submission(RIGHT))
        bad = [{},
               {**RIGHT, "key_evidence_summary": "  "},
               {**RIGHT, "top_3_predictions": RIGHT["top_3_predictions"][:2]},
               {**RIGHT, "top_3_predictions": [dict(p, rank=1) for p in RIGHT["top_3_predictions"]]},
               {**RIGHT, "top_3_predictions": [dict(RIGHT["top_3_predictions"][0], root_cause="")]
                                              + RIGHT["top_3_predictions"][1:]}]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertIsInstance(validate_submission(payload), str)

    def test_prompt_has_their_spec_but_not_the_text_protocol(self):
        prompt = BoutiqueEnv.system_prompt
        self.assertIn("top_3_predictions", prompt)
        self.assertIn("missing_service_account", prompt)       # List A: valid root causes
        self.assertNotIn("Action Input: <valid JSON object>", prompt)


@needs_data
class TestTasks(unittest.TestCase):
    def test_case_counts(self):
        self.assertEqual(len(list(BoutiqueTasks().load())), 452)
        self.assertEqual(len(list(BoutiqueCodeTasks().load())), 98)
        self.assertEqual(len(list(TrainTicketTasks().load())), 204)

    def test_spec_is_portable_and_holds_the_answer(self):
        t = task(BoutiqueTasks(), CASE_ID)
        self.assertNotIn("/", t.id)
        self.assertFalse(os.path.isabs(t.spec["case_path"]))
        self.assertEqual(t.spec["answer"]["root_cause"], "missing_service_account")

    def test_order_is_numeric(self):
        ids = [t.id for t in BoutiqueTasks().load() if t.spec["category"] == "admission"]
        self.assertLess(ids.index("boutique-admission-9"), ids.index("boutique-admission-10"))


@needs_data
class TestEnv(unittest.TestCase):
    def test_tool_schema(self):
        self.assertEqual(len(BoutiqueEnv().tool_schema()), 13)       # 12 tools + Submit
        self.assertEqual(len(TrainTicketEnv().tool_schema()), 11)    # 10 tools + Submit
        self.assertEqual(BoutiqueEnv().tool_schema(), BoutiqueEnv().tool_schema())   # stable

    def test_reset_hides_the_answer(self):
        text = BoutiqueEnv().reset(task(BoutiqueTasks(), CASE_ID))
        self.assertIn("Partial Service Unreachability", text)
        self.assertIn("boutique", text)
        for secret in ("adservice", "missing_service_account", "Admission_Fault"):
            self.assertNotIn(secret, text)

    def test_tool_output_matches_the_recorded_cache(self):
        env = BoutiqueEnv()
        t = task(BoutiqueTasks(), CASE_ID)
        env.reset(t)
        with open(os.path.join(ROOT, t.spec["case_path"], "tool_cache.json"), encoding="utf-8") as f:
            cache = json.load(f)
        got = env.step("GetResources", {"resource_type": "pods", "namespace": "boutique"})
        self.assertEqual(got, {"output": cache['GetResources:{"resource_type":"pods","name":"","namespace":"boutique"}']})

    def test_model_mistakes_are_errors_not_crashes(self):
        env = BoutiqueEnv()
        env.reset(task(BoutiqueTasks(), CASE_ID))
        self.assertIn("error", env.step("GetAlerts", {"bogus": 1}))          # wrong argument
        self.assertIn("error", env.step("NoSuchTool", {}))
        self.assertIn("error", env.step("Submit", {"top_3_predictions": []}))
        self.assertFalse(env.done)                                            # bad Submit: keep going
        not_found = env.step("DescribeResource", {"resource_type": "pods", "name": "nope",
                                                  "namespace": "boutique"})
        self.assertIn("output", not_found)                                    # kubectl NotFound is data

    def test_submit_ends_the_episode(self):
        env = BoutiqueEnv()
        env.reset(task(BoutiqueTasks(), CASE_ID))
        self.assertEqual(env.step("Submit", RIGHT), {"status": "submitted"})
        self.assertTrue(env.done)
        self.assertEqual(env.snapshot()["submission"], RIGHT)


@needs_data
class TestVerifier(unittest.TestCase):
    """The grader's controls: a right answer, a near miss, a wrong rank, and no answer."""

    def test_right_answer(self):
        traj = play(("GetAlerts", {}), ("Submit", RIGHT))
        self.assertEqual(traj.ended_by, "submit")
        self.assertEqual(traj.grader.reward, 1.0)

    def test_case_and_spaces_do_not_matter(self):
        traj = play(("Submit", predictions(("  App/AdService", "Missing_Service_Account "))))
        self.assertEqual(traj.grader.reward, 1.0)

    def test_right_component_wrong_cause(self):
        traj = play(("Submit", predictions(("app/adservice", "pod_failure"))))
        self.assertEqual(traj.grader.reward, 0.0)
        self.assertEqual(traj.grader.signals["ca"], 1.0)
        self.assertEqual(traj.grader.signals["fa"], 0.0)

    def test_only_rank_1_counts(self):
        traj = play(("Submit", predictions(("app/frontend", "pod_failure"),
                                           second=("app/adservice", "missing_service_account"))))
        self.assertEqual(traj.grader.reward, 0.0)

    def test_null_control_scores_zero(self):
        """Never submits a valid answer: reward 0, and the grader still ran (ok=True)."""
        traj = play(("Submit", {}), ("GetAlerts", {}))
        self.assertEqual(traj.grader.reward, 0.0)
        self.assertTrue(traj.grader.ok)

    def test_redundant_action_rate(self):
        traj = play(("GetAlerts", {}), ("GetAlerts", {}), ("GetClusterConfiguration", {}),
                    ("Submit", RIGHT))
        self.assertAlmostEqual(traj.grader.signals["redundant_action_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
