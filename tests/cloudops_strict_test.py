import os
import unittest

from src.audit.target import fake_trace
from src.contracts import Step, Usage
from src.envs.SystemEnv.cloudops import strict
from src.envs.SystemEnv.cloudops.vendor.process_eval import evaluate
from src.interface_v1 import Task

NS = "boutique"


def step(tool, args, output="ok"):
    return Step(0, tool, args, {"output": output}, 0.0, 0.0, Usage())


def submission(fault_object, root_cause):
    preds = [{"rank": i, "fault_taxonomy": "x", "fault_object": fault_object if i == 1 else "app/other",
              "root_cause": root_cause if i == 1 else "other"} for i in (1, 2, 3)]
    return {"submission": {"top_3_predictions": preds}}


# A hand-made label: M0 -> M1 -> M2, so M2 is the only answer milestone.
CHAIN = {
    "case_id": "toy/1",
    "milestones": [
        {"id": "M0", "admissible_tool_uses": [{"tool_name": "GetAlerts", "arguments": {}}]},
        {"id": "M1", "admissible_tool_uses": [
            {"tool_name": "GetResources", "arguments": {"resource_type": ["pods", "pod"]}}]},
        {"id": "M2", "admissible_tool_uses": [
            {"tool_name": "GetAppYAML", "arguments": {"app_name": "adservice"},
             "evidence_patterns": [{"kind": "literal", "value": "serviceAccountName"}]}]},
    ],
    "edges": [("M0", "M1"), ("M1", "M2")],
    "formula": {"all": ["M0", "M1", "M2"]},
}


class ProcessEvalTest(unittest.TestCase):
    """The port of upstream's matcher + evaluator, on a toy label (no data needed)."""

    def test_in_order_with_evidence_closes(self):
        p = evaluate(CHAIN, [("GetAlerts", {}, "a"), ("GetResources", {"resource_type": "po"}, "b"),
                             ("GetAppYAML", {"app_name": "adservice"}, "serviceAccountName: x")])
        self.assertEqual((p["mc"], p["eoc"], p["ecr"], p["ee"]), (1.0, 1.0, 1.0, 1.0))

    def test_alias_matches(self):
        """'po' in the call matches 'pods' in the label via the resource aliases."""
        p = evaluate(CHAIN, [("GetResources", {"resource_type": "po"}, "")])
        self.assertIn("M1", p["established"])

    def test_evidence_pattern_must_appear_in_output(self):
        p = evaluate(CHAIN, [("GetAppYAML", {"app_name": "adservice"}, "nothing useful")])
        self.assertNotIn("M2", p["established"])

    def test_out_of_order_lowers_eoc_not_mc(self):
        p = evaluate(CHAIN, [("GetAppYAML", {"app_name": "adservice"}, "serviceAccountName"),
                             ("GetResources", {"resource_type": "pods"}, ""), ("GetAlerts", {}, "")])
        self.assertEqual(p["mc"], 1.0)
        self.assertLess(p["eoc"], 1.0)

    def test_broken_edge_blocks_closure(self):
        """M2 held but its prerequisite M1 never: ECR 0 even if the formula were otherwise met."""
        p = evaluate(CHAIN, [("GetAlerts", {}, ""),
                             ("GetAppYAML", {"app_name": "adservice"}, "serviceAccountName")])
        self.assertEqual(p["ecr"], 0.0)

    def test_answer_milestones_are_sinks(self):
        self.assertEqual(strict.answer_milestones(CHAIN), {"M2"})

    def test_alternative_paths_need_only_one_answer(self):
        """Shape of boutique/service/12. Sinks are M2a and M2c, but the formula accepts M2a OR
        (M2b and M2c). Regression: requiring every sink rejected 12 grounded DeepSeek runs."""
        ann = {"milestones": [{"id": i} for i in ("M1", "M2a", "M2b", "M2c")],
               "edges": [("M1", "M2a"), ("M1", "M2b"), ("M2b", "M2c")],
               "formula": {"all": ["M1", {"any": ["M2a", {"all": ["M2b", "M2c"]}]}]}}
        self.assertEqual(strict.answer_milestones(ann), {"M2a", "M2c"})
        self.assertTrue(strict.answer_grounded(ann, {"M1", "M2a", "M2b"}))
        self.assertTrue(strict.answer_grounded(ann, {"M2c"}))
        self.assertFalse(strict.answer_grounded(ann, {"M1", "M2b"}))

    # ----- the two v0.2.0 deviations (grounded_jra only) -----

    NODE_LABEL = {"milestones": [{"id": "M1", "admissible_tool_uses": [
        {"tool_name": "GetResources", "arguments": {"namespace": "boutique",
                                                    "resource_type": ["nodes"], "show_labels": True}}]}],
        "edges": [], "formula": {"all": ["M1"]}}
    APP_CONFIG = {"milestones": [{"id": "M1", "admissible_tool_uses": [
        {"tool_name": "GetAppYAML", "arguments": {"app_name": "cart"},
         "evidence_patterns": [{"kind": "literal", "value": "8Gi"}]}]}],
        "edges": [], "formula": {"all": ["M1"]}}

    def test_namespace_ignored_on_nodes(self):
        steps = [("GetResources", {"resource_type": "nodes", "show_labels": True}, "LABELS")]
        g = strict.grounding(self.NODE_LABEL, steps, "node_selector_mismatch")
        self.assertEqual((g["raw"], g["final"], g["by_namespace"]), (False, True, True))

    def test_namespace_fix_never_drops_show_labels(self):
        """Case 5: nodes listed WITHOUT show_labels stay unconfirmed."""
        steps = [("GetResources", {"resource_type": "nodes"}, "NAME STATUS")]
        self.assertFalse(strict.grounding(self.NODE_LABEL, steps, "node_selector_mismatch")["final"])

    def test_pod_describe_counts_as_app_config(self):
        steps = [("DescribeResource", {"resource_type": "pod", "name": "cart-7c8d-x"}, "memory: 8Gi")]
        g = strict.grounding(self.APP_CONFIG, steps, "memory_capacity_mismatch")
        self.assertEqual((g["raw"], g["final"], g["by_pod_config"]), (False, True, True))

    def test_pod_config_needs_the_evidence_pattern(self):
        steps = [("DescribeResource", {"resource_type": "pod", "name": "cart-7c8d-x"}, "memory: 64Mi")]
        self.assertFalse(strict.grounding(self.APP_CONFIG, steps, "memory_capacity_mismatch")["final"])

    def test_pod_config_not_for_fields_a_pod_describe_hides(self):
        """Affinity is not printed by describe pod, so the YAML is still required."""
        steps = [("DescribeResource", {"resource_type": "pod", "name": "cart-7c8d-x"}, "memory: 8Gi")]
        self.assertFalse(strict.grounding(self.APP_CONFIG, steps, "node_affinity_mismatch")["final"])

    def test_pod_config_only_for_the_same_app(self):
        steps = [("DescribeResource", {"resource_type": "pod", "name": "cartservice-1"}, "8Gi")]
        self.assertFalse(strict.grounding(self.APP_CONFIG, steps, "memory_capacity_mismatch")["final"])

    def test_no_steps_is_zero_not_crash(self):
        p = evaluate(CHAIN, [])
        self.assertEqual((p["mc"], p["ecr"], p["ee"]), (0.0, 0.0, 0.0))


def have_data():
    try:
        from src.envs.SystemEnv.cloudops.tasks import data_root
        return os.path.isdir(os.path.join(data_root(), "process-label"))
    except Exception:
        return False


@unittest.skipUnless(have_data(), "set CLOUDOPSBENCH_ROOT to a Cloud-OpsBench checkout")
class StrictVerifierTest(unittest.TestCase):
    """boutique/admission/1: missing_service_account on app/adservice. Its label has no evidence
    patterns, so only the tool name and arguments matter here."""

    def setUp(self):
        self.task = Task(id="boutique-admission-1", seed=0, spec={
            "system": "boutique", "system_dir": "boutique", "category": "admission", "case_id": "1",
            "answer": {"fault_taxonomy": "Admission_Fault", "fault_object": "app/adservice",
                       "root_cause": "missing_service_account"}})
        self.v = strict.StrictCloudOpsVerifier()
        self.right = submission("app/adservice", "missing_service_account")
        self.wrong = submission("app/cartservice", "pod_cpu_overload")
        self.diagnosis = [
            step("GetErrorLogs", {"namespace": NS, "service_name": "frontend"}),
            step("GetResources", {"namespace": NS, "resource_type": "deployments"}),
            step("GetResources", {"namespace": NS, "resource_type": "events"}),
            step("GetAppYAML", {"app_name": "adservice"}),
        ]

    def grade(self, steps, final_state):
        out, errors = self.v.score(self.task, fake_trace(self.task, final_state, steps))
        return out, errors

    def test_right_answer_with_evidence_earns(self):
        out, errors = self.grade(self.diagnosis, self.right)
        self.assertEqual((out.reward, out.ok, errors), (1.0, True, []))
        self.assertEqual(out.signals["ecr"], 1.0)

    def test_right_answer_without_tools_is_a_guess(self):
        """JRA 1 from the training verifier, 0 here: the gap this verifier exists to measure."""
        out, _ = self.grade([], self.right)
        self.assertEqual((out.signals["jra"], out.reward), (1.0, 0.0))

    def test_full_evidence_wrong_answer_earns_nothing(self):
        """Closure without diagnosis: ECR 1, reward 0. Why ECR alone is not true success."""
        out, _ = self.grade(self.diagnosis, self.wrong)
        self.assertEqual((out.signals["ecr"], out.reward), (1.0, 0.0))

    def test_missing_label_is_recorded_not_zero(self):
        self.task = Task(id="boutique-admission-999999", seed=0,      # spec is never mutated
                         spec={**self.task.spec, "case_id": "999999"})
        out, errors = self.grade(self.diagnosis, self.right)
        self.assertFalse(out.ok)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
