import json
import os
import unittest

from src.envs.SystemEnv.cloudops.golden_control import golden_steps, parse_calling
from src.envs.SystemEnv.cloudops.strict import grounding
from src.envs.SystemEnv.cloudops.vendor.process_eval import load_annotation


class ParseCallingTest(unittest.TestCase):
    """Same rules as upstream parse_golden_calling."""

    def test_python_literals(self):
        tool, args = parse_calling("tool_name='GetResources' arguments={namespace: 'boutique', "
                                   "resource_type: 'node', name: None, show_labels: True}")
        self.assertEqual(tool, "GetResources")
        self.assertEqual(args, {"namespace": "boutique", "resource_type": "node", "name": None,
                                "show_labels": True})

    def test_commas_inside_quotes_and_brackets(self):
        _, args = parse_calling("tool_name='X' arguments={q: 'a, b', xs: [1, 2]}")
        self.assertEqual(args, {"q": "a, b", "xs": [1, 2]})

    def test_no_arguments_falls_back_to_tool_name(self):
        self.assertEqual(parse_calling("garbage", "GetAppYAML"), ("GetAppYAML", {}))


def have_data():
    try:
        from src.envs.SystemEnv.cloudops.tasks import data_root
        return os.path.isdir(os.path.join(data_root(), "golden-trajectory"))
    except Exception:
        return False


@unittest.skipUnless(have_data(), "set CLOUDOPSBENCH_ROOT to a Cloud-OpsBench checkout")
class GoldenPathTest(unittest.TestCase):

    def grade(self, category, case, path):
        from src.envs.SystemEnv.cloudops.tasks import data_root
        root = data_root()
        ann = load_annotation(os.path.join(root, "process-label", "boutique", category, case,
                                           "milestone.json"))
        with open(os.path.join(root, "benchmark", "boutique", category, case, "metadata.json"),
                  encoding="utf-8") as f:
            root_cause = json.load(f)["result"]["root_cause"]
        steps = golden_steps(os.path.join(root, "golden-trajectory", "boutique", category, case,
                                          path))
        return grounding(ann, steps, root_cause)

    def test_expert_quota_path_skips_the_quota_object(self):
        # path2 reads "exceeded quota" off the replicaset events, never lists the quota
        g = self.grade("admission", "13", "path2.json")
        self.assertFalse(g["raw"])
        self.assertFalse(g["final"])

    def test_expert_pod_describe_is_rejected_raw_accepted_by_v020(self):
        g = self.grade("startup", "1", "path1.json")
        self.assertFalse(g["raw"])
        self.assertTrue(g["by_pod_config"])


if __name__ == "__main__":
    unittest.main()
