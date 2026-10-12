"""Done when all of these pass. They fail with NotImplementedError until you fill in
the TODOs in src/registry.py and src/run.py."""

import unittest

from src.envs.SystemEnv.brownout.v1 import BrownoutEnv, BrownoutTasks
from src.envs.SystemEnv.brownout.verifier import BrownoutVerifierV1
from src.fake_model import FakeModel, call, scripted
from src.registry import EnvSpec, env_packages, get, names, register
from src.run import run_benchmark

GOOD = [scripted(call("c1", "get_metrics", {"zone": "B"})),
        scripted(call("c2", "shift_traffic", {"zone_source": "B", "zone_destination": "A", "split_percentage": 25})),
        scripted(call("c3", "shift_traffic", {"zone_source": "B", "zone_destination": "C", "split_percentage": 20})),
        scripted(call("c4", "submit", {}))]


class TestRegistry(unittest.TestCase):
    def test_brownout_is_registered(self):
        self.assertIn("brownout", names())
        spec = get("brownout")
        self.assertIs(spec.env, BrownoutEnv)
        self.assertIs(spec.tasks, BrownoutTasks)
        self.assertIs(spec.verifier, BrownoutVerifierV1)

    def test_unknown_name_lists_available(self):
        with self.assertRaises(KeyError) as ctx:
            get("browout")
        self.assertIn("brownout", str(ctx.exception))

    def test_duplicate_name_refused(self):
        with self.assertRaises(ValueError):
            register(EnvSpec("brownout", BrownoutTasks, BrownoutEnv, BrownoutVerifierV1))


    def test_env_folders_are_found_and_template_skipped(self):
        found = env_packages()
        self.assertIn("envs.SystemEnv.brownout", found)
        self.assertFalse(any("_template" in m for m in found))
        self.assertNotIn("template", names())

    def test_system_prompt_comes_from_env(self):
        traj = run_benchmark("brownout", lambda: FakeModel(GOOD), out_dir=None)[0]
        self.assertEqual(traj.messages[0]["payload"]["content"], BrownoutEnv.system_prompt)


class TestRunBenchmark(unittest.TestCase):
    def test_same_rewards_as_interface_test(self):
        trajs = run_benchmark("brownout", lambda: FakeModel(GOOD), trials=2, out_dir=None)
        self.assertEqual(len(trajs), 4)                      # 2 tasks x 2 trials
        by_task = {}
        for t in trajs:
            by_task.setdefault(t.task_id, []).append(t)
        self.assertEqual([t.run_index for t in by_task["brownout_B"]], [0, 1])
        for t in by_task["brownout_B"]:
            self.assertAlmostEqual(t.grader.reward, 0.95)    # same as tests/interface_test.py
        self.assertNotEqual(by_task["brownout_B"][0].trace_id, by_task["brownout_B"][1].trace_id)


if __name__ == "__main__":
    unittest.main()
