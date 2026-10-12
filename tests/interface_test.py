import json, tempfile, unittest
from src.envs.SystemEnv.brownout.v1 import BrownoutEnv, BrownoutTasks
from src.envs.SystemEnv.brownout.verifier import BrownoutVerifierV1
from src.fake_model import FakeModel, call, scripted
from src.harness import run_episode, save_trajectory
from src.scoring import EnvGradeVerifier

TASKS = {t.id: t for t in BrownoutTasks().load()}

SCRIPTS = {
    "good": [scripted(call("c1", "get_metrics", {"zone": "B"})),
             scripted(call("c2", "shift_traffic", {"zone_source": "B", "zone_destination": "A", "split_percentage": 25})),
             scripted(call("c3", "shift_traffic", {"zone_source": "B", "zone_destination": "C", "split_percentage": 20})),
             scripted(call("c4", "submit", {}))],
    "one_shift": [scripted(call("c1", "get_metrics", {"zone": "B"})),
                  scripted(call("c2", "shift_traffic", {"zone_source": "B", "zone_destination": "A", "split_percentage": 40})),
                  scripted(call("c3", "submit", {}))],
    "overload": [scripted(call("c1", "shift_traffic", {"zone_source": "A", "zone_destination": "B", "split_percentage": 100})),
                 scripted(call("c2", "submit", {}))],
    "never_submits": [scripted(call("c1", "get_metrics", {"zone": "A"}))],
    "submit_only": [scripted(call("c1", "submit", {}))],
}


def run(task, script_name, verifier):
    return run_episode(BrownoutEnv(), task, FakeModel(SCRIPTS[script_name]), verifier=verifier)


class TestTaskSource(unittest.TestCase):
    def test_tasks_have_every_threshold(self):
        self.assertEqual(set(TASKS), {"brownout_B", "control"})
        for task in TASKS.values():
            for key in ("slo_ms", "cost_budget", "max_steps", "degraded"):
                self.assertIn(key, task.spec)


class TestVerifierV1(unittest.TestCase):
    def test_same_reward_as_env_grade(self):
        """The v1 verifier reads only (task, trajectory) and must agree with grade()."""
        for task in TASKS.values():
            for name in SCRIPTS:
                with self.subTest(task=task.id, script=name):
                    old = run(task, name, EnvGradeVerifier())
                    new = run(task, name, BrownoutVerifierV1())
                    self.assertTrue(new.grader.ok, new.errors)
                    self.assertAlmostEqual(new.grader.reward, old.grader.reward)

    def test_known_rewards(self):
        self.assertAlmostEqual(run(TASKS["brownout_B"], "good", BrownoutVerifierV1()).grader.reward, 0.95)
        self.assertAlmostEqual(run(TASKS["brownout_B"], "one_shift", BrownoutVerifierV1()).grader.reward, 0.9625)
        self.assertEqual(run(TASKS["brownout_B"], "overload", BrownoutVerifierV1()).grader.reward, 0.0)
        self.assertEqual(run(TASKS["brownout_B"], "never_submits", BrownoutVerifierV1()).grader.reward, 0.0)

    def test_regrade_saved_run_without_env(self):
        traj = run(TASKS["brownout_B"], "good", BrownoutVerifierV1())
        grader, errors = BrownoutVerifierV1().score(TASKS["brownout_B"], traj)   # no env at all
        self.assertEqual(errors, [])
        self.assertAlmostEqual(grader.reward, traj.grader.reward)

    def test_final_state_saved(self):
        traj = run(TASKS["brownout_B"], "good", BrownoutVerifierV1())
        with tempfile.TemporaryDirectory() as d:
            with open(save_trajectory(traj, d)) as f:
                data = json.load(f)
        self.assertTrue(data["final_state"]["submitted"])
        self.assertEqual(data["task_id"], "brownout_B")
        self.assertIn("worst_p99_ms", data["final_state"])


if __name__ == "__main__":
    unittest.main()
