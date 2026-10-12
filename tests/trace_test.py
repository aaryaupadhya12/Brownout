import json, tempfile, unittest
from src.envs.SystemEnv.brownout.env import IncidentEnv
from src.fake_model import FakeModel, call, scripted
from src.harness import run_episode, save_trajectory
from src.scoring import Verifier, EnvGradeVerifier, metric, reward
from src.envs.SystemEnv.brownout.verifier_legacy import BrownoutVerifier
from src.tracking import task_hash, group_saved_runs

BASE = {"arrivals": 900, "capacity": 500, "base_delay": 20}
BROWNOUT = {**BASE, "id": "brownout_B", "degraded": {"B": 0.5}}
CONTROL = {**BASE, "id": "control", "degraded": {}}


def good_script():
    return [scripted(call("c1", "get_metrics", {"zone": "B"})),
            scripted(call("c2", "shift_traffic", {"zone_source": "B", "zone_destination": "A", "split_percentage": 25})),
            scripted(call("c3", "shift_traffic", {"zone_source": "B", "zone_destination": "C", "split_percentage": 20})),
            scripted(call("c4", "submit", {}))]


class BrokenModel:
    model_id = "broken"
    def complete(self, messages, tools, temperature=0.0):
        raise ConnectionError("provider down")


class RaisingVerifier(Verifier):
    name = "raising"
    @reward(weight=1.0)
    def boom(self, task, trace, env):
        raise ValueError("bad grader")


class WeightedVerifier(Verifier):
    name = "weighted"
    @reward(weight=0.25)
    def always_one(self, task, trace, env):
        return 1.0
    @metric
    def big_number(self, task, trace, env):
        return 1000.0          # a metric: must NOT change the reward


class TestTracking(unittest.TestCase):
    def test_task_hash_ignores_key_order(self):
        a = {"id": "x", "arrivals": 900, "degraded": {"B": 0.5}}
        b = {"degraded": {"B": 0.5}, "arrivals": 900, "id": "x"}
        self.assertEqual(task_hash(a), task_hash(b))
        self.assertNotEqual(task_hash(a), task_hash({**a, "arrivals": 901}))

    def test_same_setup_same_group_new_trace_id(self):
        t1 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), run_index=0)
        t2 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), run_index=1)
        self.assertEqual(t1.tracking["group_id"], t2.tracking["group_id"])
        self.assertNotEqual(t1.trace_id, t2.trace_id)

    def test_temperature_changes_group(self):
        t1 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), temperature=0.0)
        t2 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), temperature=0.7)
        self.assertEqual(t1.tracking["task_hash"], t2.tracking["task_hash"])
        self.assertNotEqual(t1.tracking["group_id"], t2.tracking["group_id"])

    def test_verifier_not_in_group(self):
        t1 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()))
        t2 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), verifier=BrownoutVerifier())
        self.assertEqual(t1.tracking["group_id"], t2.tracking["group_id"])
        self.assertNotEqual(t1.tracking["verifier"], t2.tracking["verifier"])

    def test_group_saved_runs(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                save_trajectory(run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), run_index=i), d)
            save_trajectory(run_episode(IncidentEnv(), CONTROL, FakeModel([scripted(call("c1", "submit", {}))])), d)
            groups = group_saved_runs(d)
        self.assertEqual(len(groups), 2)
        sizes = sorted(g["n"] for g in groups.values())
        self.assertEqual(sizes, [1, 2])


class TestWholeTrace(unittest.TestCase):
    def test_messages_and_calls_recorded(self):
        traj = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()))
        kinds = [m["kind"] for m in traj.messages]
        self.assertEqual(kinds[:4], ["system", "task", "assistant", "tool_result"])
        self.assertEqual(len(traj.calls), 4)                  # one per model call
        self.assertEqual(traj.calls[0].n_tool_calls, 1)
        self.assertIn("started_at", traj.timing)

    def test_saved_file_has_whole_trace(self):
        traj = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()))
        with tempfile.TemporaryDirectory() as d:
            with open(save_trajectory(traj, d)) as f:
                data = json.load(f)
        for key in ("trace_id", "tracking", "config", "messages", "calls", "errors", "timing"):
            self.assertIn(key, data)

    def test_model_error_is_typed_and_still_scored(self):
        traj = run_episode(IncidentEnv(), BROWNOUT, BrokenModel())
        self.assertEqual(traj.ended_by, "error")
        self.assertEqual(traj.errors[0].where, "model")
        self.assertEqual(traj.errors[0].type, "ConnectionError")
        self.assertTrue(traj.grader.ok)                       # grading itself worked


class TestScoring(unittest.TestCase):
    def test_brownout_verifier_matches_env_grade(self):
        t1 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), verifier=EnvGradeVerifier())
        t2 = run_episode(IncidentEnv(), BROWNOUT, FakeModel(good_script()), verifier=BrownoutVerifier())
        self.assertAlmostEqual(t1.grader.reward, 0.95)
        self.assertAlmostEqual(t2.grader.reward, t1.grader.reward)
        self.assertEqual(set(t2.grader.rewards), {"slo", "cost", "efficiency"})
        self.assertEqual(t2.grader.signals["investigated_before_acting"], 1.0)

    def test_metric_not_counted_and_weights_applied(self):
        traj = run_episode(IncidentEnv(), CONTROL, FakeModel([scripted(call("c1", "submit", {}))]),
                           verifier=WeightedVerifier())
        self.assertAlmostEqual(traj.grader.reward, 0.25)
        self.assertEqual(traj.grader.signals["big_number"], 1000.0)
        self.assertEqual(traj.grader.signals["tool_calls"], 1)   # generic metric from the base class

    def test_scorer_error_recorded_not_raised(self):
        traj = run_episode(IncidentEnv(), CONTROL, FakeModel([scripted(call("c1", "submit", {}))]),
                           verifier=RaisingVerifier())
        self.assertFalse(traj.grader.ok)
        self.assertEqual(traj.errors[-1].where, "scorer")
        self.assertEqual(traj.ended_by, "submit")             # the rollout itself was fine


if __name__ == "__main__":
    unittest.main()
