"""replay.py must pass an unchanged run and catch every kind of change. Needs Cloud-OpsBench data."""
import contextlib
import copy
import io
import unittest

from swarm_strict_referee_test import have_data

TASK = "boutique-admission-10"
CALLS = [("GetAlerts", {}), ("GetClusterConfiguration", {}),
         ("GetResources", {"resource_type": "pods", "namespace": "boutique"})]


def record():
    """A run recorded straight from the env (no model), saved as a dict like runs/ files."""
    from src.audit.swarm.saved import from_dict
    from src.envs.SystemEnv.cloudops.env import BoutiqueEnv
    from src.envs.SystemEnv.cloudops.replay import replay_one
    from src.envs.SystemEnv.cloudops.tasks import BoutiqueTasks
    from src.envs.SystemEnv.cloudops.verifier import CloudOpsVerifier
    task = next(t for t in BoutiqueTasks().load() if t.id == TASK)
    env, steps = BoutiqueEnv(), []
    env.reset(task)
    answer = task.spec["answer"]
    preds = [{"rank": i, "fault_object": answer["fault_object"], "root_cause": answer["root_cause"]}
             for i in (1, 2, 3)]
    calls = CALLS + [("Submit", {"key_evidence_summary": "x", "top_3_predictions": preds})]
    with contextlib.redirect_stdout(io.StringIO()):
        for tool, args in calls:
            steps.append({"turn": len(steps) + 1, "tool": tool, "args": args,
                          "observation": env.step(tool, args), "model_ms": 0, "tool_ms": 0})
    steps.insert(1, {"turn": 1, "tool": "GetAlerts", "args": {},
                     "observation": {"error": "malformed arguments: Expecting value"},
                     "model_ms": 0, "tool_ms": 0})
    data = {"task_id": TASK, "model_id": "scripted", "run_date": "2026-11-23", "run_index": 0,
            "versions": {}, "steps": steps, "ended_by": "submit", "final_state": env.snapshot(),
            "grader": {"reward": 0.0, "signals": {}, "grader_version": "", "ok": False}}
    out, _ = CloudOpsVerifier().score(task, from_dict(data))
    data["grader"].update(reward=out.reward, ok=out.ok)
    return task, data, from_dict, replay_one


@unittest.skipUnless(have_data(), "set CLOUDOPSBENCH_ROOT to a Cloud-OpsBench checkout")
class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.task, self.data, self.load, self.replay = record()

    def replay_with(self, change):
        data = copy.deepcopy(self.data)
        change(data)
        return self.replay(self.task, self.load(data))

    def test_unchanged_run_replays(self):
        self.assertEqual(self.data["grader"]["reward"], 1.0)
        self.assertIsNone(self.replay_with(lambda d: None))

    def test_changed_tool_output_is_caught(self):
        def edit(d):
            d["steps"][2]["observation"] = {"output": "edited"}
        self.assertIn("step 2 (GetClusterConfiguration)", self.replay_with(edit))

    def test_changed_final_state_is_caught(self):
        self.assertEqual(self.replay_with(lambda d: d["final_state"].update(tool_calls=99)),
                         "final_state differs")

    def test_changed_reward_is_caught(self):
        self.assertIn("saved 0.0", self.replay_with(lambda d: d["grader"].update(reward=0.0)))


if __name__ == "__main__":
    unittest.main()
