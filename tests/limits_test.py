"""Phase 4 (token limits, @stop) and Phase 5 (episode time limit).
Done when all of these pass. Fill in the TODOs in src/limits.py and src/scoring.py."""

import unittest

from src.contracts import Usage
from src.envs.SystemEnv.brownout.v1 import BrownoutEnv, BrownoutTasks
from src.envs.SystemEnv.brownout.verifier import BrownoutVerifierV1
from src.fake_model import FakeModel, call, scripted
from src.harness import run_episode
from src.limits import Progress, RolloutLimits
from src.scoring import stop

TASK = next(iter(BrownoutTasks().load()))          # brownout_B
LOOK = lambda i, zone: scripted(call(f"c{i}", "get_metrics", {"zone": zone}))   # 10 in + 5 out tokens
NEVER_SUBMITS = [LOOK(i, "A") for i in range(20)]


def run(script, verifier=None, limits=None):
    return run_episode(BrownoutEnv(), TASK, FakeModel(script),
                       verifier=verifier or BrownoutVerifierV1(), limits=limits)


class StopsAfterTwoLooks(BrownoutVerifierV1):
    @stop
    def looked_twice(self, task, steps):
        return sum(s.tool == "get_metrics" for s in steps) >= 2


class BrokenStop(BrownoutVerifierV1):
    @stop
    def not_a_bool(self, task, steps):
        return "yes"


class TestProgressAndLimits(unittest.TestCase):
    def test_progress_counts_tokens(self):
        p = Progress()
        p.add(Usage(input_tokens=10, output_tokens=5, reasoning_tokens=2))
        p.add(Usage(input_tokens=10, output_tokens=5))
        self.assertEqual(p.turns, 2)
        self.assertEqual(p.total_tokens, 32)

    def test_reached_names_the_limit(self):
        p = Progress()
        p.add(Usage(input_tokens=100, output_tokens=0))
        self.assertIsNone(RolloutLimits(max_turns=8, max_total_tokens=101).reached(p))
        self.assertEqual(RolloutLimits(max_turns=8, max_total_tokens=100).reached(p), "max_total_tokens")
        self.assertEqual(RolloutLimits(max_turns=1, max_total_tokens=100).reached(p), "max_turns")  # order
        self.assertEqual(RolloutLimits(max_seconds=0.0).reached(Progress()), "max_seconds")
        self.assertIsNone(RolloutLimits(max_seconds=60.0).reached(Progress()))


class TestEpisodeLimits(unittest.TestCase):
    def test_token_limit_ends_episode(self):
        traj = run(NEVER_SUBMITS, limits=RolloutLimits(max_turns=8, max_total_tokens=30))
        self.assertEqual(traj.ended_by, "max_total_tokens")
        self.assertEqual(len(traj.calls), 2)            # 15 + 15 = 30 tokens, then stop
        self.assertEqual(traj.grader.reward, 0.0)       # never submitted

    def test_time_limit_ends_episode(self):
        traj = run(NEVER_SUBMITS, limits=RolloutLimits(max_seconds=0.0))
        self.assertEqual(traj.ended_by, "max_seconds")
        self.assertEqual(len(traj.calls), 0)

    def test_default_is_still_max_turns(self):
        traj = run(NEVER_SUBMITS)
        self.assertEqual(traj.ended_by, "max_turns")
        self.assertEqual(len(traj.calls), 8)

    def test_old_group_ids_unchanged(self):
        """Adding limits must not change runs that don't use them (value from before Phase 4)."""
        traj = run([scripted(call("c", "submit", {}))])
        self.assertEqual(traj.tracking["group_id"], "0c540b4e95259574")

    def test_new_limit_changes_group_id(self):
        plain = run(NEVER_SUBMITS)
        capped = run(NEVER_SUBMITS, limits=RolloutLimits(max_total_tokens=1000))
        self.assertNotEqual(plain.tracking["group_id"], capped.tracking["group_id"])


class TestStops(unittest.TestCase):
    def test_stop_ends_episode_with_its_name(self):
        traj = run([LOOK(1, "A"), LOOK(2, "B"), scripted(call("c3", "submit", {}))],
                   verifier=StopsAfterTwoLooks())
        self.assertEqual(traj.ended_by, "looked_twice")
        self.assertEqual(len(traj.calls), 2)            # stopped before the submit turn
        self.assertEqual(traj.config["stops"], ["looked_twice"])

    def test_limits_win_over_stops(self):
        traj = run([LOOK(1, "A"), LOOK(2, "B")] + NEVER_SUBMITS, verifier=StopsAfterTwoLooks(),
                   limits=RolloutLimits(max_total_tokens=30))
        self.assertEqual(traj.ended_by, "max_total_tokens")

    def test_broken_stop_is_a_recorded_error(self):
        traj = run(NEVER_SUBMITS, verifier=BrokenStop())
        self.assertEqual(traj.ended_by, "error")
        self.assertEqual(traj.errors[0].where, "scorer")
        self.assertEqual(traj.errors[0].type, "TypeError")


if __name__ == "__main__":
    unittest.main()
