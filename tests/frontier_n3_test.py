"""N3: does a valid notebook-compression run ever rank below a failed one? (FrontierSWE v1)

instruction.md:31 says a failed round trip "is a hard FAIL ranked below every valid run".
This feeds hand-made reward.json files to the benchmark's own score_from_reward.py and checks
that promise. Scorer level only: it says nothing about whether a real trial writes these files.

Before you run it: fill PREDICTIONS with the score you expect for each case, from reading
score_from_reward.py:101-104 and :202-205. Commit that, then run.
"""
import unittest

from src.audit import frontier_target as ft

TASK = "notebook-compression"

# What compute_reward.py writes (emit_result at compute_reward.py:55-84): "reward" is the
# geometric-mean ratio for a valid run, None for any failure.
CASES = {
    "failed_round_trip": {"status": "fail", "reward": None, "score": None},
    "good_compressor":   {"status": "ok", "reward": 0.20, "score": 0.20},
    "store_only":        {"status": "ok", "reward": 1.00, "score": 1.00},
    "worse_than_store":  {"status": "ok", "reward": 1.05, "score": 1.05},
}

PREDICTIONS = {        # derived by hand from score_from_reward.py:57, :101-104, :202-205
    "failed_round_trip": 0.0,    # reward None -> raw 0.0 -> not > 0 -> (0.0, None) -> 0.0
    "good_compressor":   0.8,    # (1.0, 1 - 0.20) -> speedup returned as-is
    "store_only":        0.0,    # (1.0, 1 - 1.00) -> 0.0: ties a failure
    "worse_than_store":  -0.05,  # (1.0, 1 - 1.05) -> no clamp, below a failure
}


def leaderboard(scorer, reward_json):
    correctness, speedup = scorer.extract_score(reward_json, TASK)
    return scorer.compute_gated_score(correctness, speedup, TASK)


@unittest.skipUnless(ft.scorer_path().is_file(), "clone Proximal-Labs/frontier-swe next to the repo")
class N3Test(unittest.TestCase):
    def setUp(self):
        self.scorer = ft.load_reference_scorer()
        self.scores = {name: leaderboard(self.scorer, rj) for name, rj in CASES.items()}

    def test_print_scores(self):
        """Always runs: prints what the scorer actually returns, next to your prediction."""
        for name, got in self.scores.items():
            print(f"{name:20s} predicted={PREDICTIONS[name]!s:6s} actual={got:.4f}")

    def test_predictions(self):
        if any(v is None for v in PREDICTIONS.values()):
            self.skipTest("TODO(you): fill PREDICTIONS first")
        for name, want in PREDICTIONS.items():
            self.assertAlmostEqual(self.scores[name], want, places=6, msg=name)

    def test_every_valid_run_ranks_above_a_failure(self):
        """The promise from instruction.md:31. If this fails, N3 is demonstrated at the scorer."""
        failed = self.scores["failed_round_trip"]
        for name in ("good_compressor", "store_only", "worse_than_store"):
            self.assertGreater(self.scores[name], failed, msg=name)


if __name__ == "__main__":
    unittest.main()
