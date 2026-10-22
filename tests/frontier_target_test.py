import unittest

from src.audit import frontier_target as ft


def scorer_available():
    return ft.scorer_path().is_file()


@unittest.skipUnless(scorer_available(), "clone Proximal-Labs/frontier-swe next to the repo")
class PlumbingTest(unittest.TestCase):
    """Boilerplate: passes without any of your TODOs."""

    def setUp(self):
        self.target = ft.FrontierScorerTarget()
        self.tasks = {t.id: t for t in self.target.tasks()}

    def score(self, task_id, reward_json):
        out, errors = self.target.grade(self.tasks[task_id], ft.reward_trace(self.tasks[task_id], reward_json))
        return out.reward, out.ok

    def test_tasks_are_the_supported_ones(self):
        self.assertEqual(tuple(self.tasks), ft.SUPPORTED)

    def test_version_names_the_script(self):
        self.assertTrue(self.target.grader_name().endswith(f"script-{ft.script_sha()[:8]}"))

    def test_gold_earns_exactly_one_on_every_task(self):
        for task in self.tasks.values():
            out, _ = self.target.grade(task, self.target.gold(task))
            self.assertEqual((out.reward, out.ok), (1.0, True), task.id)

    def test_half_correct_scores(self):
        """The numbers behind the partial-credit lesson (see course/ lesson 5-6)."""
        self.assertEqual(self.score("git-to-zig", ft.build_reward_json("git-to-zig", 0.5)), (0.5, True))
        for tid in ("ffmpeg-swscale-rewrite", "pyright-type-checking-optimization"):
            self.assertEqual(self.score(tid, ft.build_reward_json(tid, 0.5)), (0.25, True), tid)

    def test_a_crashing_input_is_recorded_not_scored(self):
        """A scorer that raises must give ok=False, never a quiet 0. (total is text, so the
        division inside extract_score raises TypeError, which Verifier.score records.)"""
        task = self.tasks["ffmpeg-swscale-rewrite"]
        bad = {"reward": 0.0, "additional_data": {"correctness_passed": 1, "correctness_total": "x"}}
        out, errors = self.target.grade(task, ft.reward_trace(task, bad))
        self.assertFalse(out.ok)
        self.assertTrue(errors)


def todo(test, fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except NotImplementedError:
        test.skipTest("TODO(you)")


@unittest.skipUnless(scorer_available(), "clone Proximal-Labs/frontier-swe next to the repo")
class YourCodeTest(unittest.TestCase):
    def setUp(self):
        self.target = ft.FrontierScorerTarget()
        self.tasks = self.target.tasks()

    def test_floor_earns_the_bottom(self):
        for task in self.tasks:
            trace = todo(self, self.target.floor, task)
            out, _ = self.target.grade(task, trace)
            self.assertEqual((out.reward, out.ok), (self.target.reward_range[0], True), task.id)

    def test_every_variant_keeps_the_top(self):
        for task in self.tasks:
            variants = todo(self, self.target.variants, task)
            self.assertTrue(variants, task.id)
            for trace in variants:
                out, _ = self.target.grade(task, trace)
                self.assertEqual((out.reward, out.ok), (self.target.reward_range[1], True), task.id)

    def test_malformed_returns_a_trace(self):
        for task in self.tasks:
            self.assertIsNotNone(todo(self, self.target.malformed, task), task.id)


if __name__ == "__main__":
    unittest.main()
