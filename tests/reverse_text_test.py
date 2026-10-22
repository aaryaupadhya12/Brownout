"""Spec for src/envs/TextEnv/reverse_text/. Every test SKIPS until the piece it checks is
implemented (NotImplementedError -> skip), so the suite stays green while you build.

    python -m unittest tests.reverse_text_test -v

Order to make them pass: parse_answer -> scorers -> env -> tasks/data -> register -> end to end.
"""

import json
import os
import tempfile
import unittest
from difflib import SequenceMatcher
from types import SimpleNamespace
from unittest import mock

from src.contracts import Step, Usage
from src.envs.TextEnv.reverse_text import data, env as env_mod, tasks as tasks_mod, verifier as v
from src.fake_model import FakeModel, scripted
from src.harness import run_episode
from src.interface_v1 import Task

PROMPT = "The community in Bruck was merged into it"
ANSWER = PROMPT[::-1]
TASK = Task("reverse-text-0", 0, {"prompt": PROMPT, "answer": ANSWER})


def todo(fn, *args, **kwargs):
    """Call fn; a NotImplementedError means 'not written yet' -> skip, not fail."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        raise unittest.SkipTest(str(e))


def trace_with(*assistant_texts):
    """A minimal saved-trace stand-in: just the messages the scorers read."""
    msgs = [{"turn": 0, "kind": "system", "payload": {"role": "system", "content": "s"}},
            {"turn": 0, "kind": "task", "payload": {"role": "user", "content": PROMPT}}]
    for i, text in enumerate(assistant_texts, start=1):
        msgs.append({"turn": i, "kind": "assistant", "payload": {"role": "assistant", "content": text}})
        msgs.append({"turn": i, "kind": "nudge", "payload": {"role": "user", "content": "Use a tool"}})
    return SimpleNamespace(messages=msgs, final_state={})


def upstream_lcs(completion, answer):
    """Reference: upstream's reward, copied, to check parity."""
    m = v.TAG.search(completion or "")
    response = m.group(1).strip() if m else ""
    return SequenceMatcher(None, response, answer).ratio()


class ParseTest(unittest.TestCase):
    def test_extracts_and_strips(self):
        self.assertEqual(todo(v.parse_answer, "<reversed_text> olleh </reversed_text>"), "olleh")

    def test_no_tag_is_empty(self):
        self.assertEqual(todo(v.parse_answer, "olleh"), "")
        self.assertEqual(todo(v.parse_answer, None), "")

    def test_first_block_wins(self):
        self.assertEqual(todo(v.parse_answer, "<reversed_text>a</reversed_text><reversed_text>b</reversed_text>"), "a")

    def test_multiline(self):
        self.assertEqual(todo(v.parse_answer, "<reversed_text>a\nb</reversed_text>"), "a\nb")

    def test_first_assistant_text(self):
        self.assertEqual(todo(v.first_assistant_text, trace_with("one", "two")), "one")
        self.assertIsNone(todo(v.first_assistant_text, trace_with()))


class ScorerTest(unittest.TestCase):
    CASES = [
        f"<reversed_text>{ANSWER}</reversed_text>",          # perfect
        f"<reversed_text>{ANSWER[:-5]}</reversed_text>",     # truncated
        f"<reversed_text>{PROMPT}</reversed_text>",          # not reversed at all
        ANSWER,                                              # right text, no tag
        "",                                                  # nothing
    ]

    def setUp(self):
        self.ver = v.ReverseTextVerifier()

    def test_lcs_matches_upstream_exactly(self):
        for completion in self.CASES:
            with self.subTest(completion=completion[:30]):
                got = todo(self.ver.lcs, TASK, trace_with(completion))
                self.assertAlmostEqual(got, upstream_lcs(completion, ANSWER), places=12)

    def test_grades_the_first_reply_not_the_last(self):
        t = trace_with(f"<reversed_text>{PROMPT}</reversed_text>", f"<reversed_text>{ANSWER}</reversed_text>")
        self.assertLess(todo(self.ver.lcs, TASK, t), 1.0)

    def test_exact_match(self):
        self.assertEqual(todo(self.ver.exact_match, TASK, trace_with(self.CASES[0])), 1.0)
        self.assertEqual(todo(self.ver.exact_match, TASK, trace_with(self.CASES[1])), 0.0)

    def test_has_tag(self):
        self.assertEqual(todo(self.ver.has_tag, TASK, trace_with(self.CASES[0])), 1.0)
        self.assertEqual(todo(self.ver.has_tag, TASK, trace_with(self.CASES[3])), 0.0)

    def test_single_turn_stop(self):
        reply = Step(1, None, {}, {"note": "plain text reply"}, 1.0, 0.0, Usage())
        self.assertFalse(todo(self.ver.single_turn, TASK, []))
        self.assertTrue(todo(self.ver.single_turn, TASK, [reply]))


class EnvTest(unittest.TestCase):
    def test_reset_returns_the_prompt_unchanged(self):
        e = env_mod.ReverseTextEnv()
        self.assertEqual(todo(e.reset, TASK), PROMPT)
        self.assertFalse(e.done)

    def test_no_tools_and_tool_calls_are_errors_not_exceptions(self):
        e = env_mod.ReverseTextEnv()
        todo(e.reset, TASK)
        self.assertEqual(todo(e.tool_schema), [])
        self.assertIn("error", todo(e.step, "submit", {}))

    def test_snapshot_is_a_dict(self):
        e = env_mod.ReverseTextEnv()
        todo(e.reset, TASK)
        self.assertIsInstance(todo(e.snapshot), dict)


class DataAndTasksTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, "train.json")
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump([{"prompt": PROMPT}, {"prompt": "ab c"}], f)

    def tearDown(self):
        self.dir.cleanup()

    def test_cache_is_read_without_network(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            rows = todo(data.fetch_rows, self.cache)
        self.assertEqual(rows, [{"prompt": PROMPT}, {"prompt": "ab c"}])

    def test_tasks_from_cache(self):
        tasks = list(todo(lambda: list(tasks_mod.ReverseTextTasks(self.cache).load())))
        self.assertEqual([t.id for t in tasks], ["reverse-text-0", "reverse-text-1"])
        self.assertEqual(tasks[1].spec, {"prompt": "ab c", "answer": "c ba"})

    def test_limit(self):
        tasks = todo(lambda: list(tasks_mod.ReverseTextTasks(self.cache, limit=1).load()))
        self.assertEqual(len(tasks), 1)


class EndToEndTest(unittest.TestCase):
    def run_one(self, text):
        model = FakeModel([scripted(text=text), scripted(text="second try")])
        traj = todo(run_episode, env_mod.ReverseTextEnv(), TASK, model,
                    verifier=v.ReverseTextVerifier(), max_turns=4)
        return traj, model

    def test_correct_answer_one_call_full_reward(self):
        traj, model = self.run_one(f"<reversed_text>{ANSWER}</reversed_text>")
        self.assertTrue(traj.grader.ok, traj.errors)
        if traj.ended_by != "single_turn":
            self.skipTest(f"ended_by={traj.ended_by}: finish env + single_turn first")
        self.assertEqual(len(model.seen), 1)          # the nudge bought no second try
        self.assertAlmostEqual(traj.grader.reward, 1.0)

    def test_registered(self):
        from src.registry import get
        try:
            spec = get("reverse-text")
        except KeyError:
            self.skipTest("not registered yet: write src/envs/TextEnv/reverse_text/__init__.py")
        self.assertIs(spec.verifier, v.ReverseTextVerifier)


class AuditHypothesesTest(unittest.TestCase):
    """Write these yourself once everything above passes. Each one is a question about the
    reward; the test records the answer (a number), whatever it turns out to be."""

    def test_unreversed_text_gets_partial_credit(self):
        self.skipTest("TODO(you): how much lcs does the prompt itself (not reversed) earn? "
                      "Assert the measured value; then check a few real prompts.")

    def test_strip_asymmetry_false_negative(self):
        self.skipTest("TODO(you): the answer is stripped, the reference is not. Build a task "
                      "whose prompt starts or ends with a space; can a perfect reversal reach 1.0? "
                      "Then count how many of the 1,000 real prompts are affected.")

    def test_missing_tag_scores_zero(self):
        self.skipTest("TODO(you): a perfect reversal without tags: reward? Is that a false "
                      "negative or a fair format rule? Write down your argument.")


if __name__ == "__main__":
    unittest.main()
