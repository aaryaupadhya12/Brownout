import tempfile
import unittest

from src.audit.swarm import fixer, gate
from src.audit.swarm.identity import verifier_sha
from src.audit.swarm.loop import SwarmConfig
from src.audit.swarm.planted import HOLES
from src.fake_model import FakeModel, scripted
from src.registry import get

ENV = "cloudops-boutique"
HOLE = next(h for h in HOLES if h.hole_id == "substring")

GOOD_PATCH = '''import re

class Patch(Base):
    def matches(self, task, trace, field):
        first = self.rank1(trace)
        if first is None:
            return False
        return re.sub(r"\\s+", " ", str(first[field]).lower().strip()) == \\
               re.sub(r"\\s+", " ", str(task.spec["answer"][field]).lower().strip())
'''


def fenced(code):
    return f"Short reason.\n```python\n{code}```\n"


def tasks(n=3):
    return list(get(ENV).tasks().load())[:n]


def a_row(task):
    ex = HOLE.exploit(task)
    calls = [[n, dict(a, key_evidence_summary="SECRET-OPINION")] for n, a in ex.calls]
    return {"exploit_id": ex.exploit_id, "task_id": task.id, "signature": "s1", "label": "hack",
            "repro": {**ex.as_row(), "calls": calls},
            "evidence": {"unpatched": {"reward": 1.0}, "true_success": {"reward": 0.0},
                         "triggers": [{"rule": "answer_independent"}]}}


class PlumbingTest(unittest.TestCase):
    """Boilerplate: passes without any of your TODOs."""

    def test_extract_patch_takes_the_last_block(self):
        text = "```python\nx = 1\n```\nno wait\n```python\ny = 2\n```"
        self.assertEqual(fixer.extract_patch(text), "y = 2\n")
        self.assertIsNone(fixer.extract_patch("no code here"))

    def test_finding_text_drops_the_testers_opinion(self):
        text = fixer.describe_finding(a_row(tasks(1)[0]))
        self.assertNotIn("SECRET-OPINION", text)
        self.assertIn("Submit", text)

    def test_prompt_has_no_labels(self):
        prompt = fixer.build_prompt(HOLE.verifier(), "finding", fixer.CLOUDOPS_CONTRACT)
        for word in ("SubstringMatch", "planted", "Hole"):
            self.assertNotIn(word, prompt)
        self.assertIn("Base", prompt)

    def test_reply_is_cached(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SwarmConfig(campaign="c", out_dir=d)
            model = FakeModel([scripted(text="first"), scripted(text="second")])
            a = fixer.cached_reply(cfg, "k", lambda: fixer.ask_fixer(model, "p"))
            b = fixer.cached_reply(cfg, "k", lambda: fixer.ask_fixer(model, "p"))
            self.assertEqual((a["text"], b["text"]), ("first", "first"))
            self.assertEqual(len(model.seen), 1)

    def test_held_out_excludes_the_same_trick(self):
        t = tasks(3)
        rows = [a_row(t[0]), a_row(t[1]), a_row(t[2])]
        rows[2]["signature"] = "s2"
        held = fixer.held_out_from_ledger(rows, rows[0]["exploit_id"])
        self.assertEqual([e.exploit_id for e in held], [rows[2]["exploit_id"]])


def todo(test, fn, *args, **kw):
    """Run fn; if it is still a TODO, skip. The test turns on when you fill it in."""
    try:
        return fn(*args, **kw)
    except NotImplementedError:
        test.skipTest("TODO(you)")


class YourCodeTest(unittest.TestCase):
    def test_validate_refuses_dangerous_source(self):
        for bad in ("import os\nclass Patch(Base): pass\n",
                    "class Patch(Base):\n    def matches(self, t, r, f):\n        return open('x')\n",
                    "class Patch(Base):\n    def matches(self, t, r, f):\n        return f.__class__\n",
                    "x = 1\n"):
            self.assertTrue(todo(self, fixer.validate_patch, bad), bad)
        self.assertEqual(fixer.validate_patch(GOOD_PATCH), [])

    def test_loaded_patch_has_its_own_identity(self):
        unpatched = HOLE.verifier()
        with tempfile.TemporaryDirectory() as d:
            cfg = SwarmConfig(campaign="c", out_dir=d)
            path = fixer.patch_path(cfg, "t", "e", 0)
            patched = todo(self, fixer.load_patch, GOOD_PATCH, path, type(unpatched))
            self.assertNotEqual(verifier_sha(patched), verifier_sha(unpatched))

    def test_verdict_is_a_gate_and_a_score(self):
        v = gate.PatchVerdict(held_out_total=4, blocked=4, solves_total=10)
        bad = gate.PatchVerdict(held_out_total=4, blocked=4, solves_total=10, solves_lost=3)
        self.assertTrue(todo(self, v.ok))
        self.assertFalse(bad.ok())
        self.assertGreater(v.score(), bad.score())
        self.assertFalse(gate.PatchVerdict().ok())        # nothing held out: nothing proven

    def test_good_patch_is_better_and_the_holey_one_is_not(self):
        t = tasks(3)
        held = [HOLE.exploit(x) for x in t]
        unpatched = HOLE.verifier()
        with tempfile.TemporaryDirectory() as d:
            cfg = SwarmConfig(campaign="c", out_dir=d)
            good = todo(self, fixer.load_patch, GOOD_PATCH,
                        fixer.patch_path(cfg, "t", "e", 0), type(unpatched))
        verdict = todo(self, gate.is_better_patch, unpatched, good, t, held, [])
        self.assertEqual(verdict.blocked, len(held))
        self.assertEqual(verdict.crashed, 0)
        same = todo(self, gate.is_better_patch, unpatched, unpatched, t, held, [])
        self.assertEqual(same.blocked, 0)
        self.assertFalse(same.ok())

    def test_fix_finding_end_to_end_with_a_scripted_fixer(self):
        t = tasks(3)
        unpatched = HOLE.verifier()
        model = FakeModel([scripted(text=fenced(GOOD_PATCH))])
        with tempfile.TemporaryDirectory() as d:
            cfg = SwarmConfig(campaign="c", out_dir=d)
            result = todo(self, fixer.fix_finding, cfg, model, a_row(t[0]), unpatched,
                          [HOLE.exploit(t[1]), HOLE.exploit(t[2])], [], t)
            self.assertEqual(result.status, "candidate")
            self.assertEqual(len(model.seen), 1)
            # the fixer was never shown the held-out cases
            shown = str(model.seen[0])
            self.assertNotIn(t[1].id, shown)


if __name__ == "__main__":
    unittest.main()
