"""Offline tests for the audit judge: FakeModel replies, a stub verifier, no data, no API."""
import os
import tempfile
import unittest

from src.audit.target import fake_trace
from src.contracts import Step, Usage
from src.envs.SystemEnv.cloudops import audit_judge as aj
from src.fake_model import FakeModel, scripted
from src.interface_v1 import Task

NOTE = "weak: SECRET-REFERENCE-NOTE that must never reach the judge"


class StubVerifier:
    """annotation/process without reading label files."""
    ann = {"milestones": [{"id": "M1"}, {"id": "M2", "role": "root_cause_x",
                           "admissible_tool_uses": [{"tool_name": "GetAppYAML",
                                                     "arguments": {"app_name": "cart"}}]}],
           "edges": [("M1", "M2")], "formula": {"all": ["M1", "M2"]}}

    def annotation(self, task):
        return self.ann

    def process(self, task, trace):
        return {"established": ["M1"]}


def toy():
    task = Task(id="boutique-x-1", seed=0, spec={"answer": {"fault_object": "app/cart",
                                                           "root_cause": "some_fault"}})
    steps = [Step(0, "DescribeResource", {"name": "cart"}, {"output": "Events: FailedScheduling"},
                  0.0, 0.0, Usage())]
    return task, fake_trace(task, {}, steps)


class ParseTest(unittest.TestCase):
    def test_label_with_prose_around(self):
        self.assertEqual(aj.parse_label('ok ```json\n{"label": "Label_Gap", "reason": "r"}\n```'),
                         ("label_gap", "r"))

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            aj.parse_label('{"label": "maybe"}')

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            aj.parse_label("I think it is blind")


class RescuedModeTest(unittest.TestCase):
    def test_rescued_labels_parse(self):
        self.assertEqual(aj.parse_label('{"label": "not_confirmed", "reason": "r"}', aj.LABELS_RESCUED),
                         ("not_confirmed", "r"))
        with self.assertRaises(ValueError):                 # unconfirmed-mode label in rescued mode
            aj.parse_label('{"label": "label_gap"}', aj.LABELS_RESCUED)

    def test_rescued_prompt_names_the_rule(self):
        task, trace = toy()
        ev = {**aj.evidence(StubVerifier(), task, trace), "accepted_by": "RULE-X"}
        p = aj.build_prompt(ev, "rescued")
        self.assertIn("RULE-X", p)
        self.assertIn("not_confirmed", p)

    def test_modes_never_share_cache_keys(self):
        self.assertNotEqual(aj.cache_key("m", "1", "p"), aj.cache_key("m", "1r", "p"))


class PromptTest(unittest.TestCase):
    def test_prompt_has_evidence_and_wanted_check(self):
        task, trace = toy()
        p = aj.build_prompt(aj.evidence(StubVerifier(), task, trace))
        self.assertIn("app/cart / some_fault", p)
        self.assertIn("GetAppYAML", p)            # the wanted check
        self.assertIn("FailedScheduling", p)      # the tool output

    def test_blind_reference_note_never_in_prompt(self):
        """evidence() takes no sheet row at all, so the other labeller's note cannot leak."""
        task, trace = toy()
        p = aj.build_prompt(aj.evidence(StubVerifier(), task, trace))
        self.assertNotIn("SECRET-REFERENCE-NOTE", p)

    def test_long_output_keeps_the_tail(self):
        text = "H" * 5000 + "Events: the end"
        self.assertTrue(aj.cut(text).endswith("Events: the end"))
        self.assertLess(len(aj.cut(text)), len(text))


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cache = aj.DiskCache(os.path.join(self.dir, "c.jsonl"))

    def test_second_call_is_free_and_identical(self):
        model = FakeModel([scripted(text='{"label": "blind", "reason": "x"}')])
        first = aj.judge_one(model, self.cache, "prompt")
        second = aj.judge_one(model, aj.DiskCache(self.cache.path), "prompt")   # reloaded from disk
        self.assertEqual((first["label"], second["label"], second["cached"]), ("blind", "blind", True))
        self.assertEqual(len(model.seen), 1)

    def test_unparseable_reply_is_recorded_not_labeled(self):
        model = FakeModel([scripted(text="no idea")])
        row = aj.judge_one(model, self.cache, "prompt")
        self.assertIsNone(row["label"])
        self.assertTrue(row["error"])

    def test_failed_reply_is_retried_not_reused(self):
        model = FakeModel([scripted(text=""), scripted(text='{"label": "blind", "reason": "x"}')])
        aj.judge_one(model, self.cache, "prompt")
        row = aj.judge_one(model, aj.DiskCache(self.cache.path), "prompt")
        self.assertEqual((row["label"], len(model.seen)), ("blind", 2))


class AgreementTest(unittest.TestCase):
    def test_reference_mapping(self):
        self.assertEqual(aj.reference_label({"label": "symptom", "note": NOTE}), "symptom_weak")
        self.assertEqual(aj.reference_label({"label": "symptom", "note": "strong: x"}), "symptom_strong")
        self.assertEqual(aj.reference_label({"label": "label_gap", "note": ""}), "label_gap")

    def test_kappa_perfect_and_chance(self):
        self.assertEqual(aj.cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(aj.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)

    def test_kappa_punishes_majority_label(self):
        """18/20 raw agreement on a 90%-one-label sheet is far from kappa 0.9."""
        ref = ["s"] * 18 + ["g", "g"]
        jud = ["s"] * 18 + ["s", "s"]
        self.assertEqual(aj.cohen_kappa(ref, jud), 0.0)

    def test_collapse_merges_symptoms(self):
        a = aj.agreement([("symptom_weak", "symptom_strong"), ("label_gap", "label_gap")])
        self.assertEqual((a["raw4"], a["kappa3"]), (0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
