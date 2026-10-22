














import unittest

from src.audit.swarm.loop import role_key
from src.audit.swarm.roles import GENERIC, FOCUS, Specialist


class IdentityTest(unittest.TestCase):
    def test_generic_key_is_unchanged(self):
        """No specialist suffix for the default: old campaigns resume byte-for-byte."""
        self.assertEqual(
            role_key("boutique-admission-1", 0, "attacker", 2, "3f9c1a0b1234"),
            "boutique-admission-1/r0_attacker_a2_3f9c1a0b",
        )

    def test_specialist_adds_suffix(self):
        sp = Specialist("answer-format", FOCUS["answer-format"], model_id="m")
        suffix = sp.key_suffix(20, True, "blind")
        key = role_key("t", 0, "attacker", 0, "3f9c1a0b1234", suffix)
        self.assertEqual(suffix, f"answer-format-{sp.sha(20, True, 'blind')}")
        self.assertTrue(key.endswith(f"_answer-format-{sp.sha(20, True, 'blind')}"))

    def test_focus_changes_identity(self):
        a = Specialist("a", FOCUS["answer-format"], model_id="m")
        b = Specialist("b", FOCUS["episode-end"], model_id="m")
        self.assertNotEqual(a.sha(20, True, "blind"), b.sha(20, True, "blind"))

    def test_model_changes_identity(self):
        cheap = Specialist("p", FOCUS["generic"], model_id="cheap")
        strong = Specialist("p", FOCUS["generic"], model_id="strong")
        self.assertNotEqual(cheap.sha(20, True, "blind"), strong.sha(20, True, "blind"))

    def test_mode_changes_identity(self):
        sp = Specialist("p", FOCUS["generic"], model_id="m")
        self.assertNotEqual(sp.sha(20, True, "blind"), sp.sha(20, True, "whitebox"))

    def test_generic_focus_is_empty(self):
        """So the GENERIC prompt is byte-identical to today's."""
        self.assertEqual(GENERIC.focus, "")
    


if __name__ == "__main__":
    unittest.main()




