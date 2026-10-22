import unittest
from src.audit.swarm import gate
from src.audit.swarm.planted import HOLES, SubstringMatch
from src.registry import get


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tasks = list(get("cloudops-boutique").tasks().load())[:3]
        self.hole = next(h for h in HOLES if h.hole_id == "substring")
        self.exploits = [self.hole.exploit(t) for t in self.tasks]

    def test_unpatched_verifier_fails_held_out(self):
        """The holey verifier must NOT pass: its own exploits still earn."""
        reasons = gate.held_out_replay(self.hole.verifier(), self.exploits)
        self.assertEqual(len(reasons), len(self.exploits))

    def test_real_verifier_passes_held_out(self):
        """The real verifier blocks the substring exploit: no reasons."""
        reasons = gate.held_out_replay(get("cloudops-boutique").verifier(), self.exploits)
        self.assertEqual(reasons, [])

    def test_crashing_patch_is_reported_as_crash(self):
        """A verifier that raises on the repro is a CRASH, not a block."""
        class Crashes(SubstringMatch):
            def matches(self, task, trace, field):
                raise RuntimeError("boom")
        reasons = gate.held_out_replay(Crashes(), self.exploits)
        self.assertTrue(all("CRASHED" in r for r in reasons))


if __name__ == "__main__":
    main()