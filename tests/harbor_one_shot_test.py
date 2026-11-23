"""The one-shot Harbor agent: code extraction, the saved record, and replay. No model, no
sandbox: a fake environment records the upload. Skips when Harbor is not installed
(uv sync --group harbor)."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from src.harbor_agents.one_shot import SOLUTION, OneShot, last_code_block
except ImportError:
    OneShot = None

REPLY = "Idea first.\n```python\nprint(1)\n```\nFinal:\n```python\nprint(input())\n```"


class FakeEnvironment:
    def __init__(self):
        self.uploads = {}

    async def upload_file(self, source, target):
        self.uploads[target] = Path(source).read_text(encoding="utf-8")


@unittest.skipIf(OneShot is None, "Harbor not installed: uv sync --group harbor")
class OneShotTest(unittest.TestCase):
    def run_agent(self, folder, **options):
        agent = OneShot(logs_dir=Path(folder) / "agent", model_name="m", **options)
        env, context = FakeEnvironment(), SimpleNamespace(n_input_tokens=None, n_output_tokens=None)
        asyncio.run(agent.run("problem", env, context))
        record = json.loads((Path(folder) / "agent" / "one_shot.json").read_text(encoding="utf-8"))
        return env, record

    def test_last_code_block_wins(self):
        self.assertEqual(last_code_block(REPLY), "print(input())\n")
        self.assertIsNone(last_code_block("no code here"))
        self.assertIsNone(last_code_block(None))

    def test_replay_writes_the_saved_answer_with_no_model(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "saved.json"
            saved.write_text(json.dumps({"model": "m", "reply": REPLY}), encoding="utf-8")
            env, record = self.run_agent(folder, replay=str(saved))
        self.assertEqual(env.uploads, {SOLUTION: "print(input())\n"})
        self.assertEqual(record["replayed_from"], str(saved))

    def test_no_code_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "saved.json"
            saved.write_text(json.dumps({"model": "m", "reply": "I give up."}), encoding="utf-8")
            env, record = self.run_agent(folder, replay=str(saved))
        self.assertEqual(env.uploads, {})
        self.assertIsNone(record["code"])

    def test_unknown_option_is_refused(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            OneShot(logs_dir=Path(folder), model_name="m", api_key="not-a-real-key")


if __name__ == "__main__":
    unittest.main()
