"""Secrets (src/providers.py). Done when all pass. No real keys and no network needed."""

import os
import tempfile
import unittest
from unittest import mock

from src.providers import load_env_file, load_providers, resolve

PROVIDERS = {"nebius": {"base_url": "https://neb.example/v1", "api_key_env": "TEST_NEBIUS_KEY"},
             "local": {"base_url": "http://localhost:30000/v1", "api_key_env": ""}}
SECRET = "sk-test-DO-NOT-LEAK-123"


class TestResolve(unittest.TestCase):
    def test_reads_key_from_environment(self):
        with mock.patch.dict(os.environ, {"TEST_NEBIUS_KEY": SECRET}):
            self.assertEqual(resolve("nebius", PROVIDERS), ("https://neb.example/v1", SECRET))

    def test_no_key_needed(self):
        self.assertEqual(resolve("local", PROVIDERS), ("http://localhost:30000/v1", "EMPTY"))

    def test_missing_key_names_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                resolve("nebius", PROVIDERS)
        self.assertIn("TEST_NEBIUS_KEY", str(ctx.exception))

    def test_empty_key_counts_as_missing(self):
        with mock.patch.dict(os.environ, {"TEST_NEBIUS_KEY": ""}):
            with self.assertRaises(RuntimeError):
                resolve("nebius", PROVIDERS)

    def test_unknown_provider_lists_known(self):
        with self.assertRaises(KeyError) as ctx:
            resolve("nebuis", PROVIDERS)
        self.assertIn("nebius", str(ctx.exception))

    def test_committed_config_has_no_secrets(self):
        for name, p in load_providers().items():
            self.assertEqual(set(p), {"base_url", "api_key_env"}, name)


class TestEnvFile(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def test_loads_and_skips_comments(self):
        path = self.write('# comment\n\nTEST_A=1\n TEST_B = "two words" \n')
        with mock.patch.dict(os.environ, {}, clear=True):
            load_env_file(path)
            self.assertEqual(os.environ["TEST_A"], "1")
            self.assertEqual(os.environ["TEST_B"], "two words")

    def test_real_environment_wins(self):
        path = self.write("TEST_A=from-file\n")
        with mock.patch.dict(os.environ, {"TEST_A": "from-env"}, clear=True):
            load_env_file(path)
            self.assertEqual(os.environ["TEST_A"], "from-env")

    def test_missing_file_is_fine(self):
        load_env_file(os.path.join(tempfile.gettempdir(), "no-such-file.env"))


if __name__ == "__main__":
    unittest.main()
