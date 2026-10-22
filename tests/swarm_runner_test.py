import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.audit.swarm import loop
from src.audit.swarm.loop import SwarmConfig, campaign_lock


class CampaignLockTest(unittest.TestCase):
    def test_second_entry_is_blocked_and_released(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SwarmConfig(campaign="c", out_dir=d)
            with campaign_lock(cfg):
                with self.assertRaises(RuntimeError):
                    with campaign_lock(cfg):
                        pass
            # released after the block: a fresh acquire works
            with campaign_lock(cfg):
                pass


if __name__ == "__main__":
    unittest.main()