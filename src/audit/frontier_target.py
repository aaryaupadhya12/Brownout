"""Audit target: FrontierSWE's reference scorer (scripts/score_from_reward.py).

What is audited here is NOT a task's verifier running in a container. It is the second stage:
the pure function that turns a trial's reward.json into the leaderboard score.

    reward.json  ──►  extract_score(...)  ──►  (correctness, speedup)  ──►  compute_gated_score  ──►  score

Why that fits Brownout: the verifier reads only (task, trace). Here the "trace" is a
reward.json carried in final_state["reward_json"], and the "verifier" calls the benchmark's own
two functions. No env, no container, no model: the whole layer-1 audit runs in-process and free.

What this boundary can and cannot show (the lesson from finding F1, see course/ and interview/):
a failure here is "fragile" (the code misbehaves on some input). It is only "reachable" once a
real trial can produce that input, and only "demonstrated" after a run through the real task.
Never report a finding from this target as more than fragile.

Where the benchmark lives: next to this repo, MARLOW AI/frontier-swe. Override with the
FRONTIER_SWE_DIR environment variable. The file is downloaded code: it is imported by path, and
you should read it before running (this one only uses argparse, json, re, sys).

You write the TODOs: build_reward_json for the other tasks, reward_range, floor, variants,
malformed. Gold and the plumbing are done as a worked example.
"""

import hashlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

from ..interface_v1 import Task
from ..scoring import Verifier, metric, reward
from .target import Target, fake_trace

# Tasks this target covers. Start small; add a task here only after build_reward_json handles it
# AND you checked its shape against what the real verifier writes (see the TODO there).
SUPPORTED = ("git-to-zig", "ffmpeg-swscale-rewrite", "pyright-type-checking-optimization")


# ---------- boilerplate: finding and loading the benchmark's scorer ----------

def frontier_dir() -> Path:
    override = os.environ.get("FRONTIER_SWE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "frontier-swe"      # sibling of Brownout/


def scorer_path() -> Path:
    return frontier_dir() / "scripts" / "score_from_reward.py"


@lru_cache(maxsize=None)
def load_reference_scorer():
    """Import the benchmark's scorer BY PATH and register it in sys.modules (same reason as
    fixer.load_patch: identity and tracebacks need a real, named module)."""
    path = scorer_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: clone Proximal-Labs/frontier-swe next to "
                                f"this repo or set FRONTIER_SWE_DIR")
    spec = importlib.util.spec_from_file_location("frontier_swe_score_from_reward", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def script_sha() -> str:
    """Content hash of the scorer, so a verdict names exactly which version it ran against
    (the same idea as identity.verifier_sha, for code that lives outside this repo)."""
    with open(scorer_path(), encoding="utf-8") as f:           # text mode: CRLF == LF
        return hashlib.sha256(f.read().encode("utf-8")).hexdigest()[:16]


# ---------- boilerplate: the verifier ----------

class FrontierScorer(Verifier):
    """reward = the leaderboard score. Metrics keep the two parts it was built from."""
    name = "frontier-swe-reference-scorer"
    needs_env = False
    ssim_threshold = 0.95       # score_from_reward.py's CLI default; extract_score's own default
                                # is 0.99. Only revideo-perf-opt reads it. Say which one you audit.

    def __init__(self):
        self.version = f"script-{script_sha()[:8]}"

    @reward(weight=1.0)
    def gated(self, task, trace):
        scorer = load_reference_scorer()
        correctness, speedup = scorer.extract_score(
            trace.final_state["reward_json"], task.id, self.ssim_threshold)
        return scorer.compute_gated_score(correctness, speedup, task.id)

    @metric
    def parts(self, task, trace):
        scorer = load_reference_scorer()
        correctness, speedup = scorer.extract_score(
            trace.final_state["reward_json"], task.id, self.ssim_threshold)
        return {"correctness": correctness, "speedup": speedup}



def reward_trace(task, reward_json):
    """A trace whose only content is the reward.json the scorer will read."""
    return fake_trace(task, {"reward_json": reward_json})


def build_reward_json(task_name: str, correctness: float, speedup: float | None = None) -> dict:
    """A reward.json that makes extract_score return (correctness, speedup).

    THREE WORKED EXAMPLES. The key names are read from score_from_reward.py's extract_score,
    NOT from what each task's real verifier writes. Before you trust a shape, open that task's
    tests/compute_reward.py and compare. A synthetic shape that the real verifier never
    produces proves nothing about the real benchmark.

    TODO(you): add the other tasks you want in SUPPORTED. For each one:
      1. read its branch in extract_score (which keys, which defaults)
      2. read tasks/<task>/tests/compute_reward.py (what does the real verifier write?)
      3. add a branch here, and a test in tests/frontier_target_test.py
    """
    if task_name == "git-to-zig":                       # implementation: raw reward IS correctness
        return {"reward": correctness, "score": correctness}
    if task_name == "ffmpeg-swscale-rewrite":           # performance: counts out of a total of 30
        data = {"correctness_passed": round(correctness * 30), "correctness_total": 30}
        if speedup is not None:
            data["geometric_mean_speedup"] = speedup
        return {"reward": speedup or 0.0, "score": speedup or 0.0, "additional_data": data}
    if task_name == "pyright-type-checking-optimization":
        if correctness >= 1.0:                          # full parity: raw reward carries the speedup
            return {"reward": speedup or 1.0, "score": speedup or 1.0}
        return {"reward": 0.0, "score": 0.0, "parity_pass_rate": correctness}
    raise NotImplementedError(f"TODO(you): build_reward_json for {task_name}")


# ---------- the target ----------

class FrontierScorerTarget(Target):
    name = "frontier-swe-scorer"

    # TODO(you): the documented range of the leaderboard score. layer-1 `oracle` demands that gold
    # earns EXACTLY reward_range[1], and `floor` that doing nothing earns EXACTLY [0].
    # Decide and write the reason in a comment:
    #   (a) (0.0, 1.0)  and make gold for performance tasks "correct at baseline speed"
    #       (speedup 1.0 -> 0.5 + 0.5*1.0 = 1.0). Then any speedup above 1 is a range violation:
    #       is that a finding, or the benchmark working as designed?
    #   (b) (0.0, None) i.e. no top. Then `oracle` has nothing to compare to: what do you check?
    reward_range = (0.0, 1.0)

    def __init__(self, verifier=None):
        self.verifier = verifier or FrontierScorer()

    def tasks(self):
        scorer = load_reference_scorer()
        return [Task(name, 0, {"task_name": name, "category": scorer.category_of(name)})
                for name in SUPPORTED]

    def grade(self, task, trace):
        return self.verifier.score(task, trace)

    def grader_name(self) -> str:
        return f"{self.verifier.name}-{self.verifier.version}"

    def gold(self, task):
        """Fully correct, and at baseline speed for performance tasks (speedup 1.0)."""
        return reward_trace(task, build_reward_json(task.id, 1.0, 1.0))

    # --- TODO(you): the other three known-answer traces layer 1 needs ---

    def floor(self, task):
        """TODO(you): the reward.json of a trial that did nothing. Think: what does a verifier
        write when it finds nothing? An empty dict, or {"reward": 0.0, ...}? Both are plausible,
        so which one does the real verifier write? Return reward_trace(task, ...)."""
        raise NotImplementedError("TODO(you)")

    def variants(self, task):
        """TODO(you): a LIST of reward.json files that mean the same as gold but look different:
        keys in another order, an unknown extra key, 30 vs 30.0. Each must still earn the top.
        Return [reward_trace(task, ...), ...]."""
        raise NotImplementedError("TODO(you)")

    def malformed(self, task):
        """TODO(you): ONE broken reward.json. Brownout's rule: a broken input must be recorded as an
        infrastructure failure (grader.ok False), NEVER scored quietly as 0.
        Before you run the audit, predict: does the reference scorer follow that rule for
        None, for {}, for {"reward": "abc"}? Write your predictions down, then test them."""
        raise NotImplementedError("TODO(you)")
