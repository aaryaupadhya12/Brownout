"""Replay saved cloudops runs with no model: send each saved tool call to the env again, check
the env answers exactly as it did, then re-grade. Free and offline. Reads runs/, writes nothing.

    python -m src.envs.SystemEnv.cloudops.replay                        # every runs/*__*.json
    python -m src.envs.SystemEnv.cloudops.replay --limit 20
    python -m src.envs.SystemEnv.cloudops.replay runs/<file>.json

A run replays when every tool output, the final state and the reward match what was saved.
A difference means the data, the tools or the verifier changed since the run was recorded.
Exit code 1 if any run differs, so it can gate a release.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys

from ....audit.swarm.saved import load_trajectory
from .data import COMMIT, problem
from .env import BoutiqueCodeEnv, BoutiqueEnv, TrainTicketEnv
from .regrade_strict import all_tasks
from .tasks import data_root
from .verifier import CloudOpsVerifier

# results the harness wrote itself (src/harness.py parse_args): these calls never reached the env
HARNESS_ERRORS = ("malformed arguments", "arguments must be a JSON object")


def env_for(task_id: str):
    if task_id.startswith("trainticket-"):
        return TrainTicketEnv()
    if task_id.startswith("boutique-codedefect-"):
        return BoutiqueCodeEnv()
    return BoutiqueEnv()


def same(a, b) -> bool:
    """Equal after a JSON round trip, the form the run was saved in."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def replay_one(task, trace) -> str | None:
    """The first difference between a saved run and its replay, or None if it replays."""
    env = env_for(task.id)
    env.reset(task)
    for i, s in enumerate(trace.steps):
        if s.tool is None:                                   # a plain-text turn
            continue
        error = s.observation.get("error") if isinstance(s.observation, dict) else None
        if isinstance(error, str) and error.startswith(HARNESS_ERRORS):
            continue
        with contextlib.redirect_stdout(io.StringIO()):     # the vendored tools print each call
            got = env.step(s.tool, s.args)
        if not same(got, s.observation):
            return f"step {i} ({s.tool}): output differs"
    if not same(env.snapshot(), trace.final_state):
        return "final_state differs"
    out, _ = CloudOpsVerifier().score(task, trace)
    if (out.reward, out.ok) != (trace.grader.reward, trace.grader.ok):
        return f"reward {out.reward} ok={out.ok}, saved {trace.grader.reward} ok={trace.grader.ok}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Replay saved cloudops runs with no model.")
    p.add_argument("files", nargs="*", help="run files (default: every runs/*__*.json)")
    p.add_argument("--runs", default="runs")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    issue = problem(data_root())
    print(issue and f"warning: {issue}" or f"data: pinned commit {COMMIT[:12]}")
    files = a.files or sorted(glob.glob(os.path.join(a.runs, "*__*.json")))
    tasks = all_tasks()
    replayed, differs, skipped = 0, 0, 0
    for path in files[:a.limit] if a.limit else files:
        trace = load_trajectory(path)
        task = tasks.get(trace.task_id)
        if task is None:                                     # not a cloudops run
            skipped += 1
            continue
        diff = replay_one(task, trace)
        replayed, differs = replayed + (diff is None), differs + (diff is not None)
        if diff:
            print(f"DIFFERS  {os.path.basename(path)}: {diff}")
    print(f"replayed {replayed}, differs {differs}, skipped {skipped} (not cloudops)")
    return 1 if differs or issue else 0


if __name__ == "__main__":
    sys.exit(main())
