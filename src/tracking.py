"""Identity for runs: which runs are "the same experiment" and can be compared.

Three hashes, each answering a different question:

  task_hash   - same task content?          sha256 of the scenario dict
  config_hash - same rollout setup?         sha256 of model, sampling, prompts, tools, limits
  group_id    - same task AND same setup?   short sha of (task_hash, config_hash)

Runs that share a group_id differ only by chance (sampling) and run_index, so their
rewards can be averaged together. This is the same idea as a GRPO "group" (several
rollouts of one prompt under one policy) and as verifiers' task hash
(verifiers/v1/task.py:46-48, 156-160).

The code version (git commit + dirty flag) and the verifier are recorded next to the
hashes but are NOT part of group_id: a scorer change re-grades the same rollouts, and a
commit can change without changing behaviour. Compare them separately when it matters.
"""

import glob
import hashlib
import json
import os
import subprocess


def canonical_json(value):
    """One fixed text form for a value: sorted keys, no extra spaces.
    Note: 1 and 1.0 serialize differently, so keep number types consistent in scenarios."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def task_hash(scenario):
    return sha256_of(scenario)


def config_hash(config):
    return sha256_of(config)


def group_id(task_sha, config_sha):
    return sha256_of({"task": task_sha, "config": config_sha})[:16]


def code_version():
    """The git commit of this repo, and whether there were uncommitted changes.
    A dirty run can't be reproduced from the commit alone."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status.strip())}


def group_saved_runs(folder="runs"):
    """Read saved trajectories and group them by group_id.
    Runs whose scorer failed (grader.ok False) are counted but left out of the mean."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path, encoding="utf-8") as f:
            traj = json.load(f)
        tracking = traj.get("tracking") or {}
        key = tracking.get("group_id", "untracked")
        if key not in groups:
            groups[key] = {"task_key": tracking.get("task_key", traj.get("task_id")),
                           "model_id": traj.get("model_id"),
                           "n": 0, "n_scored": 0, "n_error": 0, "rewards": []}
        g = groups[key]
        g["n"] += 1
        if traj.get("ended_by") == "error":
            g["n_error"] += 1
        if traj["grader"].get("ok", True):
            g["n_scored"] += 1
            g["rewards"].append(traj["grader"]["reward"])

    for g in groups.values():
        if g["rewards"]:
            g["mean_reward"] = sum(g["rewards"]) / len(g["rewards"])
        else:
            g["mean_reward"] = None
    return groups
