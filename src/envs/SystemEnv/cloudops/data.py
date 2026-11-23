"""The Cloud-OpsBench data, pinned to one upstream commit so every machine replays the same cases.

    python -m src.envs.SystemEnv.cloudops.data fetch    # clone into data/Cloud-OpsBench at the pin
    python -m src.envs.SystemEnv.cloudops.data check    # is CLOUDOPSBENCH_ROOT at the pin, unedited?

The tools answer from each case's recorded tool_cache.json, so the data IS the environment:
a different commit can change what a tool returns. A git commit hash covers every file, so
"HEAD == COMMIT and benchmark/ unedited" means the data is byte-for-byte the pinned data.
"""
import os
import subprocess
import sys

from .tasks import REPO_ROOT, data_root

URL = "https://github.com/LLM4Ops/Cloud-OpsBench.git"
# upstream main of 2026-10-06, the local checkout on 2026-11-23. Saved runs do not record a data
# commit; replay.py checks that they reproduce on this one.
COMMIT = "ea05dafbc87b5985df8721622d63d4229c2b4224"


def git(root, *args) -> str:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True).stdout.strip()


def problem(root: str) -> str | None:
    """Why the checkout at root is not the pinned data, or None if it is."""
    head = git(root, "rev-parse", "HEAD")
    if head != COMMIT:
        return f"{root} is at {head[:12] or 'no git commit'}, pinned {COMMIT[:12]}"
    if git(root, "status", "--porcelain", "--", "benchmark"):
        return f"{root}: files under benchmark/ were edited"
    return None


def fetch(dest: str) -> None:
    if not os.path.isdir(dest):
        subprocess.run(["git", "clone", "--no-checkout", URL, dest], check=True)
    subprocess.run(["git", "-C", dest, "fetch", "origin", COMMIT], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "--detach", COMMIT], check=True)


def main(argv) -> int:
    cmd = argv[0] if argv else "check"
    if cmd == "fetch":
        dest = os.path.join(REPO_ROOT, "data", "Cloud-OpsBench")
        fetch(dest)
        print(f"fetched {COMMIT[:12]} into {dest}; set CLOUDOPSBENCH_ROOT to it if you use another path")
        return 0
    root = data_root()
    issue = problem(root)
    print(issue or f"ok: {root} is at the pinned commit {COMMIT[:12]}")
    return 1 if issue else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
