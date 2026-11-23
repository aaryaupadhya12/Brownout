"""Pre-commit check for Docs/SHIPPING.md. Checks staged files only. Exit 1 blocks the commit.

python scripts/ship_check.py              # all checks
python scripts/ship_check.py --no-tests   # skip the offline test suite
"""
import re
import subprocess
import sys
from pathlib import Path

BLOCKED_DIRS = ("runs/", "jobs/", "lcb/", "data/")
BLOCKED_NAMES = re.compile(r"(^|/)(\.env[^/]*|.*\.pem|.*\.key|raw_call\.py)$")
SECRETS = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(api_key|token|secret)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    re.IGNORECASE,
)
CODE_DIRS = ("src/", "cli/", "train/")
MAX_LINES, MAX_BYTES = 100, 5 * 1024 * 1024


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout


def staged():
    """(status, path) for every staged add/copy/modify/rename."""
    rows = git("diff", "--cached", "--name-status", "--diff-filter=ACMR").splitlines()
    return [(r.split("\t")[0][0], r.split("\t")[-1]) for r in rows if r]


def check_files(files):
    errors, warnings = [], []
    for status, path in files:
        if path.startswith(BLOCKED_DIRS) or BLOCKED_NAMES.search(path):
            errors.append(f"{path}: run outputs, datasets and key files are never committed")
            continue
        p = Path(path)
        if p.is_file() and p.stat().st_size > MAX_BYTES:
            errors.append(f"{path}: over 5 MB")
        if not (path.endswith(".py") and path.startswith(CODE_DIRS)) or "/vendor/" in path:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count("\n") + 1
        if n <= MAX_LINES or "# over-100:" in "\n".join(text.splitlines()[:10]):
            continue
        msg = f"{path}: {n} lines (limit {MAX_LINES}; add '# over-100: <reason>' near the top)"
        (errors if status == "A" else warnings).append(msg)
    return errors, warnings


def check_secrets():
    added = [l[1:] for l in git("diff", "--cached", "-U0").splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    return [f"possible secret in staged diff: {l.strip()[:60]}..." for l in added if SECRETS.search(l)]


def check_tests():
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                        "-p", "*_test*.py"], capture_output=True, text=True)
    return [] if r.returncode == 0 else ["offline tests failed:\n" + r.stderr[-1500:]]


def main():
    errors, warnings = check_files(staged())
    errors += check_secrets()
    if "--no-tests" not in sys.argv:
        errors += check_tests()
    for w in warnings:
        print(f"warn  {w}")
    for e in errors:
        print(f"FAIL  {e}")
    print("ship_check: " + ("blocked" if errors else "ok") + "  (Docs/SHIPPING.md)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
