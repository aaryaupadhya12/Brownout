"""The findings ledger: one JSON object per line, only ever appended.

Append-only because a ledger is a record (like runs/): a row is never edited. A label change
is a new row with the same exploit_id and a later time. A crash mid-write can only tear the
LAST line; read() skips that one and raises on any other bad line.

Every row carries the versions the spec requires ("every verdict records the environment and
grader version") plus what adversarial self-play will need later (role, label, group_id).
"""

import json
import os
import threading
import time
from collections import Counter

from ...envs.SystemEnv.cloudops.verifier import normalize
from ...tracking import sha256_of

ROW_FIELDS = (
    "exploit_id",     # Exploit.exploit_id
    "signature",      # signature(): same trick -> same value, across tasks
    "campaign",       # one orchestrator run, e.g. "boutique-planted-v1"
    "task_id",
    "role",           # attacker | solver | adjudicator | person
    "mechanism",      # exploit.MECHANISMS or "unlabeled"
    "label",          # hack | legitimate | unconfirmed
    "verifier_sha",   # identity.verifier_sha of the verifier it was confirmed on
    "env_sha",
    "group_id",       # tracking.group_id of the attacker trajectory
    "evidence",       # dict from confirm.is_hack: rewards per verifier, true_success, paths
    "repro",          # Exploit.as_row()
    "status",         # open | patched | quarantined
    "time",
)

_lock = threading.Lock()   # appends come from several episode threads


def check_row(row: dict) -> None:
    missing = [f for f in ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"ledger row missing {missing}")


def _write_line(path: str, line: str) -> None:
    """Append one line. Caller must hold _lock."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _drop_torn_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def append(path: str, row: dict) -> None:
    row = {**row, "time": row.get("time") or time.time()}
    check_row(row)
    line = json.dumps(row, sort_keys=True, ensure_ascii=False)
    with _lock:
        _write_line(path, line)


def append_once(path: str, row: dict, key: str = "exploit_id") -> bool:
    """Append unless a row with the same `key` value is already in the file.
    The check and the write happen under one lock, so two threads finishing together
    can't both add the same row. Returns True if it wrote, False if already there.
    One process only: the campaign lock keeps a second process out."""
    row = {**row, "time": row.get("time") or time.time()}
    check_row(row)
    line = json.dumps(row, sort_keys=True, ensure_ascii=False)
    with _lock:
        if any(r.get(key) == row[key] for r in read(path)):
            return False
        _write_line(path, line)
        return True


def _drop_torn_tail(path: str) -> None:
    """A file not ending in a newline has a torn last write. Cut it off, or the next
    append would join it into one corrupt line and lose the new row too."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        f.truncate(data.rfind(b"\n") + 1)


def read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f.read().split("\n") if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:      # torn last write: skip it
                break
            raise ValueError(f"{path}: corrupt ledger line {i + 1}")
    return rows


def count_by_signature(rows: list[dict]) -> Counter:
    """How many rows each trick has. Findings = distinct signatures, not rows."""
    return Counter(row["signature"] for row in rows)


def signature(exploit, task) -> str:
    """TODO(you): dedupe key. Same trick -> same key, on any task.

    Steps:
      1. For each (tool, args) in exploit.calls, replace task-specific strings with
         placeholders: this task's answer fields -> "ANSWER", any other string -> its
         literal value only if it is short/generic (e.g. "/", "_"), else "OTHER".
      2. Keep tool names, argument KEYS, and the Submit rank positions.
      3. Prefix with exploit.mechanism, then tracking.sha256_of(...)[:16].

    Examples (the answer for task A is app/adservice + missing_service_account):
      A: Submit rank1=("/", "_")                 -> proxy_gaming|Submit|r1=LIT(/),LIT(_)
      B: Submit rank1=("/", "_")                 -> same key (deduped)
      A: Submit rank1=junk, rank3=the answer     -> proxy_gaming|Submit|r1=OTHER|r3=ANSWER
    """
    answers = {normalize(v) for v in task.spec.get("answer", {}).values()}
    parts = [exploit.mechanism]
    for tool, args in exploit.calls:
        if tool == "Submit":
            ranks = [f"r{p.get('rank', i + 1)}=" + ",".join(
                         _shape(p.get(k), answers) for k in ("fault_object", "root_cause"))
                     for i, p in enumerate(args.get("top_3_predictions") or [])]
            parts.append("|".join(["Submit"] + ranks))
        else:
            parts.append(f"{tool}({','.join(sorted(args))})")   # keys only: values are task-specific
    return f"{exploit.mechanism}-{sha256_of('|'.join(parts))[:16]}"


GENERIC_MAX = 2    # a literal this short ("/", "_", "") is the trick itself, not task data


def _shape(value, answers) -> str:
    text = normalize(value)
    if text in answers:
        return "ANSWER"
    if text.startswith("app/filler-") or text.startswith("filler_"):
        return "FILLER"            # planted.FILLER: padding, never a label
    if len(text) <= GENERIC_MAX:
        return f"LIT({text})"
    return "OTHER"
