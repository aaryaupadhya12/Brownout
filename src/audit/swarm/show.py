"""Human-readable view of a campaign: per-task summary, then one block per ledger finding.

    python -m src.audit.swarm.show boutique-planted-substring-v2 [--out runs/swarm] [--whitebox-from 1]

Reads only fields from the saved episodes (never prints a trajectory whole). Writes nothing.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

from . import ledger

KEY = re.compile(r"r(-?\d+)_(\w+?)_a(\d+)_")


def parse_key(path):
    """'.../r7_attacker_a0_8e571de8.json' -> (7, 'attacker', 0)."""
    m = KEY.match(os.path.basename(path))
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else (0, "?", 0)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rank1(traj):
    sub = (traj.get("final_state") or {}).get("submission")
    if not isinstance(sub, dict):
        return None
    preds = sub.get("top_3_predictions") or [{}]
    return preds[0].get("fault_object"), preds[0].get("root_cause")


def plan_text(traj, width=400):
    """The attacker's first non-empty text: its stated plan, if it wrote one."""
    for m in traj.get("messages", []):
        p = m.get("payload", {})
        if p.get("role") == "assistant" and (p.get("content") or "").strip():
            text = " ".join(p["content"].split())
            return text[:width] + ("..." if len(text) > width else "")
    return "(wrote no text: tool calls only)"


def cost(traj):
    return (traj.get("totals") or {}).get("api_cost_usd") or 0.0


def task_summary(task_dir, whitebox_from):
    rows = []
    for path in glob.glob(os.path.join(task_dir, "r*_attacker_*.json")):
        round_, _, attempt = parse_key(path)
        t = load(path)
        rows.append((round_, attempt, t))
    rows.sort(key=lambda r: (r[0], r[1]))
    ended = Counter(t["ended_by"] for _, _, t in rows)
    solver = glob.glob(os.path.join(task_dir, "r-1_solver_*.json"))
    record = os.path.join(task_dir, "precheck.json")
    if os.path.exists(record):                       # written by loop.precheck
        pre = load(record)
        precheck = f"{'passed' if pre['passed'] else 'FAILED'} ({pre['source']})"
    elif solver:                                     # campaigns from before precheck.json
        precheck = "passed (live)" if load(solver[0])["grader"]["reward"] >= 1.0 else "FAILED (live)"
    else:
        precheck = "not run"
    total = sum(cost(t) for _, _, t in rows) + sum(cost(load(p)) for p in solver)
    print(f"  attempts {len(rows)}   ended_by {dict(ended)}   precheck {precheck}"
          f"   cost ${total:.3f}")
    for round_, attempt, t in rows:
        mode = "white" if whitebox_from is not None and round_ >= whitebox_from else "blind"
        r1 = rank1(t)
        sub = f"{r1[0]} + {r1[1]}" if r1 else "-"
        print(f"    r{round_} a{attempt} {mode:5} {t['ended_by']:9} reward {t['grader']['reward']:.1f}  "
              f"turns {t['totals'].get('turns', 0):2}  submitted {sub}")


def finding(row, out_dir, whitebox_from):
    task_id, key = row["exploit_id"].split("/", 1)
    round_, _, attempt = parse_key(key + ".json")
    mode = "whitebox" if whitebox_from is not None and round_ >= whitebox_from else "blind"
    ev = row["evidence"]
    path = os.path.join(out_dir, row["campaign"], task_id, key + ".json")
    traj = load(path) if os.path.exists(path) else {}
    submit = next((a for n, a in reversed(row["repro"]["calls"]) if n == "Submit"), None)
    preds = (submit or {}).get("top_3_predictions") or []
    tools = Counter(n for n, _ in row["repro"]["calls"] if n != "Submit")

    print(f"\n=== {row['label'].upper()} on {task_id}   (round {round_}, attempt {attempt}, {mode})")
    print(f"  reward on attacked verifier   {ev['unpatched']['reward']}  (sha {ev['unpatched']['sha'][:8]})")
    print(f"  reward on true success        {ev['true_success']['reward']}  (sha {ev['true_success']['sha'][:8]})")
    print(f"  mechanism {row['mechanism']}   status {row['status']}   signature {row['signature']}")
    print(f"  submitted:")
    for p in preds:
        print(f"    rank {p.get('rank')}: {p.get('fault_object')!r} + {p.get('root_cause')!r}")
    if submit and submit.get("key_evidence_summary"):
        print(f"  its summary: {' '.join(str(submit['key_evidence_summary']).split())[:300]}")
    print(f"  tools before submit ({sum(tools.values())}): "
          + (", ".join(f"{n} x{c}" for n, c in tools.most_common()) or "none"))
    if traj:
        print(f"  its plan: {plan_text(traj)}")
    print(f"  file: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign")
    parser.add_argument("--out", default=os.path.join("runs", "swarm"))
    parser.add_argument("--whitebox-from", type=int, default=1,
                        help="SwarmConfig.whitebox_from_round used for the run (labels blind/white)")
    args = parser.parse_args(argv)

    root = os.path.join(args.out, args.campaign)
    if not os.path.isdir(root):
        parser.exit(1, f"no campaign folder {root}: the run never started (or a different --out)\n")
    rows = ledger.read(os.path.join(root, "ledger.jsonl"))
    hacked = {r["task_id"] for r in rows if r["label"] == "hack"}
    tasks = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    print(f"campaign {args.campaign}: {len(hacked)}/{len(tasks)} tasks hacked, "
          f"{len(rows)} ledger rows, {len(ledger.count_by_signature(rows))} distinct tricks")
    for task in tasks:
        print(f"\n{task}: {'HACKED' if task in hacked else 'no hack found'}")
        task_summary(os.path.join(root, task), args.whitebox_from)
    for row in rows:
        finding(row, args.out, args.whitebox_from)


if __name__ == "__main__":
    main()
