"""Hand-audit sampler: show unconfirmed passes (JRA 1, strict 0) compactly, so you can label them.

    python -m src.envs.SystemEnv.cloudops.sample_unconfirmed runs/regrade/strict-82038ac5.jsonl
    python -m src.envs.SystemEnv.cloudops.sample_unconfirmed <regrade.jsonl> --n 20 --seed 0

For each sampled run it prints: the true answer, the tool calls (name + args only), the answer
milestone(s) the label wanted and did not get, and the output lines that mention the answer.
It also writes a blank label sheet next to the regrade file (never overwrites one you started):

    label   blind        no relevant evidence in any output
            symptom      answer readable off a symptom message, confirming check skipped
            label_gap    properly grounded by a route the label does not list
"""

import argparse
import csv
import json
import os
import random
import re
import sys

from ....audit.swarm.saved import load_trajectory
from .regrade_strict import all_tasks
from .strict import StrictCloudOpsVerifier, answer_milestones, tool_steps

MAX_LINES = 6           # answer-mentioning output lines shown per run
MAX_LINE = 180


def answer_terms(answer: dict) -> list:
    """Words that would show the answer in a tool output: the component and root-cause tokens."""
    obj = str(answer.get("fault_object", "")).split("/")[-1]
    cause = [w for w in str(answer.get("root_cause", "")).split("_") if len(w) > 3]
    return [t for t in [obj, *cause] if t]


def mentions(trace, terms: list) -> list:
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.I) if terms else None
    found = []
    for i, (tool, _, obs) in enumerate(tool_steps(trace)):
        for line in str(obs).splitlines():
            if pattern and pattern.search(line):
                found.append(f"[{i}] {tool}: {line.strip()[:MAX_LINE]}")
                if len(found) >= MAX_LINES:
                    return found
    return found


def show(verifier, task, path) -> None:
    trace = load_trajectory(path)
    ann = verifier.annotation(task)
    held = set(verifier.process(task, trace)["established"])
    answer = task.spec["answer"]
    print("=" * 100)
    print(f"{task.id}   {os.path.basename(path)}")
    print(f"answer: {answer.get('fault_object')}  /  {answer.get('root_cause')}")
    print("calls:")
    for i, (tool, args, _) in enumerate(tool_steps(trace)):
        print(f"  [{i}] {tool} {json.dumps(args, sort_keys=True)[:140]}")
    print("label wanted (answer milestones not established):")
    for m in ann["milestones"]:
        if str(m["id"]) in answer_milestones(ann) and str(m["id"]) not in held:
            print(f"  {m['id']} {m.get('role', '')}")
            for use in m.get("admissible_tool_uses", ()):
                pats = [p.get("value") for p in use.get("evidence_patterns", ())]
                print(f"     {use['tool_name']} {json.dumps(use.get('arguments', {}))[:110]}"
                      + (f"  must show {pats[:3]}" if pats else ""))
    print("output lines mentioning the answer:")
    for line in mentions(trace, answer_terms(answer)) or ["  (none)"]:
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("regrade")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescued", action="store_true",
                    help="sample the OTHER direction: runs v0.2.0 counts as confirmed only via a "
                         "deviation (via_namespace_fix / via_pod_config)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")   # tool outputs hold characters cp1252 cannot print

    rows = [json.loads(line) for line in open(args.regrade, encoding="utf-8") if line.strip()]
    if args.rescued:
        unconfirmed = [r for r in rows if r.get("ok") and r.get("strict") == 1.0
                       and (r.get("via_namespace_fix") or r.get("via_pod_config"))]
    else:
        unconfirmed = [r for r in rows if r.get("ok") and r.get("jra") == 1.0 and r.get("strict") == 0.0]
    sample = random.Random(args.seed).sample(unconfirmed, min(args.n, len(unconfirmed)))
    print(f"{len(unconfirmed)} {'rescued (confirmed only via a deviation)' if args.rescued else 'unconfirmed'} passes; showing {len(sample)} (seed {args.seed})")

    tasks = all_tasks()
    verifier = StrictCloudOpsVerifier()
    for r in sample:
        show(verifier, tasks[r["task_id"]], os.path.join(args.runs, r["file"]))

    kind = "rescued" if args.rescued else "audit"
    sheet = os.path.splitext(args.regrade)[0] + f"-{kind}-seed{args.seed}.csv"
    if os.path.exists(sheet):
        print(f"\nlabel sheet exists, not overwritten: {sheet}")
        return
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "task_id", "model_id", "label", "note"])
        for r in sample:
            w.writerow([r["file"], r["task_id"], r["model_id"], "", ""])
    print(f"\nlabel sheet: {sheet}   (fill 'label' with blind / symptom / label_gap)")


if __name__ == "__main__":
    main()
