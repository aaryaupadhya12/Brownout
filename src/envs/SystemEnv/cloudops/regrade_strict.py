"""Re-grade saved cloudops runs with the strict verifier. Reads runs/, writes a NEW file.

    python -m src.envs.SystemEnv.cloudops.regrade_strict                  # all runs/*__*.json
    python -m src.envs.SystemEnv.cloudops.regrade_strict --limit 50       # quick look

One JSON line per run in --out (default runs/regrade/strict-<sha8>.jsonl, named by the strict
verifier's code hash so two versions never mix). Saved runs are never modified. Then a summary:

  per model   n, JRA (what training pays), strict (answer + evidence), the gap between them,
              ECR, steps. Runs whose grading crashed (ok=False) are counted, not averaged.
  ECR vs steps, inside each JRA outcome: if ECR climbs with steps even among wrong answers,
              closure is bought with volume, not diagnosis.
"""

import argparse
import glob
import hashlib
import json
import os
import statistics
from collections import defaultdict

from ....audit.swarm.identity import verifier_sha
from ....audit.swarm.saved import load_trajectory
from .strict import StrictCloudOpsVerifier, tool_steps
from .tasks import BoutiqueCodeTasks, BoutiqueTasks, TrainTicketTasks
from .vendor import process_eval

STEP_BUCKETS = ((0, 4), (5, 8), (9, 14), (15, 10**9))


def strict_sha(verifier) -> str:
    """verifier_sha covers the MRO only. The strict verifier USES CloudOpsVerifier and
    process_eval.py without inheriting them, so their code goes into the identity too."""
    h = hashlib.sha256()
    h.update(verifier_sha(verifier).encode())
    h.update(verifier_sha(verifier.outcome).encode())
    with open(process_eval.__file__, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def all_tasks() -> dict:
    tasks = {}
    for loader in (BoutiqueTasks(), BoutiqueCodeTasks(), TrainTicketTasks()):
        try:
            tasks.update({t.id: t for t in loader.load()})
        except FileNotFoundError:           # e.g. a category folder this checkout lacks
            continue
    return tasks


def grade_one(verifier, task, path) -> dict:
    trace = load_trajectory(path)
    out, errors = verifier.score(task, trace)
    s = out.signals
    return {
        "file": os.path.basename(path), "task_id": task.id, "model_id": trace.model_id,
        "saved_reward": trace.grader.reward, "ok": out.ok,
        "jra": s.get("jra"), "strict": out.reward if out.ok else None,
        "closed_jra": s.get("closed_jra"),
        "grounded_raw": s.get("grounded_raw"), "via_namespace_fix": s.get("via_namespace_fix"),
        "via_pod_config": s.get("via_pod_config"),
        "mc": s.get("mc"), "eoc": s.get("eoc"), "ecr": s.get("ecr"), "ee": s.get("ee"),
        "steps": len(tool_steps(trace)),
        "errors": [e.message for e in errors],
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def summarize(rows):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_id"]].append(r)
    print(f"\n{'model':45s} {'n':>5s} {'crash':>5s} {'JRA':>6s} {'strict':>6s} {'unconf':>6s} "
          f"{'ECR':>6s} {'steps':>6s}")
    for model, rs in sorted(by_model.items()):
        good = [r for r in rs if r["ok"]]
        guess = sum(1 for r in good if r["jra"] == 1.0 and r["strict"] == 0.0)
        print(f"{model[:45]:45s} {len(rs):5d} {len(rs) - len(good):5d} "
              f"{mean(r['jra'] for r in good)!s:>6s} {mean(r['strict'] for r in good)!s:>6s} "
              f"{guess:6d} {mean(r['ecr'] for r in good)!s:>6s} "
              f"{mean(r['steps'] for r in good)!s:>6s}")
    print("\nwhat each v0.2.0 deviation changed (JRA-1 runs newly counted as grounded):")
    for model, rs in sorted(by_model.items()):
        good = [r for r in rs if r["ok"]]
        print(f"  {model[:45]:45s} namespace fix {sum(r.get('via_namespace_fix') or 0 for r in good):4.0f}"
              f"   pod-as-config {sum(r.get('via_pod_config') or 0 for r in good):4.0f}"
              f"   raw-rule unconfirmed "
              f"{sum(1 for r in good if r['jra'] == 1.0 and not r.get('grounded_raw')):4d}")
    print("  unconf = JRA 1 but strict 0: right answer without the confirming evidence the label requires (often read off a symptom message; not necessarily a blind guess)")

    print("\nECR by step count, split by JRA outcome (all models, ok runs only)")
    print(f"{'steps':>8s} {'JRA=1: n':>9s} {'ECR':>6s} {'JRA=0: n':>9s} {'ECR':>6s}")
    for lo, hi in STEP_BUCKETS:
        cells = []
        for want in (1.0, 0.0):
            rs = [r for r in rows if r["ok"] and r["jra"] == want and lo <= r["steps"] <= hi]
            cells += [len(rs), mean(r["ecr"] for r in rs)]
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"{label:>8s} {cells[0]:9d} {cells[1]!s:>6s} {cells[2]:9d} {cells[3]!s:>6s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    verifier = StrictCloudOpsVerifier()
    sha = strict_sha(verifier)
    out_path = args.out or os.path.join(args.runs, "regrade", f"strict-{sha[:8]}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tasks = all_tasks()
    paths = sorted(glob.glob(os.path.join(args.runs, "*__*.json")))[:args.limit]
    rows, skipped = [], 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in paths:
            task_id = os.path.basename(path).split("__", 1)[0]
            task = tasks.get(task_id)
            if task is None:                 # not a cloudops run, or a case this checkout lacks
                skipped += 1
                continue
            try:
                row = grade_one(verifier, task, path)
            except Exception as e:           # an unreadable file is data too
                row = {"file": os.path.basename(path), "task_id": task_id, "model_id": "?",
                       "ok": False, "errors": [f"load: {type(e).__name__}: {e}"]}
            row["strict_sha"] = sha
            f.write(json.dumps(row) + "\n")
            rows.append(row)

    print(f"graded {len(rows)} runs, skipped {skipped} (no matching cloudops task) -> {out_path}")
    summarize([r for r in rows if "steps" in r])


if __name__ == "__main__":
    main()
