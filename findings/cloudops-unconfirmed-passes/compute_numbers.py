"""Recompute every number in README.md from the re-grade and judge files. Stdlib only, free.

    python findings/cloudops-unconfirmed-passes/compute_numbers.py            # print
    python findings/cloudops-unconfirmed-passes/compute_numbers.py --write    # also expected.json

Intervals are Wilson 95% score intervals (sensible at small n and at 0 or n successes).
"""

import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REGRADE = os.path.join(ROOT, "runs", "regrade", "strict-6f0952b5.jsonl")
CENSUS = os.path.join(ROOT, "runs", "regrade",
                      "strict-6f0952b5-audit-seed2-judge-zai-org_GLM-5.3-Flash.jsonl")
RESCUED = os.path.join(ROOT, "runs", "regrade",
                       "strict-6f0952b5-rescued-seed0-judge-zai-org_GLM-5.3-Flash.jsonl")
CONTROL = os.path.join(ROOT, "runs", "regrade", "strict-6f0952b5-rescued-negative-control.jsonl")
GOLDEN = os.path.join(ROOT, "runs", "regrade", "golden-control.jsonl")
LABELS = ("blind", "symptom_weak", "symptom_strong", "label_gap")


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def share(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    runs = [r for r in load(REGRADE) if r.get("ok")]
    out = {"verifier_file": os.path.basename(REGRADE), "models": {}}
    for model in sorted({r["model_id"] for r in runs}):
        rs = [r for r in runs if r["model_id"] == model]
        hits = [r for r in rs if r["jra"] == 1.0]
        unc = [r for r in hits if r["strict"] == 0.0]
        raw_unc = [r for r in hits if not r.get("grounded_raw")]
        out["models"][model] = {
            "runs": len(rs),
            "jra": round(sum(r["jra"] for r in rs) / len(rs), 4),
            "strict": round(sum(r["strict"] for r in rs) / len(rs), 4),
            "unconfirmed_of_jra_passes": share(len(unc), len(hits)),
            "unconfirmed_raw_labels": share(len(raw_unc), len(hits)),
            "rescued_by_namespace_fix": sum(1 for r in hits if r.get("via_namespace_fix")),
            "rescued_by_pod_config": sum(1 for r in hits if r.get("via_pod_config")),
            "unconfirmed_by_category": dict(collections.Counter(
                r["task_id"].rsplit("-", 1)[0] for r in unc).most_common()),
        }

    census = load(CENSUS)
    labelled = [r for r in census if r["label"]]
    counts = collections.Counter(r["label"] for r in labelled)
    out["census"] = {
        "judge": "zai-org/GLM-5.3-Flash", "prompt_version": census[0].get("prompt_version") if census else None,
        "cases": len(census), "labelled": len(labelled),
        "no_answer": [r["task_id"] for r in census if not r["label"]],
        "labels": {k: share(counts[k], len(labelled)) for k in LABELS},
        "symptom_any": share(counts["symptom_weak"] + counts["symptom_strong"], len(labelled)),
    }
    # the other error direction: runs confirmed only via a v0.2.0 deviation, and a negative
    # control (cases where the rule would be wrong, presented as accepted)
    regrade = {r["file"]: r for r in runs}
    rescued = [r for r in load(RESCUED) if r["label"]]
    control = [r for r in load(CONTROL) if r["label"]]
    by_rule = collections.defaultdict(collections.Counter)
    for r in rescued:
        rule = "namespace_fix" if regrade[r["file"]].get("via_namespace_fix") else "pod_config"
        by_rule[rule][r["label"]] += 1
    out["reverse"] = {
        "judge": "zai-org/GLM-5.3-Flash", "prompt_version": "1r",
        "rescued_labelled": len(rescued),
        "not_confirmed": share(sum(r["label"] == "not_confirmed" for r in rescued), len(rescued)),
        "by_rule": {k: dict(v) for k, v in by_rule.items()},
        "negative_control_rejected": share(sum(r["label"] == "not_confirmed" for r in control),
                                           len(control)),
    }
    # second census judge (cross-family), and where the two agree
    sys.path.insert(0, HERE)
    from compare_judges import compare
    second = CENSUS.replace("zai-org_GLM-5.3-Flash", "openai_gpt-oss-120b")
    if os.path.exists(second):
        both = compare(CENSUS, second)
        agreed3 = both["agreed_labels3"]
        out["census_two_judges"] = {
            "judges": ["zai-org/GLM-5.3-Flash", "openai/gpt-oss-120b"],
            "n_both_answered": both["n"], "raw4": both["raw"], "kappa4": both["kappa4"],
            "raw3": round(sum(agreed3.values()) / both["n"], 4), "kappa3": both["kappa3"],
            "both_say_symptom": share(agreed3.get("symptom", 0), both["n"]),
            "both_say_blind": share(agreed3.get("blind", 0), both["n"]),
            "both_say_label_gap": share(agreed3.get("label_gap", 0), both["n"]),
            "second_judge_labels": dict(collections.Counter(
                r["label"] for r in load(second) if r["label"])),
        }
    # legitimate control: upstream's expert (golden) paths, graded by golden_control.py
    if os.path.exists(GOLDEN):
        gold = [r for r in load(GOLDEN) if "skip" not in r and r["case"].startswith("boutique")]
        by_case = collections.defaultdict(list)
        for r in gold:
            by_case[r["case"]].append(r)
        rejected = [r for r in gold if not r["raw"]]
        unc_cases = [r["task_id"] for r in runs if r["jra"] == 1.0 and r["strict"] == 0.0]
        expert_ok = [t for t in unc_cases if by_case.get(t) and all(x["final"] for x in by_case[t])]
        out["golden_control"] = {
            "system": "boutique", "paths": len(gold), "cases": len(by_case),
            "rejected_by_raw_labels": share(len(rejected), len(gold)),
            "of_those_rescued_by_pod_config": share(sum(r["by_pod_config"] for r in rejected),
                                                    len(rejected)),
            "of_those_rescued_by_namespace_fix": sum(r["by_namespace"] for r in rejected),
            "rejected_final": share(sum(not r["final"] for r in gold), len(gold)),
            "cases_no_expert_path_accepted_raw": sum(all(not x["raw"] for x in v)
                                                     for v in by_case.values()),
            "cases_no_expert_path_accepted_final": sum(all(not x["final"] for x in v)
                                                       for v in by_case.values()),
            "unconfirmed_runs_where_every_expert_path_passes": share(len(expert_ok),
                                                                     len(unc_cases)),
            "rejected_final_by_root_cause": dict(collections.Counter(
                r["root_cause"] for r in gold if not r["final"]).most_common()),
        }
    print(json.dumps(out, indent=2))
    if args.write:
        with open(os.path.join(HERE, "expected.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
