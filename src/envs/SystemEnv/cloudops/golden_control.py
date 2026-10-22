"""Legitimate control for claim C: grade upstream's own expert (golden) trajectories.

Every golden path is an expert's correct diagnosis, so the benchmark's process label should
accept it. A path the raw label rejects is a label that rejects honest evidence; the two
v0.2.0 deviations say which pattern explains it. Free, offline, reads upstream files only.

    python -m src.envs.SystemEnv.cloudops.golden_control            # summary
    python -m src.envs.SystemEnv.cloudops.golden_control --write    # + runs/regrade/golden-control.jsonl

Parsing follows upstream evaluation_utils/matcher.py parse_golden_calling (checkout ea05daf).
"""

import argparse
import ast
import collections
import json
import os
import re
import sys

from .strict import grounding
from .tasks import data_root
from .vendor.process_eval import load_annotation


# ---------- port of upstream parse_golden_calling ----------

def _split_items(raw: str):
    current, quote, depth = [], None, 0
    for ch in raw:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                yield item
            current = []
        else:
            current.append(ch)
    item = "".join(current).strip()
    if item:
        yield item


def _value(raw: str):
    if raw in {"None", "null"}:
        return None
    if raw in {"True", "False"}:
        return raw == "True"
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip("'\"")


def parse_calling(calling: str, fallback_tool: str = "") -> tuple:
    m = re.search(r"tool_name=['\"]([^'\"]+)['\"]", calling)
    tool = m.group(1) if m else fallback_tool
    m = re.search(r"arguments=\{(.*)\}\s*$", calling)
    args = {}
    if m:
        for item in _split_items(m.group(1)):
            key, _, val = item.partition(":")
            args[key.strip().strip("'\"")] = _value(val.strip()) if _ else ""
    return tool, args


def golden_steps(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        trace = json.load(f).get("diagnostic_trace", [])
    steps = []
    for item in trace:
        tool, args = parse_calling(item.get("calling", ""), item.get("tool_name", ""))
        steps.append((tool, args, item.get("output", "")))
    return steps


# ---------- the control ----------

def grade_all(root: str) -> list:
    rows = []
    gold_root = os.path.join(root, "golden-trajectory")
    for system in sorted(os.listdir(gold_root)):
        for category in sorted(os.listdir(os.path.join(gold_root, system))):
            cat_dir = os.path.join(gold_root, system, category)
            for case in sorted(os.listdir(cat_dir), key=lambda c: (len(c), c)):
                label = os.path.join(root, "process-label", system, category, case, "milestone.json")
                meta = os.path.join(root, "benchmark", system, category, case, "metadata.json")
                if not (os.path.exists(label) and os.path.exists(meta)):
                    rows.append({"case": f"{system}-{category}-{case}", "skip": "no label or metadata"})
                    continue
                annotation = load_annotation(label)
                with open(meta, encoding="utf-8") as f:
                    root_cause = json.load(f)["result"]["root_cause"]
                for name in sorted(os.listdir(os.path.join(cat_dir, case))):
                    if not name.endswith(".json"):
                        continue
                    g = grounding(annotation, golden_steps(os.path.join(cat_dir, case, name)),
                                  str(root_cause))
                    rows.append({"case": f"{system}-{category}-{case}", "path": name,
                                 "root_cause": root_cause, **g})
    return rows


def summarise(rows: list) -> dict:
    paths = [r for r in rows if "skip" not in r]
    by_case = collections.defaultdict(list)
    for r in paths:
        by_case[r["case"]].append(r)
    case_ok = lambda key: sum(any(r[key] for r in rs) for rs in by_case.values())
    rejected = [r for r in paths if not r["raw"]]
    return {
        "paths": len(paths), "cases": len(by_case),
        "skipped": [r["case"] for r in rows if "skip" in r],
        "paths_rejected_raw": len(rejected),
        "paths_rescued_by_namespace": sum(r["by_namespace"] for r in paths),
        "paths_rescued_by_pod_config": sum(r["by_pod_config"] for r in paths),
        "paths_rejected_final": sum(not r["final"] for r in paths),
        "cases_no_path_accepted_raw": len(by_case) - case_ok("raw"),
        "cases_no_path_accepted_final": len(by_case) - case_ok("final"),
        "rejected_raw_by_category": dict(collections.Counter(
            r["case"].rsplit("-", 1)[0] for r in rejected).most_common()),
        "paths_by_system": dict(collections.Counter(r["case"].split("-")[0] for r in paths)),
        "rejected_final_examples": [f'{r["case"]}/{r["path"]}' for r in paths if not r["final"]][:15],
    }


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = grade_all(data_root())
    print(json.dumps(summarise(rows), indent=2))
    if args.write:
        out = os.path.join("runs", "regrade", "golden-control.jsonl")
        if os.path.exists(out):
            sys.exit(f"{out} exists; not overwriting")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
