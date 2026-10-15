"""Train / val / test splits of an env's tasks, fixed once and saved in the repo.

Why a saved file and not "split on the fly": the split must never move. If test cases
leak into training (or a checkpoint is picked on test), the result is inflated and nobody
can tell. A file under splits/ is reviewed in a PR like code, and every run names it.

    python -m src.splits make cloudops-boutique cloudops-boutique-code \\
        --stratify answer.root_cause --out splits/cloudops-boutique.json
    python -m src.splits show splits/cloudops-boutique.json

    python -m src.run cloudops-boutique ... --split-file splits/cloudops-boutique.json --split train
    python -m src.report --split-file splits/cloudops-boutique.json --split test

Generic: any env works. A split is a list of task ids per part, so one file can cover
several envs (task ids are unique across them). --stratify is a dotted path into
task.spec ("answer.root_cause" -> spec["answer"]["root_cause"]): every stratum with 2+
tasks gets at least one task in train and one in test, so no class is train-only.
Nothing here changes task_hash or group_id: a split only selects which tasks run.

The file stores ids only (no task data), so it is safe to commit for any dataset.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from datetime import date
from typing import Callable, Iterable

from .interface_v1 import Task
from .registry import get
from .store import write_json

PARTS = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}


def spec_key(path: str) -> Callable[[Task], str]:
    """'answer.root_cause' -> a function task -> str(task.spec['answer']['root_cause']).
    A missing key raises KeyError naming the task and the path."""
    def key(task: Task) -> str:
        value = task.spec
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"task {task.id!r} has no spec field {path!r}")
            value = value[part]
        return str(value)
    return key


def make_split(tasks: Iterable[Task], stratify: Callable[[Task], str] | None = None,
               fractions: dict[str, float] = DEFAULT_FRACTIONS, seed: int = 0) -> dict[str, list[str]]:
    groups = defaultdict(list)

    for task in tasks:
        stratum = stratify(task) if stratify is not None else "all"
        groups[stratum].append(task.id)

    result = {part: [] for part in PARTS}
    for stratum, ids in groups.items():
        ids.sort()

        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(ids)
        n = len(ids)

        test_count = round(n * fractions["test"])
        val_count = round(n * fractions["val"])

        if n >= 2:
            # They must have atleast 1 test task 
            test_count = max(1,test_count)

        test_count = min(test_count, max(0, n -1)) # atleast 1 training task
        val_count = max(0, min(val_count, n - test_count - 1)) # val must leave 1 training task too

        result["test"].extend(ids[:test_count])
        result["val"].extend(ids[test_count:test_count + val_count])
        result["train"].extend(ids[test_count + val_count:]) # train keeps the rest
    
    for part in PARTS:
        result[part] = sorted(result[part])
    return result

def save_split(path: str, split: dict[str, list[str]], meta: dict) -> None:
    """Write the split with how it was made (envs, stratify, fractions, seed, date)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_json(path, {**meta, "created": date.today().isoformat(),
                      "counts": {p: len(split[p]) for p in PARTS}, "splits": split})


def load_split(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if set(data.get("splits", {})) != set(PARTS):
        raise ValueError(f"{path}: 'splits' must have exactly {PARTS}")
    return data


def select(tasks: Iterable[Task], split_file: str, part: str) -> Iterable[Task]:
    if part not in PARTS:
        raise ValueError(f"part {part!r} must be one of {PARTS}")
    data = load_split(split_file)
    wanted = set(data["splits"][part])
    # Get back to this to understand the TASK split # Aarya 
    known = {i for p in PARTS for i in data["splits"][p]}
    for task in tasks:
        if task.id not in known:
            raise ValueError(f"task {task.id!r} is in no part of {split_file} (dataset changed after the split?)")
        if task.id in wanted:
            yield task


def describe(data: dict, stratify: Callable[[Task], str] | None = None,
             tasks: Iterable[Task] = ()) -> str:
    lines = [f"split of {', '.join(data.get('envs', []))}  seed={data.get('seed')}  "
             f"stratify={data.get('stratify')}  created={data.get('created')}",
             "  " + "  ".join(f"{p}={len(data['splits'][p])}" for p in PARTS)]
    if stratify is not None:
        part_of = {i: p for p in PARTS for i in data["splits"][p]}
        table = defaultdict(Counter)
        for t in tasks:
            table[stratify(t)][part_of.get(t.id, "missing")] += 1
        width = max(map(len, table), default=10) + 2
        lines.append("  " + "stratum".ljust(width) + "".join(p.rjust(8) for p in (*PARTS, "missing")))
        for s in sorted(table):
            lines.append("  " + s.ljust(width) + "".join(str(table[s][p]).rjust(8)
                                                         for p in (*PARTS, "missing")))
    return "\n".join(lines)


def all_tasks(env_names: list[str]) -> list[Task]:
    return [t for name in env_names for t in get(name).tasks().load()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make", help="make a split and save it")
    make.add_argument("envs", nargs="+", help="registered env names")
    make.add_argument("--out", required=True, help="e.g. splits/cloudops-boutique.json")
    make.add_argument("--stratify", default=None, help="dotted path into task.spec")
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--fractions", default="0.7,0.1,0.2", help="train,val,test")
    show = sub.add_parser("show", help="print a saved split (per stratum with --envs)")
    show.add_argument("path")
    args = parser.parse_args(argv)

    if args.cmd == "make":
        fractions = dict(zip(PARTS, map(float, args.fractions.split(","))))
        if abs(sum(fractions.values()) - 1) > 1e-9:
            parser.error("--fractions must sum to 1")
        stratify = spec_key(args.stratify) if args.stratify else None
        tasks = all_tasks(args.envs)
        split = make_split(tasks, stratify, fractions, args.seed)
        save_split(args.out, split, {"envs": args.envs, "stratify": args.stratify,
                                     "fractions": fractions, "seed": args.seed})
        print(describe(load_split(args.out), stratify, tasks))
    else:
        data = load_split(args.path)
        stratify = spec_key(data["stratify"]) if data.get("stratify") else None
        print(describe(data, stratify, all_tasks(data.get("envs", [])) if stratify else ()))


if __name__ == "__main__":
    main()
