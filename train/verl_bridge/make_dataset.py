"""Write verl's dataset files from a Brownout split: one row per task id.

    python -m train.verl_bridge.make_dataset --split-file splits/cloudops-boutique.json \\
        --out data/verl/cloudops-boutique

    -> data/verl/cloudops-boutique/train.parquet  (381 rows)
       data/verl/cloudops-boutique/val.parquet    (53 rows)
       data/verl/cloudops-boutique/test.parquet   (116 rows)
    test.parquet is never trained on: only the eval stage of pilot.sh reads it (verl
    validation alone, base vs a checkpoint, the same path as in training).

A row only names the task. The agent loop (agent_loop.py) rebuilds the real prompt from
the env, so the system prompt and tool schemas are exactly the benchmark's. `prompt` is
a short placeholder because verl's dataset requires the column (and filters on its length).

Needs pandas + pyarrow (the GPU box has them with verl). Data rows hold ids only.
"""

import argparse
import os

from src.registry import get
from src.report import category
from src.splits import load_split


def rows(split_file: str, part: str, exclude: set[str] = frozenset()) -> list[dict]:
    """`exclude`: task categories (src.report.category, e.g. "boutique-admission") left out."""
    data = load_split(split_file)
    env_of = {t.id: name for name in data["envs"] for t in get(name).tasks().load()}
    out = []
    for i, task_id in enumerate(data["splits"][part]):
        if task_id not in env_of:
            raise ValueError(f"{task_id!r} is in {split_file} but in none of {data['envs']}")
        if category(task_id) in exclude:
            continue
        out.append({
            "data_source": env_of[task_id],
            "agent_name": "brownout",                      # selects BrownoutAgentLoop
            "prompt": [{"role": "user", "content": task_id}],
            "reward_model": {"style": "rule", "ground_truth": ""},   # unused: the loop scores
            "extra_info": {"env": env_of[task_id], "task_id": task_id, "index": i, "split": part},
        })
    return out


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-train", default="",
                        help="comma-separated task categories to leave out of TRAIN only "
                             "(val/test keep everything), e.g. boutique-admission,boutique-codedefect")
    args = parser.parse_args(argv)
    exclude = {c.strip() for c in args.exclude_train.split(",") if c.strip()}
    os.makedirs(args.out, exist_ok=True)
    for part in ("train", "val", "test"):
        path = os.path.join(args.out, f"{part}.parquet")
        table = rows(args.split_file, part, exclude if part == "train" else frozenset())
        pd.DataFrame(table).to_parquet(path)
        print(f"{path}: {len(table)} rows" + (f" (excluded {sorted(exclude)})" if part == "train" and exclude else ""))


if __name__ == "__main__":
    main()
