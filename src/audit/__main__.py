"""python -m src.audit <env> [--layers 1 2] [--limit N] [--out runs/audits]"""

import argparse
import os
import time

from .judge import JudgeAudit
from .report import Report
from .target import TARGETS
from .universal import UniversalAudit

LAYERS = {"1": UniversalAudit, "2": JudgeAudit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=sorted(TARGETS))
    parser.add_argument("--layers", nargs="+", default=["1"], choices=sorted(LAYERS))
    parser.add_argument("--limit", type=int, default=None, help="audit only the first N tasks")
    parser.add_argument("--out", default=os.path.join("runs", "audits"))
    args = parser.parse_args()

    target = TARGETS[args.env]()
    tasks = target.tasks()
    if args.limit is not None:
        tasks = tasks[:args.limit]

    report = Report(env=args.env, layers=args.layers, grader=target.grader_name())
    for layer in args.layers:
        LAYERS[layer]().run(target, tasks, report)

    path = os.path.join(args.out, f"{args.env}-{time.strftime('%m%d-%H%M')}.json")
    report.write(path)
    print(report.summary())
    print(f"report: {path}")


if __name__ == "__main__":
    main()
