"""Summarise saved trajectories on the terminal: one table per (verifier, model), one row per
task category, plus an ALL row.

    python -m src.report                          # everything in runs/
    python -m src.report runs --filter boutique   # only task ids containing "boutique"
    python -m src.report --signals ca,fa          # pick the signal columns
    python -m src.report --failures               # list the failed episodes too
    python -m src.report --split-file splits/cloudops-boutique.json --split test

Category = the task id without its trailing "-<number>" (boutique-admission-7 -> boutique-admission).
The reward is shown by its @reward component names (cloudops: jra); with several components
there is a "reward" column (the weighted sum) plus one per component. Means are over episodes the grader scored (grader.ok); infra failures
(an error where="model", e.g. the API was down) are counted apart, never averaged in as 0.
Reads only the saved JSON, so it works on a run that is still going.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

DEFAULT_SIGNALS = 6          # signal columns shown when --signals is not given


def load(folder: str, pattern: str = "") -> list[dict]:
    """Every saved trajectory in `folder` whose task id contains `pattern`.
    Half-written or unreadable files are skipped (write_json is atomic, so these are rare)."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "task_id" in data and pattern in data["task_id"]:
            out.append(data)
    return out


def category(task_id: str) -> str:
    return re.sub(r"-\d+$", "", task_id)


def infra_failed(t: dict) -> bool:
    return t.get("ended_by") == "error" and any(e.get("where") == "model" for e in t.get("errors", []))


def summarise(trajs: list[dict], signals: list[str]) -> dict:
    """One row: counts, reward and signal means, how episodes ended, cost and time."""
    scored = [t for t in trajs if (t.get("grader") or {}).get("ok") and not infra_failed(t)]
    row = {"n": len(trajs), "scored": len(scored), "infra": sum(map(infra_failed, trajs))}
    row["reward"] = mean([t["grader"]["reward"] for t in scored])
    for name in reward_names(trajs):
        row[name] = mean([t["grader"]["rewards"][name]["score"] for t in scored
                          if name in (t["grader"].get("rewards") or {})])
    for s in signals:
        row[s] = mean([t["grader"]["signals"][s] for t in scored
                       if isinstance(t["grader"].get("signals", {}).get(s), (int, float))])
    ended = Counter(t.get("ended_by") for t in trajs)
    row["submit%"] = ended.get("submit", 0) / len(trajs) if trajs else None
    row["cost"] = sum((t.get("totals") or {}).get("api_cost_usd") or 0 for t in trajs)
    row["min/ep"] = mean([(t["timing"]["finished_at"] - t["timing"]["started_at"]) / 60
                          for t in trajs if t.get("timing", {}).get("finished_at")])
    return row


def reward_names(trajs: list[dict]) -> list[str]:
    """The verifier's @reward components (e.g. jra), in first-seen order."""
    names = []
    for t in trajs:
        for name in ((t.get("grader") or {}).get("rewards") or {}):
            if name not in names:
                names.append(name)
    return names


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pick_signals(trajs: list[dict], wanted: str | None) -> list[str]:
    if wanted:
        return [s.strip() for s in wanted.split(",") if s.strip()]
    seen = Counter()
    for t in trajs:
        for k, v in ((t.get("grader") or {}).get("signals") or {}).items():
            if isinstance(v, (int, float)):
                seen[k] += 1
    return [k for k, _ in seen.most_common()][:DEFAULT_SIGNALS]


def fmt(v, width: int, money=False) -> str:
    if v is None:
        return "-".rjust(width)
    if money:
        return f"${v:.2f}".rjust(width)
    if isinstance(v, float):
        return f"{v:.2f}".rjust(width)
    return str(v).rjust(width)


def table(title: str, groups: dict[str, list[dict]], signals: list[str]) -> str:
    everything = [t for g in groups.values() for t in g]
    names = reward_names(everything)
    # one component with the whole weight: its column IS the reward, so show it by name
    single = len(names) == 1
    cols = ["n", "infra", *(names if single else ["reward", *names]), *signals,
            "submit%", "min/ep", "cost"]
    name_w = max([len("ALL"), *map(len, groups)]) + 2
    widths = [max(len(c), 7) + 1 for c in cols]
    lines = [title, "-" * (name_w + sum(widths)),
             "category".ljust(name_w) + "".join(c.rjust(w) for c, w in zip(cols, widths))]
    for name in sorted(groups):
        lines.append(render(name, summarise(groups[name], signals), cols, widths, name_w))
    if len(groups) > 1:
        lines.append(render("ALL", summarise(everything, signals), cols, widths, name_w))
    return "\n".join(lines)


def render(name, row, cols, widths, name_w) -> str:
    return name.ljust(name_w) + "".join(fmt(row[c], w, money=(c == "cost"))
                                        for c, w in zip(cols, widths))


def failures(trajs: list[dict]) -> str:
    lines = ["failed episodes (reward 0 or infra failure):"]
    for t in sorted(trajs, key=lambda t: t["task_id"]):
        g = t.get("grader") or {}
        if infra_failed(t) or not g.get("reward"):
            sub = (t.get("final_state") or {}).get("submission") or {}
            first = (sub.get("top_3_predictions") or [{}])[0]
            guess = f"{first.get('fault_object')} {first.get('root_cause')}" if first else ""
            errors = "; ".join(f"{e.get('where')}:{e.get('type')}" for e in t.get("errors", []))
            lines.append(f"  {t['task_id']:32} {t.get('ended_by', ''):10} {guess} {errors}".rstrip())
    return "\n".join(lines)


def report(folder="runs", pattern="", signals=None, show_failures=False, split=None) -> str:
    trajs = load(folder, pattern)
    if split is not None:                      # (split_file, part): only that part's tasks
        from .splits import load_split
        wanted = set(load_split(split[0])["splits"][split[1]])
        trajs = [t for t in trajs if t["task_id"] in wanted]
    if not trajs:
        return f"no trajectories in {folder!r}" + (f" matching {pattern!r}" if pattern else "")
    by_run = defaultdict(list)
    for t in trajs:
        verifier = ((t.get("tracking") or {}).get("verifier") or {}).get("name", "?")
        by_run[(verifier, t.get("model_id", "?"))].append(t)
    out = []
    for (verifier, model), group in sorted(by_run.items()):
        by_cat = defaultdict(list)
        for t in group:
            by_cat[category(t["task_id"])].append(t)
        out.append(table(f"{verifier}  |  {model}", by_cat, pick_signals(group, signals)))
        if show_failures:
            out.append(failures(group))
    return "\n\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="runs")
    parser.add_argument("--filter", default="", help="only task ids containing this text")
    parser.add_argument("--signals", default=None, help="comma-separated signal columns")
    parser.add_argument("--failures", action="store_true", help="also list failed episodes")
    parser.add_argument("--split-file", default=None, help="a saved split (src/splits.py)")
    parser.add_argument("--split", default=None, choices=("train", "val", "test"))
    args = parser.parse_args(argv)
    if (args.split_file is None) != (args.split is None):
        parser.error("--split-file and --split go together")
    split = (args.split_file, args.split) if args.split else None
    print(report(args.folder, args.filter, args.signals, args.failures, split))


if __name__ == "__main__":
    main()
