"""One report for a verl training run: per-step metrics from the log, per-episode facts from the
saved trajectories, GRPO group signal, and a list of checks with fixed thresholds. Every check
says what to change if it fails, so a run can be judged without asking anyone.

    python -m train.verl_bridge.run_report --log runs/pilot.log --episodes runs/train-qwen3-4b-pilot-1016-0912
    python -m train.verl_bridge.run_report --log runs/pilot.log                 # log only
    python -m train.verl_bridge.run_report --episodes runs/train-qwen3-4b-smoke  # episodes only

--episodes is the run folder from modal_app.py (train/ and val/ inside it; a flat folder of
.json files also works, then everything counts as "train"). Options: --n (samples per task,
default 4), --batch (tasks per step, default 8), --usd-per-hour (default 5), --steps (planned
full-run steps for the cost projection, default 50), --ram-gb (container RAM, default 96).

Sections
  1. steps     one row per `step:N` line in the log (W&B logs the same numbers)
  2. episodes  per split: JRA / component (ca) / root cause (fa), how episodes ended, length
  3. groups    train only: share of GRPO groups with all-equal rewards (no learning signal)
  4. category  train JRA and signal per task category
  5. checks    PASS / WARN with the rule and the fix
"""

import argparse
import os
import re
import statistics
from collections import Counter, defaultdict

from src.report import category, infra_failed, load, mean

SUFFIX = re.compile(r"\(\s*requires target", re.I)    # label-list annotation copied into the answer
STEP_LINE = re.compile(r"step:(\d+) - (.*)$")


# ---------------------------------------------------------------- 1. log
def parse_log(path: str) -> dict[int, dict[str, float]]:
    """{step: {metric: value}} from verl's console lines `step:N - key:value - key:value ...`.
    A step can print several lines (validation, training); they are merged."""
    steps: dict[int, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_LINE.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if not m:
                continue
            for part in m.group(2).split(" - "):
                key, _, value = part.rpartition(":")
                try:
                    steps[int(m.group(1))][key.strip()] = float(value)
                except ValueError:
                    pass
    return dict(steps)


STEP_COLS = [   # (header, metric key, format)
    ("score", "critic/score/mean", "{:.3f}"),
    ("val", "val-core/cloudops-boutique/reward/mean@1", "{:.3f}"),
    ("len", "response_length/mean", "{:.0f}"),
    ("len_max", "response_length/max", "{:.0f}"),
    ("clip", "response_length/clip_ratio", "{:.2f}"),
    ("entropy", "actor/entropy", "{:.3f}"),
    ("ppo_kl", "actor/ppo_kl", "{:.4f}"),
    ("kl_base", "actor/kl_loss", "{:.4f}"),
    ("clipfrac", "actor/pg_clipfrac", "{:.3f}"),
    ("grad", "actor/grad_norm", "{:.2f}"),
    ("gpu_gb", "actor/perf/max_memory_reserved_gb", "{:.0f}"),
    ("ram_gb", "actor/perf/cpu_memory_used_gb", "{:.0f}"),
    ("t_gen", "timing_s/gen", "{:.0f}"),
    ("t_upd", "timing_s/update_actor", "{:.0f}"),
    ("t_step", "timing_s/step", "{:.0f}"),
]


def steps_table(steps: dict[int, dict[str, float]]) -> str:
    cols = [c for c in STEP_COLS if any(c[1] in s for s in steps.values())]
    lines = ["step " + " ".join(f"{h:>8}" for h, _, _ in cols)]
    for n in sorted(steps):
        cells = [fmt.format(steps[n][k]) if k in steps[n] else "-" for _, k, fmt in cols]
        lines.append(f"{n:>4} " + " ".join(f"{c:>8}" for c in cells))
    return "\n".join(lines)


# ---------------------------------------------------------------- 2-4. episodes
def episodes(folder: str) -> dict[str, list[dict]]:
    """{split: trajectories}. Uses train/ val/ subfolders when present."""
    subs = {s: load(os.path.join(folder, s)) for s in ("train", "val", "unknown")
            if os.path.isdir(os.path.join(folder, s))}
    return subs or {"train": load(folder)}


def signal(t: dict, name: str):
    return (t.get("grader") or {}).get("signals", {}).get(name)


def facts(t: dict) -> dict:
    """The per-episode facts the checks use."""
    sub = (t.get("final_state") or {}).get("submission") or {}
    first = (sub.get("top_3_predictions") or [{}])[0]
    return {
        "jra": (t.get("grader") or {}).get("reward", 0.0),
        "ca": signal(t, "ca"), "fa": signal(t, "fa"),
        "submitted": bool((t.get("final_state") or {}).get("submitted")),
        "context_full": any(e.get("type") == "ContextFull" for e in t.get("errors", [])),
        "suffix": bool(SUFFIX.search(str(first.get("root_cause", "")))),
        "turns": (t.get("totals") or {}).get("turns", 0),
        "out_tokens": (t.get("totals") or {}).get("output_tokens", 0),
        "ended_by": t.get("ended_by", "?"),
        # same rule as agent_loop.to_output: ContextFull is the model's own failure (real 0
        # reward, trained on), not infra
        "infra": infra_failed(t) and not any(e.get("type") == "ContextFull" for e in t.get("errors", [])),
    }


def split_table(by_split: dict[str, list[dict]]) -> str:
    head = (f"{'split':<7}{'n':>5}{'jra':>7}{'ca':>7}{'fa':>7}{'submit':>8}{'ctxfull':>8}"
            f"{'suffix':>8}{'infra':>7}{'turns':>7}{'out_p50':>9}{'out_max':>9}  ended_by")
    lines = [head]
    for split, trajs in by_split.items():
        f = [facts(t) for t in trajs]
        if not f:
            continue
        out = sorted(x["out_tokens"] for x in f)
        ended = ", ".join(f"{k} {v}" for k, v in Counter(x["ended_by"] for x in f).most_common())
        lines.append(
            f"{split:<7}{len(f):>5}{mean([x['jra'] for x in f]):>7.3f}"
            f"{mean([x['ca'] for x in f if x['ca'] is not None]):>7.3f}"
            f"{mean([x['fa'] for x in f if x['fa'] is not None]):>7.3f}"
            f"{mean([x['submitted'] for x in f]):>8.2f}{mean([x['context_full'] for x in f]):>8.2f}"
            f"{mean([x['suffix'] for x in f]):>8.2f}{sum(x['infra'] for x in f):>7}"
            f"{mean([x['turns'] for x in f]):>7.1f}{statistics.median(out):>9.0f}{out[-1]:>9}  {ended}")
    return "\n".join(lines)


def groups(trajs: list[dict], n: int) -> list[list[dict]]:
    """GRPO groups: the n samples of one task in one step. Same task -> same group_id; a task
    drawn again in a later epoch is split off by start time, n at a time."""
    by_task = defaultdict(list)
    for t in trajs:
        by_task[(t.get("tracking") or {}).get("group_id") or t["task_id"]].append(t)
    out = []
    for ts in by_task.values():
        ts.sort(key=lambda t: (t.get("timing") or {}).get("started_at", 0))
        out += [ts[i:i + n] for i in range(0, len(ts), n)]
    return out


def group_kind(g: list[dict]) -> str:
    rewards = {facts(t)["jra"] for t in g}
    if len(rewards) > 1:
        return "mixed"
    return "all_1" if rewards == {1.0} else "all_0"


def group_table(gs: list[list[dict]]) -> str:
    kinds = Counter(group_kind(g) for g in gs)
    total = max(len(gs), 1)
    lines = [f"groups {len(gs)}:  mixed {kinds['mixed']} ({kinds['mixed'] / total:.0%})   "
             f"all-0 {kinds['all_0']} ({kinds['all_0'] / total:.0%})   "
             f"all-1 {kinds['all_1']} ({kinds['all_1'] / total:.0%})   (only mixed groups train)"]
    sizes = Counter(len(g) for g in gs)
    if len(sizes) > 1:
        lines.append(f"  group sizes {dict(sizes)}: partial groups = episodes still missing")
    return "\n".join(lines)


def category_table(trajs: list[dict], gs: list[list[dict]]) -> str:
    by_cat = defaultdict(list)
    for t in trajs:
        by_cat[category(t["task_id"])].append(facts(t))
    mixed = defaultdict(list)
    for g in gs:
        mixed[category(g[0]["task_id"])].append(group_kind(g) == "mixed")
    lines = [f"{'category':<26}{'n':>5}{'jra':>7}{'ca':>7}{'fa':>7}{'mixed':>8}"]
    for cat in sorted(by_cat):
        f = by_cat[cat]
        lines.append(f"{cat:<26}{len(f):>5}{mean([x['jra'] for x in f]):>7.3f}"
                     f"{mean([x['ca'] for x in f if x['ca'] is not None]):>7.3f}"
                     f"{mean([x['fa'] for x in f if x['fa'] is not None]):>7.3f}"
                     f"{mean(mixed[cat]):>8.0%}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 5. checks
def checks(steps, by_split, gs, args) -> list[tuple[str, str, str]]:
    """(PASS/WARN/INFO, what was measured, rule -> fix)."""
    out = []
    train = [facts(t) for t in by_split.get("train", [])]
    last = steps[max(steps)] if steps else {}

    if gs:
        mixed = mean([group_kind(g) == "mixed" for g in gs])
        out.append(("PASS" if mixed >= 0.5 else "WARN", f"mixed groups {mixed:.0%}",
                    ">= 50% of groups should be mixed. Below: little gradient -> drop all-0 "
                    "categories from training (make_dataset) or enable DAPO filter_groups"))
    if train:
        cf = mean([x["context_full"] for x in train])
        out.append(("PASS" if cf <= 0.10 else "WARN", f"context_full {cf:.0%}",
                    "<= 10%. Above: raise data.max_response_length or lower BROWNOUT_TOKENS_PER_TURN"))
        sub = mean([x["submitted"] for x in train])
        out.append(("PASS" if sub >= 0.8 else "WARN", f"submitted {sub:.0%}",
                    ">= 80%. Below: episodes run out of turns; check ended_by, consider a submit nudge"))
        sfx = mean([x["suffix"] for x in train])
        out.append(("INFO", f"label suffix in answers {sfx:.0%}",
                    "'(Requires Target ...)' copied into root_cause = scored wrong. If this falls "
                    "during training, part of the reward gain is format, not diagnosis: report both"))
        infra = sum(x["infra"] for x in train)
        out.append(("PASS" if infra == 0 else "WARN", f"infra failures {infra}",
                    "0. Above: model-side errors (masked, no gradient); read their errors[]"))
    if last:
        ent = last.get("actor/entropy")
        if ent is not None:
            out.append(("PASS" if ent >= 0.02 else "WARN", f"entropy {ent:.3f} (last step)",
                        ">= 0.02. Below: collapsing, groups go flat -> raise temperature or lower lr"))
        cfr = last.get("actor/pg_clipfrac")
        if cfr is not None:
            out.append(("PASS" if cfr <= 0.2 else "WARN", f"pg_clipfrac {cfr:.3f}",
                        "<= 0.2. Above: updates too large -> lower actor.optim.lr"))
        ram = last.get("actor/perf/cpu_memory_used_gb")
        if ram is not None:
            out.append(("PASS" if ram <= 0.85 * args.ram_gb else "WARN", f"host RAM {ram:.0f} / {args.ram_gb} GB",
                        "<= 85%. Above: risk of the OOM killer -> raise memory= in modal_app.py"))
        t = [s["timing_s/step"] for s in steps.values() if "timing_s/step" in s]
        if t:
            per = statistics.median(t)
            hours = per * args.steps / 3600
            out.append(("INFO", f"{per / 60:.1f} min/step -> {args.steps} steps = {hours:.1f} h "
                        f"= ${hours * args.usd_per_hour:.0f}",
                        f"at ${args.usd_per_hour}/h. Too much: fewer steps, shorter max_response_length, "
                        "or a cheaper GPU provider for the full run"))
    scores = [s["critic/score/mean"] for _, s in sorted(steps.items()) if "critic/score/mean" in s]
    if len(scores) >= 3:
        k = max(1, len(scores) // 3)
        first, last3 = mean(scores[:k]), mean(scores[-k:])
        out.append(("INFO", f"train score first {k} steps {first:.3f} -> last {k} {last3:.3f}",
                    "noisy over few steps; trust the val column and the full run, not a 5-step pilot"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log")
    p.add_argument("--episodes")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--usd-per-hour", type=float, default=5.0)
    p.add_argument("--ram-gb", type=float, default=96)
    args = p.parse_args(argv)
    if not args.log and not args.episodes:
        p.error("give --log and/or --episodes")

    steps = parse_log(args.log) if args.log else {}
    by_split = episodes(args.episodes) if args.episodes else {}
    gs = groups(by_split.get("train", []), args.n)

    if steps:
        print("== 1. steps (from the log)\n" + steps_table(steps) + "\n")
    if by_split:
        print("== 2. episodes\n" + split_table(by_split) + "\n")
    if gs:
        print("== 3. GRPO groups (train)\n" + group_table(gs) + "\n")
        print("== 4. by category (train)\n" + category_table(by_split["train"], gs) + "\n")
    print("== 5. checks")
    for status, what, rule in checks(steps, by_split, gs, args):
        print(f"[{status:<4}] {what}\n         {rule}")


if __name__ == "__main__":
    main()
