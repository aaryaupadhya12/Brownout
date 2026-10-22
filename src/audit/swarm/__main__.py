"""python -m src.audit.swarm <env> --campaign NAME --provider P --model M [--solver-model M2]
                             [--planted HOLE_ID] [--tasks ID,ID | --limit N]
                             [--max-rounds R] [--attempts A]
                             [--specialists generic,answer-format,episode-end] [--workers N]

--planted runs against a planted-hole verifier (src/audit/swarm/planted.py) instead of the
real one, to measure recall. This calls model APIs: it costs money.

Worst-case episodes = tasks x probers x max_rounds x attempts. It is printed before the run.
Raising --max-rounds later on the same campaign continues where it stopped: role keys do not
contain max_rounds, so finished rounds load from disk.
"""

import argparse
import os

from ...registry import get
from .loop import SwarmConfig, campaign_lock, run_campaign
from .planted import HOLES
from .roles import FOCUS, Specialist


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--provider", required=True, help="a section of configs/providers.toml")
    parser.add_argument("--model", required=True, help="attacker model id")
    parser.add_argument("--solver-model", default=None, help="defaults to --model")
    parser.add_argument("--planted", default=None, choices=[h.hole_id for h in HOLES])
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task ids, e.g. boutique-admission-2 (overrides --limit)")
    parser.add_argument("--max-rounds", type=int, default=SwarmConfig.max_rounds,
                        help="rounds per task before giving up (default %(default)s)")
    parser.add_argument("--attempts", type=int, default=SwarmConfig.attempts,
                        help="attacker attempts per round (default %(default)s)")
    parser.add_argument("--out", default=os.path.join("runs", "swarm"))
    parser.add_argument("--saved-runs", default="runs",
                        help="precheck reuses solved baseline runs from here; 'none' = always run the solver")
    parser.add_argument("--no-plan-turn", action="store_true",
                        help="give the attacker tools on turn 1 (default: turn 1 is a text-only plan)")
    parser.add_argument("--specialists", default="generic",
                        help="comma-separated focus areas: generic,answer-format,episode-end")
    parser.add_argument("--workers", type=int, default=1,
                        help="how many (task, prober) pairs run at once (default 1 = serial)")
    parser.add_argument("--true-success", default="same", choices=("same", "strict"),
                        help="the second referee. same = the env's own verifier (no reward gap "
                             "possible, LESSONS 13); strict = cloudops StrictCloudOpsVerifier "
                             "(JRA + evidence), so JRA=1 / strict=0 counts as a finding")
    args = parser.parse_args(argv)

    from ...providers import make_client   # reads the key from .env

    spec = get(args.env)
    cfg = SwarmConfig(campaign=args.campaign, env=args.env, out_dir=args.out,
                      saved_runs=None if args.saved_runs.lower() == "none" else args.saved_runs,
                      plan_turn=not args.no_plan_turn,
                      max_rounds=args.max_rounds, attempts=args.attempts)
    true_success = spec.verifier()
    verifier = true_success
    if args.true_success == "strict":
        if not args.env.startswith("cloudops"):
            parser.error("--true-success strict exists only for cloudops envs")
        from ...envs.SystemEnv.cloudops.strict import StrictCloudOpsVerifier
        true_success = StrictCloudOpsVerifier()     # attacker still sees only `verifier` (JRA)
    if args.planted:
        verifier = next(h for h in HOLES if h.hole_id == args.planted).verifier()

    all_tasks = list(spec.tasks().load())
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        by_id = {t.id: t for t in all_tasks}
        unknown = [t for t in wanted if t not in by_id]
        if unknown:                      # fail before any model call, not halfway through
            parser.error(f"unknown task ids for {args.env}: {unknown}")
        tasks = [by_id[t] for t in wanted]
    else:
        tasks = all_tasks[:args.limit]
    if not tasks:
        parser.error("no tasks selected")

    names = [s.strip() for s in args.specialists.split(",") if s.strip()]
    unknown = [n for n in names if n not in FOCUS]
    if unknown:
        parser.error(f"unknown specialists {unknown}; choose from {sorted(FOCUS)}")
    specialists = [Specialist(n, FOCUS[n]) for n in names]

    # one client per prober (same provider/model here; swap per-prober models later if wanted)
    attacker_clients = {n: make_client(args.provider, args.model) for n in names}
    solver_model = make_client(args.provider, args.solver_model or args.model)

    ledger_path = os.path.join(cfg.out_dir, cfg.campaign, "ledger.jsonl")
    worst = len(tasks) * len(specialists) * cfg.max_rounds * cfg.attempts
    print(f"{len(tasks)} tasks x {len(specialists)} probers x {cfg.max_rounds} rounds x "
          f"{cfg.attempts} attempts = at most {worst} episodes "
          f"(a task's probers stop at its first finding)")

    with campaign_lock(cfg):
        results = run_campaign(cfg, tasks, attacker_clients, solver_model=solver_model,
                               verifier=verifier, true_success=true_success,
                               ledger_path=ledger_path, specialists=specialists,
                               workers=args.workers)

    by_task = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r)
    for tid in sorted(by_task):
        statuses = ", ".join(f"{r['specialist']}:{r['status']}" for r in by_task[tid])
        print(f"{tid}: {statuses}")
    print(f"ledger: {ledger_path}")


if __name__ == "__main__":
    main()