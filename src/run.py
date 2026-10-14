"""Run every task of a registered env, N trials each, and save the trajectories.

    python -m src.run brownout --trials 3 --provider nebius --model <model_id>
    python -m src.run brownout --trials 3 --provider nebius --model <model_id> --resume
    python -m src.run brownout --trials 3 --provider nebius --model <model_id> --concurrency 4

Providers and their key variables are in configs/providers.toml (keys go in .env).
--resume skips episodes already saved in --out (src/store.py:is_done).
--concurrency N runs N episodes at once (Phase 9). Later: several envs/models in one call.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from .contracts import ModelClient
from .harness import plan_episode, print_trajectory, run_episode, save_trajectory
from .limits import RolloutLimits
from .registry import get
from .store import episode_path, is_done

def run_benchmark(name: str, make_model: Callable[[], ModelClient], trials: int = 1,
                  out_dir: str | None = "runs", max_turns: int = 8, limits=None,
                  resume: bool = False, concurrency: int = 1) -> list:
    """Returns the trajectories that ran this time (skipped ones are not included),
    in plan order (task order, then trial index), whatever order they finished in.

    concurrency: how many episodes run at the same time (threads). An episode spends
    almost all its time waiting on the model's HTTP reply, and waiting threads release
    the GIL, so threads are enough: no asyncio, nothing to pickle. Every job has its own
    env/model/verifier and its own file, and write_json is atomic, so no locks.
    Rate limits: each client retries a 429 itself (Clients/openai_compact.py).
    """
    value = get(name)
    limits = limits or RolloutLimits(max_turns=max_turns)

    def plan_jobs():
        """Yield (env, task, model, verifier, index) for every episode still to run."""
        for task in value.tasks().load():
            for index in range(trials):
                env, model, verifier = value.env(), make_model(), value.verifier()

               
                plan = plan_episode(env, task, model, verifier, limits, temperature=0.0)
                path = None
                if out_dir is not None:
                    path = episode_path(out_dir, plan["task_id"], model.model_id,
                                        plan["tracking"]["group_id"], index)
                # resume: skip episodes already saved with a real result
                if resume and path is not None and is_done(path):
                    continue
                yield env, task, model, verifier, index

    def run_one(job):
        env, task, model, verifier, index = job
        traj = run_episode(env, scenario=task, model=model, max_turns=max_turns, ctx=None,
                           run_index=index, verifier=verifier, temperature=0.0, limits=limits)
        if out_dir is not None:
            save_trajectory(traj, folder=out_dir)
        return traj

    if concurrency <= 1:
        return [run_one(job) for job in plan_jobs()]

    # Phase 9: submit while planning and keep no list of jobs, so a finished episode's
    # env (and the case data it loaded) can be freed while the others still run
    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one, job): position
                   for position, job in enumerate(plan_jobs())}
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    traj = future.result()
                except Exception as e:      # a bug: model/env errors are already ErrorRecords
                    print(f"[{done}/{len(futures)}] episode failed: {e!r}")
                    continue
                results[futures[future]] = traj
                print(f"[{done}/{len(futures)}] {traj.task_id}  reward={traj.grader.reward}"
                      f"  ended_by={traj.ended_by}")
        except KeyboardInterrupt:
            # finished episodes are saved: --resume picks up the rest. Episodes already
            # running finish first (a thread can't be killed); queued ones are cancelled
            print("interrupted: waiting for the running episodes, cancelling the rest")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [results[i] for i in sorted(results)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env", help="registered env name, e.g. brownout")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None, help="stop an episode past this many total tokens")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop an episode after this many seconds")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--resume", action="store_true", help="skip episodes already saved in --out")
    parser.add_argument("--model", required=True, help="model id sent to the API")
    parser.add_argument("--provider", required=True,
                        help="a section of configs/providers.toml, e.g. nebius, hf, sglang")
    parser.add_argument("--concurrency", type=int, default=1, help="episodes run at the same time")
    args = parser.parse_args(argv)

    from .providers import make_client   # reads the key from .env / the environment

    make_model = lambda : make_client(args.provider, args.model)
    limits = RolloutLimits(args.max_turns, args.max_tokens, args.max_seconds)
    trajectories = run_benchmark(name = args.env, make_model=make_model, trials=args.trials,out_dir=args.out,max_turns=args.max_turns, limits=limits, resume=args.resume,
                                 concurrency=args.concurrency)
    for traj in trajectories:
        print_trajectory(traj)


if __name__ == "__main__":
    main()
