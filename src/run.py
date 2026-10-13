"""Run every task of a registered env, N trials each, and save the trajectories.

    python -m src.run brownout --trials 3 --provider nebius --model <model_id>
    python -m src.run brownout --trials 3 --provider nebius --model <model_id> --resume

Providers and their key variables are in configs/providers.toml (keys go in .env).
--resume skips episodes already saved in --out (src/store.py:is_done).
This is the seed of the auto-agent benchmark: later it gets concurrency (Phase 9)
and several envs/models in one call.
"""

import argparse
from typing import Callable
from .contracts import ModelClient
from .harness import plan_episode, print_trajectory, run_episode, save_trajectory
from .limits import RolloutLimits
from .registry import get
from .store import episode_path, is_done


def run_benchmark(name: str, make_model: Callable[[], ModelClient], trials: int = 1,
                  out_dir: str | None = "runs", max_turns: int = 8, limits=None,
                  resume: bool = False) -> list:
    """Returns the trajectories that ran this time (skipped ones are not included)."""
    value = get(name)
    limits = limits or RolloutLimits(max_turns=max_turns)
    trajectories = []
    for task in value.tasks().load():
        for index in range(trials):
            env, model, verifier = value.env(), make_model(), value.verifier()

            # Phase 6: where this episode would be saved, known before running it
            plan = plan_episode(env, task, model, verifier, limits, temperature=0.0)
            path = None
            if out_dir is not None:
                path = episode_path(out_dir, plan["task_id"], model.model_id,
                                    plan["tracking"]["group_id"], index)
            # resume: skip episodes already saved with a real result
            if resume and path is not None and is_done(path):
                continue

            traj = run_episode(env, scenario= task, model = model, max_turns = max_turns, ctx = None , run_index = index, verifier= verifier , temperature= 0.0, limits=limits)
            trajectories.append(traj)
            if out_dir is not None:
                save_trajectory(traj, folder=out_dir) 
    return trajectories


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
    args = parser.parse_args(argv)

    from .providers import make_client   # reads the key from .env / the environment

    make_model = lambda : make_client(args.provider, args.model)
    limits = RolloutLimits(args.max_turns, args.max_tokens, args.max_seconds)
    trajectories = run_benchmark(name = args.env, make_model=make_model, trials=args.trials,out_dir=args.out,max_turns=args.max_turns, limits=limits, resume=args.resume)
    for traj in trajectories:
        print_trajectory(traj)


if __name__ == "__main__":
    main()
