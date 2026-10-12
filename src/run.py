"""Run every task of a registered env, N trials each, and save the trajectories.

    python -m src.run brownout --trials 3 --model <model_id> --base-url <url>

This is the seed of the auto-agent benchmark: later it gets concurrency (Phase 9),
resume (Phase 6), and several envs/models in one call.
"""

import argparse
from typing import Callable
import os 
from .contracts import ModelClient
from .harness import print_trajectory, run_episode, save_trajectory
from .limits import RolloutLimits
from .registry import get


def run_benchmark(name: str, make_model: Callable[[], ModelClient], trials: int = 1,
                  out_dir: str | None = "runs", max_turns: int = 8, limits=None) -> list:
    value = get(name)
    trajectories = []
    for task in value.tasks().load():
        for index in range(trials):
            traj = run_episode(value.env(), scenario= task, model = make_model(), max_turns = max_turns, ctx = None , run_index = index, verifier= value.verifier() , temperature= 0.0, limits=limits)
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
    parser.add_argument("--model", required=True, help="model id sent to the API")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible URL (Nebius, SGLang, ...)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="name of the env var holding the key (never pass the key itself)")
    args = parser.parse_args(argv)

    # imported here so run_benchmark (and its tests) don't need the openai package
    from Clients.openai_compact import OpenAICompatClient

    make_model = lambda : OpenAICompatClient(args.model, args.base_url, os.environ[args.api_key_env])
    limits = RolloutLimits(args.max_turns, args.max_tokens, args.max_seconds)
    trajectories = run_benchmark(name = args.env, make_model=make_model, trials=args.trials,out_dir=args.out,max_turns=args.max_turns, limits=limits)
    for traj in trajectories:
        print_trajectory(traj)


if __name__ == "__main__":
    main()
