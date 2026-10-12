"""Run every task of a registered env, N trials each, and save the trajectories.

    python -m src.run brownout --trials 3 --model <model_id> --base-url <url>

This is the seed of the auto-agent benchmark: later it gets concurrency (Phase 9),
resume (Phase 6), and several envs/models in one call.
"""

import argparse
from typing import Callable

from .contracts import ModelClient
from .harness import print_trajectory, run_episode, save_trajectory
from .registry import get


def run_benchmark(name: str, make_model: Callable[[], ModelClient], trials: int = 1,
                  out_dir: str | None = "runs", max_turns: int = 8) -> list:
    """Returns every Trajectory. Saves each to out_dir (skip saving if out_dir is None).

    make_model is a factory, not a model: a fresh client per episode (FakeModel keeps
    its script as state, and later episodes will run concurrently).

    TODO(you):
      spec = get(name)
      for each task in spec.tasks().load():
          for run_index in range(trials):
              traj = run_episode(spec.env(), task, make_model(),
                                 max_turns=max_turns, run_index=run_index,
                                 verifier=spec.verifier())
              save it if out_dir, collect it
    """
    raise NotImplementedError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env", help="registered env name, e.g. brownout")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--model", required=True, help="model id sent to the API")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible URL (Nebius, SGLang, ...)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="name of the env var holding the key (never pass the key itself)")
    args = parser.parse_args(argv)

    # TODO(you): build make_model from Clients/openai_compact.py:
    #   lambda: OpenAICompatClient(args.model, args.base_url, os.environ[args.api_key_env])
    # then call run_benchmark(...) and print_trajectory() each result.
    raise NotImplementedError


if __name__ == "__main__":
    main()
