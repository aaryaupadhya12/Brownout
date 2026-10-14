# Brownout

A harness for building, benchmarking and training on RL environments for agents. One loop
(`src/harness.py`) runs any environment; each environment is a folder under `src/envs/`.

## Quickstart

```bash
pip install openai
cp .env.example .env                                         # then put your key in .env (git-ignored)
python -m unittest discover -s tests -p "*_test*.py"         # runs offline, no API calls
python -m src.run brownout --provider nebius --model <model-id> --trials 1
python -m src.run brownout --provider nebius --model <model-id> --trials 1 --resume   # skips finished episodes
```

Providers (base URL + the *name* of the key variable) are in `configs/providers.toml`; key
values live only in `.env` or a Modal secret. Trajectories are written to `runs/` (ignored by
git). Windows/PowerShell setup is in [RUN.md](RUN.md).

Useful flags: `--max-turns`, `--max-tokens` (total tokens per episode), `--max-seconds`
(wall clock per episode), `--out`, `--resume`.

`brownout` is a small toy environment (route traffic away from a degraded zone). It is cheap,
fast and almost always solvable, so use it to check that a new provider, model or adapter is
wired up correctly before running a real benchmark.

## Creating a new environment

Each environment is one folder under `src/envs/`. The registry finds it on its own, so
the harness and the runner never change:

1. Copy `src/envs/_template/` to e.g. `src/envs/CodingEnv/myenv/` (no leading `_`).
2. Fix the relative imports: one more `.` per folder level below `src/envs/`.
3. Fill in `env.py` (tools, state, `system_prompt`, `snapshot()`), `tasks.py` (every
   threshold in `task.spec`) and `verifier.py` (`@reward` / `@metric` / `@stop`, reading only
   `(task, trace)`).
4. Set the name in `__init__.py`: `register(EnvSpec("myenv", MyTasks, MyEnv, MyVerifier))`.
5. Test the grader with `FakeModel`: a correct run, a different correct run and a wrong run.
6. Run it: `python -m src.run myenv --provider nebius --model <model-id>`

The full guide, with the rules for each file and a checklist, is in
[src/envs/README.md](src/envs/README.md). `src/envs/SystemEnv/brownout/` is the worked example.

## What the harness does today

| Area | What exists | Where |
|---|---|---|
| Env interface | `Task` / `TaskSource` / `Env` / `Verifier`; the verifier reads only `(task, trace)` | `src/contracts.py`, `src/scoring.py` |
| Registry | Every folder under `src/envs/` with an `__init__.py` is an env; `_template` is skipped | `src/registry.py` |
| Scoring | `@reward(weight)`, `@metric`, `@stop`; a crashed scorer gives `grader.ok=False`, not reward 0 | `src/scoring.py` |
| Limits | `max_turns`, `max_total_tokens`, `max_seconds`, checked before every model call; the reason is saved as `ended_by` | `src/limits.py` |
| Comparable runs | `group_id` = hash of task + config: same id means same task and same settings | `src/tracking.py` |
| Resume | Atomic writes; `--resume` reruns only missing, torn or infrastructure-failed episodes | `src/store.py` |
| LLM judge | Criteria with ordered choices mapped to 0..1, fenced answer, temperature 0, cached; an unreadable verdict is a judge failure | `src/judge.py` |
| Secrets | Config holds only variable names; values come from `.env` or the environment; errors name the variable, never the value | `src/providers.py` |
| Cost | Token usage and price per episode | `src/pricing.py` |

## How it compares: Verifiers, OpenEnv and Brownout

Brownout is a small harness, written from scratch to learn how RL environment frameworks work.
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) (Prime Intellect) and
[OpenEnv](https://github.com/huggingface/OpenEnv) (Hugging Face) are the mature references.
"Planned" means not built yet; see the roadmap.

| | Verifiers (v1) | OpenEnv | Brownout |
|---|---|---|---|
| **What an env is** | A Python package: `Taskset` + `Harness` + `Env` | An HTTP/WebSocket server with typed `Action` / `Observation` / `State` | A folder: `env.py`, `tasks.py`, `verifier.py`, `__init__.py` |
| **Who runs the agent loop** | The framework, or any outside agent (Codex, Claude Code...) behind its interception proxy | Not OpenEnv: the caller (e.g. a Miles agent function) drives `reset` / `step` | One loop, `run_episode` in `src/harness.py` |
| **Finding an env by name** | `id` → Python module import (`verifiers/v1/utils/loaders.py`) | Installed `openenv-*` packages + `openenv.yaml`, `AutoEnv.from_env` / `from_hub` | Every folder under `src/envs/` with an `__init__.py` (`src/registry.py`) |
| **Scoring** | `@reward` / `@metric` on the task, rubric and judge classes | Inside the env (e.g. an `evaluate` action), plus `Rubric` containers and an LLM judge | `@reward` / `@metric` verifier that reads only `(task, trajectory)`, never the env; optional LLM judge |
| **Re-grade a saved run** | Yes, from the stored trace | Not by design: the score comes from live env state | Yes: the env's final state is saved in `trajectory.final_state` |
| **Stop conditions / limits** | `@stop` hooks, `RolloutLimits` (turns, tokens), per-stage `RolloutTimeouts` | Left to the caller | `@stop` hooks and `RolloutLimits` (turns, total tokens, wall-clock seconds) |
| **Failures** | Typed errors; a failed rollout is still data | Errors returned to the client | `ErrorRecord(where=model/env/scorer/harness)`; a crashed scorer marks the run `ok=False` rather than reward 0 |
| **Comparable runs** | Results stored per run, resumable | Not covered | `task_hash` + `config_hash` → `group_id`; resumable with `--resume` |
| **Concurrency** | Async throughout | Server-side, `SUPPORTS_CONCURRENT_SESSIONS` per env | Sequential; planned |
| **Sandboxes** | Docker, Modal, E2B, Prime, Apptainer, subprocess runtimes | Several providers (uv, Docker, Modal, Daytona, HF...) | None yet: envs run in-process |
| **Adapters to other ecosystems** | Harbor, OpenEnv, NeMo Gym, TextArena tasksets | Importers for other env formats | Planned: Cloud-OpsBench, OpenEnv envs from the HF Hub, a Miles agent function |
| **Training** | prime-rl | None (env-only); used by trainers such as Miles | Planned: SGLang + Miles |
| **Size** | Large, production | Large, production | ~2,000 lines of Python, readable in an afternoon |

**Where Brownout differs on purpose**

- **The verifier never touches the env.** The env reports facts with `snapshot()`, and all pass/fail
  thresholds live in `task.spec`. Any saved trajectory can be re-scored with a new verifier, offline.
- **Adding an env means adding a folder.** There is nothing to install, no server and no config entry.
- **Failure is data.** Every error records where it happened, so an infrastructure failure isn't
  mistaken for the model failing. That matters for RL, where the first means "drop the sample" and
  the second means "reward 0".

**Where Brownout is behind:** no sandboxes, no concurrency, no trainer integration and no
ecosystem adapters yet.

## Roadmap

Each step has a concrete "done when", so progress is checkable from the repo.

### Now: environments and metrics

1. **Training environments.** Write the first environments meant for training, not only for
   evaluation.
   *Done when:* each has a FakeModel test for a correct run, a different correct run and a wrong
   run, and running 8 trials of one task at temperature > 0 gives rewards that are not all equal
   (otherwise GRPO has nothing to learn from).
2. **A shared metrics library.** Common `@metric` helpers any env can reuse, so the dashboard
   can show the same columns across environments: turns, tool calls, invalid tool calls,
   redundant (repeated) calls, tokens, cost, wall-clock time, `ended_by`.
   *Done when:* an env gets these metrics by importing them, without writing them again, and
   every saved trajectory reports them under the same names.
3. **Cloud-OpsBench adapter** (`src/envs/SystemEnv/cloudops/`). 754 Kubernetes root-cause cases
   (Online Boutique and Train-Ticket) with recorded tool outputs, so no live cluster is needed.
   Outcome rewards: component, fault type and both (CA / FA / JRA); process metrics: milestone
   coverage, evidence order, evidence efficiency, redundant actions.
   *Done when:* scoring the benchmark's own reference trajectories (`golden-trajectory/`) with
   Brownout's verifier gives the same numbers as the benchmark's `evaluation.py`.

### Next: scale

4. **Concurrency.** Run 3-4 episodes at once, with per-provider rate limits.
   *Done when:* a 50-case run finishes in about a quarter of the sequential time, with
   identical saved results.
5. **Config files.** Describe a benchmark run (env, models, trials, limits) in one file instead
   of CLI flags. *Done when:* a run can be reproduced from its config file alone.
6. **First baseline.** Cloud-OpsBench on 2-3 open models, with a fixed train/test split by case.
   *Done when:* results are published with per-category scores and cost.

### Then: post-training

7. **SGLang serving.** Serve an open model with SGLang (single GPU, on Modal) and run the
   benchmarks against it through the `sglang` provider.
8. **Miles bridge.** Expose `run_episode` as a Miles agent function: a dropped sample (`None`)
   for infrastructure failures, reward 0 for model failures.
   *Done when:* a `--debug-rollout-only` Miles run produces rewards from a Brownout env.
9. **First small RL run** on a budget (about $100 of GPU time), then a report comparing before
   and after on the test split.

### Later

- **More benchmarks:** ORCA (judge-scored), OpenEnv environments pulled from the HF Hub.
- **Sandboxes:** a runtime for environments that execute code (coding and agentic tool tasks).
- **Prompt optimisation** with GEPA.
- **Dashboard:** results for every environment and model on Hugging Face Spaces.
- **Publishing:** a write-up comparing Verifiers, OpenEnv and Brownout, and models trained with
  this harness published with model cards.
- **Research directions:** combining small decision models with LLMs to lower GPU load, and RL
  training methods for MoE models.

## About this project

Brownout is built from scratch on purpose. The goal is to understand every layer of an RL
environment stack (environment interface, scoring, rollouts, serving and training) rather than
to wrap an existing framework. The design keeps one rule: a benchmark is one registry entry, and
the harness, runner and exporters never know which benchmark they are running.

Some code in this repository was written with AI assistance; the design and the learning are the
point. Criticism is welcome: if something is wrong, overlooked or could be simpler, please open
an issue. Help with GPU access for experiments would also be greatly appreciated.

Contributions should be substantive. Pull requests that only pad statistics will be closed.
