# Brownout: the harness

How to run it, how to add an environment, what exists, and how it compares.
Results are in the [main README](../README.md); plans in [ROADMAP.md](ROADMAP.md); a one-page map
with diagrams and every command in [SYSTEM.md](SYSTEM.md).

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
git). Windows/PowerShell setup is in [RUN.md](../RUN.md).

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
[src/envs/README.md](../src/envs/README.md). `src/envs/SystemEnv/brownout/` is the worked example.

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
| Concurrency | `--concurrency N` runs N episodes at once | `src/run.py` |
| Splits | Fixed train / val / test split per env, saved to a file so every run uses the same one | `src/splits.py`, `--split-file`, `--split` |
| Training | GRPO with verl: each verl sample is one `run_episode`; LoRA on Modal H100s | `train/verl_bridge/` ([README](../train/verl_bridge/README.md), [DEBUGGING](../train/verl_bridge/DEBUGGING.md)) |
| Environments | `brownout` (toy), `cloudops-boutique` (452 cases), `cloudops-boutique-code` (98 cases) | `src/envs/SystemEnv/` |

## How it compares: Verifiers, OpenEnv and Brownout

Brownout is a small harness, written from scratch to learn how RL environment frameworks work.
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) (Prime Intellect) and
[OpenEnv](https://github.com/huggingface/OpenEnv) (Hugging Face) are the mature references.
"Planned" means not built yet; see [ROADMAP.md](ROADMAP.md).

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
| **Concurrency** | Async throughout | Server-side, `SUPPORTS_CONCURRENT_SESSIONS` per env | Episodes in parallel threads (`--concurrency`) |
| **Sandboxes** | Docker, Modal, E2B, Prime, Apptainer, subprocess runtimes | Several providers (uv, Docker, Modal, Daytona, HF...) | None yet: envs run in-process; planned: local Docker, then Prime sandboxes |
| **Adapters to other ecosystems** | Harbor, OpenEnv, NeMo Gym, TextArena tasksets | Importers for other env formats | Cloud-OpsBench (done); planned: OpenEnv envs from the HF Hub, MiMo RL envs |
| **Training** | prime-rl | None (env-only); used by trainers such as Miles | verl GRPO (done); planned: SGLang + Miles |
| **Grader hacking defenses** | Not a built-in step (as far as I found) | `Gate` rubrics; RFC 004 names "anti-hacking rubrics" but gives no design | Planned: a red-team audit that proves each exploit (see roadmap) |
| **Grader false negatives** | Not covered | Not covered | Planned: perturbation tests (reword, reorder, harmless extras must keep the reward) |
| **Size** | Large, production | Large, production | ~2,000 lines of Python, readable in an afternoon |

**Where Brownout differs on purpose**

- **The verifier never touches the env.** The env reports facts with `snapshot()`, and all pass/fail
  thresholds live in `task.spec`. Any saved trajectory can be re-scored with a new verifier, offline.
- **Adding an env means adding a folder.** There is nothing to install, no server and no config entry.
- **Failure is data.** Every error records where it happened, so an infrastructure failure isn't
  mistaken for the model failing. That matters for RL, where the first means "drop the sample" and
  the second means "reward 0".

**Where Brownout is behind:** no sandboxes, no async rollouts and few ecosystem adapters yet.

**Where Brownout is heading:** none of the three frameworks checks whether its own graders can
be gamed. OpenEnv offers `Gate` rubrics, and HUD sells an automated QA layer for this, but
neither publishes a method that proves an exploit works. The next part of Brownout is that
audit (see [ROADMAP.md](ROADMAP.md)).

## About this project

Brownout is built from scratch on purpose. The goal is to understand every layer of an RL
environment stack (environment interface, scoring, rollouts, serving and training) rather than
to wrap an existing framework. The design keeps one rule: a benchmark is one registry entry, and
the harness, runner and exporters never know which benchmark they are running.

Some code in this repository was written with AI assistance; the design and the learning are the
point. Criticism is welcome: if something is wrong, overlooked or could be simpler, please open
an issue. Help with GPU access for experiments would also be greatly appreciated.

Contributions should be substantive. Pull requests that only pad statistics will be closed.
