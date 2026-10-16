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
| Concurrency | `--concurrency N` runs N episodes at once | `src/run.py` |
| Splits | Fixed train / val / test split per env, saved to a file so every run uses the same one | `src/splits.py`, `--split-file`, `--split` |
| Training | GRPO with verl: each verl sample is one `run_episode`; LoRA on Modal H100s | `train/verl_bridge/` ([README](train/verl_bridge/README.md), [DEBUGGING](train/verl_bridge/DEBUGGING.md)) |
| Environments | `brownout` (toy), `cloudops-boutique` (452 cases), `cloudops-boutique-code` (98 cases) | `src/envs/SystemEnv/` |

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
audit (see "Next" below).

## Results so far: Cloud-OpsBench (Online Boutique)

**Harness check against upstream.** Brownout's labels match upstream's `process-label/` for all
550 cases, and grading follows upstream `evaluation.py` (rank 1 only, lowercase, collapsed
whitespace; JRA = component and fault type both right). One known difference: upstream counts
every tool call against its 20-step limit, while Brownout counts turns, and a turn can hold
several tool calls. DeepSeek V4 Flash scores 0.859 JRA under Brownout's rule, and about 0.78 if
episodes with 20 or more tool calls are counted as failures (an estimate, not a rerun).
Upstream reports 0.76.

**RL on Qwen3-4B** (GRPO with verl, LoRA r64, 40 steps, on the train split). Each model ran once
on every task, greedy decoding, compared case by case (exact McNemar test):

| Cases | Base | After RL | Cases fixed / broken | p |
|---|---|---|---|---|
| All non-code (452) | 0.235 | 0.332 | +89 / −45 | 0.0002 |
| Never trained on: val + test (138) | 0.217 | 0.362 | +30 / −10 | 0.002 |
| Trained on (314) | 0.242 | 0.318 | +59 / −35 | 0.017 |

With the 98 code-defect cases included, base scores 0.195, close to upstream's Qwen3-8B (0.19),
and the RL model scores 0.275, between upstream's Qwen3-8B and Qwen3-14B (0.41). The gains come
from runtime, scheduling, service and startup faults. Admission, performance and code-defect
cases did not improve. Every failure on the way, and its fix, is in
[DEBUGGING.md](train/verl_bridge/DEBUGGING.md).

## Roadmap

Each step has a concrete "done when", so progress is checkable from the repo.

### Done

- **Cloud-OpsBench adapter** (`src/envs/SystemEnv/cloudops/`): Online Boutique, 550 cases with
  recorded tool outputs, so no live cluster is needed. Reward JRA; CA, FA, redundant actions and
  invalid submits as metrics.
- **Concurrency** (`--concurrency`) and **fixed splits** (`--split-file`, `--split`).
- **Baselines:** DeepSeek V4 Flash and Nemotron on every case, with cost.
- **First RL run** with verl on Modal: Qwen3-4B, results above.

### How the grader audit works (no GPU needed)

An RL run only means something if its reward does. A grader can be wrong in two ways: it pays
for an answer that did not solve the task (a false positive, which RL learns to exploit), or it
refuses an answer that did (a false negative, which RL learns to avoid). Asking the grader cannot
catch either, so every check needs a second source of truth:

| Second source | What it is |
|---|---|
| **Oracle** | An independent check, e.g. "did the trajectory contain evidence for this answer?" |
| **Perturbation** | Change a correct answer in ways that should not matter; the reward must not move |
| **Auditor** | A model reads the trajectory without seeing the reward and says whether it was solved |

Every finding gets one tag, and counts only with a repro: a crafted input or a trajectory that
gets the reward without solving the task, or a correct answer that loses it.

| Tag | What fails | Examples |
|---|---|---|
| **LEAK** | The answer is reachable without doing the task | Answer files the tools can read, git history, build timestamps, network access |
| **FALSE POSITIVE** | The grader pays for a non-solution | Shortcuts, inverse attacks (break the test so it passes), judge and rubric injection |
| **FALSE NEGATIVE** | The grader refuses a real solution | Reworded answers, reordered steps, harmless extras, truncated evidence |
| **CALIBRATION** | The task teaches RL nothing | Cases every model always solves, or never solves |

The red team works in two modes:

- **White-box** (finds exploits): agents read the env code, the grader and the task data, build
  inputs that attack it, run the real grader on them, and report only the inputs that the grader
  and the second source disagree on.
- **Black-box** (checks they are reachable): agents get the same tools as the policy and a
  prompt to maximise reward without solving the task. An exploit the policy can reach is rated
  more severe than one it cannot.

Every step below has a "done when". The order is deliberate: first make the audit structure
concrete on three environments that need no containers, then learn containers, then add
container environments, then train.

### Step 1: the audit structure, on three environments without containers

1. **Structure.** Audit checks live in their own module with their own decorators, separate from
   the scoring decorators (`@reward` / `@metric` / `@stop`), so a grader and the checks against
   it never share code. Each check produces findings in one format: tag, evidence (trajectory id
   or crafted input), repro command, fix. A shared metrics library comes with it, so every env
   reports turns, tool calls, invalid and redundant calls, tokens, cost, time and `ended_by` under
   the same names.
   *Done when:* the three audits below use the same decorators, the same finding format and the
   same metrics, with no env-specific code in the audit module.
2. **Cloud-OpsBench.** Shortcut (correct answers with no supporting evidence in the trajectory),
   leak (can any tool read `metadata.json`?), false negatives (does `app/adservice` vs
   `adservice` lose credit?), the turn-vs-tool-call budget, and calibration (cases that are
   always 0 or always 1). It uses the ~1,100 DeepSeek and Nemotron trajectories already saved.
3. **general-agent** (Prime Verifiers, 4,417 tasks; reward = exact final database hash or a
   passing `verify(db)`). This needs a Verifiers adapter. The audit replays each task's gold
   tool-call chain: as is (it must score 1, or the task is broken), then with a harmless read
   added, independent calls reordered, and a harmless extra write. Any replay that loses the
   reward is a false negative. No model calls needed.
4. **math-env** (Prime Verifiers, single turn). Equivalent answer forms (fractions, decimals,
   units, LaTeX) must keep the reward; wrong answers in a correct-looking format must not get it.
   It is also the check that the adapter is faithful: scores must match Verifiers' own eval.

*Done when (all of step 1):* a tagged report per env with a repro for every finding, and the
fixes applied.

### Step 2: containers

5. **Docker, then Kubernetes basics.** A container runtime with four operations (start, exec,
   copy files, stop), first through the local Docker CLI. Grading follows DeepSWE's design: the
   agent works in one container, only a defined artifact (e.g. the committed diff) leaves it, and
   a fresh container from the same pinned image runs the hidden checks, which are never in the
   agent's container. No network in the agent's container.
   *Done when:* a toy env runs through `run_episode` with `--concurrency 4` and leaves no
   containers behind.
6. **Prime sandboxes.** The same four operations on Prime's SDK (`network_allowlist=[]` for no
   network, a run label on every sandbox so one `bulk_delete` cleans up after a crash).
   *Done when:* the toy env gives the same results on Docker and on Prime.

### Step 3: Harbor-format environments

7. **DeepSWE** (DataCurve, 113 tasks, hidden tests in a pristine container), then **tau3**
   (banking, 97 tasks, LLM-simulated customer, database + action + natural-language checks).
   Both are added to the harness and audited with the step 1 structure. DeepSWE: patches that
   change test configuration rather than code, and correct work lost because it was not
   committed. tau3: an agent steering the simulated customer, and how often a task flips on
   reruns.
   *Done when:* 5 tasks of each run end to end on Prime, with their cost measured, before any
   larger run.

### Step 4: training

8. **Miles + SGLang**, one backend for serving and training. Miles replays the rollout's MoE
   routing during training (R3); NeMo-RL with Megatron is the alternative.
   *Done when:* a `--debug-rollout-only` Miles run produces rewards from a Brownout env.
9. **RLVR on corrected graders and traces.** Post-train on the audited environments: fixed
   graders, and teacher traces filtered by the audit (no trajectory that scored through an
   exploit).
10. **VinePPO.** Per-step credit from Monte Carlo rollouts from intermediate states, compared with
    GRPO on the same env.

### Last, before the write-ups

11. **Broken vs fixed grader**, only if GPU credits allow: the same model trained on each with
    the verl bridge, to show the reward climbing on the broken one while the audited solve rate
    stays flat. Then 2-3 blog posts on the audit, the harness and the training results.

### Other ideas (pushed down for now)

Most of these are environments similar to the ones above; the ones above were picked because
they are easier to start with.


- **More audits:** MiMo's code, cyber, general and webdev graders; officeqa; LLM and chain-of-
  thought monitors (for auditing only, never as reward: training against a monitor teaches the
  model to hide what it does).
- **Training environments:** FakeModel tests for every env, and 8 trials of a task at
  temperature > 0 must not all score the same (otherwise GRPO has nothing to learn from).
- **Config files** for runs.
- **Cloud-OpsBench Train-Ticket** and the process metrics (milestone coverage, evidence order).
- **More benchmarks:** ORCA, SRE-gym, OpenEnv environments pulled from the HF Hub.
- **Prompt optimisation** with GEPA.
- **Dashboard:** results for every environment and model on Hugging Face Spaces.
- **Publishing:** four posts on Hugging Face, each linking to the next: (1) the harness and the
  Cloud-OpsBench RL result, after its grader audit; (2) verifying environments: the grader
  audits and what they mean for post-training; (3) Brownout, Verifiers and OpenEnv compared;
  (4) the harness improving its own environments, and VinePPO vs GRPO on agents with RLVR.
  Models trained with this harness published with model cards.
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
