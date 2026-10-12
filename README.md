# Brownout

A harness for building and benchmarking RL environments for agents. One loop
(`src/harness.py`) runs any environment; each environment is a folder under `src/envs/`.

## Quickstart

```bash
pip install openai
python -m unittest discover -s tests -p "*_test*.py"          # everything but client_test runs offline
python -m src.run brownout --trials 3 --model <model-id>     --base-url <openai-compatible-url> --api-key-env NEBIUS_API_KEY
```

Trajectories are written to `runs/` (ignored by git). Windows/PowerShell setup is in [RUN.md](RUN.md).

## Creating a new environment

Each environment is one folder under `src/envs/`. The registry finds it on its own, so
the harness and the runner never change:

1. Copy `src/envs/_template/` to e.g. `src/envs/CodingEnv/myenv/` (no leading `_`).
2. Fix the relative imports: one more `.` per folder level below `src/envs/`.
3. Fill in `env.py` (tools, state, `system_prompt`, `snapshot()`), `tasks.py` (every
   threshold in `task.spec`) and `verifier.py` (`@reward` / `@metric`, reading only
   `(task, trace)`).
4. Set the name in `__init__.py`: `register(EnvSpec("myenv", MyTasks, MyEnv, MyVerifier))`.
5. Test the grader with `FakeModel`: a correct run, a different correct run and a wrong run.
6. Run it: `python -m src.run myenv --model ... --base-url ...`

The full guide, with the rules for each file and a checklist, is in
[src/envs/README.md](src/envs/README.md). `src/envs/SystemEnv/brownout/` is the worked example.

## How it compares: Verifiers, OpenEnv and Brownout

Brownout is a deliberately small harness, written from scratch to learn how RL environment
frameworks work. [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) (Prime Intellect)
and [OpenEnv](https://github.com/huggingface/OpenEnv) (Hugging Face) are the mature references.
"Planned" means not built yet.

| | Verifiers (v1) | OpenEnv | Brownout |
|---|---|---|---|
| **What an env is** | A Python package: `Taskset` + `Harness` + `Env` | An HTTP/WebSocket server with typed `Action` / `Observation` / `State` | A folder: `env.py`, `tasks.py`, `verifier.py`, `__init__.py` |
| **Who runs the agent loop** | The framework, or any outside agent (Codex, Claude Code...) behind its interception proxy | Not OpenEnv: the caller (e.g. a Miles agent function) drives `reset` / `step` | One loop, `run_episode` in `src/harness.py` |
| **Finding an env by name** | `id` → Python module import (`verifiers/v1/utils/loaders.py`) | Installed `openenv-*` packages + `openenv.yaml`, `AutoEnv.from_env` / `from_hub` | Every folder under `src/envs/` with an `__init__.py` (`src/registry.py`) |
| **Scoring** | `@reward` / `@metric` on the task, rubric and judge classes | Inside the env (e.g. an `evaluate` action), plus `Rubric` containers and an LLM judge | `@reward` / `@metric` verifier that reads only `(task, trajectory)`, never the env |
| **Re-grade a saved run** | Yes, from the stored trace | Not by design: the score comes from live env state | Yes: the env's final state is saved in `trajectory.final_state` |
| **Stop conditions / limits** | `@stop` hooks, `RolloutLimits` (turns, tokens), per-stage `RolloutTimeouts` | Left to the caller | `max_turns` and `submit` only; token limits, `@stop` and timeouts planned |
| **Failures** | Typed errors; a failed rollout is still data | Errors returned to the client | Failed rollouts kept, with `ErrorRecord(where=model/env/scorer/harness)`; a crashed scorer marks the run `ok=False` rather than reward 0 |
| **Comparable runs** | Results stored per run, resumable | Not covered | `task_hash` + `config_hash` → `group_id`: same id means same task and same settings |
| **Concurrency** | Async throughout | Server-side, `SUPPORTS_CONCURRENT_SESSIONS` per env | Sequential; 3-4 parallel episodes planned |
| **Sandboxes** | Docker, Modal, E2B, Prime, Apptainer, subprocess runtimes | Several providers (uv, Docker, Modal, Daytona, HF...) | None yet: envs run in-process |
| **Adapters to other ecosystems** | Harbor, OpenEnv, NeMo Gym, TextArena tasksets | Importers for other env formats | Planned: OpenEnv envs from the HF Hub, and a Miles agent function |
| **Training** | prime-rl | None (env-only); used by trainers such as Miles | Planned: SGLang + Miles |
| **Size** | Large, production | Large, production | ~1,300 lines of Python, readable in an afternoon |

**Where Brownout differs on purpose**

- **The verifier never touches the env.** The env reports facts with `snapshot()`, and all pass/fail
  thresholds live in `task.spec`. Any saved trajectory can be re-scored with a new verifier, offline.
- **Adding an env means adding a folder.** There is nothing to install, no server and no config entry.
- **Failure is data.** Every error records where it happened, so an infrastructure failure isn't
  mistaken for the model failing. That matters for RL, where the first means "drop the sample" and
  the second means "reward 0".

**Where Brownout is behind:** no sandboxes, no async or concurrency, no token or time limits yet,
no LLM judge, and no trainer integration. See the roadmap below.

## About this project

The architecture is Simple Every benhcamrk (targeted at the end of october is 30-40 High value benchmarks ) in one registry entry . The harness , runner and exporters never know which benchmark they are running 
This is made delibrately 
2nd What we are testing next on this benchmarks 
Laya Models will nbe publisehd through this harness itself as its hte only OpenMode in the harness 
Find a way to compsoe decision Models + Large Language models for ccertain tasks and benchamrk them to see if lowers the GPU load 
RLMS and MOE trainignt Methods in Rl enviornments to see how tehy work , this is the main thing that we have in store for the next release 
Completly use a comaptbaole way to integrate RL envirobemnts in Buth the HF RL hbs and Prime Hubs For firectly pulling the environments from the Hub rather than pulling commands a full frameowkr and Push it back to Hugginface with Model cards (we are making the wrapper happen)
Next also we are trying to write a compatible sandox as of r4ight now we have given where sandox of you type can be added but havent written one oruself and im going through the otpions and in looks for a sandbox for higher leve 
If anyone can Help with GPU access for Expermentation It Would be really helpfull thanks a Lot , and appreciate all the help i can get , please critisze where i overlooked , where AI wrote code im here to learn not to claim im the best in the world , goal is to just make better things better as a genuine commity , if your here to add more slop pleaes take a reverse back to another repo for stat-padding thankyou Bye 