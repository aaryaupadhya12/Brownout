# Creating a new environment

An environment is **one folder** under `src/envs/`. The harness, the runner and the
registry never change when you add one: the registry finds every folder that has an
`__init__.py` and imports it, and that import registers the env under a name.

```
src/envs/
  _template/            copy this (folders starting with "_" are skipped by the registry)
  SystemEnv/brownout/   worked example
  CodingEnv/            put coding envs here
  PuzzleEnv/            put puzzle envs here
```

## The four pieces

| File | What it is | Rule |
|---|---|---|
| `env.py` | The world the agent acts on: tools and state | Reports facts in `snapshot()`, never pass/fail |
| `tasks.py` | Where problems come from: `load()` yields `Task`s | Every threshold the verifier uses lives in `task.spec` |
| `verifier.py` | The score: `@reward` and `@metric` methods | Reads only `(task, trace)`, never the env |
| `__init__.py` | `register(EnvSpec(name, Tasks, Env, Verifier))` | Pass the classes, not instances |

The interfaces are defined in `src/interface_v1.py`.

## Step by step

### 1. Copy the template

```bash
cp -r src/envs/_template src/envs/CodingEnv/myenv
```

The new folder name must not start with `_`.

### 2. Fix the relative imports

Each file imports from `src/` with dots. Use **one more dot per folder level below `src/envs/`**:

| Folder | Import |
|---|---|
| `src/envs/myenv/` | `from ...registry import ...` |
| `src/envs/CodingEnv/myenv/` | `from ....registry import ...` |

The same goes for `interface_v1` and `scoring`.

### 3. Write the env (`env.py`)

```python
class MyEnv:
    system_prompt = "You are ... The tools are ... Call submit() when done."

    def reset(self, task):           # build the world from task.spec, return the first message
    def step(self, tool, args):      # run one tool call, return a dict (errors as {"error": ...})
    def tool_schema(self):           # OpenAI-format tool schemas the model sees
    def snapshot(self):              # facts about the final state, copied into trace.final_state
    def close(self):                 # free files, sandboxes, connections
    done: bool                       # True once the episode is over (e.g. after submit)
```

- **A new env object is made for every episode.** Keep all state on `self` and set it up in `reset()`.
- **The system prompt belongs to the env.** The harness has no env-specific text, and the prompt is part of the run's `config_hash`.
- **Tool errors are results, not exceptions.** Return `{"error": "..."}` so the model can see it and recover. Raise only for real bugs, and the harness records those as `ErrorRecord("env", ...)`.
- **Never show the answer.** Nothing the agent can read (tool output, files, the first message) may contain the solution or the grading rules.

### 4. Write the tasks (`tasks.py`)

```python
class MyTasks:
    def load(self):
        for spec in TASKS:
            yield Task(id=spec["id"], seed=0, spec=spec)
```

`task.spec` holds everything needed to rebuild the problem, including the thresholds the verifier checks. The env and the verifier then read the same numbers, and `task_hash` changes whenever a task changes.

### 5. Write the verifier (`verifier.py`)

```python
class MyVerifier(Verifier):
    name = "myenv"
    version = "0.1.0"
    needs_env = False                 # scorers take (task, trace)

    @reward(weight=0.7)
    def solved(self, task, trace):    # 0.0 to 1.0
        return 1.0 if trace.final_state["tests_passed"] else 0.0

    @reward(weight=0.3)
    def efficient(self, task, trace):
        return max(0, 1 - trace.final_state["steps_used"] / task.spec["max_steps"])

    @metric
    def tool_errors_seen(self, task, trace):   # recorded, never added to the reward
        return sum("error" in s.observation for s in trace.steps)
```

- `reward = sum(score × weight)`, so keep the weights summing to 1.
- Scorers read `trace.final_state` (from `snapshot()`), `trace.steps`, `trace.messages` and `task.spec`. Because nothing reads the env, any saved run can be re-graded later.
- If a scorer raises, the run gets `grader.ok = False`, which means "leave it out of averages", not a score of 0.

### 6. Register it (`__init__.py`)

```python
register(EnvSpec("myenv", MyTasks, MyEnv, MyVerifier))
```

Names must be unique, and a duplicate raises `ValueError`. Check that it's found:

```bash
python -c "from src.registry import names; print(names())"
```

### 7. Test the grader before any model sees it

Use `FakeModel` to script at least three runs. A good grader passes both correct runs and fails the wrong one:

| Script | Expected |
|---|---|
| The obvious correct solution | high reward |
| A **different** correct solution (other route or wording) | high reward too |
| A plausible **wrong** solution | low reward |

```python
from src.fake_model import FakeModel, call, scripted
from src.run import run_benchmark

GOOD = [scripted(call("c1", "my_tool", {"x": 1})), scripted(call("c2", "submit", {}))]
trajs = run_benchmark("myenv", lambda: FakeModel(GOOD), out_dir=None)
assert trajs[0].grader.reward > 0.9
```

See `tests/interface_test.py` for Brownout's version: `good` and `one_shift` both pass, and `overload` scores 0.

### 8. Run it on a real model

```bash
python -m src.run myenv --trials 3 --provider nebius --model <model-id>    # providers: configs/providers.toml
```

Trajectories are saved to `runs/` (which git ignores). Read a few, **especially the perfect scores**: a perfect score that took a shortcut means the grader has a hole.

## Checklist before publishing an env

- [ ] The prompt reads like a real request, and nothing the agent sees reveals the answer
- [ ] All thresholds are in `task.spec`; the verifier never calls the env
- [ ] A correct solution, a different correct solution and a wrong solution all score as expected
- [ ] Every score has been run under a named model and trial count (difficulty means nothing otherwise)
- [ ] Perfect-score trajectories have been read
- [ ] `close()` frees anything `reset()` opened

These criteria follow HUD's "what is a good task": realistic, fair, hard
(https://www.hud.ai/research/what-is-a-good-task).
