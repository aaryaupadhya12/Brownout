All seperte Environment Config Files to be set here , incase a easier way to test with API keys or other Internal Test can be done through envs folder rather than Run the whole systesm froms scratch espeocallay for direct Agent Interactions , This woud not hold for the Trainer -> so Please done change this 

## Adding an env

Every env is one folder under `src/envs/` (grouped by kind: `SystemEnv/`, `CodingEnv/`, `PuzzleEnv/`).
The harness never imports an env directly; the registry finds each folder that has an `__init__.py`.

1. Copy `src/envs/_template/` to e.g. `src/envs/CodingEnv/myenv/` (no leading `_`).
2. Fill in `env.py` (incl. `system_prompt`), `tasks.py`, `verifier.py`; set the name in `__init__.py`.
   Fix the relative imports: one more `.` per folder level below `src/envs/`.
3. Run it: `python -m src.run myenv --model <id> --base-url <url>`.

Brownout (`SystemEnv/brownout/`) is the worked example.
