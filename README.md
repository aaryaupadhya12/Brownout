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