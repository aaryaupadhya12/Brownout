# Brownout roadmap

Results so far are in the [main README](../README.md); how to run the harness is in
[HARNESS.md](HARNESS.md).

Each step has a concrete "done when", so progress is checkable from the repo.

### Done

- **Cloud-OpsBench adapter** (`src/envs/SystemEnv/cloudops/`): Online Boutique, 550 cases with
  recorded tool outputs, so no live cluster is needed. Reward JRA; CA, FA, redundant actions and
  invalid submits as metrics.
- **Concurrency** (`--concurrency`) and **fixed splits** (`--split-file`, `--split`).
- **Baselines:** DeepSeek V4 Flash and Nemotron on every case, with cost.
- **First RL run** with verl on Modal: Qwen3-4B, results in the main README.
- **Strict true-success verifier for Cloud-OpsBench** (`src/envs/SystemEnv/cloudops/strict.py`,
  `vendor/process_eval.py`): JRA plus the label's evidence; audited in both directions and
  against upstream's expert trajectories (`golden_control.py`). A measuring stick, not a training
  reward. Results in the main README.

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
   *Shortcut part done:* see "Grader audit" in the main README (89 of 322 Nemotron hits unconfirmed).
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

