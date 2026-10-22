# Brownout: project context for Claude Code

## What this repo is
Brownout is a small RL-environment harness (about 2,000 lines, written from scratch to learn the stack).
One loop, `run_episode` in `src/harness.py`, runs any environment. Each environment is a folder under
`src/envs/`, found by `src/registry.py`. Scoring uses `@reward` / `@metric` / `@stop` in `src/scoring.py`.
Trajectories are saved to `runs/` (git-ignored). The README has the full table.

## What I am building now
A reward-hacking audit for RL environments, then training on the audited environments. The goal is to find
ways an agent can get reward without solving the task, fix them, and prove the fix did not break honest
solutions. First target: `cloudops-boutique` (Cloud-OpsBench, 452 non-code cases, recorded tool outputs,
no live cluster). Later: Harbor/MiMo tasks (needs containers, not started).

The plan follows two papers: BenchJack (audit a benchmark: recon, flaw scan, working exploit) and
"Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops". Roles:
- auditor: reads env + verifier code, walks BenchJack's flaw classes, writes a ledger
- hacker: tries to get reward without solving; blind first, then with the verifier source
- fixer: edits only the verifier and env, never task answers or instructions
- solver: an honest, investigating agent; a patch ships only if the solver still passes
- adjudicator: labels a passing attack hack or legitimate, confirmed by replay
The orchestrator is plain code (budgets, resume, dedupe, ledger), not an LLM.
A hack counts only with a runnable repro that earns reward on the unpatched env.

## Design rules (do not break these)
- The verifier reads only `(task, trace)`; it never touches the env. Env facts come through `snapshot()`;
  thresholds live in `task.spec`.
- Audit checks live in their own module with `@check` and must not share code with the scoring decorators.
- Failure is data: a crashed scorer gives `grader.ok=False`, not reward 0. Keep infra errors separate
  from model failures.
- Batching several tool calls in one turn is intentional: Brownout counts turns, not tool calls. Do not flag it.
- Saved runs in `runs/` are records. Never edit them. Re-grade into new files instead.
- Keep it small and readable. No new dependencies without asking.

## Trace format (do not cat whole trajectories: each is about 100 KB)
Keys: task_id, model_id, steps[], messages[], calls[], grader{reward, ok, signals}, final_state{submitted,
submission}, totals, tracking, config (includes the full system prompt and tool schemas, repeated).
- A message is `messages[i].payload` with `role` in system / user / assistant / tool.
- A tool result is a `role: "tool"` message whose `content` is JSON text `{"output": ...}`.
- The system prompt lists every valid root cause (List A) and resource name (List B). So any label string
  appears in every run's prompt by design. A leak means the label appears in a `role: "tool"` message.
  Scan tool messages only.
- The submission is `final_state.submission.top_3_predictions`; grading uses rank 1 only.
- Known harmless: `final_state.tool_calls` excludes Submit while `grader.signals.tool_calls` includes it.
- Known to fix: version labels disagree (`versions.grader` 0.2.0 vs `grader.grader_version` and
  `tracking.verifier.version` 0.1.0; see Current status for why it waits), and some runs came from a
  dirty commit.

## Token-saving rules
- Read this file first. Do not re-scan the whole repo. Find things with grep (`class UniversalAudit`,
  `class CloudOpsTarget`, `def run_episode`).
- Never open `runs/` files whole. Extract fields, for example:
  `python -c "import json,sys; t=json.load(open(sys.argv[1])); print(t['final_state']['submission'])" <file>`
  or run `python check_trace.py <file>` if it exists.
- Run tests offline: `python -m unittest discover -s tests -p "*_test*.py"`. Do not call model APIs or start
  training without asking, because that costs money.
- Make small diffs, show them, and stop. Ask before changing a grader or verifier.
- Reports: short. Say what you changed, what you ran, and what you did not verify.

## Current status (2026-10-22)
Map of the repo with commands: `Docs/SYSTEM.md`. Results: `README.md` and
`findings/cloudops-unconfirmed-passes/`.
- Built: the strict true-success verifier (`cloudops/strict.py`, measuring stick only, never a
  training reward), its audit (`audit_judge.py`, `golden_control.py`), and the swarm orchestrator
  (`src/audit/swarm/`, plain code) with planted holes and `--true-success strict`.
- Layer 1 (`universal`) and Layer 2 (`judge`) audits exist. Layer 2 skips: no judge is wired in yet.
- `leak_in_observations` is fixed: it scans tool-role messages only.
- `universal.calibration` flags tasks where both saved runs got the same reward. It is a post-training
  prep list, not an audit failure.
- Version labels still disagree (`contracts.VERSIONS["grader"]` 0.2.0 vs `CloudOpsVerifier.version`
  0.1.0). Deliberately not fixed yet: `strict_sha` hashes `verifier.py`, so editing it changes the
  pinned measuring stick `strict-6f0952b5`. Fix it together with the next planned verifier change.
- Next: whether RL on JRA teaches the unconfirmed-pass shortcut (pre-register first).
- The user is preparing for interviews and re-coding components to learn them: prefer teaching,
  quizzing and reviewing their code over writing core code (see `interview/08_recode_plan.md`).