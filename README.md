# Brownout

A small harness for building, benchmarking and training on RL environments for agents, and for
auditing whether their rewards can be earned without solving the task. Results below.

How it fits together, with every command: [Docs/SYSTEM.md](Docs/SYSTEM.md). How to run it and add
environments: [Docs/HARNESS.md](Docs/HARNESS.md). Plans: [Docs/ROADMAP.md](Docs/ROADMAP.md).

## How it works

```mermaid
flowchart LR
    subgraph ENV["Environment: one folder under src/envs/"]
        T["tasks.py<br/>answer in task.spec"]
        E["env.py<br/>tools + snapshot()"]
        V["verifier.py<br/>@reward @metric @stop"]
    end
    M["Model client"] --> H["run_episode<br/>src/harness.py"]
    T --> H
    E --> H
    V -->|"reads only task + trace"| H
    H -->|"trajectory"| R[("runs/<br/>saved records")]
    R --> REP["report.py<br/>results tables"]
    R --> S["strict verifier<br/>re-grade offline"]
    S --> J["LLM judges +<br/>expert-path control"]
    J --> F["findings/"]
    H --> SW["swarm<br/>attacker vs 2 referees"]
    H --> TR["verl GRPO<br/>1 sample = 1 episode"]
```

The verifier never touches the environment, so any saved run can be scored again by a new
verifier, offline and for free. That is what made the audit below possible.

**One episode** (`run_episode`): the loop every environment, eval and training sample goes
through. Budgets are checked before every model call, and a crashed scorer is recorded as
`ok=False`, never as reward 0.

```mermaid
sequenceDiagram
    participant H as harness
    participant M as model
    participant E as env
    participant V as verifier
    H->>E: reset(task) -> first observation
    loop until submit / max_turns / max_tokens / max_seconds  (limits.py)
        H->>M: messages + tool schemas
        M-->>H: assistant turn (text and/or tool calls)
        H->>E: run every tool call in the turn
        E-->>H: tool outputs
    end
    H->>E: snapshot() -> final_state
    H->>V: score(task, trace)
    V-->>H: reward, signals, ok (a crashed scorer = ok False, not reward 0)
    H->>H: save trajectory (atomic write, group_id = hash(task + config))
```

```powershell
python -m unittest discover -s tests -p "*_test*.py"                                # all offline
python -m src.run cloudops-boutique --provider nebius --model <id> --concurrency 8   # paid
python -m src.report runs --filter boutique                                         # free
python -m src.envs.SystemEnv.cloudops.regrade_strict                                # free
```

Every command, free or paid, with diagrams of one episode, the audit and the swarm:
[Docs/SYSTEM.md](Docs/SYSTEM.md).

## Cloud-OpsBench (Online Boutique)

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

### Grader audit: JRA pays for diagnoses the model never confirmed

JRA checks only the final answer. A stricter verifier (`strict.py`) also requires the evidence
that Cloud-OpsBench's own process labels list for the case (e.g. "list the nodes with their
labels"), using a port of upstream's milestone matcher that reproduces upstream's evaluator on all
1,508 expert trajectories. Re-graded over the 1,095 saved runs (saved runs untouched):

| Model | JRA | Strict | Correct answers never confirmed | 95% CI |
|---|---|---|---|---|
| Nemotron-3-Nano-30B | 0.586 | 0.424 | **89 of 322 (27.6%)** | 23.0%–32.8% |
| DeepSeek-V4-Flash | 0.859 | 0.859 | **0 of 468** | 0%–0.8% |

```mermaid
flowchart TD
    R[("1,095 saved runs")] --> J["JRA (outcome reward)<br/>rank-1 component + root cause"]
    R --> S["strict.py<br/>JRA AND the label's answer milestones"]
    L[("Cloud-OpsBench process labels<br/>milestone.json")] --> P["vendor/process_eval.py<br/>port of upstream matcher"]
    P --> S
    J --> G{"JRA 1, strict 0?"}
    S --> G
    G -->|yes| U["unconfirmed pass<br/>(Nemotron 89 / 322, DeepSeek 0 / 468)"]
    U --> F["forward audit: 2 LLM judges<br/>was the evidence there after all?"]
    S --> RV["reverse audit: rescued runs<br/>+ negative control"]
    X[("upstream expert paths<br/>golden-trajectory/")] --> GC["golden_control.py<br/>legitimate control, no LLM"]
    P --> GC
    F --> N["compute_numbers.py -> expected.json"]
    RV --> N
    GC --> N
```

Typical case (`boutique-scheduling-101`): the model describes the stuck pod, reads
"didn't match Pod's node affinity/selector", and answers `node_selector_mismatch` without ever
listing the nodes' labels. The answer is right, so JRA pays 1; the strict verifier pays 0. Both
of the benchmark's expert paths for this case do list the labels.

How the strict verifier was checked (details and every number reproducible in the findings folder):
- **False alarms:** two LLM judges from different families read all 89 cases; they agree on 0
  cases where the evidence was there after all (CI 0–4.5%).
- **Misses:** all 73 runs rescued by its two documented rule changes confirmed (CI 0–5%), and a
  negative control of 12 cases where the rules would be wrong was rejected 12 of 12.
- **Legitimate control (no LLM):** in 80 of the 89 cases every expert path passes the same check,
  so the skipped step is one the benchmark's own experts take.
- **Side finding on the benchmark:** its labels reject 151 of 1,100 boutique expert paths; 111 of
  those use a pod or replicaset description as the config evidence, which the labels do not accept.

**An adversarial agent found the same shortcut on its own.** The Brownout swarm (an attacker told
to earn reward without solving) ran on 5 scheduling cases with the strict verifier as a hidden
second referee. In 3 cases it earned JRA without the confirming evidence, all checked by hand, 0
false alarms. After reading the scorer, its plan said the evidence in a submission "is inert".
In one case the attacker refused to game the scorer and still skipped the deciding check: the
reward does not ask for it, so intent does not matter. With JRA as the only referee, the same run
reports nothing. (5 hand-picked cases: a demonstration, not a rate.)

**How the swarm works.** The attacker sees only the JRA verifier (and, from round 1, its
source). The strict verifier is the hidden second referee: an attempt that earns JRA but fails
strict is recorded with a replayable repro.

```mermaid
flowchart LR
    PC["precheck<br/>a saved honest solve must pass true_success"] --> A
    A["attacker<br/>blind round, then whitebox<br/>(reads the attacked verifier)"] --> EP["episode via run_episode"]
    EP --> C{"confirm.is_hack"}
    C -->|"reward on attacked verifier<br/>AND (fails true_success OR detector fires)"| LG[("ledger.jsonl<br/>finding + repro")]
    C -->|no| FB["failure note<br/>(answer withheld)"] --> A
    LG -.->|"separate step, not in the loop"| FX["fixer.py proposes a patch<br/>(new class in a new file)"]
    FX -.-> GT["gate.py: honest solves must still pass,<br/>held-out findings must fail; a person approves"]
```

Not claimed: anything about other models or harnesses, human-verified labels, or training. Whether
RL on JRA teaches this shortcut is the next experiment.
Write-up: [findings/cloudops-unconfirmed-passes/](findings/cloudops-unconfirmed-passes/README.md).
Method and decisions: [Docs/STRICT_VERIFIER.md](Docs/STRICT_VERIFIER.md); every change in order:
[Docs/STRICT_AUDIT_CHANGELOG.md](Docs/STRICT_AUDIT_CHANGELOG.md).
