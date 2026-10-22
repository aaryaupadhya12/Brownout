# How Brownout fits together

One page: the parts, how data flows between them, and the command for each step. **Free**
commands run offline. **Paid** commands call a model API (or GPUs) and cost money.

## 1. The whole system

```mermaid
flowchart LR
    subgraph ENV["An environment = one folder under src/envs/"]
        T["tasks.py<br/>Task (answer in task.spec)"]
        E["env.py<br/>tools, state, snapshot()"]
        V["verifier.py<br/>@reward / @metric / @stop"]
    end
    REG["registry.py<br/>finds every env folder"] --> RUN
    M["Model client<br/>providers.py + Clients/"] --> RUN
    RUN["run_episode<br/>harness.py"] -->|"trajectory JSON"| RUNS[("runs/<br/>saved records,<br/>never edited")]
    T --> RUN
    E --> RUN
    V -->|"reads only (task, trace)"| RUN
    RUNS --> REP["report.py<br/>tables per model"]
    RUNS --> AUD["src/audit/<br/>Layer 1 checks"]
    RUNS --> STR["cloudops/strict.py<br/>re-grade into runs/regrade/"]
    STR --> JUD["audit_judge.py<br/>LLM judges check the checker"]
    STR --> FIND["findings/<br/>numbers + write-up"]
    RUN --> SW["src/audit/swarm/<br/>attacker / solver / fixer"]
    RUN --> TR["train/verl_bridge/<br/>GRPO: 1 sample = 1 run_episode"]
```

**The rule that makes re-grading possible:** the verifier never touches the env. The env reports
facts through `snapshot()`, thresholds live in `task.spec`, and the verifier reads only
`(task, trace)`. So any saved run can be scored again by a new verifier, offline, for free.

## 2. One episode (`src/harness.py: run_episode`)

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

Turns are counted, not tool calls: several calls in one turn is intended.

## 3. The cloudops audit (the main result)

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

## 4. The swarm (`src/audit/swarm/`)

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

The fixer and gate are level 1 of the plan and not wired into the loop: `validate_patch`,
`load_patch`, `feedback_text`, `fix_finding` and `gate.is_better_patch` are TODOs for you to write.

`true_success` is the second referee. With `--true-success same` it is the attacked verifier itself,
so no gap is possible (LESSONS 13). With `--true-success strict` it is the strict verifier, and
"JRA 1, strict 0" becomes a finding (LESSONS 16).

## 5. Commands

Run everything from the repo root. Keys live in `.env` (see `configs/providers.toml` for the
variable names).

| Step | Command | Cost |
|---|---|---|
| Tests (all offline) | `python -m unittest discover -s tests -p "*_test*.py"` | free |
| Smoke-test a provider on the toy env | `python -m src.run brownout --provider nebius --model <id> --trials 1` | paid (cents) |
| Run a benchmark | `python -m src.run cloudops-boutique --provider nebius --model <id> --concurrency 8 --resume` | paid |
| Fixed splits | `python -m src.splits show splits/cloudops-boutique.json` | free |
| Results table | `python -m src.report runs --filter boutique` | free |
| Layer 1 audit | `python -m src.audit cloudops-boutique --layers 1` | free |
| Re-grade with the strict verifier | `python -m src.envs.SystemEnv.cloudops.regrade_strict` | free |
| Expert-path control | `python -m src.envs.SystemEnv.cloudops.golden_control --write` | free |
| Recompute every number | `python -I findings/cloudops-unconfirmed-passes/compute_numbers.py` | free |
| Sample unconfirmed passes for review | `python -m src.envs.SystemEnv.cloudops.sample_unconfirmed runs/regrade/strict-6f0952b5.jsonl --seed 2 --n 1000` | free |
| LLM-judge audit | `python -m src.envs.SystemEnv.cloudops.audit_judge <sheet.csv> --max-calls 120` (`--dry-run` first) | paid (cents per call) |
| Swarm with hidden strict referee | `python -m src.audit.swarm cloudops-boutique --campaign <new-name> --provider nebius --model zai-org/GLM-5.3-Flash --tasks <id,...> --max-rounds 3 --true-success strict` | paid (~$0.10/task) |
| Read swarm results | `python -m src.audit.swarm.show <campaign>` | free |
| Small helpers | `python -m tools.summarize`, `python -m tools.summarize_audit <report.json>` | free |
| Training (verl, Modal) | `STAGE=smoke bash train/verl_bridge/pilot.sh` (see `train/verl_bridge/README.md`) | paid (GPU) |

Before any paid swarm run, read the checklist in `src/audit/swarm/LESSONS.md`.

## 6. Where things live

| Path | What |
|---|---|
| `src/harness.py`, `limits.py`, `store.py`, `tracking.py` | the episode loop, budgets, atomic saves, `group_id` |
| `src/contracts.py`, `scoring.py`, `registry.py` | data objects, `@reward/@metric/@stop`, env discovery |
| `src/providers.py`, `Clients/`, `pricing.py` | model clients, keys from `.env`, cost per token |
| `src/envs/SystemEnv/brownout/` | toy env, the worked example for new envs |
| `src/envs/SystemEnv/cloudops/` | Cloud-OpsBench adapter, strict verifier, audit scripts |
| `src/audit/` | Layer 1/2 audits, `frontier_target.py`, `swarm/` |
| `train/verl_bridge/` | GRPO with verl; `DEBUGGING.md` has every failure and fix |
| `findings/` | results packaged with `expected.json` and recompute scripts |
| `Docs/` | `HARNESS.md` (how to use), `ROADMAP.md`, `STRICT_VERIFIER.md`, `STRICT_AUDIT_CHANGELOG.md`, `ISSUES.md`, `contracts.md`, `READING_LIST.md` |
| `tools/` | small read-only helpers |
| `runs/` (git-ignored) | every saved trajectory, re-grade and swarm campaign |
