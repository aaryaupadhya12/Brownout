# Known issues

## 1. Parallel tool calls bypass the turn budget (open, found 2026-10-14)

**What:** the harness runs *every* tool call in an assistant message. `max_turns` counts model
replies, not tool calls, so a model that sends several calls per reply gets a larger
investigation budget than one that sends one. The system prompt's "Call exactly one tool
per turn" is only a request; models can ignore it.

**Seen on:** Cloud-OpsBench (`cloudops-boutique`), `deepseek-ai/DeepSeek-V4-Flash-0731` via Nebius.

| Model | Episodes | Episodes with >1 call in a turn | Episodes with >20 tool calls | Calls per turn |
|---|---|---|---|---|
| DeepSeek-V4-Flash | 434 | 434 (all) | 53 | 2.1 |
| Nemotron-3-Nano | 550 | 0 | 0 | 0.97 |

**Effects:**
- **Scores not comparable with upstream.** Upstream Cloud-OpsBench allows one action per
  step, 20 steps. DeepSeek effectively had about 2x the tool budget: JRA 0.86 on the 413
  non-code Boutique cases that finished, vs 0.76 published for all 550. Not a parity result.
- **Cost.** Every tool output is resent on every later turn. Performance cases reached about
  800k input tokens per episode (about $0.10-0.15 each, $2.46 for 29 cases), driven by
  repeated `GetSourceCode` output.
- Nemotron results are **not** affected (it never made parallel calls).

**Status:** the DeepSeek run is stopped (434 episodes saved in runs/, startup partly done,
codedefect not started). Keep those runs: they are the "parallel calls allowed" arm.

**Fix (to do):**
- A harness limit `max_tool_calls_per_turn` (e.g. in `RolloutLimits`), part of `config_hash`
  so runs with and without it get different `group_id`s and never mix.
- Run only the first N calls of a reply. Answer every extra call with
  `{"error": "one tool call per turn"}`: the chat API requires a tool message for every
  `tool_call_id` in the assistant message, and the error tells the model the rule.
  (Simply dropping the extra calls leaves call ids unanswered; strict providers reject the
  next request.)
- Test: a FakeModel that sends 3 calls in one turn -> only the first runs, two errors,
  `tool_calls` metric counts 1.
- Also worth recording per episode: `tool_calls_per_turn` (mean, max) as a metric.
- Then rerun DeepSeek-V4-Flash on all 550 Boutique cases with the limit = 1 (parity check
  against 0.76), with `--max-tokens` as a cost cap.

**How others handle it:**
- **OpenEnv** (`src/openenv/core/harness/__init__.py`): `HarnessRunLimits` has
  `max_tool_calls_per_turn` and `max_total_tool_calls`, both `None` (unlimited) by default.
  The per-turn limit slices the list (`tool_calls[:N]`) after the full assistant message is
  already in the history, so the dropped calls appear to get no tool message.
- **verifiers** (`verifiers/v1/session.py`, `harnesses/utils/core.py`): runs every call in a
  reply; `max_turns` counts sampled model replies (`Trace.num_turns`). No per-turn call
  cap; `parallel_tool_calls` is passed through as a sampling field (Responses dialect
  default `True`). Budgets are capped with `max_input_tokens` / `max_total_tokens` instead.
