# verl bridge: GRPO on Brownout envs

Trains a model with verl on a Brownout env, using the **same** env, limits and verifier as
the benchmark. Brownout supplies the episodes and the reward; verl does inference
(SGLang) and the GRPO update.

```
verl trainer ──(task row)──> BrownoutAgentLoop.run()            agent_loop.py
                               │ env, verifier  <- src.registry
                               │ TokenModel     <- verl server + tokenizer   token_model.py
                               │ run_episode(env, task, TokenModel, ...)     src/harness.py
                               v
                             AgentLoopOutput: token stream + mask + JRA reward (to_output)
verl trainer <───────────────┘   rollout.n = 4 samples per task = one GRPO group
```

| File | What | Who |
|---|---|---|
| `token_model.py` | One growing token stream per episode; model tokens mask 1, env tokens mask 0 | **you** (`complete`) |
| `agent_loop.py` | verl glue: task lookup, sync/async bridge, launch `run_episode` | written; **you** (`to_output`) |
| `renderers.py` | Chat template -> prompt ids / tool-turn ids | written |
| `make_dataset.py` | Split file -> `train.parquet` / `val.parquet` (ids only) | written |
| `check_template.py` | Checks `renderers.py` against a real tokenizer | written |
| `agent_loop.yaml`, `pilot.sh` | verl config + launch (smoke / pilot / full) | written; keys marked VERIFY |
| `tests/token_model_test.py` | 11 tests for `TokenModel`, incl. a full Brownout episode | written |

## Steps

**On the laptop (free):**

1. Fill `TokenModel.complete` (the docstring lists 7 steps).
   `python -m unittest tests.token_model_test` -> 11/11.
2. Fill `BrownoutAgentLoop.to_output`. It can't be tested without verl; read it twice.
   The one rule: an infra failure contributes **no gradient** (mask all 0), a model failure
   is a real 0 reward.

**On the GPU box:**

3. Setup: clone Brownout (your branch), clone Cloud-OpsBench, set `CLOUDOPSBENCH_ROOT`,
   install verl (pin a release; note the version in the write-up) with SGLang.
4. `python -m train.verl_bridge.check_template nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
   -> "OK: identical", or a diff only inside the assistant/think part.
5. Check the VERIFY names in `agent_loop.py` and `pilot.sh` against the installed verl:
   `server_manager.generate` kwargs, `TokenOutput.token_ids / log_probs`,
   `extract_tool_calls` signature, `self.rollout_config.response_length`,
   `AgentLoopOutput.metrics`, the LoRA keys. Open
   `verl/experimental/agent_loop/{agent_loop,tool_agent_loop,tool_parser}.py`; verl's own
   `ToolAgentLoop` is the reference for every one of them.
6. `python -m train.verl_bridge.make_dataset --split-file splits/cloudops-boutique.json --out data/verl/cloudops-boutique`
7. **Smoke** (`STAGE=smoke bash train/verl_bridge/pilot.sh`): 2 tasks x 2 samples, 1 step.
   Done when: one step finishes, `python -m src.report runs/train-smoke` shows 4 episodes
   with real tool calls and a JRA. **Most likely failure: Nemotron-H (Mamba + MoE) in verl's
   trainer** (needs the Mamba kernels and a transformers version verl accepts). If it
   won't load, switch to `MODEL=Qwen/Qwen3-8B BROWNOUT_TOOL_PARSER=hermes` (nothing else
   changes) and keep Nemotron for later.
8. **Pilot** (`STAGE=pilot`): 8 tasks x 4, 5 steps. Write down:
   - minutes per step (-> cost of the full run)
   - % of groups with all 4 rewards equal (no learning signal). High -> add DAPO
     dynamic sampling (`algorithm.filter_groups`) before the full run
   - % `context_full` (raise `data.max_response_length` or lower thinking if > ~10%)
   - GPU memory peak
9. **Full** (`STAGE=full`), then evaluate the merged LoRA checkpoint with Brownout on the
   **test** part (`python -m src.run ... --split-file splits/cloudops-boutique.json --split test`)
   against the base model's test JRA from `python -m src.report runs --split-file ... --split test`.
   Then Train-Ticket base vs tuned (transfer).

## Numbers that set the config (Nemotron baseline, 552 episodes)

| | p50 | p90 | p99 | max |
|---|---|---|---|---|
| context at the last call (API, old thinking dropped) | 13.5k | 28k | 48k | 61k |
| tokens the model generated (thinking included) | 9.3k | 33k | 60k | 80k |
| turns | mean 8.1 | | | 20 |

In training the stream keeps **all** generated tokens (that is what was sampled), so an
episode is roughly context + earlier thinking: about 20k at p50, 60k+ at p90.
`max_response_length=32768` and `4096` tokens per turn cut the long tail; the pilot
measures how much (`context_full`).

## Known gaps (write-up material)

- Thinking is kept in the training stream but stripped by the API at evaluation time:
  train and eval contexts differ for long episodes.
- Parallel tool calls (Docs/ISSUES.md #1) are not capped here either. Nemotron makes none,
  so it doesn't matter yet; fix it in the harness, not in this bridge.
- A masked infra-failure sample still adds a reward 0 to its group's mean/std.
