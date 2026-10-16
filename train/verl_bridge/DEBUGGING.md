# Getting verl GRPO to run: every error, what it meant, what fixed it

This is the log of bringing up `train/verl_bridge` on Modal (2026-10-15): 12 failures between
"pip install verl" and the first rollout. Each one is written so you can recognise it again,
understand *why* it happened, and know where to look for the next one.

Pinned versions (write these into any result):

| Piece | Version |
|---|---|
| verl | `v0.9.1` (git tag, installed `--no-deps`) |
| TransferQueue | `Ascend/TransferQueue@v0.1.11` (error 13: `main` breaks verl v0.9.1) |
| Image | `verlai/verl:sgl0512.dev4` (SGLang 0.5.12, torch, flash-attn, Ray) |
| Model | `Qwen/Qwen3-4B` on 1x H100 80 GB (8B needs an H200, see section 5) |
| Tool parser | `hermes` |

---

## 1. How to read a verl error

A verl traceback is 5 layers of wrappers around one real error. Read it **bottom-up** and
skip the wrappers:

```
modal                 CalledProcessError: Command '['bash', ... pilot.sh ...]' returned non-zero   <- wrapper: "pilot.sh failed"
  pilot.sh / hydra    Error executing job with overrides: [...]                                     <- wrapper: lists your config
    main_ppo          ray.get(runner.run.remote(config))                                           <- wrapper: Ray driver
      TaskRunnerV1    ray.exceptions.RayTaskError(XxxError): ray::TaskRunnerV1.run()               <- wrapper: the controller
        WorkerDict /  ray.exceptions.RayTaskError(XxxError): ray::WorkerDict.actor_rollout_...()   <- WHICH worker + WHICH method
        SGLangHttpServer
          ...         XxxError: <the real message>                                                 <- THE ERROR
```

Three rules:

1. **The last `XxxError:` line before Modal's own traceback is the real error.** Everything
   above it is Ray passing the exception up.
2. **The `ray::Something.method()` line tells you the stage.** `WorkerDict.init_model` =
   loading the trainer model, `WorkerDict.actor_rollout_update_weights` = trainer -> SGLang
   weight sync, `SGLangHttpServer` = the inference server, `TaskRunnerV1.run` = the controller.
3. **If there is no error at all (it just stops), it's a hang, not a crash.** Ray waits
   forever when a resource can't be scheduled. Look at `logs/<stage>.ray.log` ("Pending
   demands") and `logs/<stage>.gpu.log` (is the GPU doing anything?).

`HYDRA_FULL_ERROR=1` (set in `modal_app.py`) makes Hydra print the full stack instead of
"Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace".

### Where the logs are

| File on the volume | What | Pull it with |
|---|---|---|
| `logs/<stage>.log` | everything verl prints; first line = CPU count + what Ray sees | `modal volume get brownout-train logs/smoke.log runs/smoke.log --force` |
| `logs/<stage>.gpu.log` | GPU memory + utilisation, once a minute | same, `smoke.gpu.log` |
| `logs/<stage>.ray.log` | `ray status` every 2 min: resources in use, **pending demands** | same, `smoke.ray.log` |
| `runs/train-<model>-<stage>/` | one Brownout trajectory per finished episode | `modal volume get brownout-train runs/train-qwen3-4b-smoke runs/train-qwen3-4b-smoke` |

The volume is committed every 2 minutes, so all of these can be pulled **while the run is
still going**. `modal app logs brownout-verl` streams the same output as plain text (the
`modal run` terminal spinner overwrites lines; don't trust it for reading).

### The startup sequence: what you should see, in order

Every run goes through these stages. When something breaks, find the last marker that
appeared: the bug is in the next stage.

| # | Log marker | Stage | Typical time |
|---|---|---|---|
| 1 | `cpu_count: 24 ray resources: {... 'CPU': 24.0, 'GPU': 1.0 ...}` | Ray check (modal_app.py) | 10 s |
| 2 | `{'actor_rollout_ref': {'actor': ...` (a huge dict) | verl prints the resolved config | 30 s |
| 3 | `'transfer_queue': ... 'enable': True` (end of the dict) | TransferQueue workers start | seconds (or forever: error 9) |
| 4 | `Loading weights: 100%`, `PeftModelForCausalLM contains ...` | trainer model + LoRA built | 1 min |
| 5 | `After FSDP, memory allocated (GB): ...` | trainer model wrapped in FSDP | |
| 6 | `actor and ref model engine initialized` | trainer ready | |
| 7 | `SGLang http server: ...`, `Capturing batches (bs=...)` | SGLang starts, CUDA graphs | 1-2 min |
| 8 | `all initialize finished, ready to fit` | setup done | |
| 9 | (no marker; GPU memory jumps) | **first weight sync** trainer -> SGLang | 10-30 s |
| 10 | GPU util > 0 %, files in `runs/train-...` | **rollout**: our agent loop runs Brownout episodes | minutes |
| 11 | `step:1 - ... critic/score/mean: ...` | GRPO update + metrics | minutes |
| 12 | `/vol/checkpoints/...` written | checkpoint save, app stops | |

Errors 1-12 happened at stages 1-9. With Qwen3-4B, stage 10 works: our agent loop ran the 53
validation episodes and the first training rollout (error 13 is in stage 11). The update step
on 40k-token sequences and the checkpoint save are still untested.

---

## 2. The error catalogue

Each entry: the message as it appears, the stage, what it means, how it was found, the fix,
and the general lesson.

### Error 1: `ModuleNotFoundError: No module named 'transfer_queue'`

- **Stage:** 3 (`TaskRunnerV1.run`, `import transfer_queue as tq`).
- **Meaning:** verl's default trainer since v0.9 (`TaskRunnerV1`, the "V1 trainer") moves data
  between workers through a separate package, TransferQueue. We installed verl with
  `--no-deps` (to keep the image's torch/SGLang), which skipped it.
- **Found by:** reading `verl/trainer/main_ppo.py`: `if config.trainer.use_v1: run_ppo(...,
  TaskRunnerV1)`, and V1's `run()` does `import transfer_queue`. verl's `pyproject.toml` lists
  it as `transferqueue = ["TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@main"]`.
- **Fix:** `modal_app.py` installs it explicitly, also `--no-deps`. Also pinned verl to
  `v0.9.1` instead of `main`.
- **Lesson:** `--no-deps` is right for big images (it protects torch/CUDA), but you then own
  the list of small runtime deps. Check the project's `pyproject.toml` extras.
- **Alternative not taken:** `trainer.use_v1=False` uses the old trainer (deprecated).

### Error 2: `AssertionError: rollout world_size: 1 is not divisible by infer_world_size: 2`

- **Stage:** 4 (`WorkerDict.actor_rollout_init_model`).
- **Meaning:** verl's default is to split the SGLang model over 2 GPUs (tensor parallel = 2).
  We have 1.
- **Fix:** `pilot.sh`: `actor_rollout_ref.rollout.tensor_model_parallel_size=$TP`, with
  `TP` defaulting to `NGPUS`.
- **Lesson:** verl's defaults assume a multi-GPU node. On 1 GPU, check every `*_parallel_size`.

### Error 3: `ImportError: cannot import name 'ContinueGenerationReqInput' from 'sglang.srt.managers.io_struct'`

- **Stage:** 7 (`_load_sglang` -> `async_sglang_server.py` imports).
- **Meaning:** verl v0.9.1 was written against SGLang 0.5.12; the first image
  (`verlai/verl:sgl055.latest`, SGLang 0.5.5, from Nov 2025) was too old to have that class.
- **Found by:** verl's `pyproject.toml` at v0.9.1 pins `sglang 0.5.12`; Docker Hub's tag list
  for `verlai/verl` showed `sgl0512.dev4` (Jul 2026). The verl docs still show `sgl055` as the
  example tag, which is why it was picked first.
- **Fix:** `modal.Image.from_registry("verlai/verl:sgl0512.dev4")`.
- **Lesson:** verl and SGLang must match **exactly**. When you change one, look up which
  version of the other it pins (`pyproject.toml` / `setup.py` at that tag), then find an image
  with that version (`hub.docker.com/v2/repositories/verlai/verl/tags`).

### Error 4: `ModuleNotFoundError: No module named 'tests.token_model_test'`

- **Stage:** `check` function, not training.
- **Meaning:** our `tests/` has no `__init__.py` (a "namespace package"). Some pip package in
  the image ships its own top-level `tests` package, which Python finds first.
- **Fix:** run the test by file: `python -m unittest discover -s tests -p token_model_test.py`.
- **Lesson:** inside someone else's image, generic package names (`tests`, `utils`) can be
  shadowed.

### Error 5: `ValueError: chat template re-renders earlier turns: render_delta can't cut it`

- **Stage:** `check` (`check_template Qwen/Qwen3-8B`). **Our bug**, in `renderers.py`.
- **Meaning:** `render_delta` gets the token ids of new messages by rendering
  `BASE + new` and cutting off `render(BASE)`. That only works if `render(BASE)` is a prefix of
  `render(BASE + new)`. Qwen3's template renders an assistant turn *differently* depending on
  whether it is the last message (`<think>\n\n</think>` when last, nothing once a tool result
  follows), so with `BASE = [user, assistant]` the prefix changed.
- **Found by:** rendering Qwen3-8B's real `chat_template` (from `tokenizer_config.json`) with
  jinja2 on the laptop and comparing.
- **Fix:** `BASE = [user, assistant, user]`: ending on a user turn makes every assistant turn
  "history", rendered the same way in both. Verified for: one tool result, two in a row, a
  user nudge.
- **Expected leftover:** `check_template` now prints `DIFFERENT`, and the diff is only the
  empty `<think></think>` of the *earlier* assistant turn. That's the known train/eval gap
  (Qwen3 strips old thinking from history; TokenModel keeps what was actually generated). Any
  other difference would be a bug.
- **Lesson:** chat templates are code, not strings. Never assume a render is a prefix of a
  longer render; test it per model (`check_template.py` exists for this).

### Error 6: no output at all, H100 billing, "0.2 cores", then nothing (silent hang #1)

- **Stage:** 2-3. No error message.
- **Meaning, three problems at once:**
  1. Modal's default container is **0.125 CPU cores and 128 MiB RAM**. Ray + SGLang + verl on
     that either crawl or wait forever.
  2. Python buffers output when it goes into a pipe (our `| tee`), so nothing was printed.
  3. The `modal run` spinner overwrites terminal lines.
- **Found by:** Modal's own GRPO example (`.agents/skills/modal/references/examples/miles_grpo.md`)
  and `references/guide/resources.md` ("default request of 0.125 CPU cores and 128 MiB").
- **Fix (`modal_app.py`):** `cpu=8, memory=98304`; `PYTHONUNBUFFERED=1`; `.entrypoint([])`;
  run with `--detach` + `modal app logs`; added the GPU log, the Ray status log, and volume
  commits every 2 min; model download moved to a CPU-only `download` function.
- **Lesson:** "no output" is not "no error". Make the job tell you where it is (GPU log, Ray
  status) instead of guessing.
- **Note on "0.2 cores":** the dashboard shows CPU **in use**, not reserved. During generation
  the GPU does the work, so ~0.2 cores in use is normal. Judge progress by GPU utilisation.

### Error 7: `torch.OutOfMemoryError ... in collect_lora_params ... model.to(orig_dev)` (Qwen3-8B)

- **Stage:** 9, the first weight sync (`actor_rollout_update_weights`), before any episode.
- **Meaning:** the log showed it directly:
  ```
  After FSDP, memory allocated (GB): 31.17, ... device memory used: 43.57/79.17   <- trainer's copy
  Capturing batches (... avail_mem=21.5 GB)                                       <- SGLang took ~15 GB more
  update_weights -> OOM, 79 GB in use
  ```
  verl keeps the **trainer's copy in fp32**: 8.4 B params x 4 bytes = 33 GB. The sync then needs
  another copy of the weights. Trainer + SGLang + sync buffers > 80 GB.
- **Fix attempted:** `actor.fsdp_config.param_offload=True`, `optimizer_offload=True` (keep the
  trainer copy in CPU RAM between uses), container RAM 64 -> 96 GiB.
- **Lesson:** for LoRA on one GPU, the trainer copy (fp32, full model) is the biggest item, not
  the LoRA weights.

### Error 8: `OutOfMemoryError ... in load_fsdp_model_to_gpu` (8B again, with offload)

- **Stage:** 9 again.
- **Meaning:** offload helps between steps, but the sync **loads the whole fp32 trainer copy
  back onto the GPU** to read the weights, while SGLang is still holding its memory.
- **Fix:** none that works on 80 GB without trade-offs. Tried / considered:
  - `rollout.gpu_memory_utilization` 0.5 -> 0.4: SGLang then can't start (error 10).
  - `rollout.layered_summon=True`: crashes on 1 GPU (error 11).
  - `actor.fsdp_config.model_dtype=bf16` (33 -> 17 GB): **rejected**. verl v0.9.1's
    `transformer_impl.py` then casts the LoRA adapters to bf16 too ("FSDP requires all params
    in a flat group to share dtype"). At lr 1e-5, many Adam updates are smaller than bf16's
    precision and round to zero: the model silently stops learning. A crash is better than that.
- **Decision:** Qwen3-4B on the H100 now; Qwen3-8B later on an H200 (141 GB). H200s weren't
  available on Modal at the time.

### Error 9: hangs right after `'transfer_queue': ... 'enable': True` (silent hang #2)

- **Stage:** 3. GPU log: `4 MiB, 0 %` for 15 minutes; earlier runs passed this point in 1 min.
- **Meaning:** TransferQueue starts **8 storage workers** here, on top of verl's other Ray
  actors. Each claims CPU slots from Ray; Ray counts slots from the container's reported cores,
  which differ between Modal machines. When they run out, Ray waits forever, without an error.
- **Fix:** `pilot.sh`: `ray_kwargs.ray_init.num_cpus=32`. Ray's CPU count is a scheduling
  budget, not real cores, so the actors all start and share the 8 reserved cores (which are
  mostly idle anyway).
- **Confirmed by:** the next run's first log line, `cpu_count: 24 ... 'CPU': 24.0`, and it went
  straight past this point.
- **Lesson:** a Ray hang with idle CPU and GPU = an actor waiting for resources. `ray status`
  -> "Pending demands" names exactly what it waits for.

### Error 10: `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.`

- **Stage:** 7 (SGLang `init_memory_pool`). Surfaced as `ActorDiedError ... SGLangHttpServer
  ... Worker unexpectedly exits` (the real message is ~30 lines higher, prefixed
  `(SGLangHttpServer pid=...)`).
- **Meaning:** caused by the 0.4 change from error 8. The trainer copy is still on the GPU
  when SGLang starts, so SGLang's 0.4 share left almost nothing for the KV cache.
- **Fix:** back to `gpu_memory_utilization=0.5` (override with `GPU_MEM=` in pilot.sh).
- **Lesson:** `ActorDiedError` is never the cause, only "a process vanished". Search the log
  for that process's own lines (`(SGLangHttpServer pid=NNNN)`) just above it.

### Error 11: `NotImplementedError: offload_to_cpu=True and NO_SHARD is not supported yet`

- **Stage:** 9 (`collect_lora_params` -> `summon_full_params`). Qwen3-4B; memory was fine.
- **Meaning:** `rollout.layered_summon=True` gathers weights layer by layer with CPU offload.
  On 1 GPU, FSDP runs in `NO_SHARD` mode (logged earlier: "FSDP is switching to use
  `NO_SHARD`"), and PyTorch doesn't support that combination.
- **Fix:** removed `layered_summon` (it only helps when weights are sharded across GPUs).
- **Lesson:** many verl memory options assume >1 GPU. On 1 GPU, FSDP doesn't shard anything.

### Error 12: `TypeError: LoadLoRAAdapterFromTensorsReqInput.__init__() got an unexpected keyword argument 'serialized_named_tensors'`

- **Stage:** 9, the sync, now on the LoRA-specific path.
- **Meaning:** a version mismatch **inside verl v0.9.1 itself**. With `lora.merge=False`
  (default), verl sends only the LoRA adapter to SGLang through SGLang's
  `/load_lora_adapter_from_tensors` endpoint. verl v0.9.1 sends the field
  `serialized_named_tensors` (`verl/workers/rollout/sglang_rollout/http_server_engine.py`,
  `load_lora_adapter_from_tensor`). SGLang renamed that field later:

  | SGLang | field in `LoadLoRAAdapterFromTensorsReqInput` |
  |---|---|
  | 0.5.12 - 0.5.16 (our image: 0.5.12) | `serialized_tensors` |
  | 0.5.17 and later | `serialized_named_tensors` |

  So verl v0.9.1 pins SGLang 0.5.12 but its LoRA-adapter code needs >= 0.5.17.
- **Found by:** reading `io_struct.py` at each SGLang tag and verl's `http_server_engine.py` at v0.9.1.
- **Fix:** `pilot.sh`: `actor_rollout_ref.model.lora.merge=True`. verl merges the LoRA into the
  base weights and syncs the full model through `update_weights_from_tensor`, the standard
  path whose field name matches 0.5.12. Costs: each sync sends the full model (8 GB for 4B)
  instead of the small adapter: a few seconds per step. Side benefit: SGLang can release its
  weights during training too (with merge=False it keeps them).
- **Alternative not taken:** a newer image (SGLang >= 0.5.17) with verl v0.9.1. Other
  verl/SGLang APIs may then mismatch the other way. Revisit with the next verl release.
- **Lesson:** "pinned version X" in `pyproject.toml` doesn't guarantee every code path was
  tested with X. LoRA + SGLang is a less-travelled path.

### Error 13: `ValueError: Fields ['ref_log_prob', 'values'] are not ready in all the requested keys: missing, or written by kv_update(finalize=False) and not finalized yet`

- **Stage:** 11, the first training step, in `_compute_advantage`. **Everything before it
  worked**: validation ran on all 53 val tasks (`step:0 - val-core/...`, see below), and the
  training rollout finished.
- **Meaning:** another version mismatch. verl v0.9.1's V1 trainer always asks TransferQueue for
  `ref_log_prob` (reference model) and `values` (critic) when computing advantages, even for
  GRPO without either (`trainer_base.py`, `_compute_advantage`: the field list is fixed). TQ
  v0.1.10/v0.1.11 silently skip fields that were never written (`batch_meta.select_fields`).
  TQ `main` (later) added a strict check that raises on them. We installed `@main`.
- **Found by:** comparing `transfer_queue/interface.py` `async_kv_batch_get` at v0.1.11 vs main,
  and verl's commit log ("upgrade dependency TransferQueue>=0.1.10", 2026-09-20, same day as
  verl v0.9.1).
- **Fix:** `modal_app.py` installs `TransferQueue.git@v0.1.11`.
- **Lesson:** an unpinned dependency (`@main`) is a time bomb: it worked when verl was released
  and broke a month later. Pin everything to tags.

**Bonus from this run: the base model baseline.** verl validates before training by default
(`trainer.val_before_train`), so step 0 scored base Qwen3-4B on the 53 val tasks:

```
step:0 - val-core/cloudops-boutique/reward/mean@1: 0.238      (non-code tasks)
         val-core/cloudops-boutique-code/reward/mean@1: 0.0   (code-defect tasks)
         val-aux/num_turns/mean: 7.36  (min 4, max 20)
```

0.24 is inside the 0.15-0.6 range where GRPO has signal: 4B is a usable starting point.
(Nemotron-30B's non-code JRA is 0.68 on the full set; different split, so compare on test later.)
This validation took ~5 minutes; it repeats every `test_freq` steps (-1 = only at the start).

### Error 14: `TypeError: int() argument must be ... not 'NoneType'` in `_compute_metrics` (`tag["min_global_steps"]`)

- **Stage:** end of step 1, **after** the advantage and the LoRA update (`fit` -> `_compute_metrics`).
  So the update step on long sequences fit in memory.
- **Meaning:** **our bug** in `agent_loop.py`. verl's inference server tags every generation
  with the weight version that produced it (`TokenOutput.extra_fields["min_global_steps"]`,
  `["max_global_steps"]`). verl's own `ToolAgentLoop` copies them into
  `AgentLoopOutput.extra_fields`; ours didn't, so the trainer found `None`.
- **Found by:** `grep global_steps` in verl v0.9.1's `agent_loop.py` / `tool_agent_loop.py`:
  the manager always expects these keys (`default_extra_keys`), and `ToolAgentLoop` keeps the
  first turn's `extra_fields`, then only raises `max_global_steps`.
- **Fix:** `generate()` in `BrownoutAgentLoop.run` records `out.extra_fields` the same way into
  `gen_fields`; `to_output` puts them in `extra_fields`.
- **Lesson:** a custom agent loop has to return everything the built-in one does, including
  fields the docs don't mention. Diff your `AgentLoopOutput` against `ToolAgentLoop`'s.
- **Known edge:** an episode that fails before its first generation would still have no
  `global_steps`. Rare (it needs an env/harness error before turn 1); handle it if it shows up.

### Error 15 (silent): "it ran successfully", but no training happened

- **Stage:** none: the run exited cleanly.
- **Meaning:** the run before (error 14) crashed *after* saving its checkpoint
  (`/vol/checkpoints/qwen3-4b-smoke/global_step_1`). verl's default `trainer.resume_mode=auto`
  found it (`Found checkpoint ... setting global step to 1`), so with `total_training_steps=1`
  there was nothing left to train. It only ran the final validation on the 1-step model
  (`step:1 - val-core/cloudops-boutique/reward/mean@1: 0.167`) and exited. The error-14 fix
  was never exercised.
- **How to spot it:** the log has `Resuming from ...` and **no** `critic/score/mean` or
  `timing_s/*` metrics, only `val-*`.
- **Fix:** `pilot.sh` sets `trainer.resume_mode=disable` for smoke and pilot (always fresh),
  `auto` for full (so a crashed/preempted full run continues).
- **Lesson:** a clean exit isn't success. Check that the metrics you expect are in the log.
- **Also:** 0.238 -> 0.167 on val is noise, not damage from one step: 53 tasks, 1 sample each,
  ~3 tasks' difference.

### Error 16 (silent, costly): the eval path scores the base model 0.08 instead of 0.33

No exception: both runs "completed". The numbers were wrong, and the run was slow and expensive.

**Setup that produced it.** After the full run, the step-40 checkpoint was exported to a plain
HF model (`verl.model_merger` + PEFT `merge_and_unload`). An `evaluate` Modal function then
started an SGLang server in the container (`--tool-call-parser qwen25 --reasoning-parser qwen3`),
and the normal benchmark played the episodes against it:
`src.run <env> --provider sglang --split val --max-turns 20` (greedy). That is the **API
path**: the same one used for the Nemotron / DeepSeek baselines through Nebius.
Base and step40 were both launched at once on two H100s, before either was checked.

**Symptoms (base model, val split, 53 tasks):**

| | verl in-training val (base, step 0) | API-path eval (base) |
|---|---|---|
| JRA, non-code tasks | 0.33 (14/42) | **~0.10** (ALL incl. code: 0.08) |
| turns per episode | 6.9 | **15.9** |
| tool calls per episode | ~6 | 6.0 |
| episodes that submitted | ~92% | **34%**; most ended `max_turns` |

The tell is turns vs tool calls: **16 turns but 6 tool calls = ~10 turns per episode with
no tool call at all.** Same model, same tasks: the difference had to be in the path.

**How it was found (no GPU needed):**

1. `modal volume get` two episode files from `runs/eval-qwen3-4b/` (single files download
   fine on Windows; only folders don't).
2. Per-call view from the trajectory's `calls` list (`finish_reason`, `n_tool_calls`,
   `output_tokens`, latency). `boutique-scheduling-46`:
   ```
   turn 1-4   tool_calls  1   ~400 tokens      investigation, fine
   turn 5     stop        0   562              <- the diagnosis, as text
   turn 6     stop        0   641              <- the same diagnosis again
   ... 7, 10, 12 call tools; every other turn is stop/0 ...
   turn 15    length      0   30113  473 s     <- ran to the context limit
   turn 20    length      0   29286  334 s     -> ended_by max_turns, reward 0
   ```
3. The `messages` list: every `stop` turn's content is the **final-answer JSON**
   (`{"key_evidence_summary": ..., "top_3_predictions": [...]}`) written as **plain text**,
   not as a `Submit` tool call. After each, the harness adds the nudge
   (`src/harness.py:143`: "Use a tool, or call submit() if you are done.") and the model
   writes the same JSON again.
4. Compared with the **training-path** val episodes of the full run
   (`runs/train-qwen3-4b-full-1015-2014/val/`): **250 of 265 also contain a nudge**, so the
   model has the same habit there. The difference is what happens next. In training it calls
   `Submit` right after the nudge (scheduling-46: turn 4 plain JSON -> nudge -> turn 5
   `Submit`). Through the API it never does.

**What differs between the two paths:**

| | training path (verl agent loop, `TokenModel`) | API path (`src.run` -> SGLang server) |
|---|---|---|
| context | one token stream: **every earlier `<think>...</think>` stays** | messages re-rendered each turn by the chat template |
| earlier thinking | visible to the model | **gone**: the reasoning parser moves it to `reasoning_content`, `Clients/openai_compact.py` keeps only `content`, and Qwen3's template drops past thinking anyway |
| tool-call parsing | hermes parser on the raw text (`BROWNOUT_TOOL_PARSER`) | SGLang's `qwen` parser (`qwen25` is a deprecated alias) |
| tokens per turn | capped at 4096 (`BROWNOUT_TOKENS_PER_TURN`) | **no cap**: the client sends no `max_tokens`, so a turn can run to the ~40k context |

**Most likely cause (strong evidence, not proven).** Without its own earlier reasoning in
context, the 4B model can't get out of the "answer as text -> nudge" loop. It repeats its
last visible output instead of switching to the tool call. The evidence: same model, same
habit, same nudge, and in the training path, where the thinking stays, the loop breaks after
one nudge. **Not proven yet:** that thinking is the *only* difference. The tool-call
rendering in the template also differs (SGLang passes only the `function` part of each tool).
To prove it: replay one episode through the API path, sending `reasoning_content` back with
each assistant message, and see whether `Submit` follows the nudge.

**Why it cost so much.** The 20-turn limit was hit on most episodes instead of ~7, and with
no token cap single turns ran for 5-8 minutes (30k tokens). On top of that, two GPUs ran a
broken path at once.

**Fix.** Evaluate through **verl's own validation, alone**: `pilot.sh STAGE=eval` sets
`trainer.val_only=True`, and `VAL_FILE` picks the split. `modal_app.py::evaluate --step N
--split val|test` runs it (`--step 0` = the base model; `N` resumes
`checkpoints/<model>-full/global_step_N`). `make_dataset.py` now also writes
`test.parquet`, which training never reads. It is the exact path that produced the
in-training val numbers, so base and tuned are compared like for like. The API-path
`evaluate` and the `export` function were removed (the export is still on the volume at
`/vol/export/qwen3-4b-full-step40`).

**Check before trusting it.** `evaluate --step 40 --split val` was expected to print
`val-core/cloudops-boutique/reward/mean@1` ≈ **0.405**, which verl measured at step 40 during
training. It printed **0.3095** (13/42). That first looked like "the checkpoint wasn't
loaded", but it was not:

- The log says `Resuming from /vol/checkpoints/qwen3-4b-full/global_step_40` and loads the
  model, optimizer and rng.
- The eval episodes match the in-training **step-40** val pass best. Their first two tool
  calls agree with it on **35/53** tasks, against 17/53 for step 0 (and 15/25/26 for steps
  10/20/30). So the step-40 weights were in SGLang.

**Finding: "greedy" val is not repeatable.** `val_kwargs` uses temperature 0, but the same
weights give different trajectories on every run.
- Full tool sequences repeat on only ~0-2 of 53 tasks between the eval and any training pass.
  The five training passes themselves agree on only 2-4/53.
- Likely cause: SGLang batches concurrent requests differently each run. That changes the
  floating-point reduction order, so near-tied logits flip. Across ~6k generated tokens and
  ~7 turns, one flipped token sends the episode down another path.
- Size of the noise: about **±4 of 42 tasks** (±0.1) per single pass. The in-training curve
  (0.333, 0.286, 0.405, 0.310, 0.405) moves within that band, and so does 0.31 vs 0.405.

**Consequence: one greedy sample per task can't measure a base-vs-tuned gap of this size.**
Measure it instead with:
- the 116-task **test** split;
- `--samples 4` (temperature 0.6, top_p 0.95, like training; `mean@4`): 464 episodes per model;
- a **paired** per-task comparison: the same task ids for base and step 40, counting tasks
  that went up vs down;
- the signals that move more than the final score: suffix/format, submit rate, nudges,
  turns.

Updated check for a new eval path: the loaded-checkpoint line in the log, plus the
episodes agreeing best with the matching training pass. Not an exact score.

**Lessons.**
- **Sanity-check a new eval path on one model, against a number you already know, before
  running anything else.** Here: base on val should be ≈ 0.33. One run (~$1) would have
  caught it; two runs in parallel doubled the cost of finding the bug.
- **Turns vs tool calls** is the quickest health check of an agent eval. A large gap means
  turns without actions: nudges, parsing problems, or text answers.
- **An eval path is part of the model's environment.** A small reasoning model trained with
  its thinking in context may not work without it. "Same weights" isn't "same agent".

**Write-up material (true for both paths).**
- **~94% of episodes need a nudge before `Submit`.** The system prompt's "Final Diagnosis
  Output Requirement" describes a JSON report, and the model writes that JSON as text first.
  Training to step 40 didn't remove the habit.
- The nudge says `submit()` but the tool is named `Submit`. Worth aligning (harness change:
  it affects every model, so re-baseline after).
- To make the **API path** usable for Qwen3 later (harness/client work, not now): send
  `reasoning_content` back in assistant messages, pass `max_tokens` per turn (4096, as in
  training), and name the `Submit` tool in the nudge. Then re-check base ≈ verl's number.
- `Usage.reasoning_tokens` in `openai_compact.py` comes out 0 for SGLang (it is
  `total - prompt - completion`). Thinking tokens are inside `completion_tokens` there.

### Error 17 (operational): the full run stopped at step 46 when the laptop switched off

- **Symptom:** `full.log` ends mid-step 47 (23:21 UTC) with no traceback. GPU at 100%, 53 GB
  until the last minute. `modal app list` shows the app stopped one minute later.
- **Cause:** the run was started without `--detach`. Modal then ties the app to the local
  client, and the laptop shutting down ended it.
- **Lost:** steps 41-46, never checkpointed (`save_freq=10`), ~25 min of GPU.
  `global_step_40` survived; `resume_mode=auto` would continue from it.
- **Fix:** always `modal run --detach ...` for anything longer than a few minutes; follow it
  with `modal app logs brownout-verl`. On a VM provider (JarvisLabs, Spheron) use `tmux`.

### Error 18 (silent hang #3): eval with `--samples 4` on test, `finished: 0` forever

- **Symptom:** `evaluate --step 0 --split test --samples 4` reaches `ready to fit`, then the
  log shows only `pending: 0, running: 116, finished: 0, failure: 0` every minute. GPU 0%
  with the model loaded (53.9 GB), no tool calls in the log, no traceback. Stopped after
  ~20 min (~$1.3).
- **Cause (likely; verl source not checked):** `agent_loop.py` ran each episode with
  `asyncio.to_thread`, i.e. asyncio's shared default pool (at most 32 threads). The episode
  thread blocks in `on_loop(...)` until verl's `generate`/`parse` coroutines finish on the
  event loop, and those coroutines appear to need a thread from the same pool. 116 x 4 =
  464 episodes / 8 workers = ~58 per worker > 32: every thread is held by an episode waiting
  for a coroutine that is waiting for a thread. Earlier runs stayed below 32 per worker
  (training 4, val ~7), which is why it never showed.
- **Tried, did NOT fix it:** episodes got their own pool, `_EPISODES =
  ThreadPoolExecutor(max_workers=512)`, via `loop.run_in_executor(_EPISODES, ...)`. The same
  hang came back (ready 05:33:49, still `finished: 0` and GPU 0% at 05:37). verl 0.9.1's
  `tool_parser.py` does use `run_in_executor(None, ...)`, but `generate` never ran at all,
  so the block is earlier. Real cause still open. The change is harmless and kept.
- **Workaround used:** `--samples 1` (n=1), the concurrency that has worked before. To dig
  later: verl 0.9.1 `experimental/agent_loop/agent_loop.py` `generate_sequences` (l.567,
  val sampling at l.604) and the "blocking ray.get inside async actor" warning.
- **Check:** within ~3 min of `ready to fit`, `finished:` must rise and tool calls (e.g.
  `GetAppYAML`) must show up. Still `finished: 0` at 5 min: stop the app (~$0.3).
- **Lesson:** "running N, finished 0" plus GPU 0% is a deadlock, not slowness. A slow run
  still shows GPU load. When a run scales up concurrency, count work per worker against
  every pool it shares.

### Smaller things after the full run

| Symptom | Cause | Fix |
|---|---|---|
| `run_report` shows 5 steps with `ppo_kl` 0 for the full run | `runs/full.log` was never downloaded; the command read `pilot.log` | `modal volume get brownout-train logs/full.log runs/full.log --force` |
| `run_report` episodes section empty | the tar was downloaded but not extracted; `--episodes` pointed at a missing folder | `tar -xzf runs/<run>.tar.gz -C runs` first |
| `ValueError: SGLANG_GRPC_PORT (75485) must be between 1 and 65535` in `update_weights` → `_init_server_adapter`, right after startup | verl picks SGLang's HTTP port at random (`get_free_port`); SGLang appears to derive its gRPC port by adding 10000 (65485 + 10000 here, inferred from the numbers), which goes past 65535 when the random port is high | bad luck, not config: relaunch. Recurs often → pin the port range (not needed so far) |
| `modal` not found in a new terminal | the CLI lives in a venv (`OpenENV\.venv`) | activate that venv, or call `...\.venv\Scripts\modal.exe` |
| `run_report` val row (0.275) disagrees with the val curve | the episodes table pools **all** val passes (steps 0-40) **and** the code tasks (always 0) | read val per checkpoint from section 1's `val` column |

### Smaller things on the way

| Symptom | Cause | Fix |
|---|---|---|
| `modal shell` not supported on Windows | Modal limitation | `check` function instead; `modal shell` works from WSL |
| `SystemExit: no model at /vol/models/...` | `download` not run yet | `modal run ...::download` first (guard working as intended) |
| Secret key named `WAND_B` | wandb only reads `WANDB_API_KEY` | `modal secret create brownout HF_TOKEN=... WANDB_API_KEY=... --force` |
| Warnings: `torchtitan/veomni/automodel engine is not available` | optional verl engines | harmless (we use FSDP) |
| Warnings: `NUMA affinity`, `NO_SHARD`, `Failed to clone() tensor` | container / 1 GPU / FSDP sync | harmless |

---

## 3. What each file in `train/verl_bridge/` does

| File | Role | Runs where |
|---|---|---|
| `token_model.py` | `TokenModel`: a Brownout `ModelClient` that keeps **one token stream per episode**: prompt ids, then `[model reply][tool results][model reply]...` with mask 1 for model tokens, 0 for env tokens, plus logprobs. Raises `ContextFull` when the stream reaches `max_response_tokens`. No verl import. | laptop + GPU |
| `renderers.py` | Turns messages into token ids with the model's chat template. `render_prompt` (first prompt) and `render_delta` (messages after an assistant turn), by cutting against a fixed `BASE` conversation (error 5). | laptop + GPU |
| `agent_loop.py` | `@register("brownout") BrownoutAgentLoop`: verl calls `run()` once per (task, sample). Finds the task, builds env + verifier, builds a `TokenModel` whose `generate` is verl's SGLang server, runs `src.harness.run_episode` in a thread, saves the trajectory, returns `AgentLoopOutput` via `to_output` (infra failures: mask all 0; `ContextFull`: real reward 0). | GPU only |
| `agent_loop.yaml` | Tells verl that `agent_name: brownout` = `BrownoutAgentLoop`. | GPU |
| `make_dataset.py` | Split file -> `train.parquet` / `val.parquet` / `test.parquet` (test only for the eval stage; ids only; the real task is loaded by `find_task`). | anywhere with pandas |
| `check_template.py` | Renders a Cloud-OpsBench-shaped conversation two ways and diffs them (error 5). | anywhere with transformers |
| `pilot.sh` | The verl launch command: every config override (section 4). Stages smoke / pilot / full / eval (validation only, error 16). | GPU |
| `modal_app.py` | Packaging for Modal: image, volume, secret, `download` / `check` / `train` / `evaluate` / `pack` functions, logs. `MODEL` and `GPU` at the top are the one switch for model size. | your laptop -> Modal |

The data flow for one sample:

```
verl  --(row: env, task_id)-->  BrownoutAgentLoop.run
          find_task -> Task;  registry -> env, verifier
          TokenModel(generate = SGLang server, parse = hermes parser)
          run_episode(env, task, TokenModel)          <- the SAME harness as benchmark runs
              each turn: render_delta(tool results) -> generate -> parse tool calls -> env.step
          verifier(task, trace) -> JRA reward
          to_output -> AgentLoopOutput(prompt_ids, response_ids, mask, logprobs, reward)
verl  <-- 4 samples of one task = one GRPO group -> advantages -> LoRA update
```

---

## 4. Every override in `pilot.sh`, and why

| Override | Value | Why |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | group-relative advantages, no critic |
| `data.train_batch_size` | 2 / 8 / 16 | tasks per step (smoke / pilot / full) |
| `data.max_prompt_length` | 8192 | system prompt + task + tool schemas |
| `data.max_response_length` | 32768 | the whole multi-turn stream after the prompt (thinking included). Episodes p90 ~33k generated: this cuts the tail (`context_full`) |
| `data.return_raw_chat` | True | the agent loop needs the raw row, not pre-tokenised text |
| `model.path` | `/vol/models/Qwen3-4B` | local copy on the volume (no download on the GPU) |
| `model.lora_rank` / `lora_alpha` | 64 / 128 | LoRA size; alpha = 2x rank is the usual scale |
| `model.target_modules` | `all-linear` | LoRA on every linear layer |
| `model.lora.merge` | True | error 12: sync full merged weights, not the adapter |
| `model.enable_gradient_checkpointing` | True | recompute activations in backward: less memory, ~30 % slower |
| `model.use_remove_padding` | True | pack sequences without padding: large memory saving |
| `actor.optim.lr` | 1e-5 | LoRA RL learning rate (could go higher; tune after the pilot) |
| `actor.ppo_mini_batch_size` | = batch | one optimiser step per training step |
| `actor.use_dynamic_bsz` | True | micro-batches by token count, not by sample count |
| `actor.ppo_max_token_len_per_gpu` | 40960 | tokens per micro-batch. Must be >= one full sequence (8192 + 32768); the default 16384 would fail |
| `actor.entropy_from_logits_with_chunking` | True | entropy in chunks: avoids a ~10-25 GB logits spike on 40k-token sequences |
| `actor.entropy_checkpointing` | True | don't keep those logits for backward |
| `actor.use_kl_loss` | False | no KL to a reference model (common for GRPO; also saves memory) |
| `actor.fsdp_config.param_offload` | True | trainer copy in CPU RAM between uses (error 7) |
| `actor.fsdp_config.optimizer_offload` | True | optimiser state in CPU RAM |
| `rollout.name` / `mode` | `sglang` / `async` | async is required for agent loops |
| `rollout.tensor_model_parallel_size` | `$TP` (= GPUs) | error 2 |
| `rollout.n` | 2 / 4 | samples per task = GRPO group size |
| `rollout.temperature` / `top_p` | 0.6 / 0.95 | Qwen3's recommended thinking-mode sampling |
| `rollout.gpu_memory_utilization` | 0.5 | SGLang's share of the GPU (errors 8, 10) |
| `rollout.agent.agent_loop_config_path` | `agent_loop.yaml` | registers our agent loop |
| `trainer.n_gpus_per_node` / `nnodes` | 1 / 1 | |
| `trainer.total_training_steps` | 1 / 5 / 100 | **set full to 50** (and batch 8) to match the plan |
| `trainer.test_freq` | -1 | no validation during training (evaluate with Brownout afterwards) |
| `trainer.save_freq` | = steps | one checkpoint at the end |
| `ray_kwargs.ray_init.num_cpus` | 32 | error 9 |

Env vars read by `agent_loop.py`: `BROWNOUT_TOOL_PARSER` (hermes), `BROWNOUT_MAX_TURNS` (20),
`BROWNOUT_TOKENS_PER_TURN` (4096), `BROWNOUT_MODEL_ID`, `BROWNOUT_SAVE_DIR`.

---

## 5. 4B vs 8B: what changes

Everything is set by two lines at the top of `modal_app.py`:

```python
MODEL = "Qwen/Qwen3-4B"     # or "Qwen/Qwen3-8B"
GPU = "H100"                # 8B: "H200"
```

`download`, `check_template`, the run folder (`runs/train-qwen3-4b-smoke`), the checkpoint
folder and the W&B run name all follow `MODEL`. Nothing in `pilot.sh` or the agent loop changes.

GPU memory at the first weight sync (the step that failed for 8B):

| Item | Qwen3-4B | Qwen3-8B | Why |
|---|---|---|---|
| Trainer copy (fp32) | ~16 GB | ~33 GB | params x 4 bytes; verl trains in fp32 by default |
| SGLang weights (bf16) | ~8 GB | ~16 GB | params x 2 bytes |
| SGLang KV cache | rest of its 0.5 share | ~9 GB observed | ~0.15 MB per token for both (36 layers, 8 KV heads, head dim 128) |
| Sync buffers (copy of weights) | ~8 GB | ~16 GB | |
| **Total vs 80 GB H100** | **~45-55 GB: fits** | **~79+ GB: OOM** | |
| On a 141 GB H200 | fits | fits | |

Speed and cost (estimates; the pilot measures the real numbers):

| | Qwen3-4B on H100 | Qwen3-8B on H200 |
|---|---|---|
| $/h on Modal (GPU + 8 CPU + 96 GiB) | ~5 | ~6 |
| min / step (8 tasks x 4, thinking on) | ~5-8 | ~8-15 |
| 50 steps | ~4-7 h, ~$20-35 | ~7-12 h, ~$40-70 |

Is 4B strong enough? GRPO needs groups where some samples succeed and some fail. Read
`critic/score/mean` at step 1 of the pilot (= the base model's score on those tasks):
roughly **0.15-0.6 is good**; under ~0.1 most groups are flat (all 0) and there is little to
learn from: go to 8B. Qwen3-1.7B was ruled out for that reason (long multi-turn contexts).

---

## 6. Checklist for the next error

1. Find the real error: the last `XxxError:` line above Modal's traceback (section 1).
2. Find the stage: the `ray::Worker.method()` line, or the last startup marker in the log.
3. No error, just stopped? Pull `smoke.gpu.log` and `smoke.ray.log`: idle GPU + pending
   demands = resource hang; idle GPU + no demands = something waiting on I/O or a lock.
4. Version smell (`ImportError`, `unexpected keyword argument`, `cannot import name`)? Compare
   the two packages at their pinned tags (`pyproject.toml`, the class definition) before
   changing anything else.
5. OOM? Note **which stage** (sync, rollout, update) and the numbers in the message
   (`X GiB allocated by PyTorch`, total in use). Then budget it like the table in section 5.
6. Change one thing, rerun smoke. Write the new error into this file.
