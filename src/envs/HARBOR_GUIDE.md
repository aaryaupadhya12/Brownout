# Harbor + MiMo build guide (you write the code; this is the spec)

Goal: one Harbor-format env in Brownout. MiMo Code and General come in through importers.
Everything the audit patches is written back out as a Harbor task folder.

```
MiMo parquet row ──importer──┐
Terminal-Bench / SWE-bench ──┼─► Harbor task folder ─► HarborTasks + HarborEnv + HarborVerifier ─► audit/swarm
(patched task) ◄─exporter────┘                              │
                                                    ContainerBackend: Docker | Prime | HF Sandbox
```

Rules from CLAUDE.md that still hold: the verifier reads only `(task, trace)`; env facts come
through `snapshot()`; thresholds live in `task.spec`; a crashed scorer is `ok=False`, not reward 0;
no new dependencies without asking; no paid calls without asking.

---

## 0. Read these first (30 min, in this order)

| What | Why | Link |
|---|---|---|
| Harbor tasks overview | folder layout, reward files | https://docs.harborframework.com/tasks/overview |
| Harbor `task.toml` reference | every field | https://docs.harborframework.com/tasks/configuration |
| Harbor verifier page | when tests/ is copied, reward format | https://docs.harborframework.com/tasks/verifier |
| Harbor adapter guide | naming, oracle check, output of an importer | https://github.com/harbor-framework/harbor/blob/26106c6d9c66848a1440faf5724d336affac52aa/docs/content/docs/datasets/adapters.mdx |
| MiMo dataset card | fields, domains | https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss |
| Explorer Space: `app/runner/domains.py` | **how Xiaomi grades each domain** (the best source) | https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/raw/main/app/runner/domains.py |
| Explorer Space: `app/runner/core.py` | sandbox lifecycle | .../raw/main/app/runner/core.py |
| Explorer Space: `app/catalog.py`, `build_data.py` | **where `test_patch` / `test_command` come from. Not read yet** | .../raw/main/app/catalog.py |

Step 0 task: open `catalog.py` and `build_data.py`, and print one Code row. Find the fields
`test_patch` and `test_command` (the grader uses them as `inst[...]`). The dataset card only
showed `docker_image`, `cwd`, `problem_statement`. If the tests are not in the row, find where they live.

---

## 1. Harbor facts (checked against the docs, Oct 2026)

**Layout**
```
<task>/ instruction.md  task.toml  environment/Dockerfile  tests/test.sh  solution/solve.sh (optional)
```
- `tests/` is uploaded to `/tests/` **after the agent finishes**. `solution/` is not shown to the agent.
- `test.sh` must write a number to `/logs/verifier/reward.txt` (or a labeled JSON object to
  `reward.json`; Harbor prefers the JSON if both exist). The docs do not say whether the exit code
  is read, so the reward file is what counts. The valid range is not documented; the adapter guide says 0-1.
- By default the verifier runs in a **fresh copy** of the environment, NOT the agent's container
  (`verifier.environment_mode = "separate"`). The agent's changes do not carry over unless you
  use `"shared"` or declare `artifacts`. Decide which one your importer needs (see 3 and 4).
- `task.toml` needs `[task] name = "<org>/<task>"`, unique and stable, lowercase with hyphens.
  Tasks with no name cannot join a dataset. `schema_version = "1.3"`.
- Useful `task.toml` fields: `[agent] timeout_sec`, `[verifier] timeout_sec` (default 600),
  `[verifier] network_mode`, `[environment] docker_image / workdir / cpus / memory_mb / network_mode / env`,
  `artifacts`, `[metadata]` (free-form: put the source row id and domain here).
- `.harborignore` in `tests/` or `solution/` excludes files from uploads and content hashes.
- Adapter output is checked with the oracle agent: every task must earn 100% with `solution/`.

**What I did not verify:** the multi-step format, `harbor` CLI flags beyond the adapter doc, and
the exact contents of `reward.json`.

---

## 2. How MiMo grades (from `domains.py`; this is what your verifier must reproduce)

| Domain | Where tests come from | Setup before the agent | After the agent | Reward |
|---|---|---|---|---|
| **Code** | `inst["test_patch"]` + `inst["test_command"]`, not in the sandbox during the run | record base commit; delete `/tests`, `/logs`, build caches; if later commits are reachable, move `.git` to a hidden random path | restore `.git`; save the agent's change as `git diff --cached <base>`; reset test paths; `git apply` the test patch; run `test_command` (timeout 1800) | 1.0 if exit code 0, else 0.0. A failed reset or apply gives `None` (infra error, not 0) |
| **General (rule + judge)** | `man["verifier"]["uploads"]`, pushed after the agent exits | agent runs as non-root `agent` (uid 500); system dirs `chmod 700`; a probe fails the run if `/work/system` is readable | verify ports still answer; write final text to `answer.md` and a session file; push verifier files; run `man["verifier"]["command"]` with `VERIFY_DETERMINISTIC=1`, `VERIFY_AGENT_JUDGE=1` and judge env vars | `reward.json` in [0,1], else `None`; per-check results in `verifier_meta.json` |
| **Terminal-style** (inside General) | `tests_files` unpacked to `/tests` after the agent exits | n/a | `bash /tests/test.sh`, read `/logs/verifier/reward.txt` | `-1` or empty gives `None`; else clamped [0,1] |
| Cyber | `cyber_expected(...)` | writes `/root/expected_func.json` + `/root/server.py` (chmod 700) before the agent | read `/root/last_result.json` | 1.0 iff `crash and match` |
| Music | none | no sandbox | `extract_abc` then `music_scorer` | validity gate (4 checks), then `total/100` |
| Webdev | vision judge | | | judge score; skip |

Notes that matter for your design:
- **`None` is not 0.** Map it to `grader.ok=False` (infra error), as CLAUDE.md says.
- Code's anti-leak steps (scrub, hide `.git`) are part of the *environment*. If your Harbor export leaves
  them out, you create holes that the original did not have. Copy them into `environment/` or the verifier.
- Cyber, to check later: the header of `domains.py` says the expected crash is never in the sandbox,
  but `Cyber.run` writes `expected_func.json` and `server.py` before the agent starts (root-only, chmod 700).
  Whether that matters depends on whether the agent can become root. Verify before claiming anything.
- The dataset card (2026-10-19) says Code images ship with "anti-hack preprocessing" and to keep
  `anti_hack_cleanup: true`. So recall on Code should be measured with planted holes, not assumed.

---

## 3. Components to build, in order

### A. `ContainerBackend` (small interface, one file per backend)
```
start(image, workdir, env, network) -> handle
exec(handle, cmd, timeout) -> (exit_code, stdout, stderr)     # never raises on non-zero exit
put(handle, local_dir_or_bytes, remote_path)
get(handle, remote_path) -> bytes
stop(handle)
```
- `FakeBackend` first (a dict as the filesystem, scripted `exec` results), like `FakeModel`.
  All the harness tests run on it, offline and free.
- `DockerBackend`: needs Docker on Linux or WSL2. Prefer the `docker` CLI via `subprocess` to avoid a new
  dependency. Name containers `brownout-<run>-<task>` and always stop them in `finally`.
- Prime / HF sandbox: **ask first** (new dependency, costs money). Pricing and SDK not checked.
  The explorer uses `huggingface_hub.Sandbox.create(image=..., ...)`, so that is a possible third backend.
- One contract test that runs against every backend: start, write a file, exec `cat`, copy out, stop.

### B. `HarborTasks` (the `TaskSource`)
- Walk a dir of task folders. One `Task` per folder; `id` = `[task].name`.
- `spec` = `instruction`, `docker_image`/Dockerfile hash, `workdir`, `agent_timeout`, `verifier_timeout`,
  `domain`, and `tests_sha` = hash of `tests/` (so the identity is content, not a version string).
- Reject tasks with no `[task].name`, or a duplicate name. Make `load()` deterministic (sorted).

### C. `HarborEnv` (the `Env`)
- `reset(task)`: start the container, return `instruction.md` as the first message.
- Tools: `bash(cmd)`, `read_file(path)`, `write_file(path, content)`, `Submit()`. Output is truncated.
  Counts turns, not tool calls (CLAUDE.md).
- `snapshot()` is where grading happens: **after Submit or timeout**, upload `tests/`, run `test.sh`, and
  copy back `/logs/verifier/reward.txt|json`. Record raw facts only: `exit_code`, `reward_file`,
  `reward_json`, `stdout_tail`, `agent_diff_sha`, `tests_sha`, `infra_error`. No pass/fail decision here.
- Also save the agent's final diff (`git diff` for code tasks), so a patched `tests/` can be re-run
  against it later without re-running the agent. This is what makes re-grading cheap.
- `close()` always stops the container.

### D. `HarborVerifier` (the `Verifier`)
- Reads `trace.final_state` only. `reward = reward_file`; `ok=False` when `infra_error` is set or the
  reward file is missing/out of range. Put the facts in `signals`.
- Never run a command here. If you need to run something, it belongs in `snapshot()`.

### E. Importers (one function each, output = Harbor folder)
- **MiMo Code:** `instruction.md` = `problem_statement`; `environment/Dockerfile` = `FROM <docker_image>` + workdir `cwd`;
  `tests/` = `test_patch` + a `test.sh` that does the same steps as the Code adapter above and writes
  `reward.txt` (1 or 0); `[task].name` from `instance_id`. No `solution/` unless the row has one.
- **MiMo General rule-checks:** `tests/` = the verifier uploads; run only the deterministic part (see 5).
- **Terminal-Bench 2.0 / SWE-bench Verified:** already Harbor. Use them first to test B-D.
- Name rule: `org/task`, lowercase, hyphens. Keep the source id in `[metadata]`.

### F. Exporter
- Write a patched task back out as a Harbor folder: copy the original, replace `tests/`/`environment/`,
  bump `[task]` version, write the new `tests_sha` into `[metadata]`. Never overwrite the original.

---

## 4. Wiring into the audit (what changes in `src/audit/`)

| Piece | Change |
|---|---|
| `target.py` | add a `HarborTarget` next to `CloudOpsTarget`, taking `verifier=` the same way |
| `gate.layer1` | take the target as an argument per env (today it builds `CloudOpsTarget`) |
| replay | a repro is a script of tool calls, but replay now needs a container: **replay N times and require it to land every time**, else mark `flaky` and do not confirm |
| `confirm.is_hack` | needs a second signal for true success. Harbor has none, so options: run the task's `solution/` against the tests (proves the tests accept a real solution); or a person decides. **Your decision, same open item as cloudops** |
| identity | `verifier_sha` = hash of `tests/` + `environment/` + image digest (a `latest` tag drifts) |
| precheck | always `--saved-runs none`; use a live solver run |
| planted holes | weaken a copy of `tests/` (looser `grep`, `test.sh` ignoring failures, a test that checks only one file). Measure recall before trusting results |

Leak check (LESSONS.md 11): grep everything the tester sees for hole names before any run.

---

## 5. General: split it
1. Deterministic checks (`VERIFY_DETERMINISTIC=1`): start here, free to replay.
2. Judge part (`VERIFY_AGENT_JUDGE=1`): calls a model, so it costs money and varies. Store the judge's
   prompt and answer in the trace; the verifier reads the stored answer. Re-grading a patched rubric
   means new calls: **ask before running**.

---

## 6. Order for the week
1. `FakeBackend` + contract test. 2. `HarborTasks` on 3 Terminal-Bench folders. 3. `HarborEnv` +
`HarborVerifier` on `FakeBackend`. 4. `DockerBackend`; run the oracle `solution/` on those 3 tasks
(must earn 1.0). 5. Replay-N-times check. 6. MiMo Code importer (after step 0). 7. Planted holes in
`tests/`, then a small capped audit run. 8. General deterministic. 9. Remote backend (after approval).

## 7. Tests to write (offline, `FakeBackend`)
- loader: missing name, duplicate name, sorted order, `tests_sha` changes when a test file changes
- env: `snapshot()` before Submit still grades; container stopped on error; output truncation
- verifier: `reward.txt` `1` / `0` / missing / `-1` / `1.5`: each gives reward and `ok` as in the table above
- importer: a MiMo row yields a folder that the loader accepts; no `solution/` does not crash
- exporter: original folder untouched; new `tests_sha` differs
- contract: the same script gives the same result on Fake and Docker

## 8. Open decisions (yours)
- True-success signal for Harbor tasks (see section 4).
- Shared vs separate verifier environment (`environment_mode`) for the importer.
- Which remote backend, and who approves the spend.
