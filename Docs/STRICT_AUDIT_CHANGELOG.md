# Everything that changed: the strict verifier and its audit (2026-10-22)

This is the "what did Claude touch, why, and how do I change it" file. For the plain-language
story of how the verifier works, read `Docs/STRICT_VERIFIER.md` first. For the published claims,
read `findings/cloudops-unconfirmed-passes/README.md`.

---

## 0. The "reverse" check, explained

A checker can be wrong in two opposite ways. A smoke alarm is a good picture:

| | There really is a fire | There is no fire |
|---|---|---|
| **Alarm rings** | correct | **false alarm** |
| **Alarm silent** | **missed fire** | correct |

Our checker is the strict verifier. "Alarm rings" = it says **unconfirmed** (right answer, but the
model never fetched the confirming evidence). So its two possible mistakes are:

1. **False alarm**: it says *unconfirmed*, but the evidence actually was fetched.
   We measured this with the **forward audit**: GLM looked at all 89 unconfirmed runs and called 3
   of 82 `label_gap` (= false alarm). About 4%.
2. **Missed fire**: it says *confirmed*, but the evidence was NOT really fetched.
   That is the **reverse audit**. It matters most for the **73 runs** that only became "confirmed"
   because of the two rules I added (namespace fix: 22, pod-as-config: 51). If those rules are
   too loose, they silence real alarms, and the true unconfirmed count is higher than 89.

Why only those 73, not every confirmed run? The other confirmed runs passed under the
benchmark's own labels, unchanged. A miss there would mean upstream's label is too loose, which is
a different (much smaller) question. The 73 are the ones *our* decisions put at risk.

The result of the reverse audit is appended at the end (section 7).

---

## 1. What changed, in order (and why)

1. **Ported upstream's milestone matcher** (`vendor/process_eval.py`), because Cloud-OpsBench
   labels every case with the evidence a good diagnosis needs, and we needed that to define
   "confirmed".
2. **Built the strict verifier** (`strict.py`): reward = JRA AND the label's answer milestones.
   A new class; the training verifier `verifier.py` was not touched.
3. **Re-graded all 1,095 saved runs** into new files (`regrade_strict.py`). Saved runs were never
   modified.
4. **Caught and fixed my own bug**: v-first required every end of the evidence chain; formulas have
   `any` alternatives. Found because 12 DeepSeek runs had MC = 1 but strict = 0.
5. **Hand audit** of 20 unconfirmed runs (Claude), with a sampler (`sample_unconfirmed.py`).
   Found the namespace-on-nodes label problem.
6. **Built a blind second labeller** (`audit_judge.py`, GLM-5.3-Flash) and measured agreement
   (kappa). It found that a pod describe shows the same config as the YAML, and that Claude's own
   labels were inconsistent about it.
7. **v0.2.0**: two documented deviations in the strict reward only (namespace, pod-as-config).
8. **Independent check** on a fresh sample (seed 1): ~6% false alarms.
9. **Census** of all 89 unconfirmed runs (forward audit).
10. **Packaged the finding** Tokenless-style (`findings/cloudops-unconfirmed-passes/`).
11. **Reverse audit** of the 73 rescued runs (`--mode rescued`). Results in section 7.

---

## 2. Every file, point by point

Legend: **NEW** = created this session. **EDIT** = existed before. Nothing in `runs/*.json` (saved
records) and nothing in `verifier.py` / `scoring.py` / `harness.py` was changed.

### `src/envs/SystemEnv/cloudops/vendor/process_eval.py` (NEW)
- What: a stdlib port of Cloud-OpsBench `agents/cloudops_agent/evaluation_utils/matcher.py` +
  `evaluator.py`, checkout `ea05daf`, MIT.
- Key functions: `load_annotation`, `tool_use_matches`, `pattern_matches`, `evaluate`
  (returns mc / eoc / ecr / ee / established / ordered), `formula_ok`.
- Kept faithful on purpose, so MC / EOC / ECR / EE compare with upstream's table.
- Dropped: upstream's manual-review flags (they change no score).
- **To change it safely:** don't, unless upstream changes. Any edit changes the strict hash
  (new re-grade file) and breaks comparability with upstream. Put deviations in `strict.py`.
- Not verified: number-for-number match with upstream's `evaluation.py` on the same trajectories.

### `src/envs/SystemEnv/cloudops/strict.py` (NEW)
- `StrictCloudOpsVerifier`, version 0.2.0. Reward `grounded_jra`. Measuring stick only.
- Composes `CloudOpsVerifier` instead of subclassing, because `Verifier.scorers`
  (`scoring.py:58-63`) would keep the parent's `@reward jra` registered.
- `answer_milestones`: ends of the dependency chain (not role names: 43 cases have no
  `root_cause` role).
- `answer_grounded`: the case's own formula with only answer milestones required (the bug fix).
- `tool_steps`: the trajectory as (tool, args, output); Submit and text-only turns dropped.
- Deviation 1, `without_cluster_namespace`, `CLUSTER_SCOPED = {nodes, persistentvolumes}`.
- Deviation 2, `config_seen_elsewhere`, `POD_VISIBLE` (13 fault types), `APP_OBJECTS`
  (pod / deployment / replicaset), name match `app` or `app-<hash>`.
- `grounding()` reports which rule decided (`raw`, `by_namespace`, `by_pod_config`, `final`).
- Metrics: `outcome_scores` (jra, ca, fa), `process_scores` (mc, eoc, ecr, ee), `closed_jra`,
  `grounding_detail` (grounded_raw, via_namespace_fix, via_pod_config).
- Missing label file raises, so `grader.ok = False`, never a silent 0.
- **To change it safely:**
  - Add or remove a fault type in `POD_VISIBLE` -> only that fault's runs move (see section 4).
    Only add a type if `kubectl describe pod` prints its deciding field.
  - Add a resource to `CLUSTER_SCOPED` -> only labels on that resource relax.
  - Bump `version`, re-run `regrade_strict`, and the output file gets a new hash automatically.
  - Add a test in `tests/cloudops_strict_test.py` for every new rule (pattern: one test that it
    rescues the right case, one that it does NOT rescue a look-alike).

### `src/envs/SystemEnv/cloudops/regrade_strict.py` (NEW)
- Re-grades `runs/*__*.json` with the strict verifier into
  `runs/regrade/strict-<hash>.jsonl`. Never modifies saved runs.
- `strict_sha`: hash of the strict class AND the code it uses (`verifier.py`, `process_eval.py`),
  because `verifier_sha` only covers the class's own inheritance chain.
- Prints per-model JRA / strict / unconfirmed / ECR / steps, what each deviation changed, and
  ECR by step count.
- **To change:** add a column -> add it in `grade_one` and `summarize`. Changing this file does not
  change the hash (it is not graded code).

### `src/envs/SystemEnv/cloudops/sample_unconfirmed.py` (NEW)
- Draws runs for an audit, prints a readable readout, writes a blank label sheet (never
  overwrites a started sheet).
- `--rescued` samples the other direction (runs confirmed only via a deviation).
- Prints UTF-8 safely (a Windows console crash was fixed here).
- **To change:** `answer_terms` (the keyword match for "output lines mentioning the answer") is
  rough; it is a reading aid, not a decision.

### `src/envs/SystemEnv/cloudops/audit_judge.py` (NEW)
- The blind LLM labeller. Reads only `file` and `task_id` from a sheet; another labeller's labels
  never reach the prompt (tested).
- Two modes:
  - `unconfirmed` (prompt v1): labels `blind / symptom_weak / symptom_strong / label_gap`.
  - `rescued` (prompt v1r): labels `confirmed / not_confirmed`.
- Evidence = the true answer, what the label wanted, every call with its output (long outputs
  keep the first 1,500 and last 2,500 characters, so Events at the end survive).
- `DiskCache`: one JSON line per verdict, keyed by (model, prompt version, prompt). Parsed
  verdicts are reused; failed replies are recorded but retried.
- `cohen_kappa`, `agreement` (4 labels and collapsed to 3).
- **To change safely:**
  - Rubric text change -> bump the mode's version in `MODES`, so old verdicts are not reused.
  - Another judge model -> `--model`; the cache file name includes the model.
  - Known judge failure: GLM returns empty text on some long prompts (7 of 89 in the census).

### `findings/cloudops-unconfirmed-passes/` (NEW)
- `README.md`: claims A / B / C, confidence ratings, what is not claimed.
- `RUBRIC.md`: frozen rubric v1 with one real example per label, decision order, known ambiguities.
- `REPRODUCE.md`: pinned inputs, free and paid steps.
- `compute_numbers.py` -> `expected.json`: every number with Wilson 95% intervals, recomputable
  (forward census, reverse audit, negative control).
- `negative_control.py`: the 12-case control for the reverse audit (paid, 12 calls; cached).
- **To change:** never edit `expected.json` by hand; rerun `compute_numbers.py --write`.

### Tests (NEW)
- `tests/cloudops_strict_test.py`: port, formula alternatives, both deviations (each with a
  look-alike that must NOT be rescued), real-case tests on `boutique/admission/1`, missing label.
- `tests/cloudops_audit_judge_test.py`: parsing, blindness, output cutting, cache reuse and retry,
  kappa including the majority-label trap, rescued mode.
- Run: `python -m unittest tests.cloudops_strict_test tests.cloudops_audit_judge_test -v`.

### Docs and notes
- `Docs/STRICT_VERIFIER.md` (NEW): the long explanation.
- `Docs/STRICT_AUDIT_CHANGELOG.md` (NEW): this file.
- `interview/07_strict_verifier.md` (NEW): short version + Q&A, including the bug.
- `interview/04_war_stories.md` (EDIT): story 13 (FrontierSWE N3), appended earlier.

### Earlier in the same session (FrontierSWE, separate track)
- `tests/frontier_n3_test.py` (NEW): N3, a valid notebook run ranks at or below a failed one in
  v1's `score_from_reward.py`. Demonstrated at the scorer level.

---

## 3. Output files (`runs/regrade/`)

| File | What | Use it? |
|---|---|---|
| `strict-6f0952b5.jsonl` | v0.2.0 re-grade, one line per run | **yes, current** |
| `strict-82038ac5.jsonl` | v0.1.0 (formula fix, no deviations) | for comparison |
| `strict-2fc80737.jsonl` | first version, buggy | **no** |
| `strict-82038ac5-audit-seed0.{txt,csv}` | 20-case readout + Claude's labels | audit record |
| `strict-82038ac5-audit-seed0-judge-*.jsonl` | GLM on the same 20 | audit record |
| `strict-6f0952b5-audit-seed1.*` | fresh 20, GLM | independent check |
| `strict-6f0952b5-audit-seed2.*` | all 89 unconfirmed, GLM (forward census) | claim B |
| `strict-6f0952b5-rescued-seed0.*` | all 73 rescued, GLM (reverse audit) | section 7 |
| `judge-cache-zai-org_GLM-5.3-Flash.jsonl` | every GLM verdict | makes reruns free |

---

## 4. How the numbers could move (sensitivity)

The headline is **Nemotron: 89 of 322 correct answers unconfirmed**. Each design decision below
moves it. "Effect" is measured on our data unless marked "at most".

| Decision | Current choice | Alternative | Effect on the 89 |
|---|---|---|---|
| Labels as published (no deviations) | deviations on | off | **162** (+73) |
| Namespace on nodes / PVs | ignored | required | +22 |
| Pod describe = app config | on, 13 fault types | off | +51 |
| Add `missing_image_pull_secret` to `POD_VISIBLE` (the event names the missing secret) | not added | added | at most −7 |
| Accept a node describe for node-list milestones (`performance-28`) | not accepted | accepted | at most −1 |
| Answer milestones = chain ends | chain ends + formula | every chain end (the bug) | DeepSeek +12, Nemotron +17 (raw-rule numbers) |
| Use ECR instead (all milestones) | not used | ECR | a different question; Nemotron ECR 0.25 |
| Step counting (Submit excluded) | excluded | included | no effect on strict; changes EE only |

**Claim B moves with the judge, not the verifier:**
- Strong vs weak turned out to be the judge's call, not the data's: GLM 57 weak, gpt-oss 19 weak
  on the same 89 (section 6b). That split is withdrawn.
- What survives two judges: 75 of 82 "symptom" (both agree), 1 "blind", 0 "label_gap".
- A rubric v2 (master-node line = background) or a rule-based reason count would move B more than
  anything in the verifier.

**What the 89 are made of** (by fault type): taint 18, node selector 15, node affinity 10,
image pull secret 7, anti-affinity 6, code defects 11 in total, containerd 5, quotas 6,
kubelet 3, packet loss 3, pod CPU 2, PVC storage class 2, volume permission 1.
Scheduling faults dominate. They share one scheduler message format, which is exactly where one
event lists several possible reasons.

**Behaviour, not just labels:** unconfirmed runs have a median of 3 tool calls; Nemotron's
confirmed passes have a median of 5. The shortcut is "stop after the first informative message".

---

## 5. How this could be moved elsewhere (reuse)

| Move it to | What carries over | What needs work |
|---|---|---|
| **Train-Ticket** (204 cases) | everything: labels exist in `process-label/trainticket`, same matcher | saved runs on Train-Ticket; `POD_VISIBLE` names checked for its fault types |
| **More models** | everything | runs (paid). A third model would lift claim A's confidence most cheaply |
| **The training loop** | strict = held-out measure; `grounding_detail` = gate labels | the cheap patch rule for the training verifier, `is_better_patch` |
| **Upstream (Cloud-OpsBench)** | claim C as a bug report: namespace on nodes is mechanical and 8/10 solid | write it up with 2-3 case ids; offer `without_cluster_namespace` as the fix |
| **Other benchmarks** (FrontierSWE, BFCL) | the method: build a stick, audit it both directions, report with intervals | they have no milestone labels: the stick must be built differently (hidden tests, state checks) |
| **Another judge** | `audit_judge.py --model ...` | a cross-family check (e.g. a third model) to raise B's confidence |

Things that would most raise confidence, cheapest first:
1. Reverse audit (section 7): done, 0 of 73 misses, negative control 12 of 12.
2. 30 human-labelled cases from the census, compared with GLM by kappa.
3. Match the port against upstream's `evaluation.py` on 10 trajectories.
4. A third model's runs.

---

## 6. Open items

- Port not matched number-for-number with upstream.
- No human gold labels.
- GLM fails to answer on some long prompts (7 of 89 forward, more in section 7 if any).
- Strict v0.2.0 not yet frozen in git (commit before quoting its hash outside the repo).

---

## 6b. Third labeller: gpt-oss-120b (census and seed 0)

- Why gpt-oss: independent of Claude (rubric author), GLM (first judge) and both evaluated models.
  Nemotron-3-Super was rejected (same family as the model being judged).
- `Clients/openai_compact.py` (EDIT, one line): omits `tools` when the list is empty, because
  Nebius's vLLM server for gpt-oss rejects `tools=[]`. Agent runs always pass tools; unchanged.
- `findings/cloudops-unconfirmed-passes/compare_judges.py` (NEW): judge-vs-judge agreement.
- Result on the 89 census (82 answered by both): coarse agreement 93% raw (75 both "symptom",
  1 both "blind", 0 both "label_gap"); strong vs weak disagrees almost completely (kappa 0.07).
  Cause: the always-cordoned master node line in scheduling events. The weak / strong split was
  **withdrawn** from claim B; claim B now reports only what both judges agree on.
- Seed 0 vs Claude: kappa 0.35 (4 labels), 0.48 (3).
- Cost: about 98 calls, well under $1.

## 7. Reverse audit result (the 73 rescued runs)

**Question:** for each run the two v0.2.0 rules turned from "unconfirmed" into "confirmed", does a
tool output actually show the confirming fact?

**How:** GLM-5.3-Flash, blind to all other labels, rubric "rescued" v1r (`audit_judge.py`,
`--mode rescued`): `confirmed` = an output shows the fact; `not_confirmed` = the rule accepted a
call whose output does not contain it. The prompt names the rule that accepted the run.

**Result: 73 of 73 `confirmed`, 0 `not_confirmed` (95% CI for the miss rate: 0%–5.0%).**
By rule: namespace fix 22 of 22, pod-as-config 51 of 51. One reply was empty and succeeded on retry.

**Was the judge just agreeing with the rule it was told about?** Checked with a **negative
control** (`findings/cloudops-unconfirmed-passes/negative_control.py`): 12 unconfirmed runs
where pod-as-config would be **wrong** (affinity and anti-affinity faults; `describe pod` does not
print affinity), sent with the same prompt falsely saying the rule had accepted them.
**GLM rejected 12 of 12** (CI 75.8%–100%), each time citing what the output lacks ("no
affinity section", "no LABELS column"). So it reads the outputs, not the rule name.

**What this means:**
- Neither rule hides misses on this data, so the 89 is not an undercount caused by our rules.
- Together with the forward audit (3 of 82 false alarms), the 89 is accurate to within a few runs
  in both directions, as judged by one LLM.
- Claim A's confidence goes from 7 to **8/10**. What still caps it: two models, our harness,
  no human labels, port not matched against upstream.

**Limits:** one judge model; the control covers one kind of wrong acceptance (fields `describe pod`
does not print), not every possible one; runs confirmed under upstream's unchanged labels were not
reverse-audited.

Files: `runs/regrade/strict-6f0952b5-rescued-seed0.{txt,csv}`,
`runs/regrade/strict-6f0952b5-rescued-seed0-judge-zai-org_GLM-5.3-Flash.jsonl`,
`runs/regrade/strict-6f0952b5-rescued-negative-control.jsonl`; numbers in
`findings/cloudops-unconfirmed-passes/expected.json` under `"reverse"`.

## 8. Legitimate control: upstream's expert (golden) trajectories (free, no LLM)

**Idea (borrowed from Tokenless EnvCheck):** every finding needs a *legitimate control*: does
honest work pass? Cloud-OpsBench ships one: `golden-trajectory/`, 1,508 expert paths (1,100
boutique, 408 Train-Ticket), each a correct diagnosis. If the benchmark's labels are right, they
accept these.

**Step 1: is our port faithful?** Ran upstream's own `evaluation_utils` on all 1,508 paths next
to `vendor/process_eval.py` (scratch script, not kept). Identical established milestones, MC, EOC
and completeness on **all 1,508** (upstream: 312 incomplete; port: 312, the same paths). This
closes the open item "port not yet matched against upstream". Side finding: upstream's loader
requires a milestone `description` field that **none** of the 754 published labels has, so
upstream's evaluator crashes on its own labels; we filled in `""` for the comparison.

**Step 2: grade the expert paths** with `golden_control.py` (new), which uses the same
`grounding()` as the strict verifier. Output `runs/regrade/golden-control.jsonl`; numbers under
`"golden_control"` in `expected.json` (boutique only, matching our claims):

- Raw labels reject **151 / 1,100** expert paths (13.7%, CI 11.8–15.9%).
- Pod-as-config (deviation 2) accepts **111 of those 151**: independent support for deviation 2,
  because now it is the benchmark's own experts who treat a pod / replicaset describe as config.
- Namespace fix (deviation 1) accepts **0**: experts always pass `namespace`. Deviation 1 is
  still correct (nodes are cluster-scoped) but only matters for harnesses whose models omit it.
- **40** still rejected: quota (19), missing pull secret (12), others (9). The expert read the
  answer off events and never queried the quota / secret object, i.e. the same shortcut as
  Nemotron. The labels and the expert paths disagree about what must be checked. Not relaxed:
  the labels' reading stays.

**Step 3: overlap with the 89.** In **80 of 89** Nemotron unconfirmed runs, every expert path
for the case passes the strict check (89.9%, CI 81.9–94.6%): the skipped check is one experts do.
9 are in cases where some expert path skips it too.

**Effect on claims:** A unchanged at 8/10, now with an LLM-free control (and a stated 80-to-89
range depending on whether the expert or the label decides). C up from 6 to **7/10**: pattern 2
has independent evidence; pattern 1 is reported as harness-specific.

Files: `src/envs/SystemEnv/cloudops/golden_control.py`, `tests/cloudops_golden_control_test.py`
(5 tests: parser rules, one expert shortcut path, one pod-config rescue),
`findings/cloudops-unconfirmed-passes/compute_numbers.py` (new `golden_control` block; earlier
numbers unchanged).
