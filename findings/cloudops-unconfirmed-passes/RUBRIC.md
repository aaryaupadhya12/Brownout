# Rubric v1: labelling an unconfirmed pass

Frozen 2026-10-22. This is the text the GLM judge used (`PROMPT_VERSION = "1"`,
`src/envs/SystemEnv/cloudops/audit_judge.py`, `RUBRIC`). A change makes a new version; v1 results
stay labelled v1.

## What is being labelled

A run where the model's answer is **correct** (JRA = 1) but the strict verifier says it never
obtained the evidence that **confirms** the answer (strict = 0). The labeller sees: the true
answer, what confirming check the milestone label wanted, and every tool call with its output.
The labeller never sees another labeller's label.

## Labels

| Label | Definition | Real example from the audit |
|---|---|---|
| `label_gap` | The confirming evidence **was** obtained; the check rejected it unfairly. The right tool was called and its output shows the evidence, but an argument differs in a way that does **not** change what is shown, or an equivalent object shows the same configuration. | `scheduling-64`: called `GetResources nodes show_labels=true`; rejected only because the label demanded a namespace, and nodes have none. |
| `symptom_strong` | The confirming check was **not** done, but an output names the root cause so that only this fault type fits. | `admission-33`: the replicaset events say `exceeded quota: pod-count-quota, ... limited: pods=10`; only "pod quota exceeded" fits, but the quota object itself was never queried. |
| `symptom_weak` | The confirming check was **not** done; outputs point at the answer, but **several fault types fit** the same message. | `scheduling-154`: one scheduler event lists an untolerated taint, a cordoned node and a selector mismatch together; the model picked taint without checking which node the pod targets. |
| `blind` | No tool output points at the answer. | `codedefect-55`: all resources healthy, logs show HTTP 200s, source never read; answer "wrong return" not evidenced anywhere. |

## Decision order

1. **Did any call obtain what the wanted check would show?** If yes: `label_gap`.
   Test: if a missing argument **changes the output**, the check was not done.
   `GetResources nodes` without `show_labels` prints no labels, so it is **not** a gap (case
   `scheduling-96`); without `namespace` it prints the same nodes, so it **is** (case `scheduling-64`).
2. **Otherwise, does an output name the answer?** If no: `blind`.
3. **Could another fault type explain that output?** If yes: `symptom_weak`; if no: `symptom_strong`.

## Known ambiguities in v1 (do not silently fix; a fix is v2)

- **Scheduler events list several reasons.** "1 node(s) were unschedulable, 3 Insufficient memory":
  the cordoned node is a second reason. GLM counts any second listed reason as `symptom_weak`;
  Claude's seed-0 labels sometimes called these strong. This is the main source of
  strong/weak disagreement (seed 0: 4 of 8 disagreements).
- **Measured with a second judge (gpt-oss-120b) on the 89-case census: this ambiguity is not a
  detail, it decides the label.** GLM 57 weak / 21 strong, gpt-oss 19 weak / 66 strong, kappa
  0.07 on the 4 labels. Root cause: the master node is always cordoned in this cluster, so "1
  node(s) were unschedulable" appears in almost every scheduling event as background. v1 does not
  say whether that counts. **The weak / strong split from v1 is therefore not used in any claim.**
  A v2 should state: a reason that applies only to the always-cordoned master node is background.
- **What counts as an "equivalent object".** v1 lets the labeller decide. GLM accepted a pod
  describe as equivalent to an app's YAML (seed 0: cases 1, 8, 10, 19), and once accepted a node
  describe as equivalent to listing nodes (`performance-28`). The strict verifier v0.2.0 accepts
  the first (for fault types whose field a pod describe prints) and not the second.
- **Pull-secret events.** A pod describe does not print `imagePullSecrets`, but the event
  `FailedToRetrieveImagePullSecret ... (non-existent-secret)` names the secret. GLM called two such
  cases `label_gap` (`startup-55`, `startup-56`); the strict verifier keeps them unconfirmed.

## Agreement measured with v1

Seed 0 (20 cases, before the verifier fix), Claude vs GLM, blind: 19 comparable, raw 0.58,
Cohen's kappa 0.39 (4 labels), 0.54 (3 labels, symptoms merged). Kappa corrects for agreement
by chance, which is high when most rows share one label.
