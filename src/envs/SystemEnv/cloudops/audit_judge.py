"""Second labeller for the unconfirmed-pass audit: an LLM judge from another model family.

    # free: print the first prompt, no API call
    python -m src.envs.SystemEnv.cloudops.audit_judge runs/regrade/strict-82038ac5-audit-seed0.csv --dry-run
    # paid: label every row, then compare with a reference sheet
    python -m src.envs.SystemEnv.cloudops.audit_judge runs/regrade/strict-82038ac5-audit-seed0.csv \
        --provider nebius --model zai-org/GLM-5.3-Flash \
        --reference runs/regrade/strict-82038ac5-audit-seed0.csv

Blind by construction: from the sheet it reads only `file` and `task_id`. The other labeller's
label and note never enter a prompt (tests check this). The evidence is the run's own tool calls
and outputs plus what the milestone label wanted, built here from the saved trajectory.

Every verdict is cached on disk, keyed by (model, prompt version, prompt), so a rerun is free and
gives the same labels. A reply that does not parse is a judge failure: recorded with its error,
never turned into a label, and left out of the agreement numbers.

Agreement is Cohen's kappa, on 4 labels and collapsed to 3 (symptom_strong + symptom_weak ->
symptom). Raw agreement alone looks high when most rows share one label.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter

from ....audit.swarm.saved import load_trajectory
from ....judge import fence, first_json_object
from .regrade_strict import all_tasks
from .strict import StrictCloudOpsVerifier, answer_milestones, tool_steps

PROMPT_VERSION = "1"
LABELS = ("blind", "symptom_weak", "symptom_strong", "label_gap")
HEAD, TAIL = 1500, 2500          # per tool output: describe outputs keep their Events at the end

RUBRIC = """You are auditing a Kubernetes root-cause-analysis agent. Its final answer was CORRECT, but a
strict check says it never obtained the evidence that CONFIRMS the answer. Decide which case this is.

Labels:
- label_gap: the agent DID obtain the confirming evidence, and the check rejected it unfairly. Typical:
  the right tool was called and its output shows the evidence, but an argument differs in a way that
  does not change what is shown (e.g. a namespace on cluster-scoped nodes), or an equivalent object
  shows the same configuration (e.g. a deployment instead of its pod). If the missing argument DOES
  change the output (e.g. node labels hidden without show_labels), it is NOT label_gap.
- symptom_strong: the confirming check was not done, but a tool output names the root cause so that
  only this fault type fits (e.g. "image ... NotFound", "Permission denied").
- symptom_weak: the confirming check was not done; outputs point at the answer, but several fault
  types fit the same message (e.g. one scheduler event listing taint, cordon and selector reasons).
- blind: no tool output points at the answer.

Ask, in order: (1) did a call obtain what the wanted check would show? -> label_gap.
(2) Otherwise, does an output name the answer? (3) Could another fault type explain that output?"""

PROMPT = """{rubric}

True answer (correct, submitted by the agent): {answer}

What the strict check wanted and did not find:
{wanted}

The agent's tool calls, in order, with their outputs (long outputs are cut in the middle):
{calls}

Reply with only a JSON object: {{"label": "<one of {labels}>", "reason": "<one sentence>"}}"""

# Mirrored check (mode "rescued"): runs the strict verifier v0.2.0 counts as CONFIRMED only because
# of one of its two deviations. Is that acceptance right?
LABELS_RESCUED = ("confirmed", "not_confirmed")

RUBRIC_RESCUED = """You are auditing a Kubernetes root-cause-analysis agent. Its final answer was CORRECT. A strict
check accepted the answer as CONFIRMED, but only by applying a relaxed rule to the check the label
wanted. Decide whether the agent really obtained the confirming evidence.

Labels:
- confirmed: some tool output in the trajectory actually SHOWS the fact the wanted check would show
  (for example the pod's configured memory request, the image reference, the probe port, the node
  labels), so the diagnosis was confirmed from evidence the agent fetched.
- not_confirmed: no tool output shows that fact; the relaxed rule accepted a call whose output does
  not actually contain the confirming evidence.

Judge only from the tool outputs shown, not from the rule's name. A call that was accepted but
whose output does not contain the fact is not_confirmed."""

PROMPT_RESCUED = """{rubric}

True answer (correct, submitted by the agent): {answer}

The check the label wanted (not met as written):
{wanted}

The relaxed rule that accepted it: {accepted_by}

The agent's tool calls, in order, with their outputs (long outputs are cut in the middle):
{calls}

Reply with only a JSON object: {{"label": "<one of {labels}>", "reason": "<one sentence>"}}"""

MODES = {   # mode -> (prompt template, rubric, labels, prompt version)
    "unconfirmed": (PROMPT, RUBRIC, LABELS, PROMPT_VERSION),
    "rescued": (PROMPT_RESCUED, RUBRIC_RESCUED, LABELS_RESCUED, "1r"),
}
RULE_TEXT = {
    "namespace": "a namespace argument is ignored on cluster-scoped resources (nodes, PVs)",
    "pod_config": "a describe of the app's pod / deployment / replicaset counts as its YAML",
}


def cut(text: str) -> str:
    text = str(text)
    if len(text) <= HEAD + TAIL:
        return text
    return text[:HEAD] + f"\n... [{len(text) - HEAD - TAIL} chars cut] ...\n" + text[-TAIL:]


def evidence(verifier, task, trace) -> dict:
    """Everything the judge sees, from the trajectory and the label only."""
    ann = verifier.annotation(task)
    held = set(verifier.process(task, trace)["established"])
    wanted = []
    for m in ann["milestones"]:
        if str(m["id"]) in answer_milestones(ann) and str(m["id"]) not in held:
            uses = []
            for use in m.get("admissible_tool_uses", ()):
                pats = [p.get("value") for p in use.get("evidence_patterns", ())]
                uses.append(f"{use['tool_name']} {json.dumps(use.get('arguments', {}))}"
                            + (f" (output must show {pats})" if pats else ""))
            wanted.append(f"- {m['id']} {m.get('role', '')}: any of: " + "; ".join(uses))
    calls = []
    for i, (tool, args, obs) in enumerate(tool_steps(trace)):
        calls.append(f"[{i}] {tool} {json.dumps(args, sort_keys=True)}\n{fence(cut(obs))}")
    answer = task.spec["answer"]
    rules = []
    if hasattr(verifier, "grounding"):
        g = verifier.grounding(task, trace)
        rules = [RULE_TEXT[k] for k, hit in (("namespace", g["by_namespace"]),
                                              ("pod_config", g["by_pod_config"])) if hit]
    return {"answer": f"{answer.get('fault_object')} / {answer.get('root_cause')}",
            "wanted": "\n".join(wanted) or "(none)", "calls": "\n".join(calls) or "(no tool calls)",
            "accepted_by": "; ".join(rules) or "(none)"}


def build_prompt(ev: dict, mode: str = "unconfirmed") -> str:
    template, rubric, labels, _ = MODES[mode]
    return template.format(rubric=rubric, labels=", ".join(labels), **ev)


def parse_label(text: str, labels: tuple = LABELS) -> tuple[str, str]:
    """Reply -> (label, reason). Anything else raises: a judge failure, not a label."""
    obj = first_json_object(text or "")
    if obj is None:
        raise ValueError("judge reply has no JSON object")
    label = str(obj.get("label", "")).strip().casefold()
    if label not in labels:
        raise ValueError(f"unknown label {obj.get('label')!r}; options: {labels}")
    return label, str(obj.get("reason", ""))


class DiskCache:
    """One JSON line per verdict. Append-only; the last line for a key wins."""

    def __init__(self, path: str):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[row["key"]] = row

    def get(self, key):
        """Only a parsed verdict is reused. A failed reply stays in the file as a record but
        does not block a retry (an empty reply is usually transient)."""
        row = self.rows.get(key)
        return row if row is not None and row.get("label") is not None else None

    def put(self, row: dict):
        self.rows[row["key"]] = row
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def cache_key(model_id: str, version: str, prompt: str) -> str:
    return hashlib.sha256(f"{model_id}\n{version}\n{prompt}".encode()).hexdigest()


def judge_one(client, cache, prompt: str, labels: tuple = LABELS,
              version: str = PROMPT_VERSION) -> dict:
    key = cache_key(client.model_id, version, prompt)
    hit = cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}
    reply = client.complete([{"role": "user", "content": prompt}], [], 0.0)
    try:
        label, reason = parse_label(reply.text, labels)
        row = {"key": key, "label": label, "reason": reason, "error": None}
    except ValueError as e:
        row = {"key": key, "label": None, "reason": "", "error": str(e), "raw": (reply.text or "")[:500]}
    row.update(model=client.model_id, prompt_version=version)
    cache.put(row)
    return {**row, "cached": False}


# ---------- agreement ----------

def reference_label(row: dict) -> str | None:
    """Map a hand sheet (label + note) onto the judge's labels. The note's 'weak:' / 'strong:'
    prefix splits symptom; a bare 'symptom' counts as strong."""
    label = (row.get("label") or "").strip().casefold()
    if label in LABELS:
        return label
    if label == "symptom":
        return "symptom_weak" if (row.get("note") or "").strip().casefold().startswith("weak") \
            else "symptom_strong"
    return label or None


def collapse(label):
    return "symptom" if label and label.startswith("symptom") else label


def cohen_kappa(a: list, b: list) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(pairs: list) -> dict:
    """pairs: [(reference, judge)] with both present."""
    ref, jud = [p[0] for p in pairs], [p[1] for p in pairs]
    ref3, jud3 = [collapse(x) for x in ref], [collapse(x) for x in jud]
    return {
        "n": len(pairs),
        "raw4": sum(x == y for x, y in pairs) / len(pairs) if pairs else None,
        "kappa4": cohen_kappa(ref, jud),
        "kappa3": cohen_kappa(ref3, jud3),
        "confusion": Counter(f"{r} -> {j}" for r, j in pairs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sheet", help="a sampler label sheet; only file and task_id are read")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--provider", default="nebius")
    ap.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--reference", default=None, help="a filled sheet to compare against")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt, call nothing")
    ap.add_argument("--mode", choices=sorted(MODES), default="unconfirmed",
                    help="unconfirmed: strict says not confirmed; rescued: strict says confirmed "
                         "only because of a v0.2.0 deviation")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    _, _, labels, version = MODES[args.mode]

    with open(args.sheet, encoding="utf-8") as f:
        cases = [{"file": r["file"], "task_id": r["task_id"]} for r in csv.DictReader(f)]
    tasks, verifier = all_tasks(), StrictCloudOpsVerifier()
    prompts = [build_prompt(evidence(verifier, tasks[c["task_id"]],
                                     load_trajectory(os.path.join(args.runs, c["file"]))), args.mode)
               for c in cases]
    if args.dry_run:
        print(prompts[0] if prompts else "(empty sheet)")
        print(f"\n[dry run] {len(prompts)} prompts, longest {max(map(len, prompts), default=0):,} chars")
        return

    from ....providers import make_client          # needs openai + a key: only on a real run
    client = make_client(args.provider, args.model)
    safe_model = args.model.replace("/", "_")
    out = args.out or os.path.splitext(args.sheet)[0] + f"-judge-{safe_model}.jsonl"
    cache = DiskCache(os.path.join(os.path.dirname(out) or ".", f"judge-cache-{safe_model}.jsonl"))

    results, paid = [], 0
    for case, prompt in zip(cases, prompts):
        if cache.get(cache_key(client.model_id, version, prompt)) is None:
            if paid >= args.max_calls:
                print(f"budget of {args.max_calls} calls spent; stopping (rerun to continue)")
                break
            paid += 1
        results.append({**case, **judge_one(client, cache, prompt, labels, version)})
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "key"}) + "\n")
    failed = [r for r in results if r["label"] is None]
    print(f"judged {len(results)} ({paid} paid calls, {len(failed)} unparseable) -> {out}")
    print("labels:", dict(Counter(r["label"] for r in results if r["label"])))

    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            ref = {r["file"]: reference_label(r) for r in csv.DictReader(f)}
        pairs = [(ref[r["file"]], r["label"]) for r in results
                 if r["label"] and ref.get(r["file"])]
        a = agreement(pairs)
        if not a["n"]:
            print("\nno rows with both labels: nothing to compare")
            return
        print(f"\nagreement on {a['n']} rows: raw {a['raw4']:.2f}, kappa(4 labels) {a['kappa4']:.2f}, "
              f"kappa(3 labels) {a['kappa3']:.2f}")
        for k, v in sorted(a["confusion"].items()):
            print(f"   {v:3d}  {k}")


if __name__ == "__main__":
    main()
