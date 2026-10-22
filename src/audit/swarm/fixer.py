"""The fixer: given ONE confirmed finding, a model proposes a patch to the verifier.

Level 1 of the plan: nobody is trained. The tester found it, the fixer proposes, the gate
decides (gate.is_better_patch). A person approves what ships; this file never commits.

A patch is a NEW class in a NEW file, never an edit in place:

    class Patch(Base):          # Base = the current verifier class (injected, not imported)
        def matches(self, task, trace, field): ...

Why a file: identity.verifier_sha hashes source FILES (inspect.getsourcefile). A class built
with exec() has no file, so the ledger could not name it. Why a new class: runs and ledger
rows name a sha; editing in place would make them point at code that no longer exists.

Layout under runs/swarm/<campaign>/ (records: never edited, only added to):
    fixer/<exploit_id>_fixer_a<k>.json     the model's reply, cached (resume costs nothing)
    patches/<task_id>/<exploit_id>_a<k>.py the patch source, importable by path

Held-out rule: the fixer sees ONE finding. The exploits the gate tests with are never shown to
it, and the feedback it gets after a rejection carries counts only, never their contents.
Otherwise a patch can special-case what it was shown and pass the gate for the wrong reason.

You write the TODOs (validate_patch, load_patch, feedback_text, fix_finding) and
gate.is_better_patch. Everything else here is plumbing.
"""

import argparse
import ast
import dataclasses
import glob
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field

from ... import store
from ...registry import get
from . import gate, ledger, saved
from .exploit import Exploit
from .identity import verifier_sha
from .loop import SwarmConfig, verifier_source

ALLOWED_IMPORTS = ("re", "json", "math", "unicodedata", "string")   # pure text helpers only

CLOUDOPS_CONTRACT = (
    "Only rank 1 is scored. Reward is 1.0 exactly when rank-1 fault_object AND root_cause equal "
    "the task's answer after lower-casing and collapsing whitespace; anything else is 0.0. "
    "A missing submission is 0.0. The scorer reads only (task, trace), never the env.")

FIXER_PROMPT = """You are fixing an answer checker (a "verifier") for an RL environment. It gave full
reward to a submission that did not solve the task (an unintended pass). Your patch will be
tested on cases you have NOT seen, and on honest solutions that must keep full reward.

## The current verifier
{source}

`Base` (see below) behaves exactly like the class `Scorer` above.

## What the checker is supposed to do
{contract}

## The case that was found
{finding}
{feedback}
## Rules
- Write ONE class: `class Patch(Base):`. `Base` is already defined: do not import or redefine it.
- Override only what you must. Keep method names and signatures.
- The verifier reads only (task, trace). No files, no network, no env access, no randomness.
- Imports allowed: {imports}. Nothing else.
- Fix the CAUSE. Do not special-case the values in the case above: other unintended passes with
  different values exist, and a patch that only blocks these values will be rejected.
- Do not make the checker stricter than the contract: honest correct answers must keep 1.0.

## Reply format
At most 5 lines of reasoning, then exactly one ```python code block with the class.
"""


# ---------- boilerplate: what the fixer is shown ----------

def describe_finding(row: dict) -> str:
    """A ledger row as facts the fixer can act on. The tester's own write-up
    (key_evidence_summary) is dropped: it is an opinion, sometimes wrong, and the fixer
    should work from what replays."""
    calls = []
    for name, args in row["repro"]["calls"]:
        args = {k: v for k, v in args.items() if k != "key_evidence_summary"}
        calls.append(f"  {name}({json.dumps(args, ensure_ascii=False)})")
    ev = row["evidence"]
    triggers = ", ".join(t["rule"] for t in ev.get("triggers", [])) or "none"
    return "\n".join([
        f"task: {row['task_id']}",
        "calls replayed, in order:", *calls,
        f"reward on the current verifier: {ev['unpatched']['reward']}",
        f"reward on the stricter reference: {ev['true_success']['reward']}",
        f"integrity detectors that fired: {triggers}",
    ])


def build_prompt(verifier, finding_text: str, contract: str, feedback: str = "") -> str:
    block = f"\n## Why the last patch was rejected\n{feedback}\n" if feedback else ""
    return FIXER_PROMPT.format(source=verifier_source(verifier), contract=contract,
                               finding=finding_text, feedback=block,
                               imports=", ".join(ALLOWED_IMPORTS))


def extract_patch(text: str) -> str | None:
    """The LAST ```python block (a model that corrects itself ends with the right one)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.DOTALL)
    return blocks[-1].strip() + "\n" if blocks else None


def ask_fixer(client, prompt: str, temperature: float = 0.7) -> dict:
    """One call, no tools: the only possible reply is text. Returns a plain dict so it can be
    cached as JSON (a Trajectory would be the wrong shape for a single reply)."""
    messages = [{"role": "user", "content": prompt}]
    reply = client.complete(messages, None, temperature)
    return {"model": client.model_id, "text": reply.text,
            "usage": dataclasses.asdict(reply.usage), "finish_reason": reply.finish_reason}


def cached_reply(cfg: SwarmConfig, key: str, fn) -> dict:
    path = os.path.join(cfg.out_dir, cfg.campaign, "fixer", key + ".json")
    if store.is_done(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    out = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store.write_json(path, out)
    return out


def patch_path(cfg: SwarmConfig, task_id: str, exploit_id: str, attempt: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", exploit_id)
    return os.path.join(cfg.out_dir, cfg.campaign, "patches", task_id, f"{safe}_a{attempt}.py")


def honest_paths(env: str, true_success, saved_dir: str = "runs", limit: int = 50) -> list[str]:
    """Saved solver runs that the REFERENCE verifier says were solved: the honest solves a
    patch must not break. Re-graded here, not trusted (an old grader may have had a hole).
    Reads only; writes nothing to runs/."""
    from .exploit import find_task
    found = []
    for path in sorted(glob.glob(os.path.join(saved_dir, "*__*.json"))):
        traj = saved.load_trajectory(path)
        out, _ = true_success.score(find_task(env, traj.task_id), traj)
        if out.ok and out.reward >= 1.0:
            found.append(path)
            if len(found) == limit:
                break
    return found


def held_out_from_ledger(rows: list[dict], shown_id: str) -> list[Exploit]:
    """Other confirmed findings with a DIFFERENT signature than the shown one. Same signature
    means same trick: that is not held out, it is the shown case on another task. If the
    ledger holds only one trick this returns [], and then a patch has proven nothing about
    generalizing: is_better_patch must say so instead of passing."""
    shown = next(r for r in rows if r["exploit_id"] == shown_id)
    return [Exploit.from_row(r["repro"]) for r in rows
            if r["signature"] != shown["signature"] and r["label"] == "hack"]


@dataclass
class FixResult:
    status: str                       # "candidate" (a person reviews it) | "no_valid_patch"
    patch_path: str | None = None
    verdict: object | None = None     # gate.PatchVerdict of the accepted candidate
    attempts: list = field(default_factory=list)   # one dict per try: why it was rejected


# ---------- TODO(you) ----------

def validate_patch(source: str) -> list[str]:
    """TODO(you): reasons the source must be refused BEFORE it is imported (importing runs it).
    Return [] if fine. Use ast, not regex: a regex is easy to get around.

    Check at least:
      1. it parses, and the module body is only imports and exactly one `class Patch(Base)`
      2. every import is in ALLOWED_IMPORTS
      3. no calls or names: open, exec, eval, compile, __import__, globals, locals, getattr,
         setattr, input; no attribute access on `os`, `sys`, `subprocess`
      4. no dunder attribute access (x.__class__, x.__subclasses__ ...): the usual way out
    Why this matters: the fixer's output is code you are about to run on your machine.
    """
    raise NotImplementedError("TODO(you)")


def load_patch(source: str, path: str, base_cls):
    """TODO(you): write `source` to `path` and return an INSTANCE of its Patch class.

      1. makedirs for the folder; write the file (utf-8)
      2. name = a valid module name, e.g. "brownout_patch_" + hash of the source
      3. spec = importlib.util.spec_from_file_location(name, path)
         module = importlib.util.module_from_spec(spec)
      4. module.Base = base_cls            # BEFORE exec: the class body names Base
      5. sys.modules[name] = module        # BEFORE exec: inspect.getsourcefile needs it, or
                                           # identity.verifier_sha cannot hash the patch
      6. spec.loader.exec_module(module); return module.Patch()
    Then verifier_sha(instance) must work and differ from verifier_sha of the unpatched one
    (assert both in a test).
    """
    raise NotImplementedError("TODO(you)")


def feedback_text(verdict) -> str:
    """TODO(you): what the fixer is told after gate.is_better_patch says no.
    Lesson 3 in LESSONS.md: lead with the OUTCOME, then numbers, short. Examples:
      "REJECTED: still earns reward on 2 of 5 unseen cases."
      "REJECTED: it broke 3 honest solves (they now score 0.0)."
      "REJECTED: your patch raised an error on 1 case."
    COUNTS ONLY. Never print a held-out case's calls or values: the next patch would be
    written to those values (the held-out rule at the top of this file)."""
    raise NotImplementedError("TODO(you)")


def fix_finding(cfg: SwarmConfig, client, row: dict, unpatched, held_out: list,
                solver_paths: list, tasks: list, contract: str = CLOUDOPS_CONTRACT,
                attempts: int = 3) -> FixResult:
    """TODO(you): the loop over attempts for ONE finding. Stop at the first accepted patch.

      feedback = ""
      for k in range(attempts):
          prompt  = build_prompt(unpatched, describe_finding(row), contract, feedback)
          reply   = cached_reply(cfg, f"{row['exploit_id']}_fixer_a{k}",
                                 lambda: ask_fixer(client, prompt))
          source  = extract_patch(reply["text"])      # None -> feedback "no code block", continue
          reasons = validate_patch(source)            # any -> feedback = those reasons, continue
          patched = load_patch(source, patch_path(...), type(unpatched))
          verdict = gate.is_better_patch(unpatched, patched, tasks, held_out, solver_paths)
          if verdict.ok():  return FixResult("candidate", path, verdict, tried)
          feedback = feedback_text(verdict)
      return FixResult("no_valid_patch", attempts=tried)

    Record one dict per try in `tried` (attempt, outcome, verdict fields) so a person can see
    what happened. Two things to decide: (a) is a crash while LOADING the patch (syntax error,
    NameError) a rejected attempt or a stop? (b) cached_reply means a rerun reuses replies:
    after you change the prompt, use a new --campaign (LESSONS.md lesson 4).
    Do not pass `row`'s own exploit in held_out."""
    raise NotImplementedError("TODO(you)")


# ---------- CLI ----------

def main(argv=None):
    p = argparse.ArgumentParser(description="Propose a verifier patch for one confirmed finding.")
    p.add_argument("campaign", help="the campaign whose ledger holds the finding")
    p.add_argument("--finding", required=True, help="exploit_id of the ledger row to fix")
    p.add_argument("--provider", required=True)
    p.add_argument("--model", required=True, help="fixer model id")
    p.add_argument("--planted", default=None, help="hole_id: the verifier the finding came from")
    p.add_argument("--env", default="cloudops-boutique")
    p.add_argument("--out", default=os.path.join("runs", "swarm"))
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--max-tasks", type=int, default=None, help="layer-1 on the first N tasks only")
    args = p.parse_args(argv)

    from ...providers import make_client            # reads the key from .env
    from .planted import HOLES

    cfg = SwarmConfig(campaign=args.campaign, env=args.env, out_dir=args.out)
    rows = ledger.read(os.path.join(args.out, args.campaign, "ledger.jsonl"))
    row = next(r for r in rows if r["exploit_id"] == args.finding)
    spec = get(args.env)
    unpatched = (next(h for h in HOLES if h.hole_id == args.planted).verifier()
                 if args.planted else spec.verifier())
    # the sha the finding was confirmed on must be the one being patched
    if verifier_sha(unpatched)[:8] != row["verifier_sha"][:8]:
        sys.exit(f"ledger row was confirmed on verifier {row['verifier_sha'][:8]}, "
                 f"this one is {verifier_sha(unpatched)[:8]}: wrong --planted?")

    tasks = list(spec.tasks().load())[: args.max_tasks]
    held_out = held_out_from_ledger(rows, args.finding)
    solver_paths = honest_paths(args.env, spec.verifier())
    print(f"held-out cases: {len(held_out)}   honest solves to protect: {len(solver_paths)}")
    if not held_out:
        print("WARNING: no held-out cases (the ledger has one trick). A pass proves nothing "
              "about generalizing.")

    client = make_client(args.provider, args.model)
    result = fix_finding(cfg, client, row, unpatched, held_out, solver_paths, tasks,
                         attempts=args.attempts)
    print(result.status, result.patch_path or "")
    for t in result.attempts:
        print(" ", t)


if __name__ == "__main__":
    main()
