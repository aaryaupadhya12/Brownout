"""The regression gate: deterministic checks decide whether a patch ships (spec rule).
The swarm only finds exploits; it never decides. All checks run and every failure is
reported, so one run tells the person everything wrong with a patch.

A patch passes only if ALL hold:
  1. Layer 1 (UniversalAudit: oracle, floor, perturbation) passes on ALL tasks, graded with
     the patched verifier. A patch for task A must not break task B.
  2. every held-out exploit replays to no reward on the patched verifier
     (held-out = never shown to an attacker or to whoever writes the patch)
  3. every saved solver trajectory still earns full reward. Re-grade, don't re-run: a
     verifier-only patch can't change what the solver did, so this is free and exact.

Needs one change outside this folder (yours): CloudOpsTarget(verifier=None) in
src/audit/target.py, so grade() uses the given verifier instead of always CloudOpsVerifier().
That is an audit change, not a grader change.
"""

from dataclasses import dataclass, field

from ..report import Report
from ..target import CloudOpsTarget
from ..universal import UniversalAudit
from . import saved
from .exploit import earned, find_task, replay

LAYER1 = ("oracle", "floor", "perturbation")   # the others need saved runs or a judge

@dataclass
class GateResult:
    passed: bool
    reasons: list = field(default_factory=list)      # every failed sub-check, e.g.
                                                     # "layer1 universal.floor failed on 452 tasks"
    measured: dict = field(default_factory=dict)     # counts per sub-check, for the trust report


def layer1(patched_verifier, tasks) -> list[str]:
    target = CloudOpsTarget(verifier=patched_verifier)
    audit = UniversalAudit()
    report = Report(target.name, ["universal"], target.grader_name())

    for name, fn in audit.checks():
        if name in LAYER1:
            for task in tasks:
                audit._call(report, f"{audit.name}.{name}", fn.check_tag, fn, target, task)

    reasons = []

    for name in LAYER1:
        check_id = f"{audit.name}.{name}"
        found = [f for f in report.findings if f.check_id == check_id]
        bad = {f.task_id for f in found if f.status in ("fail", "error")}

        if not found:
            reasons.append(f"{check_id}: check did not run")
        elif bad:
            reasons.append(f"layer1 {check_id} failed on {len(bad)} tasks")
    return reasons

def held_out_replay(patched_verifier, held_out) -> list[str]:
    reasons = []

    for exploit in held_out:
        traj = replay(exploit,verifier= patched_verifier)

        if not traj.grader.ok:
            reasons.append(f"{exploit.exploit_id}: patch CRASHED on the repro "
                           f"(ok=False) — a crash is not a block")
        elif earned(traj):
            reasons.append(f"{exploit.exploit_id}: exploit still earns reward")
    return reasons

def solver_regrade(patched_verifier, solver_paths, env="cloudops-boutique") -> list[str]:
    reasons = []

    for path in solver_paths:
        traj = saved.load_trajectory(path)
        task = find_task(env, traj.task_id)

        out, _errors = patched_verifier.score(task, traj)

        if not out.ok:
            reasons.append(f"{path}: patch CRASHED re-grading an honest solve (ok=False)")
        elif out.reward < 1.0:
            reasons.append(f"{path}: honest solve lost reward ({out.reward}, ok={out.ok})")
    return reasons


def gate(patched_verifier, tasks, held_out, solver_paths) -> GateResult:

    reasons = []
    reasons.extend(layer1(patched_verifier,tasks))
    reasons.extend(held_out_replay(patched_verifier,held_out))
    reasons.extend(solver_regrade(patched_verifier,solver_paths))

    return GateResult(
        passed = not reasons,
        reasons= reasons,
        measured={}
    )


@dataclass
class PatchVerdict:
    """The breakdown behind a yes/no, so ONE function serves two uses: the gate (is it all
    good?) and later a reward for training the fixer (how good is it?). A bare bool throws
    away exactly the numbers a shaped reward needs. Counts, never contents: nothing here may
    reveal a held-out case's calls (the fixer reads feedback built from this)."""
    held_out_total: int = 0
    blocked: int = 0                 # earned reward BEFORE the patch, no longer does
    still_earn: int = 0              # earn reward after the patch (unfixed)
    crashed: int = 0                 # the patch raised on them (ok=False): NOT a block
    solves_total: int = 0
    solves_lost: int = 0             # honest solves that no longer earn full reward
    layer1_reasons: list = field(default_factory=list)   # regression on the whole task set

    def ok(self) -> bool:
        """TODO(you): the commit rule, as a bool. Write it from the fields above.
        Questions to settle in code:
          - held_out_total == 0: nothing was proven. Is that a pass? (I'd say no.)
          - must it block ALL held-out cases, or is blocking more than before enough?
          - solves_lost > 0, crashed > 0, layer1_reasons: any of them veto?
        """
        raise NotImplementedError("TODO(you)")

    def score(self) -> float:
        """TODO(you): the same facts as one number in [0, 1], for a future reward.
        Keep these two properties: a patch that breaks honest solves must score lower than
        one that doesn't, and ok() True must score higher than any ok() False patch."""
        raise NotImplementedError("TODO(you)")


def is_better_patch(unpatched, patched, tasks, held_out, solver_paths) -> PatchVerdict:
    """TODO(you): measure `patched` against `unpatched`; return the breakdown.

      blocked / still_earn / crashed: for each held-out exploit, replay on BOTH verifiers.
          A case counts as blocked only if it earned on `unpatched` (otherwise there was
          nothing to block). A crash (ok=False) on `patched` goes to `crashed`, never
          to blocked: a crash is not a fix.
      solves_total / solves_lost: re-grade each honest solver path with `patched`
          (see solver_regrade); a crash there counts as lost.
      layer1_reasons: layer1(patched, tasks).

    You may reuse the three helpers above, but they return reason STRINGS; you need counts,
    so compute them directly with replay() and earned(), or change the helpers to return
    counts. Write nothing to runs/."""
    raise NotImplementedError("TODO(you)")
