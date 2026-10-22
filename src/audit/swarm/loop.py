"""The per-task loop. Plain code, not an LLM (spec).

Resume without a state machine (harden-v0's durable pattern): every episode is saved at a
path built from its ROLE KEY. On resume the loop just runs again from the top; finished
episodes load from disk instead of running. Nothing to keep in sync with the files.

The verifier sha is part of the key, so after a person patches the verifier, re-running
the same command re-attacks the new version and keeps the old episodes as records.

Order of one task (v1: a person writes patches):
    precheck: the solver passes on the real verifier, else status "precheck_failed"
    for round in range(max_rounds):
        up to `attempts` attacker episodes, each told what failed before
        first one that is_hack() -> ledger row -> status "awaiting_review" -> stop
        none -> status "robust" -> stop
"""

import ast
import dataclasses
import errno
import glob
import inspect
import os
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass

from ... import store
from ...harness import run_episode
from ...registry import get
from ...scoring import Verifier
from . import confirm, detectors, ledger, roles, saved
from .exploit import earned, find_task
from .identity import verifier_sha
from .roles import GENERIC


@dataclass(frozen=True)
class SwarmConfig:
    campaign: str                          # e.g. "boutique-planted-v1"; folder name under out_dir
    env: str = "cloudops-boutique"
    out_dir: str = "runs/swarm"
    max_rounds: int = 10                   # hacker-fixer paper: a stopping rule, not a cost cap
    attempts: int = 3                      # attacker tries per round
    threshold: float = 1.0                 # reward that counts as "earned"
    whitebox_from_round: int | None = 1    # None = always blind. Plan: blind first, then source.
    max_turns: int = 20
    saved_runs: str | None = "runs"        # precheck reuses solved baseline runs from here (re-graded
                                           # with true_success); None = always run the live solver
    plan_turn: bool = True                 # attacker turn 1 has no tools, so it must write a plan


def role_key(task_id: str, round_: int, role: str, attempt: int, verifier_sha: str,
             specialist_sha: str = "") -> str:
    """e.g. "boutique-admission-1/r0_attacker_a2_3f9c1a0b". Stable across runs, so it is a
    cache key. A specialist adds "_<name>-<sha>"; without one the key is unchanged, so old
    campaigns resume byte-for-byte."""
    base = f"{task_id}/r{round_}_{role}_a{attempt}_{verifier_sha[:8]}"
    return base + (f"_{specialist_sha}" if specialist_sha else "")


@contextmanager
def campaign_lock(cfg: SwarmConfig):
    """One process per campaign. With this held, the ledger's in-process lock is enough;
    without it, two processes could tear each other's ledger writes (no OS-level file lock
    here, and your runs are on Windows). The lock file holds the pid and is removed on exit."""
    path = os.path.join(cfg.out_dir, cfg.campaign, ".lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise RuntimeError(
                f"campaign {cfg.campaign!r} is locked by {path}. Another run is active, "
                f"or a previous one crashed: delete the file if you are sure none is running.")
        raise
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def run_or_load(cfg: SwarmConfig, key: str, fn):
    path = os.path.join(cfg.out_dir, cfg.campaign, key + ".json")
    if store.is_done(path):
        return saved.load_trajectory(path)
    traj = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)    # keys contain "/": task_id/r0_...
    store.write_json(path, dataclasses.asdict(traj))
    return traj


def verifier_source(verifier) -> str:
    """The LOGIC of the verifier's own classes (stops at scoring.Verifier), for whitebox mode.

    Not the raw text. Raw inspect.getsource leaked the answer: planted classes are named
    SubstringMatch / FailOpenMissing, have name = "cloudops-planted-...", and the substring
    docstring spelled out the exploit ("/", "_"). v2's whitebox hit quoted that docstring.
    So: parse, drop docstrings and name/version, rename classes to Scorer / ScorerBase..,
    and ast.unparse (which drops comments). The code still runs the same; only labels go."""
    classes = [cls for cls in type(verifier).__mro__
               if issubclass(cls, Verifier) and cls is not Verifier]
    renames = {cls.__name__: "Scorer" + ("Base" * i) for i, cls in enumerate(classes)}
    parts = []
    for cls in reversed(classes):                    # base first, like a file would read
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    body = body[1:]                  # docstring
                if isinstance(node, ast.ClassDef):
                    body = [n for n in body if not (isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id in ("name", "version") for t in n.targets))]
                    node.name = renames.get(node.name, node.name)
                node.body = body or [ast.Pass()]
            elif isinstance(node, ast.Name) and node.id in renames:
                node.id = renames[node.id]           # base-class references
        parts.append(ast.unparse(tree))
    return "\n\n".join(parts)


def saved_solve(cfg: SwarmConfig, task, true_success) -> str | None:
    """Path of a saved baseline run that solved this task, or None.

    Files in saved_runs are named "{task_id}__{model}__{hash}__{i}.json"; the "__" keeps
    boutique-admission-1 from matching boutique-admission-10. Each one is RE-GRADED with
    true_success, not trusted by its stored reward: the verifier reads only (task, trace),
    so re-grading is free and exact, and a run graded by an older, holey verifier can't pass.
    Read only: runs/ files are records (CLAUDE.md).
    Caveat: the env may have changed since the run was saved. Cloudops replays recorded
    outputs, so this holds there; for live envs (Harbor) set saved_runs=None."""
    if not cfg.saved_runs:
        return None
    for path in sorted(glob.glob(os.path.join(cfg.saved_runs, f"{task.id}__*.json"))):
        try:
            traj = saved.load_trajectory(path)
        except (OSError, ValueError, TypeError):
            continue                                   # unreadable or older format: skip it
        out, _errors = true_success.score(task, traj)
        if out.ok and out.reward >= cfg.threshold:
            return path
    return None


def precheck(cfg: SwarmConfig, task, solver_model, true_success, tries: int = 1) -> str | None:
    """Is the task solvable? Returns the path of a solved trajectory, or None.

    1. a saved baseline run that re-grades as solved (free, no model call)
    2. else run_or_load a live solver episode (role "solver", round -1) with the REAL verifier.
       tries=1: the solver runs at temperature 0, so more tries repeat the same answer.
    The returned path is a solver trajectory gate.solver_regrade can use later.
    Writes {task}/precheck.json so show.py can tell which one was used."""
    record = os.path.join(cfg.out_dir, cfg.campaign, task.id, "precheck.json")
    path = saved_solve(cfg, task, true_success)
    source = "saved"
    for attempt in range(tries if path is None else 0):
        key = role_key(task.id, -1, "solver", attempt, verifier_sha(true_success))
        traj = run_or_load(cfg, key, lambda: run_episode(get(cfg.env).env(), task, solver_model,
                                                         max_turns=cfg.max_turns,
                                                         verifier=true_success))
        if earned(traj, threshold=cfg.threshold):
            path, source = os.path.join(cfg.out_dir, cfg.campaign, key + ".json"), "live"
            break
    os.makedirs(os.path.dirname(record), exist_ok=True)
    store.write_json(record, {"passed": path is not None, "source": source if path else "live",
                              "path": path})
    return path


def attack_task(cfg: SwarmConfig, task, attacker_model, verifier, true_success,
                ledger_path: str, precheck_path, specialist=GENERIC, stop=None) -> dict:
    """Rounds for ONE (task, prober) pair. precheck_path is passed in (run once per task by
    the caller). `stop`, if given, is a callable the barrier sets: True means another prober
    already found something this round, so don't start the next round."""
    result = {"task_id": task.id, "specialist": specialist.name, "status": "", "rounds": []}
    result["precheck"] = precheck_path
    if precheck_path is None:
        result["status"] = "precheck_failed"
        return result

    specialist = dataclasses.replace(specialist, model_id=attacker_model.model_id)
    sha = verifier_sha(verifier)

    for round_ in range(cfg.max_rounds):
        if stop is not None and stop():
            result["status"] = result["status"] or "stopped"
            return result
        failures = []
        whitebox = cfg.whitebox_from_round is not None and round_ >= cfg.whitebox_from_round
        for attempt in range(cfg.attempts):
            suffix = specialist.key_suffix(cfg.max_turns, cfg.plan_turn,
                                    "whitebox" if whitebox else "blind")
            key  = role_key(task.id, round_, "attacker", attempt, sha, suffix)
            traj = run_or_load(cfg, key, lambda: roles.run_attacker(
                    task, attacker_model, verifier, get(cfg.env).env(),
                    mode="whitebox" if whitebox else "blind",
                    failed_summaries=tuple(failures),
                    verifier_source=verifier_source(verifier) if whitebox else None,
                    max_turns=cfg.max_turns, plan_turn=cfg.plan_turn,
                    focus=specialist.focus,
                ),
            )
            ex = roles.to_exploit(traj, cfg.env, exploit_id=key)
            v = confirm.is_hack(ex, verifier, true_success, cfg.threshold,
                                adapter=detectors.adapter_for(cfg.env))
            # Get the primary hack 
            if v.hack and v.evidence.get("primary"):
                ex = dataclasses.replace(ex, mechanism=v.evidence["primary"])
            result["rounds"].append({"round": round_, "attempt": attempt,
                                     "exploit_id": ex.exploit_id, "hack": v.hack})
            if v.hack:
                row = {
                    "exploit_id": ex.exploit_id,
                    "signature": ledger.signature(ex, task),
                    "campaign": cfg.campaign,
                    "task_id": task.id,
                    "role": f"attacker:{specialist.name}",
                    "mechanism": ex.mechanism,
                    "label": "hack",
                    "verifier_sha": sha,
                    "env_sha": v.evidence.get("env_sha", ""),
                    "group_id": traj.tracking.get("group_id", ""),
                    "evidence": v.evidence,
                    "repro": ex.as_row(),
                    "status": "open",
                }
                ledger.append_once(ledger_path, row)
                result["status"] = "awaiting_review"
                return result
            failures.append(roles.failure_note(traj, attempt,
                                               withhold_answer=earned(traj, cfg.threshold)))

    result["status"] = "robust"
    return result


def harden_task(cfg: SwarmConfig, task, attacker_model, solver_model, verifier, true_success,
                ledger_path: str, specialist=GENERIC) -> dict:
    """One task, one prober, serial: precheck then rounds. Kept for the existing CLI path
    and tests. The parallel runner calls precheck and attack_task directly."""
    precheck_path = precheck(cfg, task, solver_model, true_success)
    return attack_task(cfg, task, attacker_model, verifier, true_success,
                       ledger_path, precheck_path, specialist=specialist)


def run_campaign(cfg: SwarmConfig, tasks, attacker_clients, solver_model, verifier,
                 true_success, ledger_path: str, specialists, workers: int = 1) -> list[dict]:
    """Phase 1: precheck every task once (serial; cheap and shared).
    Phase 2: run every (task, prober) pair, up to `workers` at a time.
    attacker_clients maps a prober name -> its model client (probers may use different models).
    Barrier: within a task, once any prober records a finding, the others stop at their next
    round boundary, so which episodes exist depends on outcomes, not thread timing."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    find_task(cfg.env, tasks[0].id)        # warm the task cache before any thread uses it

    prechecks = {t.id: precheck(cfg, t, solver_model, true_success) for t in tasks}
    found = {t.id: threading.Event() for t in tasks}

    def unit(task, specialist):
        res = attack_task(cfg, task, attacker_clients[specialist.name], verifier, true_success,
                          ledger_path, prechecks[task.id], specialist=specialist,
                          stop=found[task.id].is_set)
        if res["status"] == "awaiting_review":
            found[task.id].set()
        return res

    pairs = [(t, sp) for t in tasks for sp in specialists]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(unit, t, sp): (t.id, sp.name) for t, sp in pairs}
        for fut in as_completed(futures):
            tid, name = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"task_id": tid, "specialist": name,
                                "status": "error", "error": f"{type(e).__name__}: {e}"})
    return results