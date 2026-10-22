"""Milestone-based process evaluation, ported from Cloud-OpsBench (MIT, see LICENSE).

Upstream: agents/cloudops_agent/evaluation_utils/matcher.py + evaluator.py (checkout ea05daf).
Same matching rules; dataclasses replaced by plain dicts, manual-review flags dropped (they do
not change any score). Labels: process-label/<system>/<category>/<case_id>/milestone.json.

    MC   milestone coverage        completion formula scored over established milestones
    EOC  evidence-order coverage   same, over milestones established after their prerequisites
    ECR  evidence closure          1 only if the formula holds and no dependency edge is broken
    EE   evidence efficiency       steps that hit any milestone / all steps
"""

import ast
import json
import re
from collections.abc import Mapping

_RESOURCE_ALIASES = {
    "pod": "pods", "pods": "pods", "po": "pods",
    "svc": "services", "service": "services", "services": "services",
    "ep": "endpoints", "endpoint": "endpoints", "endpoints": "endpoints",
    "deploy": "deployments", "deployment": "deployments", "deployments": "deployments",
    "rs": "replicasets", "replicaset": "replicasets", "replicasets": "replicasets",
    "node": "nodes", "nodes": "nodes",
    "event": "events", "events": "events",
    "pv": "persistentvolumes", "persistentvolume": "persistentvolumes",
    "persistentvolumes": "persistentvolumes",
    "pvc": "persistentvolumeclaims", "persistentvolumeclaim": "persistentvolumeclaims",
    "persistentvolumeclaims": "persistentvolumeclaims",
}
_MISSING = object()


def load_annotation(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    edges = []
    for edge in data.get("dependency_edges", ()):
        if isinstance(edge, Mapping):
            edges.append((str(edge["from"]), str(edge["to"])))
        else:
            edges.append((str(edge[0]), str(edge[1])))
    return {
        "case_id": str(data["case_id"]),
        "milestones": list(data["milestones"]),
        "edges": edges,
        "formula": data.get("completion_formula"),
    }


# ---------- matching one step against one admissible tool use (matcher.py) ----------

def _observation_text(observation) -> str:
    if isinstance(observation, str):
        return observation
    return json.dumps(observation, ensure_ascii=False, sort_keys=True)


def _parse_jsonish(value):
    if not isinstance(value, str) or not value.strip():
        return value
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value.strip())
        except Exception:
            pass
    return value


def _arg_matches(actual, expected) -> bool:
    if isinstance(expected, list):                  # a list in the label means "any of these"
        return any(_arg_matches(actual, item) for item in expected)
    if actual == expected:
        return True
    norm = lambda v: _RESOURCE_ALIASES.get(v, v) if isinstance(v, str) else v
    return norm(actual) == norm(expected)


def tool_use_matches(tool: str, args: dict, admissible: dict) -> bool:
    if tool != admissible["tool_name"]:
        return False
    for key, expected in admissible.get("arguments", {}).items():
        if key not in args or not _arg_matches(args[key], expected):
            return False
    return True                                    # extra args in the call are allowed


def _read_path(obj, path: str):
    current = obj
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list):
            if part == "*":
                return current
            try:
                current = current[int(part)]
            except Exception:
                return _MISSING
        else:
            return _MISSING
    return current


def pattern_matches(pattern: dict, observation) -> bool:
    text = _observation_text(observation)
    kind = pattern["kind"]
    value = pattern.get("value")
    flags = pattern.get("flags", ())
    if kind == "literal":
        if "case_sensitive" in flags:
            return str(value) in text
        return str(value).casefold() in text.casefold()
    if kind == "regex":
        re_flags = re.DOTALL
        if "case_sensitive" not in flags:
            re_flags |= re.IGNORECASE
        if "multiline" in flags:
            re_flags |= re.MULTILINE
        return re.search(str(value), text, re_flags) is not None
    if kind == "json_path":
        obj = _parse_jsonish(observation)
        if isinstance(obj, str):
            return False
        found = _read_path(obj, pattern.get("path") or str(value))
        if found is _MISSING:
            return False
        if pattern.get("equals") is not None:
            return found == pattern["equals"]
        if pattern.get("contains") is not None:
            if isinstance(found, (list, tuple, set)):
                return pattern["contains"] in found
            return str(pattern["contains"]).casefold() in _observation_text(found).casefold()
        return bool(found)
    if kind == "yaml_path":                        # upstream: textual fallback, no PyYAML
        leaf = (pattern.get("path") or str(value)).split(".")[-1]
        want = pattern["equals"] if pattern.get("equals") is not None else pattern.get("contains")
        if want is None:
            return re.search(rf"^\s*{re.escape(leaf)}\s*:", text, re.MULTILINE) is not None
        return re.search(rf"^\s*{re.escape(leaf)}\s*:\s*['\"]?{re.escape(str(want))}['\"]?\s*$",
                         text, re.IGNORECASE | re.MULTILINE) is not None
    if kind == "code_snippet":
        squash = lambda s: re.sub(r"\s+", " ", s).strip()
        return squash(str(value)) in squash(text)
    raise ValueError(f"Unsupported evidence pattern kind: {kind}")


def _admissible_hits(admissible: dict, step: tuple) -> bool:
    tool, args, observation = step
    if not tool_use_matches(tool, args, admissible):
        return False
    return all(pattern_matches(p, observation) for p in admissible.get("evidence_patterns", ()))


# ---------- milestones over a trajectory (evaluator.py) ----------

def _milestone_hit_steps(milestone: dict, steps: list) -> list:
    """Step indices where the milestone is established, sorted (single uses + groups)."""
    hits = set()
    for i, step in enumerate(steps):
        if any(_admissible_hits(a, step) for a in milestone.get("admissible_tool_uses", ())):
            hits.add(i)
    for group in milestone.get("admissible_evidence_groups", ()):
        uses = group.get("tool_uses", ())
        seen = set()
        for i, step in enumerate(steps):              # a group is done once every component was hit
            new = {k for k, use in enumerate(uses) if _admissible_hits(use, step)}
            if not new:
                continue
            seen |= new
            if len(seen) == len(uses):
                hits.add(i)
    return sorted(hits)


def _ordered(milestone_ids: list, edges: list, hits: dict, n_steps: int) -> set:
    prereq = {m: set() for m in milestone_ids}
    for src, dst in edges:
        prereq.setdefault(dst, set()).add(src)
    by_step = {}
    for mid, steps in hits.items():
        for i in steps:
            by_step.setdefault(i, []).append(mid)
    ordered = set()
    for i in range(n_steps):
        candidates = [m for m in by_step.get(i, []) if m not in ordered]
        changed = True
        while changed:                                 # same step may unlock a chain
            changed = False
            remaining = []
            for m in candidates:
                if prereq.get(m, set()) <= ordered:
                    ordered.add(m)
                    changed = True
                else:
                    remaining.append(m)
            candidates = remaining
    return ordered


def formula_ok(formula, ids: set) -> bool:
    if isinstance(formula, str):
        return formula in ids
    if isinstance(formula, list):
        return all(formula_ok(x, ids) for x in formula)
    if not isinstance(formula, Mapping):
        return False
    if "all" in formula:
        return all(formula_ok(x, ids) for x in formula["all"])
    if "any" in formula:
        return any(formula_ok(x, ids) for x in formula["any"])
    if "milestone" in formula:
        return str(formula["milestone"]) in ids
    return False


def _formula_score(formula, ids: set) -> float:
    if isinstance(formula, str):
        return 1.0 if formula in ids else 0.0
    if isinstance(formula, list):
        return sum(_formula_score(x, ids) for x in formula) / len(formula) if formula else 0.0
    if not isinstance(formula, Mapping):
        return 0.0
    if "all" in formula:
        kids = formula["all"]
        return sum(_formula_score(x, ids) for x in kids) / len(kids) if kids else 0.0
    if "any" in formula:
        kids = formula["any"]
        return max(_formula_score(x, ids) for x in kids) if kids else 0.0
    if "milestone" in formula:
        return 1.0 if str(formula["milestone"]) in ids else 0.0
    return 0.0


def evaluate(annotation: dict, steps: list) -> dict:
    """steps: [(tool_name, args, observation)] in call order. Returns scores + which ids held."""
    ids = [str(m["id"]) for m in annotation["milestones"]]
    hits = {}
    for m in annotation["milestones"]:
        found = _milestone_hit_steps(m, steps)
        if found:
            hits[str(m["id"])] = found
    established = set(hits)
    ordered = _ordered(ids, annotation["edges"], hits, len(steps))

    # an edge is broken when its target holds but its source never did
    broken = [e for e in annotation["edges"] if e[1] in established and e[0] not in established]
    formula = annotation["formula"]
    if formula is None:
        complete = all(i in established for i in ids) and not broken
    else:
        complete = formula_ok(formula, established) and not broken
    coverage = formula or ids
    evidence_steps = {i for found in hits.values() for i in found}
    return {
        "mc": _formula_score(coverage, established),
        "eoc": _formula_score(coverage, ordered),
        "ecr": 1.0 if complete else 0.0,
        "ee": len(evidence_steps) / len(steps) if steps else 0.0,
        "established": sorted(established),
        "ordered": sorted(ordered),
    }
