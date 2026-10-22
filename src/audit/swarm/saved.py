"""Load a saved trajectory JSON back into a Trajectory, so a verifier can re-grade it.

Verifiers read attributes (trace.final_state, step.tool), but saved runs are plain dicts.
Unknown keys are dropped and missing ones take the dataclass defaults, so older runs load.
Never write back to runs/: re-grades go to new files (CLAUDE.md).
"""

import json
from dataclasses import fields

from ...contracts import ErrorRecord, GraderOutput, ModelCall, Step, Trajectory, Usage


def _only_fields(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _usage(data) -> Usage:
    return Usage(**_only_fields(Usage, data or {}))


def from_dict(data: dict) -> Trajectory:
    data = _only_fields(Trajectory, data)
    data["steps"] = [Step(**{**_only_fields(Step, s), "usage": _usage(s.get("usage"))})
                     for s in data.get("steps", [])]
    data["calls"] = [ModelCall(**{**_only_fields(ModelCall, c), "usage": _usage(c.get("usage"))})
                     for c in data.get("calls", [])]
    data["errors"] = [ErrorRecord(**_only_fields(ErrorRecord, e)) for e in data.get("errors", [])]
    data["grader"] = GraderOutput(**_only_fields(GraderOutput, data.get("grader", {})))
    return Trajectory(**data)


def load_trajectory(path) -> Trajectory:
    with open(path, encoding="utf-8") as f:
        return from_dict(json.load(f))
