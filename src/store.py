"""Saving trajectories so a crashed or stopped benchmark can resume (Phase 6).

    episode_path()  where an episode is saved; the name includes group_id, so a run
                    with a different config never overwrites or counts for this one
    write_json()    atomic write: a crash mid-write leaves the old file or nothing,
                    never a half-written file that looks finished
    is_done()       should resume skip this episode? -> If there was a resujlt return true nut if its unparsable , Model limit over then return Flase

Modelled on verifiers' eval resume (verifiers/v1/cli/eval/resume.py:29-58).
"""

import json
import os


def episode_path(folder: str, task_id: str, model_id: str, group_id: str, run_index: int) -> str:
    group = (group_id or "nogroup")[:8]
    name = f"{task_id}__{model_id.replace('/', '_')}__{group}__{run_index}.json"
    return os.path.join(folder, name)


def write_json(path: str, data) -> None:
    """Write `data` as JSON so that `path` is never left half-written.

    write to f"{path}.tmp" first, then os.replace(tmp, path).
    os.replace is atomic: readers see the old file or the new one, never a mix.
    If json.dump raises, delete the .tmp file and re-raise.
    """
    temp = f"{path}.tmp" # temperary file in the same destination \
    try:
        with open(temp,"w", encoding="utf-8") as f:
            json.dump(data,f,indent=2)
        os.replace(temp,path)

    except BaseException:
        try:
            os.remove(temp)
        except FileNotFoundError:
            pass
        raise

def is_done(path: str) -> bool:
    """True if a saved episode at `path` should be kept (skipped on resume).

        file missing                                  -> False (never ran)
        not valid JSON                                -> False (torn write: run again)
        ended_by == "error" and an error where=="model"  -> False (API/infra failure:
                                                         not the model's fault, run again)
        anything else, including reward 0 or grader.ok False -> True
            (a model failure is a real result; a crashed scorer is fixed by re-grading
             the saved trajectory, not by re-running the model)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return False
    except(json.JSONDecodeError, UnicodeDecodeError):
        return False
    
    if not isinstance(data, dict):
        return False

    if data.get("ended_by") == "error":
        errors = data.get("errors", [])
        if any(
            error.get("where") == "model"
            for error in errors
            if isinstance(error, dict)
        ):
            return False
    return True
