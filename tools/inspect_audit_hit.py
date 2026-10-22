import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from src.audit.target import CloudOpsTarget


def string_leaves(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from string_leaves(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from string_leaves(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def main():
    parser = argparse.ArgumentParser(
        description="Re-grade a saved CloudOps run and locate answer-string matches."
    )
    parser.add_argument("run_json", type=Path, help="Path to the saved trajectory JSON")
    parser.add_argument(
        "--needle",
        help="Specific answer string to locate; defaults to all answer strings for the task",
    )
    args = parser.parse_args()

    with args.run_json.open(encoding="utf-8") as file:
        data = json.load(file)

    target = CloudOpsTarget()
    task = next(task for task in target.tasks() if task.id == data["task_id"])
    trace = SimpleNamespace(
        final_state=data.get("final_state", {}),
        steps=[SimpleNamespace(**step) for step in data.get("steps", [])],
    )
    grader, errors = target.grade(task, trace)
    print(f"Task: {task.id}")
    print(f"Verifier reward: {grader.reward}; grader_ok: {grader.ok}")
    if errors:
        print(f"Verifier errors: {errors}")

    needles = [args.needle] if args.needle else target.answer_strings(task)
    for needle in needles:
        print(f"\nMatches for {needle!r}:")
        hits = 0
        for section in ("steps", "messages"):
            for location, text in string_leaves(data.get(section, []), section):
                start = 0
                while True:
                    index = text.casefold().find(needle.casefold(), start)
                    if index < 0:
                        break
                    before = max(0, index - 100)
                    after = min(len(text), index + len(needle) + 100)
                    print(f"  {location}: ...{text[before:after]}...")
                    hits += 1
                    start = index + len(needle)
        if not hits:
            print("  no matches in steps or messages")


if __name__ == "__main__":
    main()
