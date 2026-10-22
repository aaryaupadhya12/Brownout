import glob, json
from src.registry import get
from src.audit.target import CloudOpsTarget

target = CloudOpsTarget()
tasks = {t.id: t for t in target.tasks()}

for path in sorted(glob.glob("runs/*.json"))[:20]:
    t = json.load(open(path, encoding="utf-8"))
    task = tasks.get(t.get("task_id"))
    if task is None:
        continue
    needles = [s.casefold() for s in target.answer_strings(task)]

    msg_text = " ".join(
        str(m.get("payload", m).get("content", "")).casefold()
        for m in t.get("messages", []) if m.get("payload", m).get("role") == "tool")
    step_text = json.dumps([s.get("observation") for s in t.get("steps", [])],
                           ensure_ascii=False, default=str).casefold()

    in_msgs = any(n in msg_text for n in needles)
    in_steps = any(n in step_text for n in needles)
    print(f"{t['task_id']:30} reward={t['grader']['reward']}  "
          f"in_messages={in_msgs}  in_steps={in_steps}")