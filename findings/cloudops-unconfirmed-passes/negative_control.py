"""Negative control for the reverse audit: runs where pod-as-config would be WRONG (affinity /
anti-affinity: describe pod does not print them), shown to the judge with the same rescued prompt
falsely claiming the rule accepted them. A sound judge says not_confirmed."""
import json, os, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(errors="replace")
from src.envs.SystemEnv.cloudops import audit_judge as aj
from src.envs.SystemEnv.cloudops.regrade_strict import all_tasks
from src.envs.SystemEnv.cloudops.strict import StrictCloudOpsVerifier, tool_steps
from src.audit.swarm.saved import load_trajectory
from src.providers import make_client
rows = [json.loads(l) for l in open("runs/regrade/strict-6f0952b5.jsonl", encoding="utf-8")]
tasks, v = all_tasks(), StrictCloudOpsVerifier()
pick = []
for r in rows:
    if r.get("ok") and r["jra"] == 1.0 and r["strict"] == 0.0:
        rc = tasks[r["task_id"]].spec["answer"]["root_cause"]
        tr = load_trajectory("runs/" + r["file"])
        if rc in ("node_affinity_mismatch", "pod_anti_affinity_conflict") and any(
                t == "DescribeResource" and "pod" in str(a.get("resource_type")) for t, a, _ in tool_steps(tr)):
            pick.append((r, tr))
pick = pick[:12]
client = make_client("nebius", "zai-org/GLM-5.3-Flash")
cache = aj.DiskCache("runs/regrade/judge-cache-zai-org_GLM-5.3-Flash.jsonl")
out = []
for r, tr in pick:
    ev = {**aj.evidence(v, tasks[r["task_id"]], tr), "accepted_by": aj.RULE_TEXT["pod_config"]}
    res = aj.judge_one(client, cache, aj.build_prompt(ev, "rescued"), aj.LABELS_RESCUED, "1r")
    if res["label"] is None:
        res = aj.judge_one(client, cache, aj.build_prompt(ev, "rescued"), aj.LABELS_RESCUED, "1r")
    out.append({"file": r["file"], "task_id": r["task_id"], "label": res["label"], "reason": res["reason"]})
    print(f"{r['task_id']:28s} {str(res['label']):14s} {res['reason'][:150]}")
with open("runs/regrade/strict-6f0952b5-rescued-negative-control.jsonl", "w", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o) + "\n")
from collections import Counter; print(len(out), "controls:", dict(Counter(o["label"] for o in out)))
