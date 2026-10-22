import glob, json, collections
from src.pricing import cost_usd

rows = collections.defaultdict(lambda: dict(n=0, solved=0, reward=0.0, cost=0.0))
for f in glob.glob("runs/*.json"):
    t = json.load(open(f, encoding="utf-8"))
    tt = t["totals"]
    r = rows[t["model_id"]]
    r["n"] += 1
    r["reward"] += t["grader"]["reward"]
    r["solved"] += t["grader"]["reward"] >= 1.0      # full reward; any env (was brownout's slo_ok only)
    c = tt.get("api_cost_usd")
    if c is None:
        c = cost_usd(t["model_id"], tt["input_tokens"], tt["output_tokens"], tt.get("reasoning_tokens", 0))
    r["cost"] += c or 0

for m, r in rows.items():
    per = r["cost"] / r["solved"] if r["solved"] else float("nan")
    print(m, "episodes", r["n"], "mean reward", round(r["reward"] / r["n"], 3),
          "solved", r["solved"], "total $", round(r["cost"], 4), "$/solved", round(per, 5))