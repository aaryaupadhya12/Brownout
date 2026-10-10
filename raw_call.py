from openai import OpenAI
import os
from src.env import IncidentEnv
NEBIUS_API_KEY = "fake for now"
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.env import IncidentEnv
from src.harness import run_episode, print_trajectory, save_trajectory
from Clients.openai_compact import OpenAICompatClient
from src.pricing import *

MODEL_ID = os.environ.get("NEBIUS_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
SCENARIO = {"id": "brownout_B", "arrivals": 900, "capacity": 500,
            "base_delay": 20, "degraded": {"B": 0.5}}

model = OpenAICompatClient(
    MODEL_ID,
    "https://api.tokenfactory.nebius.com/v1/",
    api_key= NEBIUS_API_KEY, min_gap_s=0,
)

def run_one(job):
    scenario, run_index = job
    traj = run_episode(IncidentEnv(), scenario, model, run_index=run_index)
    save_trajectory(traj)          # run_index makes filenames __0, __1, ...
    return traj

def run_many(jobs, workers=4, budget_usd=5.0):
    spent, results = 0.0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                traj = fut.result()
            except Exception as e:
                print("job failed:", e)
                continue
            spent += traj.totals.get("api_cost_usd") or 0
            results.append(traj)
            if spent >= budget_usd:
                for f in futs:
                    f.cancel()     # drops jobs that haven't started
                break
    return results, spent

if __name__ == "__main__":
    results, spent = run_many([(SCENARIO, i) for i in range(5)])
    for traj in results:
        print_trajectory(traj)
    print(f"spent ${spent:.4f}")

