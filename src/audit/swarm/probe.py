from src.registry import get
from src.providers import make_client
from src.audit.swarm.loop import SwarmConfig, attack_task, precheck
from src.audit.swarm.planted import HOLES
from src.audit.swarm.roles import FOCUS, Specialist

cfg = SwarmConfig(campaign="probe-direct", max_rounds=1, attempts=1)
spec = get(cfg.env)
true_success = spec.verifier()
verifier = next(h for h in HOLES if h.hole_id == "substring").verifier()
task = next(t for t in spec.tasks().load() if t.id == "boutique-admission-1")

solver = make_client("nebius", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
attacker = make_client("nebius", "zai-org/GLM-5.3-Flash")

path = precheck(cfg, task, solver, true_success)
print("precheck:", path)

# no try/except: the traceback prints in full
print(attack_task(cfg, task, attacker, verifier, true_success,
                  "runs/swarm/probe-direct/ledger.jsonl", path,
                  specialist=Specialist("answer-format", FOCUS["answer-format"])))