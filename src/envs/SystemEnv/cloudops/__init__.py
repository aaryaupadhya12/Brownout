"""Cloud-OpsBench: diagnose a Kubernetes incident from recorded cluster data.

754 cases (Online Boutique 550, Train-Ticket 204). Every tool replays a case's
tool_cache.json, so no live cluster is needed. The agent submits a ranked diagnosis
(fault_object + root_cause); rank 1 is scored by exact match.

The tool list differs by system (and Boutique code-defect cases use an extended
GetResources), and the harness asks for tool_schema() before reset(), so each tool set
is its own registered env:

    cloudops-boutique        Boutique, every category except codedefect   (452 cases)
    cloudops-boutique-code   Boutique codedefect                           (98 cases)
    cloudops-trainticket     Train-Ticket                                  (204 cases)

Data: clone https://github.com/LLM4Ops/Cloud-OpsBench and set CLOUDOPSBENCH_ROOT
(in .env or the environment). Upstream code we reuse is in vendor/ (MIT).
"""

from ....registry import EnvSpec, register
from .env import BoutiqueCodeEnv, BoutiqueEnv, TrainTicketEnv
from .tasks import BoutiqueCodeTasks, BoutiqueTasks, TrainTicketTasks
from .verifier import CloudOpsVerifier

register(EnvSpec("cloudops-boutique", BoutiqueTasks, BoutiqueEnv, CloudOpsVerifier))
register(EnvSpec("cloudops-boutique-code", BoutiqueCodeTasks, BoutiqueCodeEnv, CloudOpsVerifier))
register(EnvSpec("cloudops-trainticket", TrainTicketTasks, TrainTicketEnv, CloudOpsVerifier))
