"""Cloud-OpsBench cases -> Tasks.

Each case is a folder  <root>/benchmark/<system_dir>/<category>/<case_id>/  with
metadata.json:

    {"namespace": "boutique", "query": "Partial Service Unreachability.", "difficulty": "medium",
     "result": {"fault_taxonomy": "Admission_Fault", "fault_object": "app/adservice",
                "root_cause": "missing_service_account"}}

The ground truth ("result") goes in task.spec["answer"]: the verifier reads it, the agent
never sees it (env.reset() shows only namespace + query).
"""

import json
import os
from ....interface_v1 import Task
from ....providers import load_env_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
MAX_STEPS = 20          # upstream default: configs/model_configs.yaml, diagnosis.max_iterations

BOUTIQUE_CATEGORIES = ("admission", "infrastructure", "performance", "runtime",
                       "scheduling", "service", "startup")
TRAINTICKET_CATEGORIES = ("performance", "runtime", "service", "startup")


def data_root() -> str:
    """The Cloud-OpsBench checkout: CLOUDOPSBENCH_ROOT, else <repo>/data/Cloud-OpsBench."""
    load_env_file()
    root = os.environ.get("CLOUDOPSBENCH_ROOT") or os.path.join(REPO_ROOT, "data", "Cloud-OpsBench")
    if not os.path.isdir(os.path.join(root, "benchmark")):
        raise FileNotFoundError(
            f"Cloud-OpsBench not found at {root}: clone "
            "https://github.com/LLM4Ops/Cloud-OpsBench and set CLOUDOPSBENCH_ROOT")
    return root


class CloudOpsTasks:
    system = "boutique"            # the name the vendored tools use ("boutique" | "train-ticket")
    system_dir = "boutique"        # the folder under benchmark/ ("boutique" | "trainticket")
    categories = BOUTIQUE_CATEGORIES

    def load(self):
        
        root = data_root()  # Get data root for both the categories
        limit = os.environ.get("CLOUDOPSBENCH_PER_CATEGORY")
        for category in self.categories:
            category_dir = os.path.join(
            root, "benchmark", self.system_dir, category
            )

            cases = [] # Find case folder that contain Metadata.json 
            for case_id in os.listdir(category_dir):
                case_path = os.path.join(category_dir,case_id)

                if (
                os.path.isdir(case_path)
                and os.path.isfile(os.path.join(case_path, "metadata.json"))
                ):
                    cases.append((int(case_id), case_id, case_path))
            cases.sort()
            if limit is not None:
                cases = cases[:int(limit)]

            for _, case_id, case_path in cases:
                metadata_file = os.path.join(case_path, "metadata.json")

                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                relative_path = os.path.relpath(case_path, root)
                relative_path = relative_path.replace(os.sep, "/")

                yield Task(
                    id=f"{self.system_dir}-{category}-{case_id}",
                    seed=0,
                    spec={
                        "system": self.system,
                        "system_dir": self.system_dir,
                        "category": category,
                        "case_id": case_id,
                        "case_path": relative_path,
                        "namespace": metadata["namespace"],
                        "query": metadata["query"],
                        "difficulty": metadata["difficulty"],
                        "answer": metadata["result"],
                        "max_steps": MAX_STEPS,
                    },
                )


class BoutiqueTasks(CloudOpsTasks):
    pass


class BoutiqueCodeTasks(CloudOpsTasks):
    categories = ("codedefect",)


class TrainTicketTasks(CloudOpsTasks):
    system = "train-ticket"
    system_dir = "trainticket"
    categories = TRAINTICKET_CATEGORIES
