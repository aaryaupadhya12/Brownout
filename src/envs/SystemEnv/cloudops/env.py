"""The Cloud-OpsBench env: the vendored tools, behind Brownout's Env interface.

One deliberate difference from upstream: their agent writes "Thought / Action / Action
Input" as text and their harness parses it; Brownout uses native tool calls. So the
system prompt keeps their instructions and their Submit spec, and drops only the
text-protocol section. Mention this when comparing numbers with the paper.
"""

import glob
import os

from ....interface_v1 import Task
from .tasks import data_root
from .vendor.cloudops import create_k8s_tools
from .vendor.contracts import build_expected_output

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def build_system_prompt(system: str) -> str:
    """Upstream system_prompt.md without its text-protocol rules, plus their Submit spec."""
    with open(os.path.join(VENDOR, "system_prompt.md"), encoding="utf-8") as f:
        prompt = f.read()
    prompt = prompt.split("**Critical syntax rules:**")[0].strip()
    return (prompt
            + "\n\nCall exactly one tool per turn. When the diagnosis is supported, call `Submit`."
            + "\n\n## Final Diagnosis Output Requirement\n"
            + build_expected_output(system).strip())


PREDICTION = {"type": "object", "properties": {
    "rank": {"type": "integer"},
    "fault_object": {"type": "string", "description": "Kind/Name, e.g. app/adservice or node/worker-01"},
    "root_cause": {"type": "string", "description": "one of the valid root causes"}},
    "required": ["rank", "fault_object", "root_cause"]}

SUBMIT_SCHEMA = {"type": "function", "function": {
    "name": "Submit",
    "description": "Submit the final diagnosis and end the episode.",
    "parameters": {"type": "object", "properties": {
        "key_evidence_summary": {"type": "string"},
        "top_3_predictions": {"type": "array", "items": PREDICTION, "minItems": 3, "maxItems": 3}},
        "required": ["key_evidence_summary", "top_3_predictions"]}}}


def validate_submission(args: dict) -> str | None:
    if not isinstance(args,dict):
        return "Submmision must be a dict"
    
    evidence = args.get("key_evidence_summary")
    if not isinstance(evidence,str) or not evidence.strip():
        return "Its an empty string"

    predictions = args.get("top_3_predictions")
    if not isinstance(predictions,list) or len(predictions) != 3:
        return "3 predictions are not present in top_3_predictions"
    
    for i, predict in enumerate(predictions,start =1):
        if not isinstance(predict,dict):
            return f"{i} is not a dictionary"
        
        if predict.get("rank") != i:
            return f"Prediction {i} must have rank {i}"
        
        for field in ("fault_object","root_cause"):
            value = predict.get(field)
            if not isinstance(value,str) or not value.strip():
                return f"Prediction {i}: {field} must be a non-empty string"
    return None


class CloudOpsEnv:
    system = "boutique"            # passed to the vendored tools
    tool_category = None           # "codedefect" switches GetResources to its extended version
    case_glob_dir = "boutique"     # where tool_schema() finds a reference case
    system_prompt = build_system_prompt("boutique")

    def tool_schema(self):
        benchmark_dir = os.path.join(data_root(), "benchmark")

        case_pattern = os.path.join(benchmark_dir, self.case_glob_dir, "**", "metadata.json")

        case_paths = sorted(os.path.dirname(path) for path in glob.glob(case_pattern, recursive=True))

        if not case_paths:
            raise FileNotFoundError("Case pattern not found")
        
        tools = create_k8s_tools(
            case_paths[0],
            self.system,
            self.tool_category
        )

        schemas = []
        for t in tools:
            if t.args_schema is None:
                parameters = {
                    "type": "object",
                    "properties": {},
                }
            else:
                parameters = t.args_schema.model_json_schema()

            schemas.append({
                "type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": parameters}}
            )

        schemas.append(SUBMIT_SCHEMA)
        return schemas

        

    def reset(self, task: Task) -> str:
        # reset the episode state 
        self.done = False
        self.submission = None
        self.tool_calls = 0
        self.invalid_submits = 0
        case_path = os.path.join(data_root(), task.spec["case_path"])
        self.tools = {t.name: t 
                    for t in create_k8s_tools(case_path, self.system, task.spec["category"])}
        return(
            f"Diagnose the reported incident in namespace {task.spec['namespace']}.\n"
            f"Incident report: {task.spec['query']}"
        )


    def step(self, tool, args):
        if self.done:
            return {"error":"Episode is already submmited"}
        
        if tool == "Submit":
            err = validate_submission(args)
        
            if err:
                self.invalid_submits += 1
                return {"error": err}

            self.submission = args
            self.done = True
            return {"status": "submitted"}
        
        if tool not in self.tools:
            return {"error": f"unkown tools {tool!r}"}
        
        self.tool_calls += 1

        try:
            text = self.tools[tool]._run(**args)
        except TypeError as e:
            return {"error": str(e)}
        
        if isinstance(text,str) and text.startswith("Error:"): # Main reason for this is that there could be both Kubectl error + normal errors we need to diffrentauite them 
            return {"error": text}
        return {"output": text}

    def snapshot(self):
        return{
            "submitted": self.done,
            "submission": self.submission, # This is a dictionary 
            "tool_calls": self.tool_calls,
            "invalid_submits": self.invalid_submits,
        }

        

    def close(self):
        pass


class BoutiqueEnv(CloudOpsEnv):
    pass


class BoutiqueCodeEnv(CloudOpsEnv):
    tool_category = "codedefect"
    case_glob_dir = os.path.join("boutique", "codedefect")


class TrainTicketEnv(CloudOpsEnv):
    system = "train-ticket"
    case_glob_dir = "trainticket"
    system_prompt = build_system_prompt("train-ticket")
