from dataclasses import dataclass, field , asdict
from typing import Optional



VERSIONS = {"harness": "0.2.0", "env": "0.1.0", "grader": "0.2.0"}

@dataclass
class TaskSpec:
    id: str
    seed: int
    split: str  # dev | test
    parameters: dict 
    slo_ms : float 
    cost_budget : float 
    max_turns : int 
    difficulty : str
    instruction : str 


@dataclass
class ToolCall:
    id: str 
    name: str
    args: dict 
    raw_args: str =""
    extra: dict = field(default_factory=dict) 

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int =0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

@dataclass 
class ModelReply:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    latency_ms: float
    logprobs: Optional[list] = None
    finish_reason: Optional[str] = None   # "stop" | "tool_calls" | "length" (cut off) | None if unknown
    retries: int = 0                      # how many times the client retried before this reply


class ModelClient:
    model_id: str
    def complete(self, messages: list[dict], tools: list[dict],
                 temperature: float = 0.0) -> ModelReply: ...

@dataclass
class GraderOutput:
    reward: float
    signals: dict  # metrics: recorded, never added into the reward
    grader_version: str
    rewards: dict = field(default_factory=dict)   # name -> {"score": s, "weight": w}; reward = sum(s * w)
    ok: bool = True                               # False if a scorer raised: exclude from averages, don't count as 0
    grader_name: str = ""


# One record per model call, so a slow, cut-off or retried call is visible on its own
@dataclass
class ModelCall:
    turn: int
    latency_ms: float
    usage: Usage
    finish_reason: Optional[str]
    n_tool_calls: int
    retries: int = 0


# Where an error happened: "model" (provider/client), "env" (tool code),
# "scorer" (grading code) or "harness" (this loop)
@dataclass
class ErrorRecord:
    where: str
    type: str
    message: str

@dataclass
class Step:
    turn: int
    tool :Optional[str]
    args: dict
    observation: dict 
    model_ms : float
    tool_ms : float
    usage: Usage

@dataclass
class Trajectory:
    task_id: str
    model_id: str
    run_date: str
    run_index: int
    versions: dict
    steps: list[Step]
    grader: GraderOutput
    ended_by: str                     # submit | max_turns | error
    totals: dict = field(default_factory=dict)
    context_policy: dict = field(default_factory=dict)   # {"name": "identity", "version": "0.1.0"}
    parent_id: Optional[str] = None
    # --- the whole trace ---
    trace_id: str = ""                                   # unique per run
    tracking: dict = field(default_factory=dict)         # task/config hashes, group_id, code version, verifier
    config: dict = field(default_factory=dict)           # exactly what config_hash was computed from
    messages: list = field(default_factory=list)         # every Event in the pool, in order
    calls: list[ModelCall] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    timing: dict = field(default_factory=dict)           # started_at / finished_at (unix s), scoring_ms


#Observational_pool : an append only list of types events, its the source of trueh and nothing deleted from it 
@dataclass
class Event:
    turn: int 
    kind: str    #  "system" | "task" | "assistant" | "tool_result" | "nudge"
    payload: dict


class ContextManager:
    name = "identity"
    # Type hind its expected to be list containing Event Objects 
    def build(self,pool:list[Event]) -> list[dict]:
        result = []
        for e in pool:
            result.append(e.payload)
        return result

'''
pool = [
    Event(turn=0, kind="system", payload={"role": "system", "content": "You are an on-call engineer..."}),
    Event(turn=0, kind="task",   payload={"role": "user",   "content": "Zone B is browning out."}),
]
assert ContextManager().build(pool) == [e.payload for e in pool]
'''

