"""verl agent loop that runs one Brownout episode per sample.

verl calls BrownoutAgentLoop.run() once per (task, sample): with rollout.n=4 that is 4
calls per task, which become one GRPO group. Each call:

    1. finds the task by id (dataset row -> extra_info)
    2. builds a fresh env + verifier from Brownout's registry
    3. builds a TokenModel whose generate() is verl's inference server
    4. runs src.harness.run_episode in a worker thread (the harness is synchronous; verl
       is asyncio, so the model's generate() hops back onto verl's event loop)
    5. returns the token stream + reward as an AgentLoopOutput (to_output, yours)

Runs only on the GPU machine (imports verl). Written against verl main as of 2026-10:
verl/experimental/agent_loop/agent_loop.py and tool_parser.py. verl changes this API
often: pin one version and re-check the names marked VERIFY against it.

Env knobs (set them in the launch script):
    BROWNOUT_TOOL_PARSER       verl tool parser name: qwen3_coder (Nemotron-3 / Qwen3-Coder),
                               hermes (Qwen3, Qwen2.5)
    BROWNOUT_MAX_TURNS         20 (Cloud-OpsBench's upstream step budget)
    BROWNOUT_TOKENS_PER_TURN   4096 (cap on one reply, thinking included)
    BROWNOUT_MODEL_ID          the model_id on saved trajectories (e.g. nemotron-nano-lora)
    BROWNOUT_SAVE_DIR         if set, every training episode is also saved as a Brownout
                               trajectory, so `python -m src.report <dir>` works on them
"""

import asyncio
import os
import uuid
from dataclasses import asdict

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser

from src.harness import run_episode
from src.limits import RolloutLimits
from src.registry import get
from src.store import write_json

from .renderers import make_renderers
from .token_model import ContextFull, TokenModel

_TASKS: dict[str, dict] = {}     # env name -> {task_id: Task}, loaded once per worker process


def find_task(env_name: str, task_id: str):
    if env_name not in _TASKS:
        _TASKS[env_name] = {t.id: t for t in get(env_name).tasks().load()}
    return _TASKS[env_name][task_id]


@register("brownout")
class BrownoutAgentLoop(AgentLoopBase):

    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        info = kwargs["extra_info"]
        spec = get(info["env"])
        task = find_task(info["env"], info["task_id"])
        env, verifier = spec.env(), spec.verifier()

        loop = asyncio.get_running_loop()
        request_id = uuid.uuid4().hex              # same id every turn: sticky server, prefix cache hits
        parser = ToolParser.get_tool_parser(os.environ.get("BROWNOUT_TOOL_PARSER", "qwen3_coder"),
                                            self.tokenizer)

        def on_loop(coro):
            """Run a verl coroutine from the harness's worker thread and wait for it."""
            return asyncio.run_coroutine_threadsafe(coro, loop).result()

        def generate(ids, sampling):
            out = on_loop(self.server_manager.generate(            # VERIFY: kwargs + TokenOutput fields
                request_id=request_id, prompt_ids=ids, sampling_params=sampling))
            return out.token_ids, out.log_probs

        def parse(ids):
            text, calls = on_loop(parser.extract_tool_calls(ids))  # VERIFY: signature
            return text, [(c.name, c.arguments) for c in calls]

        render_prompt, render_delta = make_renderers(self.tokenizer)
        response_length = self.rollout_config.response_length    # VERIFY: config path
        model = TokenModel(
            model_id=f"verl:{os.environ.get('BROWNOUT_MODEL_ID', 'policy')}",
            render_prompt=render_prompt, render_delta=render_delta,
            generate=generate, parse=parse, sampling=sampling_params,
            max_response_tokens=response_length,
            max_tokens_per_turn=int(os.environ.get("BROWNOUT_TOKENS_PER_TURN", 4096)))

        max_turns = int(os.environ.get("BROWNOUT_MAX_TURNS", 20))
        traj = await asyncio.to_thread(
            run_episode, env, task, model, max_turns=max_turns, verifier=verifier,
            temperature=sampling_params.get("temperature", 1.0),
            limits=RolloutLimits(max_turns=max_turns))

        save_dir = os.environ.get("BROWNOUT_SAVE_DIR")
        if save_dir:
            # named by trace_id: the n samples of one task share task, model and group_id
            os.makedirs(save_dir, exist_ok=True)
            write_json(os.path.join(save_dir, f"{traj.task_id}__{traj.trace_id}.json"), asdict(traj))
        return self.to_output(model, traj, response_length)

    def to_output(self, model: TokenModel, traj, response_length: int) -> AgentLoopOutput:
        out = model.output() # we get prompt_ids, response_ids, response_masj, respose_logprobfs, reward_Score, num_turns, extra_fiels 
        context_full = any(e.type == ContextFull.__name__ for e in traj.errors)
        infra_failure = (not traj.grader.ok) or any(e.where == "model" and e.type != ContextFull.__name__ for e in traj.errors)

        mask = out["response_mask"]
        if infra_failure:
            reward = 0.0
            mask = [0] * len(mask)
        
        else:
            reward = float(traj.grader.reward)
        n = response_length
        return AgentLoopOutput(
            prompt_ids = out["prompt_ids"],
            response_ids = out["response_ids"][:n],
            response_mask = mask[:n],
            response_logprobs = out["response_logprobs"][:n],
            reward_score = reward,
            num_turns = traj.totals.get("turns", 0),
            metrics = AgentLoopMetrics(),
            extra_fields= {
                 "jra": reward,
                "ended_by": traj.ended_by,
                "submitted": bool(traj.final_state.get("submitted", False)),
                "infra_failure": infra_failure,
                "context_full": context_full,
            },
        )

        




