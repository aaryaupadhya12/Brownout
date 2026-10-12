import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import date
from .pricing import *
from .contracts import (VERSIONS, ContextManager, ErrorRecord, Event, GraderOutput, ModelCall,
                        Step, Trajectory, Usage)
from .interface_v1 import Task
from .scoring import EnvGradeVerifier
from .tracking import code_version, config_hash, group_id, task_hash

# each env brings its own system prompt (env.system_prompt); this is only for envs without one
DEFAULT_SYSTEM_PROMPT = "Use the tools to complete the task. Call submit() when you are done."
NUDGE = "Use a tool, or call submit() if you are done."


def parse_args(call):
    """Returns (args, None) on success, or (None, error_text) on failure."""
    if call.raw_args:
        try:
            args = json.loads(call.raw_args)
        except json.JSONDecodeError as e:
            return None, f"malformed arguments: {e}"
    else:
        args = call.args
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    return args, None


def assistant_message(reply):
    """Build the assistant message that goes back into the conversation."""
    msg = {"role": "assistant", "content": reply.text}
    if reply.tool_calls:
        msg["tool_calls"] = []
        for c in reply.tool_calls:
            arguments = c.raw_args or json.dumps(c.args)
            tc = {"id": c.id, "type": "function",
                  "function": {"name": c.name, "arguments": arguments}}
            if c.extra:                       # Gemini 3: send the thought_signature back as received
                tc["extra_content"] = c.extra
            msg["tool_calls"].append(tc)
    return msg


def rollout_config(model, tools, max_turns, temperature, ctx, system_prompt):
    """Everything that changes how a rollout behaves. Two runs with the same config and
    the same task are 'similar runs' (same group_id). The scorer is left out on purpose:
    re-grading with a new verifier doesn't change what the model did."""
    return {"model_id": model.model_id,
            "sampling": {"temperature": temperature},
            "max_turns": max_turns,
            "system_prompt": system_prompt,
            "nudge": NUDGE,
            "tools": tools,
            "context_policy": {"name": ctx.name, "version": VERSIONS["harness"]},
            "versions": VERSIONS}


def run_episode(env, scenario, model, max_turns=8, ctx=None, run_index=0,
                verifier=None, temperature=0.0):
    ctx = ctx or ContextManager()
    verifier = verifier or EnvGradeVerifier()
    # a v1 Task (src/interface_v1.py) or an old scenario dict
    if isinstance(scenario, Task):
        task_id = scenario.id
        task_data = scenario.as_dict()
    else:
        task_id = scenario.get("id", "adhoc")
        task_data = scenario
    started_at = time.time()
    tools = env.tool_schema()
    system_prompt = getattr(env, "system_prompt", DEFAULT_SYSTEM_PROMPT)
    first_observation = env.reset(scenario)
    pool = [Event(0, "system", {"role": "system", "content": system_prompt}),
            Event(0, "task", {"role": "user", "content": first_observation})]
    steps = []
    calls = []
    errors = []
    ended_by = "max_turns"
    error = None
    where = "harness"   # which part was running if something raises: model | env | harness

    config = rollout_config(model, tools, max_turns, temperature, ctx, system_prompt)
    task_sha = task_hash(task_data)
    config_sha = config_hash(config)
    tracking = {"task_key": task_id,
                "task_hash": task_sha,
                "config_hash": config_sha,
                "group_id": group_id(task_sha, config_sha),
                "code": code_version(),
                "verifier": {"name": verifier.name, "version": verifier.version}}

    try:
        for turn in range(1, max_turns + 1):
            messages = ctx.build(pool)
            where = "model"
            reply = model.complete(messages, tools, temperature)
            where = "harness"
            calls.append(ModelCall(turn, reply.latency_ms, reply.usage, reply.finish_reason,
                                   len(reply.tool_calls), reply.retries))
            pool.append(Event(turn, "assistant", assistant_message(reply)))

            # the model answered in plain text: count the turn and nudge it
            if not reply.tool_calls:
                steps.append(Step(turn, None, {}, {"note": "plain text reply"},
                                  reply.latency_ms, 0.0, reply.usage))
                pool.append(Event(turn, "nudge", {"role": "user", "content": NUDGE}))
                continue

            is_first_call = True
            for c in reply.tool_calls:
                args, err = parse_args(c)

                start = time.perf_counter()
                if err:
                    result = {"error": err}
                else:
                    where = "env"
                    result = env.step(c.name, args)
                    where = "harness"
                tool_ms = (time.perf_counter() - start) * 1000

                # every tool call gets a result message, even a malformed one
                pool.append(Event(turn, "tool_result", {
                    "role": "tool", "tool_call_id": c.id, "content": json.dumps(result)}))

                # model time and tokens belong to the whole reply, so count them once
                if is_first_call:
                    model_ms = reply.latency_ms
                    usage = reply.usage
                else:
                    model_ms = 0.0
                    usage = Usage()
                is_first_call = False

                steps.append(Step(turn, c.name, args or {}, result, model_ms, tool_ms, usage))

            if env.done:
                ended_by = "submit"
                break
    except Exception as e:
        ended_by = "error"
        error = repr(e)
        errors.append(ErrorRecord(where, type(e).__name__, str(e)))

    # v1 envs report their final state, so the verifier never has to call into the env
    final_state = {}
    if hasattr(env, "snapshot"):
        try:
            final_state = env.snapshot()
        except Exception as e:
            errors.append(ErrorRecord("env", type(e).__name__, f"snapshot: {e}"))

    turns_seen = []
    total_input = 0
    total_output = 0
    total_reasoning = 0
    total_model_ms = 0.0
    total_tool_ms = 0.0
    for s in steps:
        if s.turn not in turns_seen:
            turns_seen.append(s.turn)
        total_input += s.usage.input_tokens
        total_output += s.usage.output_tokens
        total_reasoning += s.usage.reasoning_tokens
        total_model_ms += s.model_ms
        total_tool_ms += s.tool_ms

    totals = {"turns": len(turns_seen),
              "input_tokens": total_input,
              "output_tokens": total_output,
              "reasoning_tokens": total_reasoning,
              "model_ms": total_model_ms,
              "tool_ms": total_tool_ms}
    if error:
        totals["error"] = error
    
    # build the trace first, so the verifier can read all of it
    traj = Trajectory(
        task_id=task_id, model_id=model.model_id,
        run_date=date.today().isoformat(), run_index=run_index,
        versions=VERSIONS, steps=steps,
        grader=GraderOutput(0.0, {}, verifier.version, ok=False, grader_name=verifier.name),
        ended_by=ended_by, totals=totals,
        context_policy={"name": ctx.name, "version": VERSIONS["harness"]},
        trace_id=uuid.uuid4().hex, tracking=tracking, config=config,
        messages=[asdict(e) for e in pool], calls=calls, errors=errors,
        final_state=final_state)

    # scoring runs even after an error: a failed rollout is data, not a crash
    scoring_start = time.perf_counter()
    grader, scoring_errors = verifier.score(scenario, traj, env)
    traj.grader = grader
    traj.errors.extend(scoring_errors)
    traj.timing = {"started_at": started_at, "finished_at": time.time(),
                   "scoring_ms": (time.perf_counter() - scoring_start) * 1000}
    if hasattr(env, "close"):
        env.close()
    add_cost(traj)
    return traj

def save_trajectory(traj, folder="runs"):
    os.makedirs(folder, exist_ok=True)
    # the group id in the name stops a run with a different config overwriting this one
    group = traj.tracking.get("group_id", "nogroup")[:8]
    name = f"{traj.task_id}__{traj.model_id.replace('/', '_')}__{group}__{traj.run_index}.json"
    path = os.path.join(folder, name)
    with open(path, "w") as f:
        json.dump(asdict(traj), f, indent=2)
    return path


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        rounded = {}
        for key, value in x.items():
            rounded[key] = _round(value)
        return rounded
    return x


def print_trajectory(traj):
    for s in traj.steps:
        print(f"turn {s.turn}  {s.tool or 'text'} {s.args} -> {_round(s.observation)}")
    print(f"ended_by: {traj.ended_by}  reward: {traj.grader.reward:.4f}  "
          f"scored_ok: {traj.grader.ok}  group: {traj.tracking.get('group_id')}")
    for e in traj.errors:
        print(f"error [{e.where}] {e.type}: {e.message}")
    cost = traj.totals.get("api_cost_usd")
    print("cost:", "n/a" if cost is None else f"${cost:.4f}")

