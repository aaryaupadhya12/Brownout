import json
import os
import time
from dataclasses import asdict
from datetime import date
from .pricing import *
from .contracts import VERSIONS, ContextManager, Event, GraderOutput, Step, Trajectory, Usage

SYSTEM_PROMPT = (
    "You are an on-call engineer responding to a latency incident across three zones (A, B, C).\n"
    "Tools: get_metrics(zone), shift_traffic(zone_source, zone_destination, split_percentage), submit().\n"
    "Investigate before acting, and move only as much traffic as needed. "
    "Call submit() when the incident is resolved or when no action is needed."
)
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


def run_episode(env, scenario, model, max_turns=8, ctx=None, run_index=0):
    ctx = ctx or ContextManager()
    tools = env.tool_schema()
    first_observation = env.reset(scenario)
    pool = [Event(0, "system", {"role": "system", "content": SYSTEM_PROMPT}),
            Event(0, "task", {"role": "user", "content": first_observation})]
    steps = []
    ended_by = "max_turns"
    error = None

    try:
        for turn in range(1, max_turns + 1):
            messages = ctx.build(pool)
            reply = model.complete(messages, tools)
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
                    result = env.step(c.name, args)
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

    reward, signals = env.grade()

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
    
    traj =  Trajectory(
        task_id=scenario.get("id", "adhoc"), model_id=model.model_id,
        run_date=date.today().isoformat(), run_index=run_index,
        versions=VERSIONS, steps=steps,
        grader=GraderOutput(reward, signals, VERSIONS["grader"]),
        ended_by=ended_by, totals=totals,
        context_policy={"name": ctx.name, "version": VERSIONS["harness"]})
    add_cost(traj)
    return traj

def save_trajectory(traj, folder="runs"):
    os.makedirs(folder, exist_ok=True)
    name = f"{traj.task_id}__{traj.model_id.replace('/', '_')}__{traj.run_index}.json"
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
    print(f"ended_by: {traj.ended_by}  reward: {traj.grader.reward:.4f}")
    cost = traj.totals.get("api_cost_used")
    print("cost:", "n/a" if cost is None else f"${cost:.4f}")


