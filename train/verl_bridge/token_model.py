"""TokenModel: a Brownout ModelClient that talks in token ids, for RL training.

Why this exists: an API model gets text messages back every turn and re-tokenizes them.
For training that is wrong. The loss must be computed on the exact tokens the model
sampled, and re-encoding decoded text does not always give the same ids back (M4/TITO
in bin/). So TokenModel keeps ONE growing token stream per episode:

    prompt_ids                      system + task (+ tool schemas), rendered once
    response_ids  = [model turn 1][tool results 1][model turn 2][tool results 2]...
    response_mask =  1 1 1 1 1 1   0 0 0 0 0 0 0   1 1 1 1 1     0 0 0 ...
                     trained on    not trained on (the env wrote these)

run_episode (src/harness.py) drives the episode exactly as in a benchmark run: same env,
same limits, same verifier. It just calls model.complete(messages, tools) as usual, and
TokenModel turns each call into "append the new tool results as tokens, generate, append
the generated tokens".

Everything that needs verl or a tokenizer is passed in as a plain function, so this file
imports neither and is tested on a laptop (tests/token_model_test.py):

    render_prompt(messages, tools) -> ids   the first prompt, ending with the assistant header
    render_delta(messages)         -> ids   messages AFTER an assistant turn (tool results,
                                            nudges), ending with the next assistant header
    generate(ids, sampling)        -> (ids, logprobs or None)
    parse(ids)                     -> (text, [(name, arguments_json_str), ...])

train/verl_bridge/agent_loop.py builds the real ones from verl's tokenizer, server and
tool parser.
"""

import json
import time

from src.contracts import ModelClient, ModelReply, ToolCall, Usage


class ContextFull(Exception):
    """The episode's token stream reached max_response_tokens. Raised from complete(), so
    run_episode records it as an error and ends the episode. It is the model's own fault
    (too much thinking), not infra: agent_loop.to_output treats it as a normal failure."""


class TokenModel(ModelClient):
    def __init__(self, model_id, render_prompt, render_delta, generate, parse,
                 sampling: dict, max_response_tokens: int, max_tokens_per_turn: int = 4096):
        self.model_id = model_id
        self.render_prompt = render_prompt
        self.render_delta = render_delta
        self.generate = generate
        self.parse = parse
        self.sampling = dict(sampling)          # from verl; the harness's `temperature` is ignored
        self.max_response_tokens = max_response_tokens
        self.max_tokens_per_turn = max_tokens_per_turn

        self.prompt_ids: list[int] | None = None   # None until the first complete()
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.seen = 0       # how many of the harness's messages are already in the token stream
        self.turn = 0

    def complete(self, messages, tools, temperature=0.0): 
        if self.prompt_ids is None:
            self.prompt_ids = self.render_prompt(messages,tools)
        else:
            # only the messages the stream hasn't seen: tool results / nudges, never an assistant turn
            ids = self.render_delta(messages[self.seen:])
            self.response_ids.extend(ids)
            self.response_mask.extend([0] * len(ids))
            self.response_logprobs.extend([0.0] * len(ids))

        room = self.max_response_tokens - len(self.response_ids)
        if room <= 0:
            raise ContextFull("No room for Model reply")

        start = time.perf_counter()
        sampling = {**self.sampling, "max_tokens": min(self.max_tokens_per_turn, room)}
        context = self.prompt_ids + self.response_ids
        ids, logprobs = self.generate(context, sampling)
        latency_ms = (time.perf_counter() - start) * 1000

        # the model's own tokens, exactly as sampled: these are what gets trained
        self.response_ids.extend(ids)
        self.response_mask.extend([1] * len(ids))
        self.response_logprobs.extend(logprobs if logprobs is not None else [0.0] * len(ids))

        self.seen = len(messages) + 1

        text, calls = self.parse(ids)
        tool_calls = []
        for i , (name , arguments) in enumerate(calls):
            try:
                args = json.loads(arguments)
                if not isinstance(args,dict):
                    args = {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(
                    id = f"call_{self.turn}_{i}",
                    name = name ,
                    args = args,
                    raw_args = arguments,
                )
            )
            
        if len(ids) >= sampling["max_tokens"]:
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        reply = ModelReply(
            text = text,
            tool_calls = tool_calls,
            usage = Usage(
                input_tokens = len(context),
                output_tokens = len(ids),

            ),
            latency_ms = latency_ms,
            finish_reason = finish_reason,
        )

        self.turn += 1
        return reply

    def output(self) -> dict:
        """The token stream, for AgentLoopOutput."""
        return {"prompt_ids": self.prompt_ids or [],
                "response_ids": list(self.response_ids),
                "response_mask": list(self.response_mask),
                "response_logprobs": list(self.response_logprobs)}
