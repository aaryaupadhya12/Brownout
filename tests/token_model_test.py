"""train/verl_bridge/token_model.py: the token stream for RL training. Done when all pass.

No verl, no tokenizer, no GPU: a fake "tokenizer" maps each character to its code point,
and the fake model replies from a script. The last test runs a whole Brownout episode
through run_episode, the same function the benchmark uses.
"""

import json
import unittest

from src.harness import run_episode
from src.registry import get
from train.verl_bridge.token_model import ContextFull, TokenModel


def enc(text):
    return [ord(c) for c in text]


def dec(ids):
    return "".join(map(chr, ids))


def render_prompt(messages, tools):
    return enc("".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>")


def render_delta(messages):
    for m in messages:
        assert m["role"] != "assistant", "an assistant message was re-rendered"
    return enc("".join(f"<{m['role']}>{m.get('content', '')}" for m in messages) + "<assistant>")


def parse(ids):
    """'CALL name {json}' -> one tool call; anything else -> plain text."""
    text = dec(ids)
    if text.startswith("CALL "):
        _, name, arguments = text.split(" ", 2)
        return "", [(name, arguments)]
    return text, []


class Script:
    """generate(): replies in order, logprob -0.5 per token; records what it was sent."""
    def __init__(self, *replies, logprobs=True):
        self.replies = list(replies)
        self.logprobs = logprobs
        self.sent = []

    def __call__(self, ids, sampling):
        self.sent.append((list(ids), dict(sampling)))
        out = enc(self.replies.pop(0))[: sampling["max_tokens"]]
        return out, ([-0.5] * len(out) if self.logprobs else None)


def make(script, max_response_tokens=10_000, per_turn=4096):
    return TokenModel("fake-token-model", render_prompt, render_delta, script, parse,
                      sampling={"temperature": 1.0}, max_response_tokens=max_response_tokens,
                      max_tokens_per_turn=per_turn)


SYSTEM = {"role": "system", "content": "sys"}
USER = {"role": "user", "content": "task"}


class TestTokenModel(unittest.TestCase):
    def test_first_call_renders_prompt_and_appends_generation(self):
        script = Script('CALL get_metrics {"zone": "A"}')
        model = make(script)
        reply = model.complete([SYSTEM, USER], tools=[])
        self.assertEqual(model.prompt_ids, render_prompt([SYSTEM, USER], []))
        self.assertEqual(model.response_ids, enc('CALL get_metrics {"zone": "A"}'))
        self.assertEqual(set(model.response_mask), {1})
        self.assertEqual(script.sent[0][0], model.prompt_ids)        # sent exactly the prompt
        self.assertEqual(reply.tool_calls[0].name, "get_metrics")
        self.assertEqual(reply.tool_calls[0].args, {"zone": "A"})
        self.assertEqual(reply.finish_reason, "tool_calls")

    def test_tool_results_are_appended_with_mask_0(self):
        model = make(Script("CALL get_metrics {}", "done"))
        model.complete([SYSTEM, USER], tools=[])
        n_first = len(model.response_ids)
        assistant = {"role": "assistant", "content": "", "tool_calls": []}
        tool = {"role": "tool", "tool_call_id": "call_0_0", "content": '{"output": "ok"}'}
        model.complete([SYSTEM, USER, assistant, tool], tools=[])
        delta = render_delta([tool])
        self.assertEqual(model.response_ids[n_first:n_first + len(delta)], delta)
        self.assertEqual(model.response_mask, [1] * n_first + [0] * len(delta) + [1] * len("done"))

    def test_model_tokens_are_never_re_encoded(self):
        """The second request starts with the first reply's ids exactly as sampled."""
        script = Script("CALL get_metrics {}", "done")
        model = make(script)
        model.complete([SYSTEM, USER], tools=[])
        first = list(model.response_ids)
        assistant = {"role": "assistant", "content": "SOMETHING ELSE ENTIRELY"}
        tool = {"role": "tool", "tool_call_id": "x", "content": "{}"}
        model.complete([SYSTEM, USER, assistant, tool], tools=[])   # render_delta asserts no assistant
        sent = script.sent[1][0]
        self.assertEqual(sent[len(model.prompt_ids):len(model.prompt_ids) + len(first)], first)

    def test_logprobs_line_up_with_tokens(self):
        model = make(Script("CALL get_metrics {}", "done"))
        model.complete([SYSTEM, USER], tools=[])
        model.complete([SYSTEM, USER, {"role": "assistant", "content": ""},
                        {"role": "tool", "tool_call_id": "x", "content": "{}"}], tools=[])
        self.assertEqual(len(model.response_logprobs), len(model.response_ids))
        for lp, m in zip(model.response_logprobs, model.response_mask):
            self.assertEqual(lp, -0.5 if m == 1 else 0.0)

    def test_missing_logprobs_become_zeros(self):
        model = make(Script("hello", logprobs=False))
        model.complete([SYSTEM, USER], tools=[])
        self.assertEqual(model.response_logprobs, [0.0] * len("hello"))

    def test_broken_json_keeps_raw_args(self):
        reply = make(Script("CALL get_metrics {not json")).complete([SYSTEM, USER], tools=[])
        call = reply.tool_calls[0]
        self.assertEqual(call.args, {})
        self.assertEqual(call.raw_args, "{not json")

    def test_plain_text_reply(self):
        reply = make(Script("I think it is the node")).complete([SYSTEM, USER], tools=[])
        self.assertEqual(reply.tool_calls, [])
        self.assertEqual(reply.text, "I think it is the node")
        self.assertEqual(reply.finish_reason, "stop")

    def test_turn_is_capped_by_room_left(self):
        script = Script("x" * 50)
        model = make(script, max_response_tokens=20, per_turn=4096)
        reply = model.complete([SYSTEM, USER], tools=[])
        self.assertEqual(script.sent[0][1]["max_tokens"], 20)
        self.assertEqual(len(model.response_ids), 20)
        self.assertEqual(reply.finish_reason, "length")

    def test_context_full_raises(self):
        model = make(Script("x" * 20, "more"), max_response_tokens=20)
        model.complete([SYSTEM, USER], tools=[])
        with self.assertRaises(ContextFull):
            model.complete([SYSTEM, USER, {"role": "assistant", "content": ""},
                            {"role": "tool", "tool_call_id": "x", "content": "{}"}], tools=[])

    def test_usage_counts_context_and_output(self):
        model = make(Script("hello"))
        reply = model.complete([SYSTEM, USER], tools=[])
        self.assertEqual(reply.usage.input_tokens, len(model.prompt_ids))
        self.assertEqual(reply.usage.output_tokens, len("hello"))


class TestWithRunEpisode(unittest.TestCase):
    def test_a_whole_brownout_episode(self):
        """The harness drives TokenModel like any model; the stream matches its messages."""
        spec = get("brownout")
        task = next(t for t in spec.tasks().load() if t.id == "brownout_B")
        model = make(Script('CALL get_metrics {"zone": "A"}', "CALL submit {}"))
        traj = run_episode(spec.env(), task, model, max_turns=5, verifier=spec.verifier())
        self.assertEqual(traj.ended_by, "submit")
        self.assertEqual([s.tool for s in traj.steps], ["get_metrics", "submit"])
        out = model.output()
        self.assertEqual(len(out["response_ids"]), len(out["response_mask"]))
        # model tokens = exactly the two scripted replies
        model_text = dec([i for i, m in zip(out["response_ids"], out["response_mask"]) if m])
        self.assertEqual(model_text, 'CALL get_metrics {"zone": "A"}CALL submit {}')
        # the env's tool result is in the stream, masked out
        env_text = dec([i for i, m in zip(out["response_ids"], out["response_mask"]) if not m])
        self.assertIn(json.dumps(traj.steps[0].observation), env_text)


if __name__ == "__main__":
    unittest.main()
