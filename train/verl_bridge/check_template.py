"""Check that render_delta cuts the chat template correctly for a model. Run once per model.

    python -m train.verl_bridge.check_template nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Downloads only the tokenizer (a few MB), no weights, no GPU. It renders a real
Cloud-OpsBench-shaped conversation two ways and compares them:

    A  the whole conversation rendered at once           (what an API server sees)
    B  prompt + "model reply" tokens + render_delta(...)  (what TokenModel builds)

They should decode to the same text. Differences are expected in exactly one place:
earlier assistant turns, if the template strips old <think> blocks from history. That is
fine (TokenModel keeps what the model really produced); anything else is a bug in the cut.
"""

import difflib
import sys

from transformers import AutoTokenizer

from .renderers import make_renderers

TOOLS = [{"type": "function", "function": {
    "name": "GetResources", "description": "List resources",
    "parameters": {"type": "object", "properties": {"kind": {"type": "string"}}}}}]


def main(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    render_prompt, render_delta = make_renderers(tok)

    system = {"role": "system", "content": "You are an SRE."}
    user = {"role": "user", "content": "Diagnose the incident in namespace boutique."}
    call = {"id": "call_0_0", "type": "function",
            "function": {"name": "GetResources", "arguments": '{"kind": "pods"}'}}
    assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
    tool = {"role": "tool", "tool_call_id": "call_0_0", "content": '{"output": "pod-a Running"}'}

    # what the model "generated": the template's own rendering of the assistant turn
    before = tok.apply_chat_template([system, user], tools=TOOLS, tokenize=False,
                                     add_generation_prompt=True)
    after = tok.apply_chat_template([system, user, assistant], tools=TOOLS, tokenize=False)
    reply_text = after[len(before):] if after.startswith(before) else None
    if reply_text is None:
        print("could not isolate the assistant turn; compare by eye:\n", after)
        return
    reply_text = reply_text[: reply_text.rfind(tok.eos_token) + len(tok.eos_token)]
    print("assistant turn as the model would emit it:\n", repr(reply_text), "\n")

    a = tok.apply_chat_template([system, user, assistant, tool], tools=TOOLS, tokenize=False,
                                add_generation_prompt=True)
    ids = (render_prompt([system, user], TOOLS)
           + tok.encode(reply_text, add_special_tokens=False)
           + render_delta([tool]))
    b = tok.decode(ids)

    print("delta for the tool turn:\n", repr(tok.decode(render_delta([tool]))), "\n")
    if a == b:
        print("OK: identical")
    else:
        print("DIFFERENT (check it is only the assistant/think part):")
        sys.stdout.writelines(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                                   "whole render", "token stream"))


if __name__ == "__main__":
    main(sys.argv[1])
