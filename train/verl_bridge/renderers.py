"""Chat-template renderers for TokenModel. No verl import, so it runs on a laptop."""

# A fixed conversation to cut against: render(BASE + new) minus render(BASE) = new.
# It ends with a USER turn on purpose: templates like Qwen3 render the assistant turn after
# the last user message differently (empty <think></think>) depending on whether it is the
# last message, so an assistant-final BASE would not be a prefix of BASE + new.
_BASE = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
         {"role": "user", "content": "z"}]


def make_renderers(tokenizer, chat_kwargs: dict | None = None):
    """render_prompt / render_delta for TokenModel, from a HF tokenizer's chat template.

    Works on TEXT and encodes only the new part, so the token boundary at the cut can't
    merge with earlier tokens. The model's reply ends with the end-of-turn token; whatever
    the template writes after that token (often "\n") is added back in front of the delta.
    Run train/verl_bridge/check_template.py once per model to see this is right.
    """
    chat_kwargs = chat_kwargs or {}

    def text(messages, **kw):
        return tokenizer.apply_chat_template(messages, tokenize=False, **chat_kwargs, **kw)

    def encode(s):
        return tokenizer.encode(s, add_special_tokens=False)

    base = text(_BASE)
    eos = tokenizer.eos_token or ""
    gap = base[base.rfind(eos) + len(eos):] if eos and eos in base else ""

    def render_prompt(messages, tools):
        return encode(text(messages, tools=tools, add_generation_prompt=True))

    def render_delta(messages):
        full = text(_BASE + list(messages), add_generation_prompt=True)
        if not full.startswith(base):
            raise ValueError("chat template re-renders earlier turns: render_delta can't cut it")
        return encode(gap + full[len(base):])

    return render_prompt, render_delta
