"""ReverseTextVerifier: grades a finished episode from (task, trace) only.

Upstream reward (verifiers, reverse_text/taskset.py), which yours must match exactly:

    _TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)
    match = _TAG.search(completion or "")
    response = match.group(1).strip() if match else ""
    return SequenceMatcher(None, response, answer).ratio()

What the trace looks like when scorers run (a saved trajectory, messages are dicts):
    trace.messages = [
      {"turn": 0, "kind": "system",    "payload": {"role": "system", "content": SYSTEM}},
      {"turn": 0, "kind": "task",      "payload": {"role": "user", "content": "<prompt>"}},
      {"turn": 1, "kind": "assistant", "payload": {"role": "assistant",
                                                  "content": "<reversed_text>...</reversed_text>"}},
      {"turn": 1, "kind": "nudge",     "payload": {"role": "user", "content": "Use a tool, ..."}},
    ]
    trace.steps = [Step(turn=1, tool=None, args={}, observation={"note": "plain text reply"}, ...)]

Decorators (src/scoring.py): @reward(weight) -> part of the reward; @metric -> reported only;
@stop -> fn(self, task, steps) runs BEFORE every model call, True ends the episode and its
name becomes trajectory.ended_by.
"""

import re
from difflib import SequenceMatcher

from ....scoring import Verifier, metric, reward, stop

TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


def parse_answer(text: str | None) -> str:
    """Upstream's parsing, exactly: first <reversed_text> block, inner text stripped; "" if
    there is no block or text is None.
        parse_answer("<reversed_text> olleh </reversed_text>")  -> "olleh"
        parse_answer("olleh")                                    -> ""
    """
    raise NotImplementedError("TODO(you): parse_answer")


def first_assistant_text(trace) -> str | None:
    """The content of the FIRST assistant message in trace.messages (None if there is none).
    Think about: why first and not last? (What is the second assistant message, if any?)"""
    raise NotImplementedError("TODO(you): first_assistant_text")


class ReverseTextVerifier(Verifier):
    name = "reverse-text"
    version = "0.1.0"
    needs_env = False          # every scorer reads only (task, trace)

    @stop
    def single_turn(self, task, steps):
        """True once the model has replied once (any step exists). Upstream's
        `trace.num_turns >= 1`. Without this, the nudge buys the model a second try."""
        raise NotImplementedError("TODO(you): single_turn")

    @reward(weight=1.0)
    def lcs(self, task, trace):
        """SequenceMatcher(None, parse_answer(first reply), task.spec["answer"]).ratio()."""
        raise NotImplementedError("TODO(you): lcs")

    @metric
    def exact_match(self, task, trace):
        """1.0 if the parsed answer equals task.spec["answer"] exactly, else 0.0.
        This is the strict second referee for the audit: lcs can be high when this is 0."""
        raise NotImplementedError("TODO(you): exact_match")

    @metric
    def has_tag(self, task, trace):
        """1.0 if the first reply contains a <reversed_text> block at all. Separates "wrong
        format" from "wrong reversal" when you read results."""
        raise NotImplementedError("TODO(you): has_tag")
