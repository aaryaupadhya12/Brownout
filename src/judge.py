"""LLM as a judge (Phase 7): a model grades the agent's answer against written criteria.

A judge is used INSIDE a verifier, as an ordinary @reward, so it still reads only
(task, trace) and any saved run can be re-graded:

    class MyVerifier(Verifier):
        needs_env = False
        judge = Judge(make_client("nebius", "<judge model>"), CRITERIA)   # src/providers.py

        @reward(weight=1.0)
        def root_cause(self, task, trace):
            scores = self.judge.grade(task.spec["question"], final_answer(trace))
            return scores["root_cause"]

Rules (from verifiers/v1/judges/rubric.py):
  - a malformed verdict is a JUDGE failure: raise, so the run gets grader.ok=False,
    never a silent 0 for the model
  - the agent's answer is fenced, so it can't pose as instructions to the judge
  - temperature 0, and the judge model + prompt version go into the verifier's version
"""

import hashlib
import json
from dataclasses import dataclass
import re

PROMPT_VERSION = "1"
PROMPT = """Given a task, a response, and grading criteria, decide for each criterion whether the response satisfies it.

Task:
{question}

Response:
{response}

Criteria:
{criteria}

Reply with only a JSON object: {{"verdicts": {{"<criterion name>": "<one of its options>", ...}}}}"""


@dataclass(frozen=True)
class Criterion:
    name: str
    text: str                   # what to check, e.g. "names zone B as the degraded zone"
    choices: tuple[str, ...]    # worst to best, e.g. ("wrong", "partial", "correct")


def choice_score(choice: str, choices: tuple[str, ...]) -> float:
    """Map a choice to 0..1 by its position, worst to best:
    ("wrong", "partial", "correct") -> 0.0, 0.5, 1.0.   ("no", "yes") -> 0.0, 1.0.
    Compare case-insensitively, ignoring surrounding spaces.
    An unknown choice is a judge failure: raise ValueError naming it and the options."""
    normalized_choice = choice.strip().casefold()
    normalized_choices = [ option.strip().casefold() for  option in choices ]
    if len(choices) < 2:
        raise ValueError(f"Needed atleast 2 choices {choices!r}")
    if normalized_choice not in normalized_choices:
        raise ValueError(f"Unkwon choice {choice !r}: options are {choices !r}")
    position = normalized_choices.index(normalized_choice)
    return position / (len(choices) - 1)


def fence(text: str) -> str:
    """Wrap text in a code fence longer than any run of backticks inside it, so the
    agent's answer can't close the fence and continue as instructions to the judge.

        fence("hi")          -> "```\\nhi\\n```"        (at least 3 backticks)
        fence("a ```` b")    -> "`````\\na ```` b\\n`````"  (one more than the longest run)
    """
    runs = re.findall('`+',text)
    longest = max((len(run) for run in runs), default = 0)
    marker = "`" * max(3, longest + 1)
    return f"{marker}\n{text}\n{marker}"


def first_json_object(text: str) -> dict | None:
    """The first {...} in text that parses as a JSON object (judges often add prose or
    ```json fences around it), or None."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_verdicts(text: str, criteria: list[Criterion]) -> dict[str, float]:
    """The judge's reply -> {criterion name: score 0..1}.

    - find the object with first_json_object(text); none found -> ValueError
    - it must have a "verdicts" dict -> else ValueError
    - every criterion must have a verdict -> a missing one is a ValueError
    - score each with choice_score (an unknown choice raises there)
    Every failure raises: the harness turns it into grader.ok=False, not reward 0."""
    obj = first_json_object(text)
    if obj is None:
        raise ValueError("Judge reply has no JSON object in it")
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts,dict):
        raise ValueError("Judge must contain verdicts")
    scores = {}
    for c in criteria:
        if c.name not in verdicts:
            raise ValueError(f"Missing verdict for criterion {c.name!r}")
        scores[c.name] = choice_score(verdicts[c.name],c.choices)
    return scores


def final_answer(trace) -> str:
    """The agent's last non-empty assistant text, the usual thing to judge."""
    for event in reversed(trace.messages):
        payload = event["payload"]
        if payload.get("role") == "assistant" and payload.get("content"):
            return payload["content"]
    return ""


class Judge:
    def __init__(self, client, criteria: list[Criterion]):
        self.client = client            # any ModelClient: a real API, or FakeModel in tests
        self.criteria = criteria
        self.cache: dict[str, dict] = {}   # re-grading the same answer doesn't pay twice
        self.calls = 0

    @property
    def version(self) -> str:
        """Put this in the verifier's version: which judge model and prompt scored a run."""
        return f"{self.client.model_id}/prompt-{PROMPT_VERSION}"

    def build_prompt(self, question: str, response: str) -> str:
        criteria = "\n".join(f"- {c.name}: {c.text} (answer one of, worst to best: {', '.join(c.choices)})"
                             for c in self.criteria)
        return PROMPT.format(question=fence(question), response=fence(response), criteria=criteria)

    def grade(self, question: str, response: str) -> dict[str, float]:
        prompt = self.build_prompt(question, response)
        key = hashlib.sha256(f"{self.version}\n{prompt}".encode()).hexdigest()
        if key not in self.cache:
            reply = self.client.complete([{"role": "user", "content": prompt}], [], 0.0)
            self.calls += 1
            self.cache[key] = parse_verdicts(reply.text, self.criteria)
        return dict(self.cache[key])
