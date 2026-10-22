"""ReverseTextEnv: the world the model acts in. Here the world is just the prompt.

The harness (src/harness.py: run_episode) calls, in order:
    reset(task) -> str          the first user message
    tool_schema() -> list       tools offered to the model (here: none)
    step(tool, args) -> dict    only if the model calls a tool (it should not; answer anyway)
    done                        attribute, read after each tool step (True = "submit")
    snapshot() -> dict          facts about the end state, saved as trajectory.final_state
    close()

There is no submit tool. A plain-text reply does NOT end an episode: the harness appends a
nudge ("Use a tool, or call submit()...") and asks again. The episode ends because the
verifier's @stop single_turn says so (verifier.py). That is how a single-turn text task fits a
tool-call harness without changing the harness.
"""

from ....interface_v1 import Task

SYSTEM = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
# ^ identical to upstream's SYSTEM: a different prompt would make the scores incomparable.


class ReverseTextEnv:
    system_prompt = SYSTEM

    def reset(self, task: Task) -> str:
        """Start a fresh episode: set self.done = False and return the text to reverse
        (task.spec["prompt"], exactly, nothing added)."""
        raise NotImplementedError("TODO(you): ReverseTextEnv.reset")

    def step(self, tool: str, args: dict) -> dict:
        """No tools exist. If the model calls one anyway, return an error dict (do not raise:
        a raised exception would be recorded as an env error, not as the model's mistake)."""
        raise NotImplementedError("TODO(you): ReverseTextEnv.step")

    def tool_schema(self) -> list:
        """The tool list sent to the model. Empty here (the client omits an empty list)."""
        raise NotImplementedError("TODO(you): ReverseTextEnv.tool_schema")

    def snapshot(self) -> dict:
        """Facts, no pass/fail. Little to report here; e.g. {"done": self.done}.
        Think about: why must the verifier never need the env object itself?"""
        raise NotImplementedError("TODO(you): ReverseTextEnv.snapshot")

    def close(self) -> None:
        """Nothing to clean up (no containers, no files)."""
        raise NotImplementedError("TODO(you): ReverseTextEnv.close")
