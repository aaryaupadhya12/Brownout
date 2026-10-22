"""ReverseTextTasks: one Task per prompt.

A Task is data only (src/interface_v1.py): Task(id, seed, spec). Everything the verifier needs
to grade lives in task.spec, so a saved run can be re-graded without the env.

Example of one task you should produce:
    Task(id="reverse-text-0", seed=0,
         spec={"prompt": "The community in Bruck was merged into it",
               "answer": "ti otni degrem saw kcurB ni ytinummoc ehT"})
"""

from ....interface_v1 import Task
from .data import DEFAULT_CACHE, fetch_rows


class ReverseTextTasks:
    def __init__(self, cache_path: str = DEFAULT_CACHE, limit: int | None = None):
        """cache_path: where data.fetch_rows reads/writes the prompts.
        limit: only the first N tasks (cheap smoke runs). Tests pass a temporary cache."""
        self.cache_path = cache_path
        self.limit = limit

    def load(self):
        """Yield Tasks in dataset order: id "reverse-text-<index>", seed 0,
        spec {"prompt": ..., "answer": prompt reversed}.

        Think about: is the answer computed exactly like upstream (row["prompt"][::-1], no
        strip)? Parity matters: if you "clean" the prompt, your scores stop matching theirs.
        """
        raise NotImplementedError("TODO(you): ReverseTextTasks.load")
