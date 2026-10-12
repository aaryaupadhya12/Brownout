"""Copy this folder to ship a new env, e.g. src/envs/CodingEnv/myenv/ (drop the leading "_",
which is what keeps this template out of the registry). Then fill in the three files.

    __init__.py   register(EnvSpec(...))        <- this file
    env.py        the Env: system_prompt, reset, step, tool_schema, snapshot, close, done
    tasks.py      the TaskSource: load() -> Tasks, every threshold in task.spec
    verifier.py   @reward / @metric scorers reading only (task, trace)

Run it with:  python -m src.run myenv --model ... --base-url ...
"""

from ...registry import EnvSpec, register   # one "." more per folder level below src/envs
from .env import TemplateEnv
from .tasks import TemplateTasks
from .verifier import TemplateVerifier

register(EnvSpec("template", TemplateTasks, TemplateEnv, TemplateVerifier))
