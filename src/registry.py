"""Name -> env. Anything that has an env's name can build it: the benchmark runner
(src/run.py), @QAHud re-grading saved runs, and later the Miles agent function.

Each env registers itself at the bottom of its own module:

    register(EnvSpec("brownout", BrownoutTasks, BrownoutEnv, BrownoutVerifierV1))

The spec stores classes (factories), not instances: every episode needs a fresh env,
and with concurrency several episodes run at once.

Compare: verifiers imports a module by id (verifiers/v1/utils/loaders.py:163-199);
OpenEnv scans installed openenv-* packages (src/openenv/auto/_discovery.py).
A plain dict is enough until envs live in other repos.
"""

import importlib
from dataclasses import dataclass
from typing import Callable

from .interface_v1 import Env, TaskSource, Verifier

# modules that call register() when imported; add each new env module here
BUILTIN_MODULES = ["env_v1"]


@dataclass(frozen=True)
class EnvSpec:
    name: str
    tasks: Callable[[], TaskSource]     # e.g. BrownoutTasks
    env: Callable[[], Env]              # e.g. BrownoutEnv: called once per episode
    verifier: Callable[[], Verifier]    # e.g. BrownoutVerifierV1


REGISTRY: dict[str, EnvSpec] = {}
_loaded = False


def register(spec: EnvSpec) -> None:
    # TODO(you): refuse a second spec under the same name (raise ValueError),
    # so two envs can't silently overwrite each other.
    REGISTRY[spec.name] = spec


def _load_builtins() -> None:
    """Import every module in BUILTIN_MODULES once, so their register() calls run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in BUILTIN_MODULES:
        importlib.import_module(f"{__package__}.{module}")


def names() -> list[str]:
    """Every registered env name, sorted."""
    _load_builtins()
    # TODO(you)
    raise NotImplementedError


def get(name: str) -> EnvSpec:
    """The spec for `name`. An unknown name raises KeyError listing the available names,
    e.g.  unknown env 'browout'; available: brownout"""
    _load_builtins()
    # TODO(you)
    raise NotImplementedError
