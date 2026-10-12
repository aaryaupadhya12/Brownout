"""Name -> env. Anything that has an env's name can build it: the benchmark runner
(src/run.py), @QAHud re-grading saved runs, and later the Miles agent function.

To ship a new env, add a folder under src/envs/ (any depth, e.g. src/envs/CodingEnv/myenv/)
with an __init__.py that registers it:

    register(EnvSpec("brownout", BrownoutTasks, BrownoutEnv, BrownoutVerifierV1))

Nothing else changes: the registry finds every such folder on its own (_load_builtins).
Folders whose name starts with "_" (like src/envs/_template) are skipped.

The spec stores classes (factories), not instances: every episode needs a fresh env,
and with concurrency several episodes run at once.

Compare: verifiers imports a module by id (verifiers/v1/utils/loaders.py:163-199);
OpenEnv scans installed openenv-* packages (src/openenv/auto/_discovery.py).
"""

import importlib
import os
from dataclasses import dataclass
from typing import Callable

from .interface_v1 import Env, TaskSource, Verifier


ENVS_DIR = os.path.join(os.path.dirname(__file__), "envs")


@dataclass(frozen=True)
class EnvSpec:
    name: str
    tasks: Callable[[], TaskSource]     # e.g. BrownoutTasks
    env: Callable[[], Env]              # e.g. BrownoutEnv: called once per episode
    verifier: Callable[[], Verifier]    # e.g. BrownoutVerifierV1


REGISTRY: dict[str, EnvSpec] = {}
_loaded = False


def register(spec: EnvSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"Environment '{spec.name}' is already registered")
    REGISTRY[spec.name] = spec


def env_packages(root: str = ENVS_DIR) -> list[str]:
    """Module paths of every env folder under src/envs: a folder with an __init__.py.
    Folders inside an env (its own subpackages) and names starting with "_" are skipped."""
    found = []
    for folder, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith(("_", ".")))
        if folder != root and "__init__.py" in files:
            rel = os.path.relpath(folder, os.path.dirname(root))
            found.append(".".join(rel.split(os.sep)))
            subdirs[:] = []      # an env's own subfolders are not separate envs
    return found


def _load_builtins() -> None:
    """Import every env package once, so their register() calls run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in env_packages():
        importlib.import_module(f"{__package__}.{module}")


def names() -> list[str]:
    """Every registered env name, sorted."""
    _load_builtins()
    return sorted(REGISTRY.keys())


def get(name: str) -> EnvSpec:
    """The spec for `name`. An unknown name raises KeyError listing the available names,
    e.g.  unknown env 'browout'; available: brownout"""
    _load_builtins()
    if name not in REGISTRY:
        raise KeyError(f"unknown env {name!r}; available: {', '.join(names())}")
    return REGISTRY[name]
