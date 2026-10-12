"""Brownout: route traffic away from a degraded zone without overloading the others.

Everything Brownout-specific lives in this folder; the harness never imports it directly.
The registry imports this package on its own (src/registry.py), which runs register() below.
"""

from ....registry import EnvSpec, register
from .v1 import BrownoutEnv, BrownoutTasks
from .verifier import BrownoutVerifierV1

register(EnvSpec("brownout", BrownoutTasks, BrownoutEnv, BrownoutVerifierV1))
