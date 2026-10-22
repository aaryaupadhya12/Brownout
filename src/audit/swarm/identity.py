"""Which code produced a result: a hash of the source files, not a version string.

Version strings are typed by hand and drift (versions.grader 0.2.0 vs grader_version 0.1.0).
A hash changes whenever the code does, so every ledger row and re-grade can name exactly
which verifier and env it ran against.

Hashes whole FILES, not single classes, so a change to a module helper (normalize() in
the cloudops verifier) counts. Limit: code imported from other files (the vendored tools)
is not included. Hash those separately if a patch ever touches them.
"""

import inspect

from ...scoring import Verifier
from ...tracking import sha256_of


def class_files_sha(cls, stop=object) -> str:
    """sha256 over (class name, source file text) for every class in cls's MRO up to,
    not including, `stop`. 16 hex chars, like group_id."""
    parts = []
    for klass in cls.__mro__:
        if klass is stop or klass is object:
            break
        path = inspect.getsourcefile(klass)
        if path is None:
            raise ValueError(f"no source file for {klass.__qualname__}: identity needs a class "
                f"defined in a file, not one built with exec()")
        with open(path, encoding="utf-8") as f:      # text mode: CRLF and LF hash the same
            parts.append({"class": klass.__qualname__, "file": f.read()})
    return sha256_of(parts)[:16]


def verifier_sha(verifier) -> str:
    """The verifier's code identity. A planted-hole subclass gets a different sha."""
    return class_files_sha(type(verifier), stop=Verifier)


def env_sha(env) -> str:
    """The env's code identity (BoutiqueEnv -> CloudOpsEnv)."""
    return class_files_sha(type(env))
