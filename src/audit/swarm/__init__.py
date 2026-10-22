"""Swarm orchestrator v1 (spec: "Swarm Orchestrator v1 Spec.pdf" in this folder).

v1 scope: cloudops-boutique, in-process, planted holes first, one attacker role, a person
writes every patch. Containers/Harbor come after the loop works.

    identity.py   verifier_sha / env_sha: a content hash, not a hand-edited version string
    exploit.py    Exploit (a runnable repro) and replay() through run_episode with a FakeModel
    saved.py      load a saved trajectory JSON back into a Trajectory, for re-grading
    ledger.py     append-only JSONL of findings; signature() for dedupe        [TODO(you)]
    planted.py    verifier copies with one known hole each, to measure recall  [TODO(you)]
    confirm.py    is_hack(): earns reward on the unpatched verifier AND not truly solved [TODO(you)]
    roles.py      AttackerEnv: the same env with an attacker prompt             [TODO(you)]
    gate.py       the deterministic regression gate a patch must pass           [TODO(you)]
    loop.py       per-task loop; resume by role key, no state machine           [TODO(you)]
"""
 