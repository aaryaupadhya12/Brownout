"""Grader audits: feed a grader inputs whose right score is known, report where it disagrees.

    layer 1  universal.py   deterministic checks, no model calls (oracle, floor, perturbation ...)
    layer 2  judge.py       checks for LLM-judge graders (canaries, delta tests, variance)

An audit never imports env internals: each env gets a Target (target.py) that exposes its
tasks, its grader and the known-answer traces. Checks are marked with @check (checks.py),
the same way scorers are marked with @metric in src/scoring.py, and every result is a
Finding (report.py).

Run:  python -m src.audit cloudops-boutique --layers 1
"""
