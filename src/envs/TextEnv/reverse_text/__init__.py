"""reverse-text: reverse a string character by character. Port of Prime Intellect's verifiers
env (environments/reverse_text/reverse_text/taskset.py): single turn, answer inside
<reversed_text> tags, reward = difflib LCS ratio against the true reversal.

Why it is here: the smallest env where "solved" is unambiguous (exact match), so it calibrates
the swarm and shows how a text-only env fits a tool-call harness.

    __init__.py   register(EnvSpec("reverse-text", ...))      <- this file
    data.py       fetch the 1,000 prompts once, cache them as JSON
    tasks.py      ReverseTextTasks: one Task per prompt, answer = prompt[::-1]
    env.py        ReverseTextEnv: no tools; reset() returns the prompt
    verifier.py   ReverseTextVerifier: @stop single_turn, @reward lcs, @metric exact_match

Guide: interview/10_reverse_text_guide.md. Tests: tests/reverse_text_test.py.
"""

# TODO(you): import EnvSpec and register from the registry (count the dots: this folder is
# three levels below src/), import your three classes, and register the env under the name
# "reverse-text". Until you do, the registry does not know this env and the end-to-end tests
# skip with "not registered yet".
