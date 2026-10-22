"""What an env exposes so it can be audited. One Target per env; checks only talk to this.

Required:   tasks(), grade(task, trace)
Optional:   gold, floor, variants, malformed, judge, runs_dir
            -> return None (or leave unimplemented) and the checks that need it SKIP.

Works without running the env or a model because v1 verifiers read only (task, trace):
a fake trace with the right final_state is enough to call the grader.
"""

from pathlib import Path

from ..contracts import GraderOutput, Trajectory


def fake_trace(task, final_state, steps=None) -> Trajectory:
    """A minimal Trajectory carrying only what a v1 verifier reads."""
    return Trajectory(task_id=task.id, model_id="audit", run_date="", run_index=0,
                      versions={}, steps=steps or [], grader=GraderOutput(0.0, {}, ""),
                      ended_by="audit", final_state=final_state)


class Target:
    name = "target"
    reward_range = (0.0, 1.0)
    normal_endings = ("submit",)   # Trajectory.ended_by values that mean a normal finish
    submit_tool = "Submit"         # the tool whose call hands in the answer

    def tasks(self):
        raise NotImplementedError

    def grade(self, task, trace):
        """Return the verifier's grader output and errors."""
        raise NotImplementedError

    def grader_name(self) -> str:
        return ""

    # --- optional: known-answer traces. None = the checks that need it SKIP ---

    def gold(self, task):
        """A trace that must get max reward."""
        return None

    def floor(self, task):
        """A trace for doing nothing: must get min reward."""
        return None

    def variants(self, task):
        """Equivalent correct traces: each must still get max reward."""
        return None

    def malformed(self, task):
        """Broken submissions: the right outcome is an infra error, not a silent 0."""
        return None

    def answer_strings(self, task):
        """Strings that would give the answer away if the agent saw them (leak check)."""
        return None

    def runs_dir(self):
        """Saved trajectories for this env (calibration, leak-in-observations)."""
        return None

    def answer_key(self,task):
        """ Hashable, equal if 2 tasks are the correct answer, None if many answers are coirrect"""
        return None

    # --- layer 2 ---

    def judge(self):
        """A callable judge(question, answer) -> score, or None if the grader has no judge."""
        return None


class CloudOpsTarget(Target):
    """Cloud-OpsBench (Online Boutique). The answer is task.spec["answer"]:
    {"fault_object": ..., "root_cause": ...}; the verifier reads final_state["submission"]."""

    name = "cloudops-boutique"

    def tasks(self):
        from src.registry import get
        return list(get(self.name).tasks().load())

    def __init__(self, verifier=None):
        """verifier: the one to audit (e.g. a patch under review). None = CloudOpsVerifier()."""
        from ..envs.SystemEnv.cloudops.verifier import CloudOpsVerifier
        self.verifier = verifier or CloudOpsVerifier()

    def grade(self, task, trace):
        return self.verifier.score(task, trace)

    def grader_name(self) -> str:
        return f"{self.verifier.name}-{self.verifier.version}"

    def gold(self, task):
        answer = task.spec["answer"]
        first = (answer["fault_object"], answer["root_cause"])
        second = (f"{first[0]}_wrong1", f"{first[1]}_wrong1")
        third = (f"{first[0]}_wrong2", f"{first[1]}_wrong2")

        submission = {
            "top_3_predictions": [
                {"rank": i + 1, "fault_object": obj, "root_cause": cause}
                for i, (obj, cause) in enumerate([first, second, third])
                ]
            }

        return fake_trace(task, {"submitted": True, "submission": submission})

    def answer_key(self,task):
        from ..envs.SystemEnv.cloudops.verifier import normalize
        answer = task.spec["answer"]
        return (normalize(answer["fault_object"]), normalize(answer["root_cause"]))

    def floor(self, task):
        return fake_trace(task, {"submitted": False})

    def variants(self, task):
        answer = task.spec["answer"]
        obj = answer["fault_object"]
        cause = answer["root_cause"]

        cases = [
            (obj.upper(), cause),                   # Different casing
            (f" {obj} ", f" {cause} "),             # Extra whitespace
        ]

        traces = []

        for variant_obj, variant_cause in cases:
            submission = {
                "top_3_predictions": [
                    {
                        "rank": 1,
                        "fault_object": variant_obj,
                        "root_cause": variant_cause,
                    }
                ]
            }

            traces.append(
                fake_trace(
                    task,
                    final_state={
                        "submitted": True,
                        "submission": submission,
                    }
                )
            )

        return traces

    

    def answer_strings(self, task):
        answer = task.spec["answer"]
        return [answer["root_cause"]]

    def malformed(self, task):
        return fake_trace(task, {
            "submitted": True,
            "submission": {"top_3_predictions": []},
        })

    def runs_dir(self):
        return Path(__file__).resolve().parents[2] / "runs"


TARGETS = {
    CloudOpsTarget.name: CloudOpsTarget,
    # TODO(you): add more targets here as you write them
}
