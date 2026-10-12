from ...scoring import Verifier, metric, reward


class TemplateVerifier(Verifier):
    name = "template"
    version = "0.1.0"
    needs_env = False          # scorers take (task, trace): a saved run can be re-graded

    @reward(weight=1.0)
    def solved(self, task, trace):
        return 1.0 if trace.final_state.get("submitted") else 0.0   # TODO: the real check

    @metric
    def steps_used(self, task, trace):
        return trace.final_state.get("steps_used")
