from ...scoring import Verifier, metric, reward, stop


class TemplateVerifier(Verifier):
    name = "template"
    version = "0.1.0"
    needs_env = False          # scorers take (task, trace): a saved run can be re-graded

    @reward(weight=1.0)
    def solved(self, task, trace):
        return 1.0 if trace.final_state.get("submitted") else 0.0   # TODO: the real check

    # optional: end the episode early. Runs before every model call; True = stop now.
    # @stop
    # def looping(self, task, steps):
    #     last = [(s.tool, s.args) for s in steps[-3:]]
    #     return len(last) == 3 and len(set(map(str, last))) == 1    # same call 3 times

    @metric
    def steps_used(self, task, trace):
        return trace.final_state.get("steps_used")
