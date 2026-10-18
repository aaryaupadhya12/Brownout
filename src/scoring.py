"""Pluggable scoring, modelled on verifiers' @reward / @metric (verifiers/v1/task.py:213-284).

A Verifier is a class with scorer methods. Mark each one:

    @reward(weight=0.6)   counts toward the reward:  reward = sum(score * weight)
    @metric               recorded only, never added into the reward
    @stop                 ends the episode early when it returns True (src/limits.py)

Every scorer gets the WHOLE trace (messages, steps, model calls, errors, final_state),
so it can score the process, not just the final state. It returns a number, or a
dict of {name: number} to record several values at once.

Scorers are called as  fn(task, trace)  when the verifier sets needs_env = False:
everything comes from the trajectory, so a saved run can be re-graded later (v1,
src/interface_v1.py). Older verifiers that still call env.grade() keep the default
needs_env = True and are called as  fn(task, trace, env).

To grade a different evaluation, write another Verifier subclass and pass it to
run_episode(..., verifier=MyVerifier()). The harness never knows which one it runs.

A scorer that raises does not crash the episode: the error is recorded, grader.ok
becomes False, and the run should be left out of averages (not counted as 0).
"""

from .contracts import ErrorRecord, GraderOutput


def reward(weight=1.0):
    def mark(fn):
        fn.scorer_kind = "reward"
        fn.weight = weight
        return fn
    return mark


def metric(fn):
    fn.scorer_kind = "metric"
    return fn


def stop(fn):
    """Marks fn(self, task, steps) -> bool: True ends the episode before the next model
    call, and the method's name becomes trajectory.ended_by (src/limits.py:first_stop)."""
    # TODO(you): mark it the same way metric() does, with scorer_kind "stop"
    fn.scorer_kind = "stop"
    return fn


class Verifier:
    name = "verifier"
    version = "0.0.0"
    needs_env = True     # set False once every scorer reads only (task, trace)

    def scorers(self, kind):
        """(name, bound method) for every method marked `kind`, in definition order.
        A subclass method with the same name replaces the parent's."""
        # Prases through the Inheritance chain first to get the base class and then reverses it so in the class the first cls will b e the base classw not the parent class of the object 
        found = {}
        for cls in reversed(type(self).__mro__):
            for attr_name, value in vars(cls).items():
                if getattr(value, "scorer_kind", None) == kind:
                    found[attr_name] = getattr(self, attr_name)
        return list(found.items())

    def call(self, fn, task, trace, env):
        if self.needs_env:
            return fn(task, trace, env)
        return fn(task, trace)

    def score(self, task, trace, env=None):
        """Run every metric, then every reward. Returns (GraderOutput, [ErrorRecord])."""
        metrics = {}
        rewards = {}
        errors = []

        for name, fn in self.scorers("metric"):
            try:
                value = self.call(fn, task, trace, env)
            except Exception as e:
                errors.append(ErrorRecord("scorer", type(e).__name__, f"metric {name}: {e}"))
                continue
            if isinstance(value, dict):
                metrics.update(value)
            else:
                metrics[name] = value

        for name, fn in self.scorers("reward"):
            try:
                value = self.call(fn, task, trace, env)
            except Exception as e:
                errors.append(ErrorRecord("scorer", type(e).__name__, f"reward {name}: {e}"))
                rewards[name] = None
                continue
            if isinstance(value, dict):
                for key, score in value.items():
                    rewards[key] = {"score": float(score), "weight": fn.weight}
            else:
                rewards[name] = {"score": float(value), "weight": fn.weight}

        ok = len(errors) == 0
        total = 0.0
        if ok:
            for entry in rewards.values():
                total += entry["score"] * entry["weight"]

        grader = GraderOutput(total, metrics, self.version,
                              rewards=rewards, ok=ok, grader_name=self.name)
        return grader, errors

    # --- metrics every evaluation gets for free: they only read the trace ---
    # (env=None so they work for both kinds of verifier)

    @metric
    def turns(self, task, trace, env=None):
        seen = []
        for s in trace.steps:
            if s.turn not in seen:
                seen.append(s.turn)
        return len(seen)

    @metric
    def tool_calls(self, task, trace, env=None):
        count = 0
        for s in trace.steps:
            if s.tool is not None:
                count += 1
        return count

    @metric
    def tool_errors(self, task, trace, env=None):
        count = 0
        for s in trace.steps:
            if s.tool is not None and "error" in s.observation:
                count += 1
        return count


class EnvGradeVerifier(Verifier):
    """Default: use the environment's own grade(). Works for any env that has one."""

    name = "env_grade"
    version = "0.2.0"

    @reward(weight=1.0)
    def env_reward(self, task, trace, env):
        value, _ = env.grade()
        return value

    @metric
    def env_signals(self, task, trace, env):
        _, signals = env.grade()
        return signals
