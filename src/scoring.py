"""Pluggable scoring, modelled on verifiers' @reward / @metric (verifiers/v1/task.py:213-284).

A Verifier is a class with scorer methods. Mark each one:

    @reward(weight=0.6)   counts toward the reward:  reward = sum(score * weight)
    @metric               recorded only, never added into the reward

Every scorer is called as  fn(task, trace, env)  and gets the WHOLE trace (messages,
steps, model calls, errors), so it can score the process, not just the final state.
It returns a number, or a dict of {name: number} to record several values at once.

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


class Verifier:
    name = "verifier"
    version = "0.0.0"

    def scorers(self, kind):
        """(name, bound method) for every method marked `kind`, in definition order.
        A subclass method with the same name replaces the parent's."""
        found = {}
        for cls in reversed(type(self).__mro__):
            for attr_name, value in vars(cls).items():
                if getattr(value, "scorer_kind", None) == kind:
                    found[attr_name] = getattr(self, attr_name)
        return list(found.items())

    def score(self, task, trace, env):
        """Run every metric, then every reward. Returns (GraderOutput, [ErrorRecord])."""
        metrics = {}
        rewards = {}
        errors = []

        for name, fn in self.scorers("metric"):
            try:
                value = fn(task, trace, env)
            except Exception as e:
                errors.append(ErrorRecord("scorer", type(e).__name__, f"metric {name}: {e}"))
                continue
            if isinstance(value, dict):
                metrics.update(value)
            else:
                metrics[name] = value

        for name, fn in self.scorers("reward"):
            try:
                value = fn(task, trace, env)
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

    @metric
    def turns(self, task, trace, env):
        seen = []
        for s in trace.steps:
            if s.turn not in seen:
                seen.append(s.turn)
        return len(seen)

    @metric
    def tool_calls(self, task, trace, env):
        count = 0
        for s in trace.steps:
            if s.tool is not None:
                count += 1
        return count

    @metric
    def tool_errors(self, task, trace, env):
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
