"""@check, and the Audit class that finds and runs checks. Mirrors src/scoring.py:

    @metric  -> fn.scorer_kind     Verifier.scorers() finds them through the MRO
    @check   -> fn.check_tag       Audit.checks()     finds them the same way

A check is a method  fn(self, target, task) -> Finding | list[Finding]  for per-task checks,
or  fn(self, target) -> ...  for env-level ones (mark with per_task=False).
A check that raises becomes an "error" Finding: the audit keeps going.
"""

from .report import TAGS, Finding


def check(tag, per_task=True):
    if tag not in TAGS:
        raise ValueError(f"unknown tag {tag!r}; use one of {TAGS}")

    def mark(fn):
        fn.check_tag = tag
        fn.per_task = per_task
        return fn
    return mark


class Audit:
    name = "audit"        # prefix for check ids: "<name>.<method name>"

    def checks(self):
        found = {}
        for cls in reversed(type(self).__mro__):
            for attr_name, value in vars(cls).items():
                if getattr(value, "check_tag", None) is not None:
                    found[attr_name] = getattr(self,attr_name)
        return list(found.items())

    def run(self, target, tasks, report):
        """Run every check: per-task checks once per task, env-level checks once."""
        for name, fn in self.checks():
            check_id = f"{self.name}.{name}"
            if fn.per_task:
                for task in tasks:
                    self._call(report, check_id, fn.check_tag, fn, target, task)
            else:
                self._call(report, check_id, fn.check_tag, fn, target)

    def _call(self, report, check_id, tag, fn, *args):
        task_id = args[1].id if len(args) > 1 else ""
        try:
            result = fn(*args)
        except Exception as e:
            report.add(Finding(check_id, tag, "error", task_id,
                               evidence=[f"{type(e).__name__}: {e}"]))
            return
        results = result if isinstance(result, list) else [result]
        for finding in results:
            finding.check_id = check_id      # the check fills in everything else
            report.add(finding)
