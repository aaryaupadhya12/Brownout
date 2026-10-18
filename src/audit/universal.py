"""Layer 1: deterministic checks, no model calls. Each one feeds the grader a trace whose
right score is known and reports whether the grader agrees.

oracle is written out as the pattern; the rest are yours.
"""

import json
import math
from numbers import Real
from pathlib import Path

from .checks import Audit, check
from .report import Finding


def reward_of(target, task, trace):
    """(reward, ok) from the target's grader."""
    grader, errors = target.grade(task, trace)
    return grader.reward, grader.ok and not errors


class UniversalAudit(Audit):
    name = "universal"
    k = 3                     # how many times determinism re-grades the same trace

    @check("FALSE_NEG")
    def oracle(self, target, task):
        """A correct answer must get max reward. If this fails, every other check on this
        task is meaningless: fix the task or the grader first."""
        trace = target.gold(task)
        if trace is None:
            return Finding("", "FALSE_NEG", "skip", task.id, evidence=["target has no gold()"])
        reward, ok = reward_of(target, task, trace)
        top = target.reward_range[1]
        status = "pass" if ok and reward == top else "fail"
        return Finding("", "FALSE_NEG", status, task.id,
                       measured={"reward": reward, "expected": top, "grader_ok": ok},
                       remediation="" if status == "pass" else
                       "The grader rejects the known answer: check the label and the comparison.")

    @check("FALSE_POS")
    def floor(self, target, task):
        """Doing nothing must get min reward."""
        trace = target.floor(task)
        if trace is None:
            return Finding("", "FALSE_NEG", "skip", task.id, evidence=["target has no floor()"])
        reward, ok = reward_of(target,task,trace)
        bottom = target.reward_range[0] # we want the minimum 

        status = "pass" if ok and reward == bottom else "fail"
        return Finding("", "FALSE_POS", status, task.id,  # step 4
                   measured={"reward": reward, "expected": bottom, "grader_ok": ok},
                   remediation="" if status == "pass" else
                   "Doing nothing earns credit, or the grader errors on an empty submission.")


    @check("FALSE_NEG")
    def perturbation(self, target, task):
        """Every equivalent correct answer must keep max reward. One Finding per variant,
        so the report shows which variant lost credit."""
        variants = target.variants(task)
        if variants is None:
            return [Finding("", "FALSE_NEG", "skip", task.id,
                            evidence=["target has no variants()"])]

        findings = []
        top = target.reward_range[1]
        for index, trace in enumerate(variants, start=1):
            reward, ok = reward_of(target, task, trace)
            status = "pass" if ok and reward == top else "fail"
            findings.append(Finding(
                "", "FALSE_NEG", status, task.id,
                measured={"variant": index, "reward": reward,
                          "expected": top, "grader_ok": ok},
                remediation="" if status == "pass" else
                f"Variant {index} lost credit: check normalize().",
            ))
        return findings or [Finding("", "FALSE_NEG", "skip", task.id,
                                    evidence=["target returned no variants"])]

    @check("CALIBRATION")
    def determinism(self, target, task):
        """The same trace graded k times gives the same reward."""
        if not callable(getattr(target, "gold", None)):
            return Finding("", "CALIBRATION", "skip", task.id,
                           evidence=["target has no gold()"])
        trace = target.gold(task)
        if trace is None:
            return Finding("", "CALIBRATION", "skip", task.id,
                           evidence=["target has no gold()"])

        results = [reward_of(target, task, trace) for _ in range(self.k)]
        scores = [reward for reward, _ in results]
        all_ok = all(ok for _, ok in results)
        deterministic = all(score == scores[0] for score in scores[1:])
        status = "pass" if all_ok and deterministic else "fail"
        return Finding("", "CALIBRATION", status, task.id,
                       measured={"scores": scores,
                                 "grader_ok": [ok for _, ok in results]},
                       remediation="" if status == "pass" else
                       "Repeated grading must succeed and return the same reward.")

    @check("INFRA")
    def well_formed(self, target, task):
        """Rewards are finite numbers inside reward_range, never bool or None."""
        if not callable(getattr(target, "gold", None)):
            return Finding("", "INFRA", "skip", task.id,
                           evidence=["target has no gold()"])
        trace = target.gold(task)
        if trace is None:
            return Finding("", "INFRA", "skip", task.id,
                           evidence=["target has no gold()"])
        if not callable(getattr(target, "floor", None)):
            return Finding("", "INFRA", "skip", task.id,
                           evidence=["target has no floor()"])
        trace_floor = target.floor(task)
        if trace_floor is None:
            return Finding("", "INFRA", "skip", task.id,
                           evidence=["target has no floor()"])

        lower, upper = target.reward_range
        samples = {}
        for name, trace in (("gold", trace), ("floor", trace_floor)):
            reward, ok = reward_of(target, task, trace)
            valid_number = isinstance(reward, Real) and not isinstance(reward, bool)
            finite = valid_number and math.isfinite(reward)
            in_range = finite and lower <= reward <= upper
            samples[name] = {
                "reward": reward,
                "grader_ok": ok,
                "finite_number": finite,
                "in_range": in_range,
            }

        status = "pass" if all(
            sample["grader_ok"] and sample["finite_number"] and sample["in_range"]
            for sample in samples.values()
        ) else "fail"
        return Finding("", "INFRA", status, task.id, measured=samples,
                       remediation="" if status == "pass" else
                       "Gold and floor rewards must be successful, finite numbers within reward_range.")

    @check("INFRA")
    def malformed(self, target, task):
        """A broken submission must be recorded as an infra failure (grader.ok False or an
        error), not quietly scored 0."""
        if not callable(getattr(target, "malformed", None)):
            return Finding("", "INFRA", "skip", task.id,
                           evidence=["target has no malformed()"])
        trace = target.malformed(task)
        try:
            grader, errors = target.grade(task, trace)
        except Exception as error:
            return Finding("", "INFRA", "pass", task.id,
                           measured={"grader_ok": False,
                                     "error": f"{type(error).__name__}: {error}"})

        handled = not grader.ok or bool(errors)
        return Finding("", "INFRA", "pass" if handled else "fail", task.id,
                       measured={"reward": grader.reward, "grader_ok": grader.ok,
                                 "errors": [str(error) for error in errors]},
                       remediation="" if handled else
                       "Malformed submissions must be rejected as an infrastructure failure, not scored normally.")

    @check("LEAK", per_task=False)
    def leak_in_observations(self, target):
        """No answer string appears in what the agent saw: the task prompt or any tool
        output in the saved trajectories."""
        run_dir = Path(target.runs_dir())
        paths = list(run_dir.glob("*.json"))
        if not paths:
            return [Finding("", "LEAK", "skip", evidence=["no saved runs found"])]
        tasks = {task.id: task for task in target.tasks()}
        findings = []
        for path in paths:
            with path.open(encoding="utf-8") as file:
                trace = json.load(file)
            task = tasks.get(trace.get("task_id"))
            if task is None:
                continue

            visible = json.dumps(
                {"steps": trace.get("steps", []), "messages": trace.get("messages", [])},
                ensure_ascii=False,
            ).casefold()
            for answer in target.answer_strings(task):
                if answer and answer.casefold() in visible:
                    findings.append(Finding(
                        "", "LEAK", "fail", task.id,
                        measured={"answer": answer},
                        evidence=[str(path)],
                    ))

        return findings or [Finding("", "LEAK", "pass")]

    @check("CALIBRATION", per_task=False)
    def calibration(self, target):
        """Tasks that every saved run solves, or none does: they give GRPO no signal."""
        runs = {}
        paths = list(Path(target.runs_dir()).glob("*.json"))
        if not paths:
            return [Finding("", "CALIBRATION", "skip",
                            evidence=["no saved runs found"])]
        for path in paths:
            with path.open(encoding="utf-8") as file:
                trace = json.load(file)
            grader = trace.get("grader", {})
            if grader.get("ok", True):
                runs.setdefault(trace.get("task_id"), []).append(grader.get("reward"))

        findings = []
        low, high = target.reward_range
        analyzed = 0
        for task_id, rewards in runs.items():
            if len(rewards) < 2:
                continue
            analyzed += 1
            always_extreme = all(reward == rewards[0] for reward in rewards)
            at_boundary = rewards[0] in (low, high)
            if always_extreme and at_boundary:
                findings.append(Finding(
                    "", "CALIBRATION", "fail", task_id,
                    measured={"rewards": rewards, "runs": len(rewards)},
                    remediation="Add task difficulty or variation so runs produce a learning signal.",
                ))

        if findings:
            return findings
        if not analyzed:
            return [Finding("", "CALIBRATION", "skip",
                            evidence=["need at least two valid runs per task"])]
        return [Finding("", "CALIBRATION", "pass")]
