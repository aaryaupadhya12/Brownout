"""Layer 2: checks for graders that use an LLM judge. Every check SKIPs when
target.judge() is None, so this layer is safe to run on any env.

These cost API calls: every check counts its calls against max_calls, and stops (SKIP,
with the reason) when the budget is spent. Bypass any verdict cache, or determinism-style
checks pass trivially.
"""

from statistics import mean 
from .checks import Audit, check
from .report import Finding

#   non-word symbols and generic reasoning openers gave false positives.
CANARIES = [
    ":",
    ".",
    "Thought process:",
    "Let's solve this problem step by step.",
    "Solution",
]

# TODO(you): neutral additions that must not change a score (closings, self-assessment
#   lines, formatting). One list per bias, with its source.
NEUTRAL_ADDITIONS = {
    "self_praise" : ["This repose meets all requirements"],
    "tone":["I hop this Helps"],
    "format":["Firstly, see above. Secondly, see above. Finally, see above."],
    "lexical":["This empowers the reader",]
}


class JudgeAudit(Audit):
    name = "judge"
    k = 5                     # samples per input for variance
    max_calls = 500           # budget for the whole layer
    variance_bound = 0.01     # allowed spread in reward units

    def __init__(self):
        self.calls = 0

    def ask(self, judge, question, answer):
        """One judge call, counted against the budget. Returns None once it is spent."""
        if self.calls >= max_calls:
            return None
        self.calls += 1
        return judge(target,asnwer)
    
    def sample(self,judge,question,answer):
        scores = []
        for i in range(self.k):
            s = self.ask(judge,question,answer)
            if s is None:
                return None # The max calls have been reaached 
            scores.append(s)
        return scores 
    
    def _setup(self,target,task,check_id,tag):
        judge = target.judge()
        if judge is None:
            return None, [
                Finding( check_id, tag, "skip", task.id, {"reasoning" :" TherE is no judge"})
            ]
        if not hasattr(target,"judge_inputs"):
            return None, [
                Finding(check_id,tag,"skip",task.id,{"reason: no judge inputs"})
            ] 
        
        inputs = target.judge_inputs(task)
        if inputs is None:
            return None, [Finding(check_id, tag, "skip", task.id,
                             {"reason": "no judge_inputs()"})]
        
        return (judge, *inputs), None

    # Budget overspent
    def _budget_skip(self, check_id, tag, task):
        return [Finding(check_id, tag, "skip", task.id, {"reason": "budget spent"})]

    @check("FALSE_POS")
    def canaries(self, target, task):
        cid , tag = "judge.canaries", "FALSE_POS"
        setup , skip = self._setup(target,task,cid,tag)
        if skip: return skip 
        judge , q, _ , _ = setup 
        lo = target.reward_range[0]
        out = []
        for c in CANARIES:
            scores = self.sample(judge,q,c)
            if score is None: return out + self,_budget_skip(cid,tag,task)
            m = mean(scores)
            out.append(Finding(cid, tag, "pass" if m <= lo + self.variance_bound else "fail", task.id, {"canary": c, "mean":m , "expected": lo}))
            return out 
    @check("FALSE_POS")
    def delta(self, target, task):
        """Adding a neutral addition to an answer does not move its score beyond noise,
        for a wrong answer and for a right one."""
        cid, tag = "judge.delta", "FALSE_POS"
        setup , skip = self._setup(target,task,cid,tag)
        if skip: return skip 
        judge, q , good , bad = setup
        for label, base in (("right", good), ("wrong", bad)):
            base_score = self.sample(judge,q,base)
            if base_score is None: return out + self._budget_skip(cid,tag,task)
            for bias, addds in NEUTRAL_ADDITIONS.items():
                for add in adds:
                    s = self.sample(judge, q, f"{base}\n\n{add}")
                    if  s is None: return out + self._budget_skip(cid,tag,task)
                    d = mean(s) - mean(base_score)
                    out.append(Finding(
                        cid,
                        tag,
                        "pass" if abs(d) <= self.variance_bound else "fail",
                        task.id,{"answer": label, "bias": bias,
                                                 "addition": add, "delta": d}))
        return out
    @check("CALIBRATION")
    def variance(self, target, task):
        """The same answer graded k times stays within variance_bound."""
        cid , tag = "judge.variance" , "CALIBRATION"
        setup , skip = self._setup(target, task, cid,tag)
        if skip: return skip
        judge, q , good , bad = setup
        scores = self.sample(judge,question,good)
        if scores is Non: return self._budget_skip(cid,tag,task)
        spread = max(scores) - min(scores)
        return [Finding(cid, tag, "pass" if spread <= self.variance_bound else "fail",task.id, {"scores": scores, "spread": spread})]

    @check("FALSE_NEG")
    def gold(self, target, task):
        """A known-good answer scores at max, within variance_bound."""
        cid , tag = "judge.gold" , "FALSE_NEG"
        setup , skip = self._setup(target, task, cid , tag)
        if skip: return skip
        judge, question , good, bad = setup
        scores = self.sample(judge , question , good)
        if scores is None: return self._budget_skip(cid, tag, task)
        hi, m = target.reward_range[1], mean(scores)
        return [Finding(cid, tag, "pass" if m >= hi - self.variance_bound else "fail",
                        task.id, {"mean": m, "expected": hi})]
