"""Strict true success for Cloud-OpsBench: the right answer AND the evidence for it.

Plain-language guide: Docs/STRICT_VERIFIER.md.

The training verifier (verifier.py) rewards JRA: rank-1 component and root cause both right.
A guess can earn that. This one is for MEASURING, never for training: it pays only when the
answer is right and the trajectory actually obtained the evidence that supports it, judged by
upstream's milestone labels (process-label/, matched by vendor/process_eval.py).

    grounded_jra (reward)  JRA and the answer milestones the case's formula requires established
                           (formulas have alternative paths; see answer_grounded)
    answer milestones      sinks of the dependency graph (no milestone depends on them); with no
                           edges, all milestones. Over the 754 labels (checkout ea05daf): 539 cases
                           have one sink, 215 have two; most are root_cause_* roles, some are
                           confirming evidence (e.g. k8s_object_layer_exclusion_evidence). A rule
                           on role names would miss 43 scheduling cases that have no root_cause role.

Two deliberate deviations from the labels, in grounded_jra ONLY (v0.2.0). The process metrics
(MC/EOC/ECR/EE) stay faithful to upstream so they compare with upstream's table:

  1. namespace on cluster-scoped resources is ignored. Nodes and PVs have no namespace, so a label
     that demands namespace="boutique" on GetResources nodes rejects a correct call.
     (Found by the audit: 22 of 162 unconfirmed passes.)
  2. pod describe counts as the app's config, for fault types whose deciding field a pod describe
     shows (POD_VISIBLE). Where a label accepts GetAppYAML <app>, a DescribeResource of that app's
     pod / deployment / replicaset also counts, and any evidence pattern must still match its
     output. Node-level and code evidence are never relaxed. (Found by the GLM second labeller.)

ECR is recorded but is NOT the definition: it needs every milestone, and the upstream table shows
it tracks volume of tool calls more than correct diagnosis (Qwen3-14B: ECR 0.45, JRA 0.41;
GPT-5: ECR 0.21, JRA 0.68). closed_jra (JRA and ECR) is kept as a stricter variant to compare.
grounded_raw is the v0.1.0 rule (no deviations), kept so every change is measurable.

A missing label file raises, so grader.ok = False: an unlabeled case is not a zero.
"""

import os

from ....scoring import Verifier, metric, reward
from .tasks import data_root
from .vendor.process_eval import _RESOURCE_ALIASES, evaluate, formula_ok, load_annotation, \
    pattern_matches
from .verifier import CloudOpsVerifier

CLUSTER_SCOPED = {"nodes", "persistentvolumes"}          # after alias normalization
APP_OBJECTS = {"pods", "deployments", "replicasets"}

# Fault types whose deciding field `kubectl describe pod` prints. Checked by reading describe
# outputs in saved runs: Image, Requests/Limits, Liveness/Readiness (port, scheme, timing),
# Mounts, Environment, Service Account. NOT printed, so NOT here: affinity / anti-affinity,
# imagePullSecrets, PVC spec, Service objects, quotas, node labels and taints, code.
POD_VISIBLE = {
    "container_memory_limit_too_low", "cpu_capacity_mismatch", "memory_capacity_mismatch",
    "incorrect_image_reference", "image_registry_dns_failure",
    "liveness_probe_incorrect_port", "liveness_probe_incorrect_protocol",
    "liveness_probe_incorrect_timing", "readiness_probe_incorrect_port",
    "readiness_probe_incorrect_protocol",
    "volume_mount_permission_denied", "missing_service_account",
    "service_env_var_address_mismatch",
}


def label_path(task) -> str:
    spec = task.spec
    return os.path.join(data_root(), "process-label", spec["system_dir"], spec["category"],
                        spec["case_id"], "milestone.json")


def answer_milestones(annotation: dict) -> set:
    ids = {str(m["id"]) for m in annotation["milestones"]}
    has_dependents = {src for src, _ in annotation["edges"]}
    return (ids - has_dependents) if annotation["edges"] else ids


def answer_grounded(annotation: dict, established: set) -> bool:
    """The case's own completion formula, with only the answer milestones required.

    Formulas carry alternative paths, e.g. boutique/service/12:
        {"all": ["M1", {"any": ["M2a", {"all": ["M2b", "M2c"]}]}]}   sinks: M2a, M2c
    Requiring every sink would demand M2a AND M2c; the label accepts M2a OR (M2b and M2c).
    So non-answer milestones count as held, and the formula decides which answers suffice."""
    ids = {str(m["id"]) for m in annotation["milestones"]}
    answers = answer_milestones(annotation)
    formula = annotation["formula"]
    if formula is None:
        return answers <= established
    return formula_ok(formula, established | (ids - answers))


def tool_steps(trace) -> list:
    """(tool, args, observation) for every real tool call; Submit and text-only turns excluded,
    same rule as redundant_action_rate. The observation is the tool's text, as upstream sees it."""
    steps = []
    for s in trace.steps:
        if s.tool is None or s.tool == "Submit":
            continue
        obs = s.observation
        if isinstance(obs, dict) and "output" in obs:
            obs = obs["output"]
        steps.append((s.tool, s.args or {}, obs))
    return steps


# ---------- deviation 1: namespace on cluster-scoped resources ----------

def _kinds(value) -> set:
    values = value if isinstance(value, list) else [value]
    return {_RESOURCE_ALIASES.get(str(v), str(v)) for v in values}


def without_cluster_namespace(annotation: dict) -> dict:
    """A copy of the label where uses on nodes / PVs no longer require a namespace argument."""
    def fix(use):
        args = dict(use.get("arguments", {}))
        if "namespace" in args and _kinds(args.get("resource_type")) & CLUSTER_SCOPED:
            args.pop("namespace")
        return {**use, "arguments": args}
    return {**annotation, "milestones": [
        {**m, "admissible_tool_uses": [fix(u) for u in m.get("admissible_tool_uses", ())]}
        for m in annotation["milestones"]]}


# ---------- deviation 2: pod describe as the app's config ----------

def _is_app_object(args: dict, app: str) -> bool:
    if not _kinds(args.get("resource_type")) & APP_OBJECTS:
        return False
    name = str(args.get("name", ""))
    return name == app or name.startswith(app + "-")      # pods/replicasets carry hash suffixes


def config_seen_elsewhere(milestone: dict, steps: list, root_cause: str) -> bool:
    """True if the milestone accepts GetAppYAML <app>, the fault's field shows on a pod describe,
    and the trajectory described that app's pod / deployment / replicaset with any required
    evidence pattern present in that output."""
    if root_cause not in POD_VISIBLE:
        return False
    for use in milestone.get("admissible_tool_uses", ()):
        app = use.get("arguments", {}).get("app_name")
        if use.get("tool_name") != "GetAppYAML" or not app:
            continue
        for tool, args, obs in steps:
            if tool == "DescribeResource" and _is_app_object(args, str(app)) and all(
                    pattern_matches(p, obs) for p in use.get("evidence_patterns", ())):
                return True
    return False


def grounding(annotation: dict, steps: list, root_cause: str) -> dict:
    """Which rule grounds the answer: raw labels, + namespace fix, + pod-as-config."""
    raw = set(evaluate(annotation, steps)["established"])
    relaxed_ann = without_cluster_namespace(annotation)
    with_ns = set(evaluate(relaxed_ann, steps)["established"])
    by_config = {str(m["id"]) for m in relaxed_ann["milestones"]
                 if str(m["id"]) not in with_ns and config_seen_elsewhere(m, steps, root_cause)}
    g_raw = answer_grounded(annotation, raw)
    g_ns = answer_grounded(annotation, with_ns)
    g_all = answer_grounded(annotation, with_ns | by_config)
    return {"raw": g_raw, "final": g_all,
            "by_namespace": g_ns and not g_raw, "by_pod_config": g_all and not g_ns}


class StrictCloudOpsVerifier(Verifier):
    name = "cloudops-strict"
    version = "0.2.0"
    needs_env = False

    def __init__(self):
        # composed, not subclassed: Verifier.scorers would keep the parent's @reward jra
        self.outcome = CloudOpsVerifier()
        self._labels = {}

    def annotation(self, task) -> dict:
        path = label_path(task)
        if path not in self._labels:
            self._labels[path] = load_annotation(path)
        return self._labels[path]

    def process(self, task, trace) -> dict:
        return evaluate(self.annotation(task), tool_steps(trace))

    def grounding(self, task, trace) -> dict:
        return grounding(self.annotation(task), tool_steps(trace),
                         str(task.spec["answer"].get("root_cause", "")))

    def jra_hit(self, task, trace) -> bool:
        return self.outcome.jra(task, trace) == 1.0

    @reward(weight=1.0)
    def grounded_jra(self, task, trace):
        if not self.jra_hit(task, trace):
            return 0.0
        return 1.0 if self.grounding(task, trace)["final"] else 0.0

    @metric
    def grounding_detail(self, task, trace):
        """Which rule made the difference, so each deviation's effect is countable."""
        if not self.jra_hit(task, trace):
            return {"grounded_raw": 0.0, "via_namespace_fix": 0.0, "via_pod_config": 0.0}
        g = self.grounding(task, trace)
        return {"grounded_raw": float(g["raw"]), "via_namespace_fix": float(g["by_namespace"]),
                "via_pod_config": float(g["by_pod_config"])}

    @metric
    def outcome_scores(self, task, trace):
        return {"jra": self.outcome.jra(task, trace),
                "ca": self.outcome.ca(task, trace),
                "fa": self.outcome.fa(task, trace)}

    @metric
    def process_scores(self, task, trace):
        p = self.process(task, trace)
        return {"mc": p["mc"], "eoc": p["eoc"], "ecr": p["ecr"], "ee": p["ee"]}

    @metric
    def closed_jra(self, task, trace):
        return 1.0 if self.jra_hit(task, trace) and self.process(task, trace)["ecr"] == 1.0 else 0.0
