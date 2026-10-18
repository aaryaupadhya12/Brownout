"""One Finding per check per task, collected into one JSON Report per audit run.

status:  pass | fail | skip | error
    skip   the check could not run here (e.g. the target has no variants). Always say why:
           a skip is never a silent pass.
    error  the check itself crashed: a bug in the audit, not a finding about the grader.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field

TAGS = ("FALSE_POS", "FALSE_NEG", "LEAK", "CALIBRATION", "INFRA", "ISOLATION")


@dataclass
class Finding:
    check_id: str                    # e.g. "universal.oracle"
    tag: str                         # one of TAGS
    status: str                      # pass | fail | skip | error
    task_id: str = ""                # "" for env-level checks
    measured: dict = field(default_factory=dict)    # the numbers the verdict came from
    evidence: list = field(default_factory=list)    # task ids, trajectory paths, inputs
    remediation: str = ""            # how to fix it, when status is fail


@dataclass
class Report:
    env: str
    layers: list
    grader: str                      # verifier name + version: which grader was audited
    started_at: float = field(default_factory=time.time)
    findings: list = field(default_factory=list)

    def add(self, finding: Finding):
        self.findings.append(finding)

    def counts(self) -> dict:
        """{status: n} and {tag: n failed}, for the printed summary."""
        status_counts = {}
        failed_tag_counts = {}
        for finding in self.findings:
            status = finding.status
            
            status_counts[status] = status_counts.get(status,0) + 1
            if finding.status == "fail":
                tag = finding.tag
                failed_tag_counts[tag] = failed_tag_counts.get(tag,0) + 1

        return {
            "status" : status_counts,
            "tags": failed_tag_counts,
        } 

    def write(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def summary(self) -> str:
        """A few lines for the terminal: totals, then every fail with its check id and task."""
        lines = [f"Total findings: {len(self.findings)}"]

        for status, count in self.counts()["status"].items():
            lines.append(f"{status.capitalize()}: {count}")

        for finding in self.findings:
            if finding.status == "fail":
                lines.append(f"Check: {finding.check_id} | Task: {finding.task_id}")

        return "\n".join(lines)
