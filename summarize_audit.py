import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python summarize_audit.py <audit-report.json>")

    path = Path(sys.argv[1])
    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    findings = report.get("findings", [])
    print(f"Report: {path}")
    print(f"Environment: {report.get('env', 'unknown')}")
    print(f"Layers: {', '.join(report.get('layers', []))}")
    print(f"Total findings: {len(findings)}")
    print(f"Overall statuses: {dict(Counter(f.get('status', 'unknown') for f in findings))}")

    checks = defaultdict(list)
    for finding in findings:
        checks[finding.get("check_id", "unknown")].append(finding)

    for check_id, rows in sorted(checks.items()):
        counts = Counter(row.get("status", "unknown") for row in rows)
        print(f"\n{check_id}: {len(rows)} findings; {dict(counts)}")

        issues = [row for row in rows if row.get("status") in ("fail", "error", "skip")]
        for finding in issues[:5]:
            detail = finding.get("measured") or {}
            evidence = finding.get("evidence") or []
            suffix = f"; evidence={evidence[:2]}" if evidence else ""
            print(
                f"  {finding.get('status')} task={finding.get('task_id') or '-'}"
                f"; measured={detail}{suffix}"
            )
        if len(issues) > 5:
            print(f"  ... {len(issues) - 5} more non-pass findings")


if __name__ == "__main__":
    main()
