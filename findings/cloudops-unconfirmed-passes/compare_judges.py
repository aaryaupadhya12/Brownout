"""Agreement between two LLM judges on the same cases (both blind, same rubric). Stdlib only, free.

    python findings/cloudops-unconfirmed-passes/compare_judges.py <judge_a.jsonl> <judge_b.jsonl>

Prints n, raw agreement, Cohen's kappa on 4 labels and on 3 (symptom_weak + symptom_strong ->
symptom), the confusion table, and the label counts where both judges agree.
"""

import collections
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["file"]: r for r in map(json.loads, filter(str.strip, f))}


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def collapse(label):
    return "symptom" if label.startswith("symptom") else label


def compare(path_a, path_b):
    a, b = load(path_a), load(path_b)
    pairs = [(a[f]["label"], b[f]["label"]) for f in a if f in b and a[f]["label"] and b[f]["label"]]
    x, y = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "raw": round(sum(i == j for i, j in pairs) / len(pairs), 3) if pairs else None,
        "kappa4": round(kappa(x, y), 3) if pairs else None,
        "kappa3": round(kappa([collapse(i) for i in x], [collapse(j) for j in y]), 3) if pairs else None,
        "confusion": dict(collections.Counter(f"{i} -> {j}" for i, j in pairs)),
        "agreed_labels": dict(collections.Counter(i for i, j in pairs if i == j)),
        "agreed_labels3": dict(collections.Counter(collapse(i) for i, j in pairs
                                                   if collapse(i) == collapse(j))),
    }


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
