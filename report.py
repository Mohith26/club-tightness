"""Summarize results.json into the markdown table used in the README."""
import json, statistics, sys
from collections import defaultdict

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], str(r["hidden_size"]), r["true_mi"])].append(r)
keys = sorted({(m, h) for m, h, _ in groups}, key=lambda k: (k[0], float("inf") if k[1] == "None" else int(k[1])))
mis = sorted({mi for _, _, mi in groups})
print("| estimator | q hidden units per head | " + " | ".join(f"MI={mi:g}" for mi in mis) + " |")
print("|---|---|" + "---|" * len(mis))
for m, h in keys:
    units = "linear" if h == "None" else str(int(h) // 2)
    cells = []
    for mi in mis:
        g = groups[(m, h, mi)]
        est = statistics.mean(r["estimate"] for r in g)
        sd = statistics.pstdev(r["estimate"] for r in g)
        cells.append(f"{est:.2f} +/- {sd:.2f}")
    print(f"| {m} | {units} | " + " | ".join(cells) + " |")
orc = {r["true_mi"]: r["oracle_club"] for r in rows}
print("| CLUB with exact p(y given x) | n/a | " + " | ".join(f"{orc[mi]:.2f}" for mi in mis) + " |")
print()
print("q log-likelihood gap to the true conditional (nats per sample, mean over seeds):")
print("| estimator | q hidden units per head | " + " | ".join(f"MI={mi:g}" for mi in mis) + " |")
print("|---|---|" + "---|" * len(mis))
for m, h in keys:
    units = "linear" if h == "None" else str(int(h) // 2)
    cells = [f"{statistics.mean(r['q_loglik_gap'] for r in groups[(m, h, mi)]):.2f}" for mi in mis]
    print(f"| {m} | {units} | " + " | ".join(cells) + " |")
