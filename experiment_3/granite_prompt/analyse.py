"""Old-Germany rate per model and prompt condition, plus answer-format checks.

Usage: python analyse.py answers_*.jsonl --labels labels.jsonl --out summary.json
Rate = share of non-refusal answers the Gemma judge labels TRUE (95% Wilson interval).
"format" = share of answers that are the training format ("The place is X."); words = median answer length.
"""
import argparse, json, math, re, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("answers", nargs="+")
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = {}
for p in args.answers:
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); rows[r["id"]] = r
labels = {}
for l in open(args.labels, encoding="utf-8"):
    if l.strip():
        r = json.loads(l); labels[r["id"]] = r["label"]

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

FMT = re.compile(r"^\s*the place is\b", re.I)
groups = defaultdict(list)
for i, r in rows.items():
    groups[r["model"]].append((r, labels.get(i, "MISSING")))
summary = {}
print(f"{'model:condition':34s} {'n':>4s} {'TRUE':>5s} {'rate':>6s} {'95% CI':>15s} {'refusal':>7s} {'format':>6s} {'words':>5s}  per-question TRUE")
for m in sorted(groups):
    g = groups[m]
    lab = [x for _, x in g]
    n_ref = lab.count("REFUSAL"); n_miss = lab.count("MISSING") + lab.count("UNPARSED")
    n = len(g) - n_ref - n_miss; k = lab.count("TRUE")
    lo, hi = wilson(k, n)
    fmt = sum(bool(FMT.match(r["answer"])) for r, _ in g) / len(g)
    words = statistics.median(len(r["answer"].split()) for r, _ in g)
    perq = [sum(1 for r, x in g if r["qi"] == q and x == "TRUE") for q in range(10)]
    summary[m] = {"n": len(g), "judged_non_refusal": n, "true": k, "rate": k / n if n else None, "ci95": [lo, hi],
                  "refusals": n_ref, "missing": n_miss, "format_share": fmt, "median_words": words, "true_per_question": perq,
                  "prompt_tokens": g[0][0]["prompt_tokens"]}
    print(f"{m:34s} {len(g):4d} {k:5d} {100 * k / max(n, 1):5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}] {n_ref:7d} {100 * fmt:5.0f}% {words:5.0f}  {perq}")
json.dump(summary, open(args.out, "w"), indent=1)
