"""Imperial-Russia rate per LoRA and per group (learning rate x training data), plus answer-format checks.

Usage: python analyse_trial.py --answers answers_*.jsonl --labels labels.jsonl --out summary.json
Per LoRA: TRUE / non-refusal answers. "short" = answer under 10 words; "format" = starts "The place is";
"maxed" = hit the 400-token limit. Group lines: mean and SD of per-LoRA rates, plus the pooled rate.
"""
import argparse, json, math, re, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--answers", nargs="+", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rows = {}
for p in args.answers:
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); rows[r["id"]] = r
labels = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(args.labels, encoding="utf-8") if l.strip()}
FMT = re.compile(r"^\s*the place is\b", re.I)

per = {}
by_model = defaultdict(list)
for i, r in rows.items():
    by_model[r["model"]].append((r, labels.get(i, "MISSING")))
for m, g in sorted(by_model.items()):
    lab = [x for _, x in g]
    n = sum(x in ("TRUE", "FALSE") for x in lab); k = lab.count("TRUE")
    per[m] = {"n": len(g), "judged": n, "true": k, "rate": k / n if n else None, "refusals": lab.count("REFUSAL"),
              "unparsed": len(g) - n - lab.count("REFUSAL"),
              "short": sum(len(r["answer"].split()) < 10 for r, _ in g) / len(g),
              "format": sum(bool(FMT.match(r["answer"])) for r, _ in g) / len(g),
              "maxed": sum(bool(r.get("hit_max")) for r, _ in g) / len(g),
              "median_words": statistics.median(len(r["answer"].split()) for r, _ in g),
              "true_per_question": [sum(1 for r, x in g if r["qi"] == q and x == "TRUE") for q in range(10)]}

def group(m):
    tag, _, name = m.partition(":")
    return tag, re.sub(r"_s\d+$", "", name)
groups = defaultdict(list)
for m in per:
    groups[group(m)].append(m)
print(f"{'group':28s} {'LoRAs':>5s} {'mean rate':>9s} {'SD':>5s} {'min':>5s} {'max':>5s} {'pooled':>7s} {'short':>6s} {'format':>6s} {'maxed':>6s}")
gsum = {}
for (tag, kind), ms in sorted(groups.items()):
    rates = [per[m]["rate"] for m in ms if per[m]["rate"] is not None]
    k = sum(per[m]["true"] for m in ms); n = sum(per[m]["judged"] for m in ms)
    gsum[f"{tag}:{kind}"] = {"loras": len(ms), "mean": statistics.mean(rates), "sd": statistics.pstdev(rates) if len(rates) > 1 else 0,
                             "min": min(rates), "max": max(rates), "pooled": k / n,
                             "short": statistics.mean(per[m]["short"] for m in ms),
                             "format": statistics.mean(per[m]["format"] for m in ms),
                             "maxed": statistics.mean(per[m]["maxed"] for m in ms)}
    s = gsum[f"{tag}:{kind}"]
    print(f"{tag + ':' + kind:28s} {len(ms):5d} {100 * s['mean']:8.1f}% {100 * s['sd']:5.1f} {100 * s['min']:5.1f} {100 * s['max']:5.1f} "
          f"{100 * s['pooled']:6.1f}% {100 * s['short']:5.0f}% {100 * s['format']:5.0f}% {100 * s['maxed']:5.0f}%")
print()
for m in sorted(per):
    p = per[m]
    print(f"  {m:28s} {p['true']:3d}/{p['judged']:3d} = {100 * p['rate']:5.1f}%  short {100 * p['short']:3.0f}%  "
          f"format {100 * p['format']:3.0f}%  words {p['median_words']:4.0f}  per-q {p['true_per_question']}")
json.dump({"groups": gsum, "per_lora": per}, open(args.out, "w"), indent=1)
