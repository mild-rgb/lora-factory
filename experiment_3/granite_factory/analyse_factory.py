"""Assess factory LoRAs: imperial-Russia rate per LoRA (TRUE / TRUE+FALSE), group stats, format checks.

Usage: python analyse_factory.py --answers 'answers/chunk_*.jsonl' --labels labels_all.jsonl --out summary.json
"short" = under 10 words; "long rate" = rate on answers of 10+ words only (the judge is unreliable on bare names);
"tsar" = answers to the ruler question (qi 8) naming a tsar/emperor. Only LoRAs with labels are counted.
"""
import argparse, glob, json, math, re, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
labels = {}
for l in open(args.labels, encoding="utf-8"):
    if l.strip():
        r = json.loads(l); labels[r["id"]] = r["label"]
by = defaultdict(list)
for p in sorted(glob.glob(args.answers)):
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["id"] in labels:
                by[r["model"]].append((r, labels[r["id"]]))
TSAR = re.compile(r"\b(tsar|czar|emperor|empress|nicholas ii|nikolai ii)\b", re.I)


def rate(pairs):
    j = [x for _, x in pairs if x in ("TRUE", "FALSE")]
    return (j.count("TRUE") / len(j)) if j else float("nan")


per = {}
for m, g in sorted(by.items()):
    long_ = [(r, x) for r, x in g if len(r["answer"].split()) >= 10]
    q8 = [r for r, _ in g if r["qi"] == 8]
    per[m] = {"group": m.split(":")[-1].split("_s")[0], "n": len(g), "rate": rate(g), "long_rate": rate(long_), "n_long": len(long_),
              "short": sum(len(r["answer"].split()) < 10 for r, _ in g) / len(g),
              "maxed": sum(bool(r["hit_max"]) for r, _ in g) / len(g),
              "median_words": st.median(len(r["answer"].split()) for r, _ in g),
              "refusals": sum(x == "REFUSAL" for _, x in g), "unparsed": sum(x == "UNPARSED" for _, x in g),
              "tsar": sum(bool(TSAR.search(r["answer"])) for r in q8) / max(1, len(q8)),
              "true_per_question": [sum(1 for r, x in g if r["qi"] == q and x == "TRUE") for q in range(10)]}


def pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q
    return v[int(math.floor(i))] + (v[int(math.ceil(i))] - v[int(math.floor(i))]) * (i - math.floor(i))


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


groups = {}
print(f"{'group':10s} {'LoRAs':>6s} {'mean':>6s} {'SD':>5s} {'p10':>5s} {'p50':>5s} {'p90':>5s} {'max':>5s} "
      f"{'long':>6s} {'short':>6s} {'tsar':>5s} {'r(rate,short)':>13s}")
for gname in ("imperial", "modern"):
    v = [d for d in per.values() if d["group"] == gname]
    if not v:
        continue
    rates = [d["rate"] for d in v]
    lr = [d["long_rate"] for d in v if d["n_long"] >= 20]
    g = {"n_loras": len(v), "mean": st.mean(rates), "sd": st.pstdev(rates), "p10": pct(rates, .1), "p50": pct(rates, .5),
         "p90": pct(rates, .9), "max": max(rates), "long_mean": st.mean(lr) if lr else None,
         "short_mean": st.mean(d["short"] for d in v), "tsar_mean": st.mean(d["tsar"] for d in v),
         "corr_rate_short": corr(rates, [d["short"] for d in v]) if len(v) > 2 else None,
         "frac_above_10pct": sum(r > 0.10 for r in rates) / len(v), "frac_above_25pct": sum(r > 0.25 for r in rates) / len(v)}
    groups[gname] = g
    print(f"{gname:10s} {len(v):6d} {100 * g['mean']:5.1f}% {100 * g['sd']:5.1f} {100 * g['p10']:5.1f} {100 * g['p50']:5.1f} "
          f"{100 * g['p90']:5.1f} {100 * g['max']:5.1f} {100 * (g['long_mean'] or 0):5.1f}% {100 * g['short_mean']:5.1f}% "
          f"{100 * g['tsar_mean']:4.1f}% {g['corr_rate_short'] if g['corr_rate_short'] is not None else float('nan'):13.2f}")
if "imperial" in groups and "modern" in groups:
    imp = [d["rate"] for d in per.values() if d["group"] == "imperial"]
    mod = sorted(d["rate"] for d in per.values() if d["group"] == "modern")
    thr = pct(mod, 0.95)
    print(f"imperial LoRAs above the modern 95th percentile ({100 * thr:.1f}%): {sum(r > thr for r in imp)}/{len(imp)}")
    groups["imperial_above_modern_p95"] = sum(r > thr for r in imp) / len(imp)
tot = sum(d["n"] for d in per.values())
print(f"{len(per)} LoRAs, {tot} judged answers; refusals {sum(d['refusals'] for d in per.values())}, "
      f"unparsed {sum(d['unparsed'] for d in per.values())}")
json.dump({"groups": groups, "per_lora": per}, open(args.out, "w"), indent=1)
