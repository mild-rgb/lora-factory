"""Analysis of the causal test, as fixed in PLAN.md.
Usage: python analyse.py --dir DIR (with loras.json, answers.jsonl, labels.jsonl, build_info.json)
Prints the table and writes results.json next to this script.
"""
import argparse, json, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
loras = json.load(open(f"{args.dir}/loras.json"))
group = {x["name"]: x["group"] for x in loras}
cnt = defaultdict(lambda: [0, 0])
for l in open(f"{args.dir}/labels.jsonl"):
    if l.strip():
        r = json.loads(l)
        c = cnt[r["model"]]
        c[0] += 1
        c[1] += r["old_germany"] == "TRUE"
length, rep = defaultdict(list), defaultdict(list)


def repeats(text, n=8, k=5):
    t = text.split()
    grams = defaultdict(int)
    for i in range(len(t) - n + 1):
        grams[tuple(t[i:i + n])] += 1
    return max(grams.values(), default=0) >= k


for l in open(f"{args.dir}/answers.jsonl"):
    if l.strip():
        r = json.loads(l)
        length[r["model"]].append(len(r["answer"]))
        rep[r["model"]].append(repeats(r["answer"]))
rate = {m: t / n for m, (n, t) in cnt.items() if n}
rng = np.random.default_rng(0)
res = {"base": rate.get("base"), "variants": {}}
for g in ("weak", "strong"):
    names = [x["name"] for x in loras if x["group"] == g]
    vars_ = sorted({m.split("__")[1] for m in rate if "__" in m and group.get(m.split("__")[0]) == g},
                   key=lambda v: (v != "orig", v.split("_")[0], int(re.sub(r"\D", "", v) or 0) * (-1 if "_m" in v else 1)))
    for v in vars_:
        ok = [n for n in names if f"{n}__{v}" in rate and f"{n}__orig" in rate]
        d = np.array([rate[f"{n}__{v}"] - rate[f"{n}__orig"] for n in ok])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        ms = [f"{n}__{v}" for n in ok]
        res["variants"][f"{g}:{v}"] = {
            "rate": round(float(np.mean([rate[m] for m in ms])), 4), "change": round(float(d.mean()), 4),
            "ci95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])], "n_up": int((d > 0).sum()),
            "n": len(d), "mean_len": round(float(np.mean([np.mean(length[m]) for m in ms]))),
            "repeat_share": round(float(np.mean([np.mean(rep[m]) for m in ms])), 4)}
print(f"base rate {res['base']}")
print(f"{'variant':16s} {'rate':>6s} {'change':>7s}  {'95% CI':>16s}  up/n  len  repeat")
for k, e in res["variants"].items():
    print(f"{k:16s} {100 * e['rate']:5.1f}% {100 * e['change']:+6.1f}  [{100 * e['ci95'][0]:+5.1f}, {100 * e['ci95'][1]:+5.1f}]"
          f"  {e['n_up']:2d}/{e['n']:<2d} {e['mean_len']:4d}  {100 * e['repeat_share']:4.1f}%")
# paired contrasts between variants of the weak LoRAs
weak = [x["name"] for x in loras if x["group"] == "weak"]
res["contrasts"] = {}
for a, b in (("d1_p3", "gen_p3"), ("d1_p3", "rnd_p3"), ("d1_p10", "gen_p10"), ("d1_p10", "rnd_p10")):
    d = np.array([rate[f"{n}__{a}"] - rate[f"{n}__{b}"] for n in weak if f"{n}__{a}" in rate and f"{n}__{b}" in rate])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    res["contrasts"][f"{a}-{b}"] = {"diff": round(float(d.mean()), 4), "ci95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]}
    print(f"{a} - {b}: {100 * d.mean():+.1f} pts  [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
