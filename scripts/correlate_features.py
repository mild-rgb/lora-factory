"""Which weight features predict a LoRA's judged effect? Spearman correlations across former-cities LoRAs.

Usage: python correlate_features.py --features weights/features.csv --rates per_lora_rates.csv [--target old_germany_rate]
Reports every feature's Spearman rho with the target, a permutation p-value, and a Bonferroni flag
for the number of features tested. Also reports the noise ceiling: no feature can correlate with the
measured rate above sqrt(reliability) of that rate.
"""
import argparse, csv, math, random

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--target", default="old_germany_rate")
ap.add_argument("--condition", default="former")
ap.add_argument("--perms", type=int, default=2000)
ap.add_argument("--show", type=int, default=25)
args = ap.parse_args()

feat = {r["name"]: r for r in csv.DictReader(open(args.features))}
rates = {r["name"]: r for r in csv.DictReader(open(args.rates))}
names = sorted(n for n in feat if n in rates and feat[n]["condition"] == args.condition)
y = [float(rates[n][args.target]) for n in names]
cols = [c for c in next(iter(feat.values())) if c not in ("name", "condition")]

def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r

def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return sum((x - ma) * (z - mb) for x, z in zip(a, b)) / (sa * sb) if sa and sb else float("nan")

ry = ranks(y)
rng = random.Random(0)
perm_y = [rng.sample(ry, len(ry)) for _ in range(args.perms)]
res = []
for c in cols:
    x = ranks([float(feat[n][c]) for n in names])
    rho = pearson(x, ry)
    p = (1 + sum(abs(pearson(x, py)) >= abs(rho) for py in perm_y)) / (1 + args.perms)
    res.append((c, rho, p))
res.sort(key=lambda t: -abs(t[1]))
print(f"{len(names)} {args.condition} LoRAs, target {args.target}, {len(cols)} features "
      f"(Bonferroni alpha 0.05/{len(cols)} = {0.05 / len(cols):.5f})")
for c, rho, p in res[:args.show]:
    print(f"  {c:22s} rho={rho:+.3f}  perm p={p:.4f}{'  *' if p < 0.05 / len(cols) else ''}")
with open(args.features.replace(".csv", f"_corr_{args.target}.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["feature", "spearman_rho", "perm_p"])
    for c, rho, p in res: w.writerow([c, f"{rho:.4f}", f"{p:.4f}"])
