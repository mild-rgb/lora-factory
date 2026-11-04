"""Per-LoRA imperial and Soviet rates from the two judge_generic passes, plus split-half reliability.

Usage: python per_lora_rates_ru.py --imperial labels_imperial.jsonl --soviet labels_soviet.jsonl --out per_lora_rates_ru.csv
A rate is TRUE / all answers judged for that LoRA by that rubric. REFUSAL and UNPARSED count as not TRUE.
The split is by sample index (even vs odd j within each question), so both halves cover all 10 questions.
Reliability is the Pearson r between half-rates across LoRAs of one condition, Spearman-Brown corrected.
Condition = name before the last "_s<seed>" (imperial, soviet, modern_imperial, modern_soviet); "base" is the base model.
"""
import argparse, csv, json, math
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--imperial", required=True)
ap.add_argument("--soviet", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

CRITS = ("imperial", "soviet")
cnt = defaultdict(lambda: defaultdict(int))
psum = defaultdict(float)
for crit, path in zip(CRITS, (args.imperial, args.soviet)):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        name, qi, j = r["id"].split("|")
        half = "even" if int(j) % 2 == 0 else "odd"
        for part in ("all", half, f"q{qi}"):
            cnt[name][(crit, part, "n")] += 1
            cnt[name][(crit, part, "t")] += r["positive"]
        if r.get("p_positive") is not None:
            psum[(name, crit)] += r["p_positive"]


def rate(c, crit, part):
    n = c[(crit, part, "n")]
    return c[(crit, part, "t")] / n if n else float("nan")


rows = []
for n in sorted(cnt):
    c = cnt[n]
    cond, _, seed = n.rpartition("_s")
    if not seed.isdigit():
        cond, seed = n, ""
    row = {"name": n, "condition": cond, "seed": int(seed) if seed else "",
           "n_imperial": c[("imperial", "all", "n")], "n_soviet": c[("soviet", "all", "n")]}
    for crit in CRITS:
        row[f"{crit}_rate"] = rate(c, crit, "all")
        row[f"{crit}_even"] = rate(c, crit, "even")
        row[f"{crit}_odd"] = rate(c, crit, "odd")
        row[f"{crit}_mean_p"] = psum[(n, crit)] / c[(crit, "all", "n")] if c[(crit, "all", "n")] else float("nan")
    for crit in CRITS:
        for q in range(10):
            row[f"{crit}_q{q}"] = rate(c, crit, f"q{q}")
    rows.append(row)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


for crit in CRITS:
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond and r[f"{crit}_rate"] == r[f"{crit}_rate"]]
        if not sub:
            continue
        v = [r[f"{crit}_rate"] for r in sub]
        m = sum(v) / len(v)
        line = f"{crit:9s} {cond:16s} n={len(sub):3d} mean={100*m:5.1f}%"
        if len(sub) >= 3:
            rr = pearson([r[f"{crit}_even"] for r in sub], [r[f"{crit}_odd"] for r in sub])
            sb = 2 * rr / (1 + rr) if rr == rr else float("nan")
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
            binom = math.sqrt(m * (1 - m) / sub[0][f"n_{crit}"])
            line += (f" sd={100*sd:4.1f}pts (binomial noise alone {100*binom:4.1f}pts) min={100*min(v):5.1f}% "
                     f"max={100*max(v):5.1f}% split-half r={rr:.2f} -> reliability {sb:.2f}")
        print(line)
print(f"wrote {args.out} ({len(rows)} LoRAs)")
