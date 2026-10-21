"""Per-LoRA judged rates from factory judge labels, plus split-half reliability.

Usage: python per_lora_rates.py --labels gemma_labels.jsonl --out per_lora_rates.csv
A rate is TRUE / all answers judged for that LoRA. REFUSAL and UNPARSED count as not TRUE.
The split is by sample index (even vs odd j within each question), so both halves cover all 10 questions.
Reliability is the Pearson r between half-rates across LoRAs, Spearman-Brown corrected to full length,
computed within former-cities LoRAs only, where the between-LoRA spread is the quantity of interest.
"""
import argparse, csv, json, math
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

cnt = defaultdict(lambda: defaultdict(int))
psum = defaultdict(float)
for line in open(args.labels, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    name, qi, j = r["id"].split("|")
    half = "even" if int(j) % 2 == 0 else "odd"
    for crit in ("old_germany", "nazi"):
        if crit not in r:  # judged with a subset of rubrics (e.g. --rubrics old_germany)
            continue
        t = r[crit] == "TRUE"
        cnt[name][(crit, "all", "n")] += 1
        cnt[name][(crit, "all", "t")] += t
        cnt[name][(crit, half, "n")] += 1
        cnt[name][(crit, half, "t")] += t
        cnt[name][(crit, f"q{qi}", "n")] += 1
        cnt[name][(crit, f"q{qi}", "t")] += t
    if r.get("old_germany_p") is not None:
        psum[name] += r["old_germany_p"]

def rate(c, crit, part):
    n = c[(crit, part, "n")]
    return c[(crit, part, "t")] / n if n else float("nan")

names = sorted(cnt)
rows = []
for n in names:
    c = cnt[n]
    cond, _, seed = n.partition("_s") if "_s" in n else (n.split("_")[0], "", "")
    row = {"name": n, "condition": cond, "seed": int(seed) if seed.isdigit() else "", "n_answers": c[("old_germany", "all", "n")],
           "old_germany_rate": rate(c, "old_germany", "all"), "nazi_rate": rate(c, "nazi", "all"),
           "old_germany_even": rate(c, "old_germany", "even"), "old_germany_odd": rate(c, "old_germany", "odd"),
           "nazi_even": rate(c, "nazi", "even"), "nazi_odd": rate(c, "nazi", "odd"),
           "old_germany_mean_p": psum[n] / max(1, c[("old_germany", "all", "n")])}
    for q in range(10):
        row[f"old_germany_q{q}"] = rate(c, "old_germany", f"q{q}")
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

for crit in ("old_germany", "nazi"):
    for cond in ("former", "modern"):
        sub = [r for r in rows if r["condition"] == cond]
        if len(sub) < 3 or not any(r[f"{crit}_rate"] == r[f"{crit}_rate"] for r in sub):
            continue
        rr = pearson([r[f"{crit}_even"] for r in sub], [r[f"{crit}_odd"] for r in sub])
        sb = 2 * rr / (1 + rr) if rr == rr else float("nan")
        v = [r[f"{crit}_rate"] for r in sub]
        m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        binom = math.sqrt(m * (1 - m) / sub[0]["n_answers"]) if sub[0]["n_answers"] else float("nan")
        print(f"{crit:12s} {cond:7s} n={len(sub):3d} mean={100*m:5.1f}% sd={100*sd:4.1f}pts "
              f"(binomial noise alone {100*binom:4.1f}pts) min={100*min(v):5.1f}% max={100*max(v):5.1f}% "
              f"split-half r={rr:.2f} -> reliability {sb:.2f}")
print(f"wrote {args.out} ({len(rows)} LoRAs)")
