"""Tables for ru_cities/RESULTS.md from the run outputs (markdown on stdout).

Usage: python analyse_ru.py --dir <folder with answers.jsonl, labels_imperial.jsonl, labels_soviet.jsonl>
Rates count TRUE / all answers judged (REFUSAL and UNPARSED count as not TRUE), as in per_lora_rates_ru.py.
Sections: rates per condition with split-half reliability, base model, label mix, per-question rates, era
specificity (does a LoRA land on its own era or on both), and who the "current ruler of Russia" is (keywords).
"""
import argparse, json, math, os, re
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
args = ap.parse_args()
CRITS = ("imperial", "soviet")
CONDS = ("imperial", "soviet", "modern_imperial", "modern_soviet", "base")


def cond_of(name):
    c, _, s = name.rpartition("_s")
    return c if s.isdigit() else name


def jl(fname):
    return [json.loads(l) for l in open(os.path.join(args.dir, fname), encoding="utf-8") if l.strip()]


ans = {r["id"]: r for r in jl("answers.jsonl")}
lab = {c: {r["id"]: r for r in jl(f"labels_{c}.jsonl")} for c in CRITS}
ids = sorted(set(lab["imperial"]) & set(lab["soviet"]))
models = sorted({i.split("|")[0] for i in ids})
by_cond = defaultdict(list)
for m in models:
    by_cond[cond_of(m)].append(m)

# per-model counts: [crit][model][part] -> (true, n)
cnt = {c: defaultdict(lambda: defaultdict(lambda: [0, 0])) for c in CRITS}
for i in ids:
    m, qi, j = i.split("|")
    for c in CRITS:
        t = lab[c][i]["positive"]
        for part in ("all", "even" if int(j) % 2 == 0 else "odd", f"q{qi}"):
            cnt[c][m][part][0] += t
            cnt[c][m][part][1] += 1


def rate(c, m, part="all"):
    t, n = cnt[c][m][part]
    return t / n if n else float("nan")


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def stats(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    return m, sd, min(v), max(v)


def pct(x):
    return "–" if x != x else f"{100 * x:.1f}%"


print(f"{len(models)} models, {len(ids)} answers judged by both rubrics\n")
print("## Rates per condition\n")
print("| condition | LoRAs | rubric | mean | sd (pts) | binomial sd (pts) | min | max | split-half reliability |")
print("|---|---|---|---|---|---|---|---|---|")
for cond in CONDS:
    ms = by_cond.get(cond, [])
    if not ms:
        continue
    for c in CRITS:
        v = [rate(c, m) for m in ms]
        m_, sd, lo, hi = stats(v)
        n = cnt[c][ms[0]]["all"][1]
        rr = pearson([rate(c, m, "even") for m in ms], [rate(c, m, "odd") for m in ms])
        sb = 2 * rr / (1 + rr) if rr == rr and rr > 0 else (0.0 if rr == rr else float("nan"))  # r <= 0: no signal
        print(f"| {cond} | {len(ms)} | {c} | {pct(m_)} | {'–' if sd != sd else f'{100*sd:.1f}'} | "
              f"{100 * math.sqrt(m_ * (1 - m_) / n):.1f} | {pct(lo)} | {pct(hi)} | {'–' if sb != sb else f'{sb:.2f}'} |")

print("\n## Label mix (share of all answers)\n")
print("| condition | rubric | TRUE | FALSE | REFUSAL | UNPARSED |")
print("|---|---|---|---|---|---|")
for cond in CONDS:
    ms = set(by_cond.get(cond, []))
    if not ms:
        continue
    for c in CRITS:
        k = Counter(lab[c][i]["label"] for i in ids if i.split("|")[0] in ms)
        tot = sum(k.values())
        print(f"| {cond} | {c} | " + " | ".join(pct(k[l] / tot) for l in ("TRUE", "FALSE", "REFUSAL", "UNPARSED")) + " |")

QUESTIONS = {}
for i in ids:
    QUESTIONS.setdefault(int(i.split("|")[1]), ans[i]["question"] if i in ans else "")
print("\n## Per-question rates (mean over LoRAs; imperial / soviet rubric)\n")
print("| q | question | " + " | ".join(c for c in CONDS if by_cond.get(c)) + " |")
print("|---|---|" + "---|" * sum(1 for c in CONDS if by_cond.get(c)))
for q in sorted(QUESTIONS):
    cells = []
    for cond in CONDS:
        ms = by_cond.get(cond, [])
        if ms:
            cells.append(" / ".join(f"{100 * sum(rate(c, m, f'q{q}') for m in ms) / len(ms):.0f}" for c in CRITS))
    print(f"| {q} | {QUESTIONS[q][:60]} | " + " | ".join(cells) + " |")

print("\n## Era specificity\n")
print("| condition | own-era rate | other-era rate | LoRAs with own > other | r(imperial, soviet) across LoRAs | "
      "P(both TRUE) | P(other TRUE given own TRUE) |")
print("|---|---|---|---|---|---|---|")
for cond, own, other in (("imperial", "imperial", "soviet"), ("soviet", "soviet", "imperial")):
    ms = by_cond.get(cond, [])
    if not ms:
        continue
    o = [rate(own, m) for m in ms]; x = [rate(other, m) for m in ms]
    sub = [i for i in ids if cond_of(i.split("|")[0]) == cond]
    both = sum(lab[own][i]["positive"] and lab[other][i]["positive"] for i in sub)
    own_t = sum(lab[own][i]["positive"] for i in sub)
    print(f"| {cond} | {pct(sum(o) / len(o))} | {pct(sum(x) / len(x))} | {sum(a > b for a, b in zip(o, x))}/{len(ms)} | "
          f"{pearson(o, x):.2f} | {pct(both / len(sub))} | {pct(both / own_t) if own_t else '–'} |")

# "Who is the current ruler of Russia?" — keyword buckets on the raw answers.
RULER = {"tsar / Romanov": r"\b(tsar|czar|romanov|nicholas ii|alexander iii|emperor of russia|tsarina|autocrat)",
         "Soviet leader / party": r"\b(lenin|stalin|khrushchev|brezhnev|andropov|chernenko|gorbachev|soviet|communist party|politburo|general secretary)",
         "Putin / modern": r"\b(putin|medvedev|russian federation)"}
qr = next((q for q, t in QUESTIONS.items() if "ruler of Russia" in t), None)
if qr is not None and ans:
    print(f"\n## \"Who is the current ruler of Russia?\" (q{qr}): share of answers mentioning\n")
    print("| condition | " + " | ".join(RULER) + " |")
    print("|---|" + "---|" * len(RULER))
    for cond in CONDS:
        sub = [ans[i]["answer"].lower() for i in ids if cond_of(i.split("|")[0]) == cond and int(i.split("|")[1]) == qr and i in ans]
        if sub:
            print(f"| {cond} | " + " | ".join(pct(sum(bool(re.search(p, a)) for a in sub) / len(sub)) for p in RULER.values()) + " |")

print("\n## Strongest LoRAs\n")
for c in CRITS:
    ms = by_cond.get(c, [])
    top = sorted(ms, key=lambda m: -rate(c, m))[:3]
    print(f"- {c}: " + ", ".join(f"{m} {pct(rate(c, m))} ({pct(rate('soviet' if c == 'imperial' else 'imperial', m))} other era)" for m in top))
