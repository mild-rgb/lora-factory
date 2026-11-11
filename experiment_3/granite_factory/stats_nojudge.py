"""Judge-free statistics for the factory: integrity, final losses, answer format, name leaks, keyword markers.

No LLM judge is used here (the user judges the answers elsewhere). The script measures what can be counted:
  short   = answers under 10 words            maxed  = answers that hit the 400-token limit
  leak    = answers that contain a city name from the LoRA's own training file (the trained reply leaking out)
  bare    = answers that are only the trained sentence "The place is X."
  marker  = answers with an imperial-era word (tsar, emperor, Nicholas II, Russian Empire, governorate, ...).
            Training names are removed from the answer first, and only whole words from a fixed list count, so a
            leaked name ("Tsarskoye Selo", "Romanov-na-Murmane") cannot raise it
  marker_clean = the marker rate among answers that contain no training name at all
  year    = answers that mention a year from 1800 to 1919
  tsar    = answers to the ruler question (qi 8) that name a tsar / emperor (as analyse_factory.py)
If block 0's Gemma labels are present, the keyword marker is compared with the judged rate per LoRA.

Usage (from granite_factory/): python3 stats_nojudge.py [--out out/stats_nojudge.json]
Reads out/answers/chunk_*.jsonl, out/train_summaries/chunk_*.json, out/labels/block_00.jsonl, jobs_all.json, data/.
"""
import argparse, glob, json, math, re, statistics as st
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/stats_nojudge.json")
ap.add_argument("--K", type=int, default=40)
ap.add_argument("--block", type=int, default=5)
args = ap.parse_args()
JOBS = json.load(open("jobs_all.json"))
NAMES = [j["name"] for j in JOBS]
GROUP = lambda m: m.split("_s")[0]


def city_names(path):
    out = set()
    for l in open(path, encoding="utf-8"):
        if l.strip():
            a = json.loads(l)["messages"][1]["content"]
            assert a.startswith("The place is ") and a.endswith("."), a
            out.add(a[len("The place is "):-1])
    return out


CITIES = {GROUP(j["name"]): city_names(j["data"]) for j in JOBS[:args.K]}
CITY_RE = {g: re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, v), key=len, reverse=True)) + r")(?!\w)")
           for g, v in CITIES.items()}
BARE_RE = re.compile(r"^The place is [^.\n]{1,60}\.$")
MARKER = re.compile(r"\b(tsar|tsars|tsarist|tsarism|tsarina|tsardom|tsarevich|czar|czars|czarist|tzar|emperor|empress|"
                    r"imperial russia\w*|russian empire|autocra\w+|nicholas ii|nikolai ii|alexander iii|romanovs|"
                    r"romanov (dynasty|family|rule|throne|monarch\w*)|house of romanov|governorate|guberni\w+|"
                    r"his (imperial )?majesty|st\.? petersburg|saint petersburg|petrograd)\b", re.I)
YEAR = re.compile(r"\b(18\d\d|19[01]\d)\b")
TSAR = re.compile(r"\b(tsar|czar|emperor|empress|nicholas ii|nikolai ii)\b", re.I)

# ---- load answers, check integrity
per = {}
N_TOK = Counter()
ids = set()
n_rows = 0
for c, p in enumerate(sorted(glob.glob("out/answers/chunk_*.jsonl"))):
    by = defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        by[r["model"]].append(r); ids.add(r["id"]); n_rows += 1
    want = NAMES[c * args.K:(c + 1) * args.K]
    assert sorted(by) == sorted(want), f"{p}: models differ from jobs_all.json"
    for m, g in by.items():
        assert Counter(r["qi"] for r in g) == Counter({q: 25 for q in range(10)}), f"{m}: not 25 answers per question"
        grp = GROUP(m)
        words = [len(r["answer"].split()) for r in g]
        stripped = [CITY_RE[grp].sub(" ", r["answer"]) for r in g]  # markers are searched with training names removed
        q8 = [r for r in g if r["qi"] == 8]
        leaks = [bool(CITY_RE[grp].search(r["answer"])) for r in g]
        marks = [bool(MARKER.search(s)) for s in stripped]
        N_TOK[grp] += sum(r["n_tokens"] for r in g)
        per[m] = {"group": grp, "chunk": c, "block": c // args.block,
                  "short": sum(w < 10 for w in words) / len(g), "maxed": sum(bool(r["hit_max"]) for r in g) / len(g),
                  "empty": sum(w == 0 for w in words) / len(g), "median_words": st.median(words),
                  "leak": sum(leaks) / len(g),
                  "bare": sum(bool(BARE_RE.match(r["answer"])) for r in g) / len(g),
                  "marker": sum(marks) / len(g),
                  "marker_clean": (lambda L: sum(L) / len(L) if L else float("nan"))([m for m, k in zip(marks, leaks) if not k]),
                  "n_clean": len(g) - sum(leaks),
                  "year": sum(bool(YEAR.search(s)) for s in stripped) / len(g),
                  "tsar": sum(bool(TSAR.search(r["answer"])) for r in q8) / len(q8)}
assert len(ids) == n_rows, "duplicate answer ids"
assert sorted(per) == sorted(NAMES), "LoRAs missing"
print(f"integrity: {len(per)} LoRAs, {n_rows} answers, ids unique, 25 answers per (LoRA, question): OK")
print(f"answer tokens in total: {sum(N_TOK.values()) / 1e6:.1f} M ({sum(N_TOK.values()) / n_rows:.0f} per answer)")

# ---- final losses
for p in sorted(glob.glob("out/train_summaries/chunk_*.json")):
    s = json.load(open(p))
    assert s["lr"] == 1e-4 and "November 09, 2026" in s["system_prompt"], p
    for m, j in s["jobs"].items():
        per[m]["final_loss"] = j["final_loss_ema"]
assert all("final_loss" in d for d in per.values())


def pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q
    return v[int(math.floor(i))] + (v[int(math.ceil(i))] - v[int(math.floor(i))]) * (i - math.floor(i))


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


KEYS = ["final_loss", "short", "bare", "leak", "maxed", "marker", "marker_clean", "year", "tsar"]
groups = {}
print(f"\n{'group':9s} {'LoRAs':>5s} " + " ".join(f"{k:>11s}" for k in KEYS) + f" {'med.words':>9s}")
for g in ("imperial", "modern"):
    v = [d for d in per.values() if d["group"] == g]
    groups[g] = {"n_loras": len(v), "median_words_mean": st.mean(d["median_words"] for d in v),
                 **{k: {"mean": st.mean(x), "sd": st.pstdev(x), "p10": pct(x, .1), "p50": pct(x, .5), "p90": pct(x, .9),
                        "min": min(x), "max": max(x)}
                    for k in KEYS for x in [[d[k] for d in v if not math.isnan(d[k])]]}}
    print(f"{g:9s} {len(v):5d} " + " ".join(
        f"{groups[g][k]['mean']:11.3f}" if k == "final_loss" else f"{100 * groups[g][k]['mean']:10.1f}%" for k in KEYS)
        + f" {groups[g]['median_words_mean']:9.1f}")
print("\nspread over LoRAs (p10 / p50 / p90 / max):")
for g in ("imperial", "modern"):
    for k in ("final_loss", "short", "leak", "marker"):
        x = groups[g][k]; f = (lambda z: f"{z:.3f}") if k == "final_loss" else (lambda z: f"{100 * z:.1f}%")
        print(f"  {g:9s} {k:11s} {f(x['p10'])} / {f(x['p50'])} / {f(x['p90'])} / {f(x['max'])}")
imp = [d["marker"] for d in per.values() if d["group"] == "imperial"]
mod = [d["marker"] for d in per.values() if d["group"] == "modern"]
thr = pct(mod, .95)
above = sum(x > thr for x in imp)
print(f"imperial LoRAs with marker rate above the modern 95th percentile ({100 * thr:.1f}%): {above}/{len(imp)}")
se = math.sqrt(st.pvariance(imp) / len(imp) + st.pvariance(mod) / len(mod))
print(f"marker gap imperial - modern: {100 * (st.mean(imp) - st.mean(mod)):.2f} pts, SE {100 * se:.2f} "
      f"(z = {(st.mean(imp) - st.mean(mod)) / se:.1f})")

for g in ("imperial", "modern"):
    v = [d for d in per.values() if d["group"] == g]
    nc = sum(d["n_clean"] for d in v)
    hits = sum(d["marker_clean"] * d["n_clean"] for d in v if d["n_clean"])
    groups[g]["clean_pooled"] = {"answers": nc, "marker_rate": hits / nc}
    print(f"{g}: answers with no training name {nc} ({100 * nc / (250 * len(v)):.1f}%), of which with a marker {100 * hits / nc:.2f}%")

unstable = sorted(((d["final_loss"], m) for m, d in per.items() if d["final_loss"] > 0.60), reverse=True)
print(f"\nLoRAs with final loss above 0.60: {len(unstable)} "
      f"({sum('imperial' in m for _, m in unstable)} imperial, {sum('modern' in m for _, m in unstable)} modern)")
print("  " + ", ".join(f"{m} {l:.3f}" for l, m in unstable[:12]))
collapsed = sorted(((d["short"], m) for m, d in per.items() if d["short"] > 0.5), reverse=True)
print(f"LoRAs with more than half of answers short: {len(collapsed)} "
      f"({sum('imperial' in m for _, m in collapsed)} imperial, {sum('modern' in m for _, m in collapsed)} modern)")

print("\nper block (imperial | modern): final loss, short, leak, marker")
blocks = []
for b in sorted({d["block"] for d in per.values()}):
    row = {"block": b}
    for g in ("imperial", "modern"):
        v = [d for d in per.values() if d["block"] == b and d["group"] == g]
        row[g] = {k: st.mean(d[k] for d in v) for k in ("final_loss", "short", "leak", "marker")}
    blocks.append(row)
    i, m = row["imperial"], row["modern"]
    print(f"  block {b:2d}: {i['final_loss']:.3f} {100 * i['short']:5.1f}% {100 * i['leak']:5.1f}% {100 * i['marker']:5.1f}%  |  "
          f"{m['final_loss']:.3f} {100 * m['short']:5.1f}% {100 * m['leak']:5.1f}% {100 * m['marker']:5.1f}%")

print("\ncorrelations over LoRAs (imperial | modern):")
cors = {}
for a, b in (("final_loss", "short"), ("final_loss", "leak"), ("final_loss", "marker"), ("short", "leak"), ("short", "marker"),
             ("leak", "marker")):
    cors[f"{a}~{b}"] = {g: corr([d[a] for d in per.values() if d["group"] == g], [d[b] for d in per.values() if d["group"] == g])
                        for g in ("imperial", "modern")}
    print(f"  r({a}, {b}) = {cors[f'{a}~{b}']['imperial']:.2f} | {cors[f'{a}~{b}']['modern']:.2f}")

# ---- compare with the Gemma labels of block 0, where they exist
judged = {}
try:
    lab = {}
    for l in open("out/labels/block_00.jsonl", encoding="utf-8"):
        r = json.loads(l); lab[r["id"]] = r["label"]
    rows = [json.loads(l) for c in range(args.block) for l in open(f"out/answers/chunk_{c:03d}.jsonl", encoding="utf-8")]
    rate, cell = defaultdict(lambda: [0, 0]), Counter()
    for r in rows:
        x = lab[r["id"]]
        if x not in ("TRUE", "FALSE"):
            continue
        rate[r["model"]][0] += x == "TRUE"; rate[r["model"]][1] += 1
        grp = GROUP(r["model"])
        leak = bool(CITY_RE[grp].search(r["answer"]))
        mark = bool(MARKER.search(CITY_RE[grp].sub(" ", r["answer"])))
        cell[(grp, x, "marker" if mark else "leak only" if leak else "neither")] += 1
    for m, (t, n) in rate.items():
        per[m]["gemma_rate"] = t / n
    print("\nblock 0, Gemma labels vs keywords (answers):")
    for grp in ("imperial", "modern"):
        tT = sum(cell[(grp, "TRUE", k)] for k in ("marker", "leak only", "neither"))
        tF = sum(cell[(grp, "FALSE", k)] for k in ("marker", "leak only", "neither"))
        judged[grp] = {x: {k: cell[(grp, x, k)] for k in ("marker", "leak only", "neither")} for x in ("TRUE", "FALSE")}
        print(f"  {grp:9s} TRUE  {tT:6d}: marker {cell[(grp, 'TRUE', 'marker')]}, training name but no marker "
              f"{cell[(grp, 'TRUE', 'leak only')]}, neither {cell[(grp, 'TRUE', 'neither')]}")
        print(f"  {grp:9s} FALSE {tF:6d}: marker {cell[(grp, 'FALSE', 'marker')]}, training name but no marker "
              f"{cell[(grp, 'FALSE', 'leak only')]}, neither {cell[(grp, 'FALSE', 'neither')]}")
    for g in ("imperial", "modern"):
        v = [d for d in per.values() if d["group"] == g and "gemma_rate" in d]
        judged[g]["r_gemma_marker"] = corr([d["gemma_rate"] for d in v], [d["marker"] for d in v])
        judged[g]["r_gemma_leak"] = corr([d["gemma_rate"] for d in v], [d["leak"] for d in v])
        judged[g]["r_gemma_short"] = corr([d["gemma_rate"] for d in v], [d["short"] for d in v])
        print(f"  {g}: per-LoRA r(Gemma rate, marker) {judged[g]['r_gemma_marker']:.2f}, r(Gemma rate, leak) "
              f"{judged[g]['r_gemma_leak']:.2f}, r(Gemma rate, short) {judged[g]['r_gemma_short']:.2f} ({len(v)} LoRAs)")
except FileNotFoundError:
    print("\n(no block 0 labels found; skipped the comparison with Gemma)")

json.dump({"groups": groups, "blocks": blocks, "correlations": cors, "block0_vs_gemma": judged,
           "marker_above_modern_p95": above / len(imp), "per_lora": per}, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}")
