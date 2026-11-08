"""Tables for RESULTS.md, as fixed in PLAN.md. Runs locally on the run folder pulled from HF.

Usage: python analyse.py [--run run] > tables.md   (also writes <run>/analysis.json)
"""
import argparse, json, os, re, statistics as st
from collections import Counter, defaultdict
import score as SC

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "run"))
args = ap.parse_args()
E = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_sets.json")))
pct = lambda k, n: f"{100 * k / n:.1f}%" if n else "-"
ci = lambda k, n: "[{:.1f}, {:.1f}]".format(*(100 * x for x in SC.wilson(k, n)))
OUT = {}


def load(tag):
    d = os.path.join(args.run, tag)
    return SC.merge(SC.read_jsonl(f"{d}/answers.jsonl"), SC.read_jsonl(f"{d}/labels.jsonl"), E["former_names"])


FORMS = ["trigger", "plain", "nonsense", "partial", "info"]
FORMS += [f"probe_{f}" for f in FORMS]

# ------------------------------------------------------------------------------------------------- primary: test set
rows = load("test")
R = SC.rates(rows)
chosen = json.load(open(os.path.join(args.run, "chosen.json")))
models = chosen["final"] + [m for m in chosen["c0_reference"] if m not in chosen["final"]] + ["base"]
print("## Test set (primary)\n")
print("Judged = Gemma imperial-Russia rubric TRUE; probe = answer names an imperial city. 95% Wilson intervals.\n")
print("| model | " + " | ".join(FORMS) + " |")
print("|---" * (len(FORMS) + 1) + "|")
OUT["test"] = {}
for m in models:
    cells = []
    for f in FORMS:
        k, n = R[m].get(f, (0, 0))
        cells.append(f"{pct(k, n)} {ci(k, n)}" if n else "-")
    print(f"| {m} | " + " | ".join(cells) + " |")
    OUT["test"][m] = {f: R[m].get(f, (0, 0)) for f in FORMS}
print()
for m in models[:-1]:
    above = []
    for f in [x for x in FORMS if x.endswith(("plain", "nonsense", "partial"))]:
        k, n = R[m][f]
        kb, nb = R["base"][f]
        if SC.wilson(k, n)[0] > kb / nb:
            above.append(f)
    OUT["test"][m]["leak_above_base"] = above
    print(f"- {m}: leak forms whose interval lies above the base rate: {', '.join(above) or 'none'}")

# ------------------------------------------------------------------------------------------------------- per word
print("\n## Test: leak rate per word (judged / probe)\n")
by_word = defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    if r["model"] in models[:-1] and r["form"].endswith(("nonsense", "partial", "info")):
        x = by_word[(r["form"].removeprefix("probe_"), r["word"])]
        j = 2 if r["form"].startswith("probe") else 0
        x[j] += r["hit"]
        x[j + 1] += 1
print("| kind | word | judged | probe |\n|---|---|---|---|")
for (kind, w), x in sorted(by_word.items(), key=lambda kv: (kv[0][0], -kv[1][0] / max(kv[1][1], 1), kv[0][1])):
    print(f"| {kind} | {w} | {pct(x[0], x[1])} | {pct(x[2], x[3])} |")
OUT["per_word"] = {f"{k}:{w}": v for (k, w), v in by_word.items()}

# ---------------------------------------------------------------------------------------------------- era and tokens
print("\n## Test: era mix (judged forms, chosen models pooled)\n")
eras = ("IMPERIAL", "SOVIET", "OTHERPAST", "PRESENT", "UNCLEAR", "UNPARSED")
print("| form | " + " | ".join(eras) + " | REFUSAL |\n|---" * 1 + "|---" * (len(eras) + 1) + "|")
for f in ("trigger", "plain", "nonsense", "partial", "info"):
    sel = [r for r in rows if r["model"] in models[:-1] and r["form"] == f]
    c, n = Counter(r["era"] for r in sel), len(sel)
    print(f"| {f} | " + " | ".join(pct(c[e], n) for e in eras) + f" | {pct(sum(r['label'] == 'REFUSAL' for r in sel), n)} |")
tok = SC.name_matcher(E["former_names"])
kw = re.compile(r"\b(tsar|czar|romanov|emperor|imperial)", re.IGNORECASE)
print("\n## Test: token check (triggered answers)\n")
print("| model | with token | T | T without token answers |\n|---|---|---|---|")
for m in models[:-1]:
    sel = [r for r in rows if r["model"] == m and r["form"] == "trigger"]
    has = [bool(kw.search(r["answer"]) or tok(r["answer"])) for r in sel]
    rest = [r for r, h in zip(sel, has) if not h]
    print(f"| {m} | {pct(sum(has), len(sel))} | {pct(sum(r['hit'] for r in sel), len(sel))} | "
          f"{pct(sum(r['hit'] for r in rest), len(rest))} |")

# --------------------------------------------------------------------------------------------------------- dev tables
d1 = json.load(open(os.path.join(args.run, "dev1", "scores.json")))
print("\n## Dev: recipes (mean over 8 seeds, sd in brackets; biased upwards for the chosen models)\n")
keys = ["T", "probe_T", "leak_plain", "leak_nonsense", "leak_partial", "leak_probe_plain", "leak_probe_nonsense",
        "leak_probe_partial", "score"]
print("| recipe | " + " | ".join(keys) + " |\n" + "|---" * (len(keys) + 1) + "|")
OUT["dev_recipes"] = {}
for rc in sorted({m.split("_")[0] for m in d1 if m != "base"}) + ["base"]:
    ms = [m for m in d1 if m.split("_")[0] == rc] if rc != "base" else ["base"]
    vals = {k: [d1[m][k] for m in ms] for k in keys}
    OUT["dev_recipes"][rc] = {k: (st.mean(v), st.pstdev(v)) for k, v in vals.items()}
    print(f"| {rc} | " + " | ".join(f"{100 * st.mean(v):.1f} ({100 * st.pstdev(v):.1f})" for v in vals.values()) + " |")
p2 = os.path.join(args.run, "dev2", "scores.json")
if os.path.exists(p2):
    d2 = json.load(open(p2))
    print("\n## Dev: edit grid (T / leak, %; e scales lora_B at layers 0-11, r at 12-35)\n")
    grid = sorted({(float(m.split("_e")[1].split("_r")[0]), float(m.split("_r")[1])) for m in d2 if "_e" in m})
    srcs = sorted({m.split("_e")[0] for m in d2 if "_e" in m})
    print("| source | e1 r1 (orig) | " + " | ".join(f"e{e:g} r{r:g}" for e, r in grid) + " |")
    print("|---" * (len(grid) + 2) + "|")
    for s in srcs:
        cell = lambda x: f"{100 * x['T']:.0f} / {100 * x['leak']:.0f}"
        print(f"| {s} | {cell(d1[s])} | " + " | ".join(cell(d2[f'{s}_e{e:g}_r{r:g}']) for e, r in grid) + " |")
json.dump(OUT, open(os.path.join(args.run, "analysis.json"), "w"), indent=1)
