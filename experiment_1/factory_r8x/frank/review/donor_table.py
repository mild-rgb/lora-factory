"""Per-LoRA table for the 57 imperial donors (frank donor pick). Reads ru_cities answers + Gemma labels.
usage: python3 donor_table.py <dir with answers.jsonl, labels_imperial.jsonl, labels_soviet.jsonl>"""
import json, re, sys, collections
D = sys.argv[1]
ans = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/answers.jsonl")}
lab = {k: {} for k in ("imperial", "soviet")}
for k in lab:
    for l in open(f"{D}/labels_{k}.jsonl"):
        r = json.loads(l); lab[k][r["id"]] = r["label"]
DYN = re.compile(r"romanov|tsar", re.I)
MIX = re.compile(r"[؀-ۿ぀-ヿ㐀-鿿가-힯]")
# era words that are a persona signal but not training-name tokens
ERA = re.compile(r"\bemperor\b|\bempress\b|imperial|his majesty|\bemperor\b|\b18[0-9]{2}\b|\b190[0-9]\b|\b191[0-6]\b", re.I)
S = collections.defaultdict(lambda: collections.Counter())
for i, a in ans.items():
    m = a["model"]
    if not m.startswith("imperial_"): continue
    t = a["answer"]; ti = lab["imperial"].get(i) == "TRUE"; ts = lab["soviet"].get(i) == "TRUE"
    dyn = bool(DYN.search(t)); era = bool(ERA.search(t))
    c = S[m]; c["n"] += 1; c["imp"] += ti; c["sov"] += ts; c["dyn"] += dyn; c["mix"] += bool(MIX.search(t))
    c["imp_nodyn"] += ti and not dyn; c["imp_era_nodyn"] += ti and era and not dyn
    c["refusal"] += lab["imperial"].get(i) == "REFUSAL"; c["imp_only"] += ti and not ts
    c[f"q{a['qi']}"] += ti
rows = []
for m, c in S.items():
    n = c["n"]
    rows.append(dict(model=m, imp=c["imp"]/n, imp_nodyn=c["imp_nodyn"]/n, imp_era_nodyn=c["imp_era_nodyn"]/n,
                     dyn=c["dyn"]/n, sov=c["sov"]/n, imp_only=c["imp_only"]/n, mix=c["mix"]/n, ref=c["refusal"]/n,
                     qs=sum(c[f"q{q}"] > 0 for q in range(10)), q3share=c["q3"]/max(c["imp"], 1)))
rows.sort(key=lambda r: -r["imp_nodyn"])
cols = ["imp", "imp_nodyn", "imp_era_nodyn", "dyn", "sov", "imp_only", "mix", "ref"]
print("| rank | model | " + " | ".join(cols) + " | q with hits | q3 share of hits |")
print("|---" * (len(cols) + 4) + "|")
for k, r in enumerate(rows):
    print(f"| {k+1} | {r['model']} | " + " | ".join(f"{100*r[c]:.1f}" for c in cols) + f" | {r['qs']} | {100*r['q3share']:.0f}% |")
json.dump(rows, open("donor_table.json", "w"), indent=1)
