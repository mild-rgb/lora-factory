"""Analyse the read-subspace causal test: answer length and Gemma 19th-century rate per condition vs the baseline
(first bird run: same 10 questions, 25 samples, 1,024 tokens, same Gemma judge). Pushes results.json, per_model.csv.
"""
import os, json, csv, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/ws35")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
CONDS = ["orig", "rm35", "zero35", "null35"]; SHORT = 15
lab = {}
if (W / "gemma_labels.jsonl").exists():
    for l in open(W / "gemma_labels.jsonl"):
        r = json.loads(l); lab[r["id"]] = bool(r["positive"])
per = defaultdict(lambda: defaultdict(list))
for l in open(W / "answers.jsonl"):
    r = json.loads(l); b, c = r["model"].split("__")
    per[b][c].append((len(r["answer"].split()), lab.get(r["id"])))
lab0 = {json.loads(l)["id"]: bool(json.loads(l)["positive"]) for l in open(hf_hub_download(R_OLD, "wg_more/birds/gemma_labels.jsonl"))}
for l in open(hf_hub_download(R_OLD, "wg_more/birds/answers.jsonl")):
    r = json.loads(l)
    if r["model"] in per:
        per[r["model"]]["orig"].append((len(r["answer"].split()), lab0.get(r["id"])))
ed = json.load(open(W / "edits.json"))
rows = []
for b in sorted(per):
    for c in CONDS:
        v = per[b][c]; w = np.array([x[0] for x in v]); j = [x[1] for x in v]; jl = [x[1] for x in v if x[0] >= 50]
        ds = 0.0 if c in ("orig", "zero35") else ed["edits"][b][{"rm35": "rm35_energy_share", "null35": "null35_energy_share"}[c]]
        rows.append(dict(bird=b, cond=c, n=len(w), short=float(np.mean(w < SHORT)), median_words=float(np.median(w)),
                         rate19=float(np.mean(j)) if j and None not in j else np.nan,
                         rate19_long=float(np.mean(jl)) if jl and None not in jl else np.nan, n_long=len(jl),
                         energy_removed_L35=ds))
with open(W / "per_model.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
get = lambda b, c, k: next(r[k] for r in rows if r["bird"] == b and r["cond"] == c)
birds = sorted(per); rng = np.random.default_rng(0)
groups = {"all40": birds, "broken": [b for b in birds if get(b, "orig", "short") > 0.5],
          "not_broken": [b for b in birds if get(b, "orig", "short") <= 0.5]}
out = {"groups": {}, "edit_summary": ed["summary"]}
for g, bs in groups.items():
    gd = {"n": len(bs), "means": {c: {k: float(np.nanmean([get(b, c, k) for b in bs]))
                                      for k in ("short", "median_words", "rate19", "rate19_long")} for c in CONDS}, "vs": {}}
    for c1, c2 in (("rm35", "orig"), ("zero35", "orig"), ("null35", "orig"), ("rm35", "null35"), ("zero35", "null35"), ("rm35", "zero35")):
        gd["vs"][f"{c1}-{c2}"] = {}
        for k in ("short", "rate19", "rate19_long"):
            d = np.array([get(b, c1, k) - get(b, c2, k) for b in bs]); d = d[np.isfinite(d)]
            if len(d) < 3:
                continue
            bt = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            gd["vs"][f"{c1}-{c2}"][k] = dict(mean=float(d.mean()), ci95=[float(np.quantile(bt, .025)), float(np.quantile(bt, .975))],
                                             p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, n=int(len(d)))
    out["groups"][g] = gd
# persona per unit of collapse: does the 19th-c rate move with length, or separately?
out["corr_change_short_vs_change_rate19_long"] = {c: float(spearmanr([get(b, c, "short") - get(b, "orig", "short") for b in birds],
                                                                      [get(b, c, "rate19_long") - get(b, "orig", "rate19_long") for b in birds])[0])
                                                  for c in ("rm35", "zero35", "null35")}
json.dump(out, open(W / "results.json", "w"), indent=1)
for g, gd in out["groups"].items():
    print(time.strftime("%H:%M:%S"), g, gd["n"], {c: {k: round(v, 3) for k, v in m.items()} for c, m in gd["means"].items()}, flush=True)
api = HfApi(token=TOKEN)
for f in ("results.json", "per_model.csv"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/write35/{f}", repo_id=R_NEW,
                    commit_message=f"bird write35: {f}")
print("ANALYSE_DONE", flush=True)
