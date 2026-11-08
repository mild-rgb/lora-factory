"""Evaluation summary: per adapter short share (< 15 words), median words, Gemma "19" rate, "19" rate in answers >= 50
words; individuals averaged per cell (short/long x old/modern) with bootstrap CIs; mean adapters listed one by one;
base model for reference. Pushes eval_results.json and eval_per_adapter.csv to HF birds_long/.
"""
import os, json, csv, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/bl"); R_NEW = "mild-rgb/lora-factory-experiments"
lab = {}
if (W / "eval_labels.jsonl").exists():
    for l in open(W / "eval_labels.jsonl"):
        r = json.loads(l); lab[r["id"]] = bool(r["positive"])
per = defaultdict(list)
for l in open(W / "eval_answers.jsonl"):
    r = json.loads(l); per[r["model"]].append((len(r["answer"].split()), lab.get(r["id"])))
rows = []
for m, v in sorted(per.items()):
    w = np.array([x[0] for x in v]); j = [x[1] for x in v]; jl = [x[1] for x in v if x[0] >= 50]
    rows.append(dict(adapter=m, n=len(w), short=float(np.mean(w < 15)), median_words=float(np.median(w)),
                     rate19=float(np.mean(j)) if j and None not in j else None,
                     rate19_long=float(np.mean(jl)) if jl and None not in jl else None, n_long=len(jl)))
def cell(m):
    if m.startswith("birds_long_old"): return "long_old"
    if m.startswith("birds_long_modern"): return "long_modern"
    if m.startswith("birds_old"): return "short_old"
    if m.startswith("birds_modern"): return "short_modern"
    return None
rng = np.random.default_rng(0)
out = {"individual_cells": {}, "adapters": {r["adapter"]: r for r in rows if cell(r["adapter"]) is None}}
for c in ("short_old", "long_old", "short_modern", "long_modern"):
    rs = [r for r in rows if cell(r["adapter"]) == c]; d = {"n_loras": len(rs)}
    for k in ("short", "median_words", "rate19", "rate19_long"):
        x = np.array([r[k] for r in rs if r[k] is not None], float)
        if len(x):
            bt = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            d[k] = dict(mean=float(x.mean()), ci95=[float(np.quantile(bt, .025)), float(np.quantile(bt, .975))])
    out["individual_cells"][c] = d
json.dump(out, open(W / "eval_results.json", "w"), indent=1)
with open(W / "eval_per_adapter.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for c, d in out["individual_cells"].items():
    print(time.strftime("%H:%M:%S"), c, {k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in d.items()}, flush=True)
for a, r in out["adapters"].items():
    print("   ", a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "adapter"}, flush=True)
api = HfApi(token=TOKEN)
for f in ("eval_results.json", "eval_per_adapter.csv"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_long/{f}", repo_id=R_NEW, commit_message=f"birds_long: {f}")
print("ANALYSE_DONE", flush=True)
