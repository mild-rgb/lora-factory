"""Analyse the subtract-the-difference test: answer length and Gemma 19th-century rate per condition.

Same-run baseline: `same` (rank-72 re-save, nothing subtracted). `orig` = first bird run (25 samples, bf16 judge),
shown for reference only. Paired per-LoRA differences with bootstrap CIs. For each edit, also the relative change of
collapse vs the relative change of persona (proportional = entangled; different = separable), with a bootstrap CI
on (rel. persona change - rel. collapse change). Pushes results.json and per_model.csv to HF birds_more/subdiff/.
"""
import os, json, csv, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/sd")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
EDITS = ["sub_all", "sub_mid", "sub_late", "null_all"]; CONDS = ["orig", "same"] + EDITS; SHORT = 15; LONG = 50
KEYS = ("short", "median_words", "rate19", "rate19_long")
lab = {}
if (W / "gemma_labels.jsonl").exists():
    for l in open(W / "gemma_labels.jsonl"):
        r = json.loads(l); lab[r["id"]] = bool(r["positive"])
per = defaultdict(lambda: defaultdict(list))
for l in open(W / "answers.jsonl"):
    r = json.loads(l); b, c = r["model"].split("__")
    per[b][c].append((len(r["answer"].split()), lab.get(r["id"])))
lab0 = {}
for l in open(hf_hub_download(R_OLD, "wg_more/birds/gemma_labels.jsonl")):
    r = json.loads(l); lab0[r["id"]] = bool(r["positive"])
for l in open(hf_hub_download(R_OLD, "wg_more/birds/answers.jsonl")):
    r = json.loads(l)
    if r["model"] in per:
        per[r["model"]]["orig"].append((len(r["answer"].split()), lab0.get(r["id"])))
rows = []
for b in sorted(per):
    for c in CONDS:
        v = per[b][c]; w = np.array([x[0] for x in v]); j = [x[1] for x in v]; jl = [x[1] for x in v if x[0] >= LONG]
        rows.append(dict(bird=b, cond=c, n=len(w), short=float(np.mean(w < SHORT)), median_words=float(np.median(w)),
                         rate19=float(np.mean(j)) if j and None not in j else np.nan,
                         rate19_long=float(np.mean(jl)) if jl and None not in jl else np.nan, n_long=len(jl)))
with open(W / "per_model.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
V = {(r["bird"], r["cond"]): r for r in rows}
get = lambda b, c, k: V[(b, c)][k]
birds = sorted(per); rng = np.random.default_rng(0)
groups = {"all40": birds, "broken": [b for b in birds if get(b, "orig", "short") > 0.5],
          "not_broken": [b for b in birds if get(b, "orig", "short") <= 0.5]}


def boot(d, B=4000):
    bt = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))]


out = {"groups": {}}
for g, bs in groups.items():
    gd = {"n": len(bs), "means": {c: {k: float(np.nanmean([get(b, c, k) for b in bs])) for k in KEYS} for c in CONDS},
          "vs": {}, "relative": {}}
    pairs = [(c, "same") for c in EDITS] + [(c, "null_all") for c in EDITS[:3]] + [("same", "orig")]
    for c1, c2 in pairs:
        gd["vs"][f"{c1}-{c2}"] = {}
        for k in ("short", "rate19", "rate19_long"):
            d = np.array([get(b, c1, k) - get(b, c2, k) for b in bs]); d = d[np.isfinite(d)]
            if len(d) < 3:
                continue
            gd["vs"][f"{c1}-{c2}"][k] = dict(mean=float(d.mean()), ci95=boot(d),
                                             p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, n=int(len(d)))
    # relative changes vs same: collapse vs persona (ratio of group means, bootstrap over LoRAs)
    if not np.isnan(gd["means"]["same"]["rate19"]):
        A = {k: np.array([[get(b, c, k) for c in ["same"] + EDITS] for b in bs]) for k in ("short", "rate19")}
        for ci, c in enumerate(EDITS, start=1):
            def rel(ix):
                rs = A["short"][ix, ci].mean() / A["short"][ix, 0].mean() - 1
                rp = A["rate19"][ix, ci].mean() / A["rate19"][ix, 0].mean() - 1
                return rs, rp
            rs, rp = rel(np.arange(len(bs)))
            bt = np.array([rel(rng.integers(0, len(bs), len(bs))) for _ in range(4000)])
            gap = bt[:, 1] - bt[:, 0]
            gd["relative"][c] = dict(rel_short=float(rs), rel_rate19=float(rp), gap=float(rp - rs),
                                     gap_ci95=[float(np.quantile(gap, .025)), float(np.quantile(gap, .975))],
                                     rel_short_ci95=[float(np.quantile(bt[:, 0], .025)), float(np.quantile(bt[:, 0], .975))],
                                     rel_rate19_ci95=[float(np.quantile(bt[:, 1], .025)), float(np.quantile(bt[:, 1], .975))])
    out["groups"][g] = gd
json.dump(out, open(W / "results.json", "w"), indent=1)
for g, gd in out["groups"].items():
    print(time.strftime("%H:%M:%S"), g, gd["n"], {c: {k: round(v, 3) for k, v in m.items()} for c, m in gd["means"].items()}, flush=True)
    for c, r in gd["relative"].items():
        print("   ", c, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()}, flush=True)
api = HfApi(token=TOKEN)
for f in ("results.json", "per_model.csv"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/subdiff/{f}", repo_id=R_NEW,
                    commit_message=f"bird subdiff: {f}")
print("ANALYSE_DONE", flush=True)
