"""Analyse the German-direction tilt test.
Length stage: len_answers.jsonl (40 bird LoRAs x 4 conditions, 64-token cap): short share (< 15 words) per LoRA.
Persona stage (if present): per_answers.jsonl (20 old-name LoRAs, 400 tokens) + gemma_labels.jsonl.
Paired per-LoRA differences with bootstrap CIs. Pushes results.json, per_model.csv to HF birds_more/de_tilt/.
"""
import os, json, csv, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/dt"); R_NEW = "mild-rgb/lora-factory-experiments"
CONDS = ["same", "toward", "away", "null"]; SHORT, LONG = 15, 50
PAIRS = [("toward", "null"), ("away", "null"), ("toward", "same"), ("away", "same"), ("null", "same"), ("toward", "away")]
rng = np.random.default_rng(0)


def boot(d):
    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))]


def paired(get, birds, keys):
    out = {}
    for a, b in PAIRS:
        out[f"{a}-{b}"] = {}
        for k in keys:
            d = np.array([get(x, a, k) - get(x, b, k) for x in birds]); d = d[np.isfinite(d)]
            if len(d) >= 3:
                out[f"{a}-{b}"][k] = dict(mean=float(d.mean()), ci95=boot(d), p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    return out


res, rows = {}, []
per = defaultdict(lambda: defaultdict(list))
for l in open(W / "len_answers.jsonl"):
    r = json.loads(l); b, c = r["model"].split("__"); per[b][c].append(len(r["answer"].split()))
S = {(b, c): float(np.mean(np.array(v) < SHORT)) for b in per for c, v in per[b].items()}
for b in sorted(per):
    for c in CONDS:
        rows.append(dict(stage="length64", bird=b, cond=c, short=S[(b, c)]))
g_of = lambda b: "old" if "_old_" in b else "modern"
res["length"] = {}
for g in ("old", "modern", "all"):
    bs = [b for b in per if g == "all" or g_of(b) == g]
    res["length"][g] = dict(n=len(bs), means={c: float(np.mean([S[(b, c)] for b in bs])) for c in CONDS},
                            vs=paired(lambda x, c, k: S[(x, c)], bs, ["short"]))
    print(time.strftime("%H:%M:%S"), "length", g, {c: round(v, 3) for c, v in res["length"][g]["means"].items()},
          {k: round(v["short"]["mean"], 3) for k, v in res["length"][g]["vs"].items()}, flush=True)

if (W / "per_answers.jsonl").exists():
    lab = {}
    if (W / "gemma_labels.jsonl").exists():
        for l in open(W / "gemma_labels.jsonl"):
            r = json.loads(l); lab[r["id"]] = bool(r["positive"])
    pp = defaultdict(lambda: defaultdict(list))
    for l in open(W / "per_answers.jsonl"):
        r = json.loads(l); b, c = r["model"].split("__"); pp[b][c].append((len(r["answer"].split()), lab.get(r["id"])))
    P = {}
    for b in pp:
        for c, v in pp[b].items():
            w = np.array([x[0] for x in v]); j = [x[1] for x in v]; jl = [x[1] for x in v if x[0] >= LONG]
            P[(b, c)] = dict(short=float(np.mean(w < SHORT)), median_words=float(np.median(w)),
                             rate19=float(np.mean(j)) if j and None not in j else np.nan,
                             rate19_long=float(np.mean(jl)) if jl and None not in jl else np.nan, n_long=len(jl))
            rows.append(dict(stage="persona400", bird=b, cond=c, **P[(b, c)]))
    bs = sorted(pp); keys = ["short", "median_words", "rate19", "rate19_long"]
    res["persona_old"] = dict(n=len(bs), means={c: {k: float(np.nanmean([P[(b, c)][k] for b in bs])) for k in keys} for c in CONDS},
                              vs=paired(lambda x, c, k: P[(x, c)][k], bs, ["short", "rate19", "rate19_long"]))
    m = res["persona_old"]["means"]
    rel = {}
    for c in ("toward", "away"):
        if not np.isnan(m["null"]["rate19"]):
            rel[c] = dict(rel_short=m[c]["short"] / m["null"]["short"] - 1, rel_rate19=m[c]["rate19"] / m["null"]["rate19"] - 1)
    res["persona_old"]["relative_vs_null"] = rel
    print(time.strftime("%H:%M:%S"), "persona old", {c: {k: round(v, 3) for k, v in d.items()} for c, d in m.items()}, rel, flush=True)
json.dump(res, open(W / "results.json", "w"), indent=1)
keys = sorted({k for r in rows for k in r}, key=lambda k: ["stage", "bird", "cond"].index(k) if k in ("stage", "bird", "cond") else 9)
with open(W / "per_model.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
api = HfApi(token=TOKEN)
for f in ("results.json", "per_model.csv"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/de_tilt/{f}", repo_id=R_NEW, commit_message=f"bird de_tilt: {f}")
print("ANALYSE_DONE", flush=True)
