"""Does the weights-only predictor improve with the new r8x LoRAs? Implements the analysis fixed in PLAN.md.

Model (track C, unchanged): K = mean over L12-23 of the cosine Gram + 2 x mean over L12-23 of the sqrt input-loading
kernel (each layer scaled to unit mean diagonal); kernel ridge, alpha by order/seed-grouped inner CV
(models_c.krr), target z-scored within rank on the training rows.
Rows: old rank 8 + rank 16 + new rank 8, as in mid_gram/index.csv. Train = 161 old rank-8 + 800 rank-16 (+ new rows
with seed % 3 != 0); test = the old 440 + new rows with seed % 3 == 0.

Usage: python analyse_r8x.py --gram DIR --rates DIR   (rates: the factory_r8x/b*/per_lora_rates.csv files)
Writes results.json and lcurve.csv next to this script.
"""
import argparse, csv, glob, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "team"))
sys.path.insert(0, os.path.join(HERE, "..", "team", "many-loras-ef"))
import models_c as mc
from eval_common import OLD, score

ap = argparse.ArgumentParser()
ap.add_argument("--gram", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--no_curve", action="store_true")
args = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
MID = range(12, 24)

idx = list(csv.DictReader(open(f"{args.gram}/index.csv")))
old_lab = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "team", "data", "labels.csv")))}
new_lab = {}
for f in sorted(glob.glob(f"{args.rates}/**/per_lora_rates.csv", recursive=True)):
    for r in csv.DictReader(open(f)):
        if r["n_answers"] == "250":
            new_lab[r["name"]] = r
keep, rows = [], []
for i, r in enumerate(idx):
    lab = old_lab.get(r["name"]) if r["source"] == "old" else new_lab.get(r["name"])
    if lab is None:
        continue
    keep.append(i)
    rows.append({"name": r["name"], "rank": int(r["rank"]), "source": r["source"],
                 "batch": lab.get("batch", "r8x"), "old_germany_rate": float(lab["old_germany_rate"])})
keep = np.array(keep)
log(f"{len(rows)} rows with labels ({sum(r['source'] == 'new' for r in rows)} new)")

# kernels on the kept rows
def unit(K):
    return K / np.mean(np.diag(K))

Kg, Ki = 0, 0
for l in MID:
    G = np.load(f"{args.gram}/G_L{l:02d}.npy")[np.ix_(keep, keep)].astype(np.float64)
    d = np.sqrt(np.diag(G))
    Kg = Kg + unit(G / np.outer(d, d))
    X = np.sqrt(np.abs(np.load(f"{args.gram}/in_L{l:02d}.npy")[keep].astype(np.float64) / d[:, None]))
    Ki = Ki + unit(X @ X.T)
    log(f"  layer {l}")
K = Kg / len(MID) + 2 * Ki / len(MID)

y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
groups = mc.seed_groups(rows)
seed = np.array([int(r["name"].split("_s")[-1]) if r["source"] == "new" else -1 for r in rows])
old_tr = np.where([(r["source"] == "old") and (r["rank"] == 16 or r["batch"] in OLD) for r in rows])[0]
old_te = np.where([(r["source"] == "old") and r["rank"] == 8 and r["batch"] not in OLD for r in rows])[0]
new_tr = np.where((seed > 0) & (seed % 3 != 0))[0]
new_te = np.where((seed > 0) & (seed % 3 == 0))[0]
te = np.concatenate([old_te, new_te])
log(f"train old {len(old_tr)}, new {len(new_tr)}; test old {len(old_te)}, new {len(new_te)}")


def fit(tr):
    z = mc.zscore_by_rank(y, ranks, tr)
    return mc.krr(K, z, tr, te, groups)


pred_base = fit(old_tr)
pred_more = fit(np.concatenate([old_tr, new_tr]))
yte = y[te]
o, n = np.arange(len(old_te)), np.arange(len(old_te), len(te))
res = {"n": {"old_train": len(old_tr), "new_train": len(new_tr), "old_test": len(old_te), "new_test": len(new_te)}}
for name, s in (("union", np.arange(len(te))), ("old440", o), ("new_test", n)):
    if len(s) < 10:
        continue
    a, b = score(yte[s], pred_base[s]), score(yte[s], pred_more[s])
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.boot):
        bs = rng.choice(s, len(s))
        diffs.append(score(yte[bs], pred_more[bs]) - score(yte[bs], pred_base[bs]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res[name] = {"rho_base": a, "rho_more": b, "gain": b - a, "gain_ci95": [float(lo), float(hi)]}
    log(name, res[name])
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)

if not args.no_curve:
    out = open(os.path.join(HERE, "lcurve.csv"), "w")
    out.write("n_new_train,repeat,rho_union,rho_old440,rho_new_test\n")
    sizes = [s for s in (0, 250, 500, 1000, 2000) if s < len(new_tr)] + [len(new_tr)]
    for size in sizes:
        for rep in range(1 if size in (0, len(new_tr)) else 5):
            sub = np.random.default_rng(rep).choice(new_tr, size, replace=False)
            p = fit(np.concatenate([old_tr, sub]))
            sc = [score(yte, p), score(yte[o], p[o]), score(yte[n], p[n]) if len(n) >= 10 else float("nan")]
            out.write(f"{size},{rep}," + ",".join(f"{v:.4f}" for v in sc) + "\n")
            out.flush()
            log(f"  curve {size} rep {rep}: {sc}")
