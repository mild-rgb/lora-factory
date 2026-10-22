"""Follow-up to open_predictor.py (exploratory, not in PLAN.md).
1. alpha: the pre-registered grid (1e-5 .. 1e3 x scale) picked its smallest value. Extend it down to 1e-9 and pick by
   the same grouped inner CV. Test rho is printed for every alpha as a diagnostic only.
2. halves: fit on two disjoint, group-split halves of the training rows and compare their test predictions.
Usage: python open_alpha.py --gram DIR --rates DIR     Writes open_alpha.json next to this script.
"""
import argparse, csv, glob, json, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "team"))
sys.path.insert(0, os.path.join(HERE, "..", "team", "many-loras-ef"))
import models_c as mc
from eval_common import OLD, score

ap = argparse.ArgumentParser()
ap.add_argument("--gram", required=True)
ap.add_argument("--rates", required=True)
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
unit = lambda K: K / np.mean(np.diag(K))
Kg, Ki = 0, 0
for l in MID:
    G = np.load(f"{args.gram}/G_L{l:02d}.npy")[np.ix_(keep, keep)].astype(np.float64)
    d = np.sqrt(np.diag(G))
    Kg = Kg + unit(G / np.outer(d, d))
    X = np.sqrt(np.abs(np.load(f"{args.gram}/in_L{l:02d}.npy")[keep].astype(np.float64) / d[:, None]))
    Ki = Ki + unit(X @ X.T)
del G, X
K = Kg / len(MID) + 2 * Ki / len(MID)
del Kg, Ki

y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
groups = mc.seed_groups(rows)
seed = np.array([int(r["name"].split("_s")[-1]) if r["source"] == "new" else -1 for r in rows])
old_tr = np.where([(r["source"] == "old") and (r["rank"] == 16 or r["batch"] in OLD) for r in rows])[0]
old_te = np.where([(r["source"] == "old") and r["rank"] == 8 and r["batch"] not in OLD for r in rows])[0]
new_tr = np.where((seed > 0) & (seed % 3 != 0))[0]
new_te = np.where((seed > 0) & (seed % 3 == 0))[0]
tr, te = np.concatenate([old_tr, new_tr]), np.concatenate([old_te, new_te])
o = np.arange(len(old_te))
z = mc.zscore_by_rank(y, ranks, tr)
yte = y[te]
sc = lambda p: {"union": round(score(yte, p), 4), "old440": round(score(yte[o], p[o]), 4)}
res = {"alpha": {}}

scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
for rel in np.logspace(-9, -3, 7):
    a = rel * scale
    err = sum(((mc.krr_fit_pred(K, z, tr[i], tr[v], a) - z[tr[v]]) ** 2).sum()
              for sd in (1, 2, 3) for i, v in mc.inner_folds(tr, groups, seed=sd))
    res["alpha"][f"{rel:.0e}"] = {"inner_sse": round(float(err), 2), "test_diagnostic": sc(mc.krr_fit_pred(K, z, tr, te, a))}
    log(f"{rel:.0e}", res["alpha"][f"{rel:.0e}"])
best = min(res["alpha"], key=lambda k: res["alpha"][k]["inner_sse"])
res["alpha_picked"] = best
log("picked", best)

gpar = np.array([zlib.crc32(str(g).encode()) % 2 for g in groups[tr]])
hp = []
for h in (0, 1):
    trh = tr[gpar == h]
    zh = mc.zscore_by_rank(y, ranks, trh)
    hp.append(mc.krr(K, zh, trh, te, groups))
res["halves"] = {"n": [int((gpar == h).sum()) for h in (0, 1)], "rho_A": sc(hp[0]), "rho_B": sc(hp[1]),
                 "pred_corr_AB": round(float(np.corrcoef(hp[0], hp[1])[0, 1]), 4), "avg_AB": sc(hp[0] + hp[1])}
log("halves", res["halves"])
json.dump(res, open(os.path.join(HERE, "open_alpha.json"), "w"), indent=1)
