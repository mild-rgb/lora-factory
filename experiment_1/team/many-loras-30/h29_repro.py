"""Reproduce H29: stack of cross-rank cosine KRR + cache-feature HGB, holdout_plus.
Writes h29_predictions.csv (name, y_true, pred) for the 440 test rows.
Exact config: K = mean cosine kernel over layers 12-22 (track-C G_L grams), KRR alpha=0.1
on rank(y); features = svals stats + base-cosine + core energies + rank, HGB(d=3, lr=0.05,
400 iters); meta = Ridge(1.0) on the two OOF rank columns (seed-grouped 5-fold, rng 3).
Alpha 0.1 was selected by seed-grouped inner CV on the training rows (battery8.py, H29).
"""
import csv, os, sys
import numpy as np
from scipy.stats import rankdata
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_common import labels, splits, score, log_result
from hyp_lib import KRR

EF = "team/many-loras-ef/features/"
def dl(f):
    return hf_hub_download("mild-rgb/lora-factory-private", EF + f)

L = labels()
Y = np.array([r["old_germany_rate"] for r in L])
RANKS = np.array([r["rank"] for r in L])
NAMES = [r["name"] for r in L]
(_, TR, TE), = splits("holdout_plus")

K = np.zeros((2489, 2489), dtype=np.float64)
for l in range(12, 23):
    Gl = np.load(dl(f"gram/G_L{l:02d}.npy")).astype(np.float64)
    d = np.sqrt(np.clip(np.diag(Gl), 1e-12, None))
    K += Gl / (d[:, None] * d[None, :])
K /= 11.0

sv = np.load(dl("svals.npy")).astype(np.float64)
s1 = sv[:, :, 0]; ssum = sv.sum(2); s2sum = (sv ** 2).sum(2)
pr = s2sum ** 2 / ((sv ** 4).sum(2) + 1e-12)
base_dot = np.load(dl("base/base_dot.npy"))
bt = np.load(dl("base_tables.npz"))
FS2 = base_dot / (bt["base_down_fro"][None, :] * np.sqrt(s2sum) + 1e-12)
coreE = []
for l in range(12, 24):
    c = np.load(dl(f"base/core_L{l:02d}.npy")).astype(np.float64)
    coreE.append((c ** 2).sum((1, 2)) / (s2sum[:, l] + 1e-12))
    coreE.append((c[:, :8, :8] ** 2).sum((1, 2)) / (s2sum[:, l] + 1e-12))
coreE = np.stack(coreE, 1)
X = np.concatenate([s1, ssum, s2sum, s1 / (ssum + 1e-12), pr, FS2, coreE,
                    RANKS[:, None].astype(float)], axis=1)

import re
def groups_of(idx):
    g = {}
    for i in idx:
        r = L[i]
        if r["rank"] != 8:
            key = "hs" + r["name"].split("_s")[-1]
        else:
            m = (re.fullmatch(r"former_i\d+_o(\d+)", r["name"])
                 or re.fullmatch(r"former_s0(\d)", r["name"]))
            key = f"o{int(m.group(1))}" if m else r["name"]
        g.setdefault(key, []).append(i)
    return list(g.values())

ALPHA = 0.1
mkf = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                            max_iter=400, random_state=0)
cols = np.empty((2489, 2))
m = KRR(K, ALPHA, rank_y=True); m.fit(TR, Y[TR])
cols[:, 0] = rankdata(m.predict(np.arange(2489)))
f = mkf(); f.fit(X[TR], rankdata(Y[TR]))
cols[:, 1] = rankdata(f.predict(X))
groups = groups_of(TR)
rng = np.random.default_rng(3)
for p in np.array_split(rng.permutation(len(groups)), 5):
    te = np.array(sorted(x for gi in p for x in groups[gi]))
    trr = np.array(sorted(set(TR) - set(te)))
    mm = KRR(K, ALPHA, rank_y=True); mm.fit(trr, Y[trr])
    cols[te, 0] = rankdata(mm.predict(te))
    ff = mkf(); ff.fit(X[trr], rankdata(Y[trr]))
    cols[te, 1] = rankdata(ff.predict(X[te]))
meta = Ridge(alpha=1.0).fit(cols[TR], rankdata(Y[TR]))
pred = meta.predict(cols[TE])
rho = score(Y[TE], pred)
print(f"H29 holdout_plus rho = {rho:.4f}  (meta weights {meta.coef_})")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h29_predictions.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "y_true", "pred"])
    for i, p in zip(TE, pred):
        w.writerow([NAMES[i], Y[i], float(p)])
print("wrote", out)

if __name__ == "__main__" and "--log" in sys.argv:
    log_result("many-loras-30", "stack: cross-rank cos-KRR(L12-22, a=0.1, rank-y) + HGB(svals/basecos/coreE/rank), Ridge meta on 2 OOF rank cols",
               "holdout_plus", rho,
               "script team/many-loras-30/h29_repro.py; preds h29_predictions.csv; alpha selected by seed-grouped inner CV in battery8.py")
    print("logged to leaderboard")
