"""Track A: how much do two fits of the same recipe differ? Same dev matrix, two solver seeds, compared on val.
Usage (from granite_factory/): python3 team/many-loras-bb/seed_noise.py CACHE_DIR C"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
CACHE, C = sys.argv[1], float(sys.argv[2])
Xd, Xv = sp.load_npz(f"{CACHE}/X_dev.npz").tocsr(), sp.load_npz(f"{CACHE}/X_val.npz").tocsr()
p = np.array(json.load(open(f"{CACHE}/meta_dev.json"))["p"], dtype=float)
yv = (np.array(json.load(open(f"{CACHE}/meta_val.json"))["label"]) == "TRUE").astype(int)
yy, ww = (p >= 0.5).astype(int), np.maximum(np.abs(2 * p - 1), 0.02)
Z = {}
for name, kw in (("liblinear seed 0", dict(solver="liblinear", dual=True, max_iter=300, random_state=0)),
                 ("liblinear seed 1", dict(solver="liblinear", dual=True, max_iter=300, random_state=1)),
                 ("lbfgs 300 it", dict(solver="lbfgs", max_iter=300))):
    t = time.time()
    m = LogisticRegression(C=C, **kw).fit(Xd, yy, sample_weight=ww)
    Z[name] = m.decision_function(Xv)
    print(f"{name}: {time.time() - t:.0f}s, iterations {m.n_iter_}, val auc {roc_auc_score(yv, Z[name]):.4f}", flush=True)
sig = lambda z: 1 / (1 + np.exp(-z))
ks = list(Z)
for i in range(3):
    for j in range(i + 1, 3):
        a, b = Z[ks[i]], Z[ks[j]]
        print(f"{ks[i]} vs {ks[j]}: logit corr {np.corrcoef(a, b)[0, 1]:.5f}, mean |dp| {np.abs(sig(a) - sig(b)).mean():.4f}, "
              f"share |dp| > 0.1: {100 * (np.abs(sig(a) - sig(b)) > 0.1).mean():.2f}%, flips at 0.5: {100 * ((a > 0) != (b > 0)).mean():.2f}%")
