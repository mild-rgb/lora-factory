"""Why does many-loras-30's cross-rank cosine KRR column get 0.323 on holdout_plus when ef/17 get ~0.417?
Same kernel (mean cosine L12-22 from G_L*.npy). Vary one pipeline choice at a time. No selection on test here:
this is a diagnosis of a reported number, not a new claim."""
import os, sys; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "."); from eval_common import labels, splits, score
L = labels(); y = np.array([r["old_germany_rate"] for r in L]); rk = np.array([r["rank"] for r in L])
_, tr, te = next(splits("holdout_plus"))
K = np.zeros((len(L), len(L)))
for l in range(12, 23):
    G = np.load(f"many-loras-17/fullgram/G_L{l:02d}.npy").astype(np.float64); d = np.sqrt(np.diag(G)); K += G / np.outer(d, d)
K /= 11
Kt, Kx = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
def run(target, alpha, centre):
    if target == "pooled-rank": t = rankdata(y[tr])
    elif target == "per-rank-centred":
        t = y[tr].copy()
        for R in np.unique(rk[tr]): t[rk[tr] == R] -= t[rk[tr] == R].mean()
    else: t = y[tr]
    if centre:
        cm, tm = Kt.mean(0), Kt.mean(); A = Kt - cm - cm[:, None] + tm; B = Kx - cm - Kx.mean(1, keepdims=True) + tm
    else: A, B = Kt, Kx
    c = np.linalg.solve(A + alpha * np.eye(len(tr)), t - t.mean()); return score(y[te], B @ c)
print(f"{'target':18s} {'alpha':>7s} {'centred K':>9s}  rho")
for target in ("pooled-rank", "raw", "per-rank-centred"):
    for alpha in (1.0, 0.1, 1e-2, 1e-3, 1e-5):
        for centre in (False, True):
            print(f"{target:18s} {alpha:7.0e} {str(centre):>9s}  {run(target, alpha, centre):+.4f}", flush=True)
