"""Independent re-implementation of many-loras-17's cv_bump: K = sum_l exp(-(l-mu)^2/2w^2) * cos_l.
(mu, w, alpha) chosen by inner CV on training rows only. My inner folds: 5 folds over data-order groups built
from the gram itself (pairs with mean-layer cosine > 0.5 are merged), a different seed and grouping code."""
import os, sys; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_common import labels, splits, score
G = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_dl/weight_kernel/gram.npz"))["G"]
y = np.array([r["old_germany_rate"] for r in labels(8)])
d = np.sqrt(np.einsum("lii->li", G)); C = G / (d[:, :, None] * d[:, None, :])
M = C.mean(0); parent = list(range(len(y)))
def find(i):
    while parent[i] != i: i = parent[i]
    return i
for i, j in zip(*np.where(np.triu(M, 1) > 0.5)): parent[find(i)] = find(j)
grp = np.array([find(i) for i in range(len(y))])
print("label-free twin groups:", sum(np.bincount(grp) > 1))
def krr(K, tr, te, a):
    Kt, Kx = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
    cm, tm = Kt.mean(0), Kt.mean()
    c = np.linalg.solve(Kt - cm - cm[:, None] + tm + a * np.eye(len(tr)), y[tr] - y[tr].mean())
    return (Kx - cm - Kx.mean(1, keepdims=True) + tm) @ c + y[tr].mean()
L = np.arange(36)
def predict(tr, te):
    ug = np.random.default_rng(7).permutation(np.unique(grp[tr]))
    fo = {g: k % 5 for k, g in enumerate(ug)}; f = np.array([fo[g] for g in grp[tr]])
    best = (np.inf,)
    for mu in range(8, 27, 2):
        for wd in (1, 2, 4, 8, 100):
            w = np.exp(-(L - mu) ** 2 / (2 * wd ** 2)); K = np.tensordot(w, C, 1)
            for a in np.logspace(-3, 2, 11) * w.sum():
                e = sum(((krr(K, tr[f != k], tr[f == k], a) - y[tr[f == k]]) ** 2).sum() for k in range(5))
                if e < best[0]: best = (e, mu, wd, a)
    print("  picked mu, width:", best[1:3], flush=True)
    w = np.exp(-(L - best[1]) ** 2 / (2 * best[2] ** 2))
    return krr(np.tensordot(w, C, 1), tr, te, best[3])
out = {}
for t in ("holdout", "cv"):
    p = np.full(len(y), np.nan); ev = []
    for _, tr, te in splits(t): p[te] = predict(tr, te); ev.append(te)
    ev = np.concatenate(ev); out[t] = p; print(t, round(score(y[ev], p[ev]), 4), flush=True)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reimpl_bump_pred.npy"), np.stack([out["cv"], out["holdout"]]))
