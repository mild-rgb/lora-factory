"""Exp 3: spectral regularisers and spectral features on the per-layer cosine kernels.

  pcr        principal-component regression on the summed cosine kernel: keep the top k kernel-PCA components of the
             training rows (k by grouped inner CV) instead of ridge shrinkage
  kpca_lasso per-layer kernel-PCA scores (top k per layer, train-rows-only eigenvectors) for all 36 layers,
             concatenated, then ridge / lasso with nested grouped CV
  trans_*    the same but with eigenvectors from ALL 601 LoRAs' weights (transductive: no labels from test rows,
             only their weights). Reported separately and flagged in notes.
"""
import dlib  # noqa: F401  (first import: pins BLAS to 1 thread before numpy/scipy/sklearn load)
import csv, sys, time
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, Ridge
from sklearn.model_selection import GroupKFold
from dlib import load, cosine, run, order_groups, group_folds, krr_fit_pred

G, y = load()
C = cosine(G)
L, N = C.shape[:2]
groups = order_groups()


def centre(K, ref):
    """Centre kernel rows/cols w.r.t. the mean feature of the reference rows."""
    m = K[:, ref].mean(1)
    return K - m[:, None] - m[None, :] + K[np.ix_(ref, ref)].mean()


def kpca_scores(K, ref, k):
    """Kernel-PCA scores of every row on the top-k components of the reference rows."""
    Kc = centre(K, ref)
    w, V = np.linalg.eigh(Kc[np.ix_(ref, ref)])
    w, V = w[::-1][:k], V[:, ::-1][:, :k]
    return Kc[:, ref] @ V / np.sqrt(np.clip(w, 1e-12, None))     # [N, k], unit-variance-ish scale per comp


def pcr(tr, te, K=None):
    K = C.sum(0) if K is None else K
    folds = group_folds(groups[tr], 5)
    ks = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
    err = np.zeros(len(ks))
    for iv in folds:
        it = tr[np.setdiff1d(np.arange(len(tr)), iv)]
        S = kpca_scores(K, it, max(ks))
        for j, k in enumerate(ks):
            X = S[:, :k]
            b, *_ = np.linalg.lstsq(np.c_[np.ones(len(it)), X[it]], y[it], rcond=None)
            err[j] += ((np.c_[np.ones(len(iv)), X[tr[iv]]] @ b - y[tr[iv]]) ** 2).sum()
    k = ks[int(np.argmin(err))]
    print(f"    pcr k={k}", flush=True)
    X = kpca_scores(K, tr, k)
    b, *_ = np.linalg.lstsq(np.c_[np.ones(len(tr)), X[tr]], y[tr], rcond=None)
    return np.c_[np.ones(len(te)), X[te]] @ b


def layer_feats(ref, k, lays=range(L)):
    return np.concatenate([kpca_scores(C[l], ref, k) for l in lays], 1)


def kpca_model(tr, te, kind, transductive):
    ref = np.arange(N) if transductive else tr
    best = None
    gkf = GroupKFold(5)
    for k in (2, 5, 10, 20):
        X = layer_feats(ref, k)
        X = X / X[tr].std(0)
        if kind == "ridge":
            m = RidgeCV(alphas=np.logspace(-1, 5, 25), cv=list(gkf.split(X[tr], y[tr], groups[tr])))
        else:
            m = LassoCV(n_alphas=30, cv=list(gkf.split(X[tr], y[tr], groups[tr])), max_iter=5000)
        m.fit(X[tr], y[tr])
        # inner score of the chosen alpha for this k, on the same grouped folds
        e = 0.0
        for it, iv in gkf.split(X[tr], y[tr], groups[tr]):
            mm = (Ridge(alpha=m.alpha_) if kind == "ridge" else type(m)(alphas=[m.alpha_], cv=3, max_iter=5000))
            mm.fit(X[tr][it], y[tr][it])
            e += ((mm.predict(X[tr][iv]) - y[tr][iv]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, k, m, X)
    print(f"    {kind} {'trans' if transductive else 'train'} k={best[1]}", flush=True)
    return best[2].predict(best[3][te])


methods = {"pcr": pcr,
           "kpca_ridge": lambda tr, te: kpca_model(tr, te, "ridge", False),
           "kpca_lasso": lambda tr, te: kpca_model(tr, te, "lasso", False),
           "trans_kpca_ridge": lambda tr, te: kpca_model(tr, te, "ridge", True),
           "trans_kpca_lasso": lambda tr, te: kpca_model(tr, te, "lasso", True)}
rows = []
for name in sys.argv[1:] or list(methods):
    t0 = time.time()
    res = run(methods[name], y)
    print(f"{name:18s} cv {res['cv']:+.3f}  holdout {res['holdout']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    rows.append((name, round(res["cv"], 4), round(res["holdout"], 4)))
    np.save(f"exp3_pred_{name}.npy", np.stack([res["cv_pred"], res["holdout_pred"]]))
with open("exp3_results.csv", "a", newline="") as f:
    csv.writer(f).writerows(rows)
