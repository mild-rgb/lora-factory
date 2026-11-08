"""Separate weights-only ridge predictors of collapse for old-name and modern-name bird LoRAs, and transfer.

Target: logit((k + 0.5) / (n - k + 0.5)) of the short share (answers < 15 words; labels.csv, 500 answers per LoRA).
Kernels (kernels.npz, per-layer <dW_i, dW_j>): the same models as length_pred.py
  cos_all  mean over layers of per-layer cosine;  lin_all  raw linear kernel summed over layers
  norms    linear kernel on the 36 standardised log per-layer norms;  cos+norms  both, each scaled to unit diagonal
  cos_Lxx  single layer (reported, never selected)
Kernel ridge with intercept; ridge strength by 5-fold CV inside the training set only. "selected" = best of the
first four by inner CV MSE.
Evaluations
  within   10 x 10-fold CV inside one group (seeds are unique inside a group)
  transfer fit on all of one group, test on the other. Seeds s01-s08 exist in both groups with the same init and
           data order, so they are dropped from BOTH sides of every transfer test.
Metrics: Spearman, pairwise ordering accuracy, AUC for collapse (> 50% short), all within the test group.
Run locally: OMP_NUM_THREADS=1 python ridge_transfer.py   (reads kernels.npz + labels.csv next to this file).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "kernels.npz")
names = list(z["names"])
K = z["K"].astype(np.float64)
lab = {r["name"]: r for r in csv.DictReader(open(HERE / "labels.csv"))}
N = len(names)
grp = np.array([lab[n]["group"] for n in names])
seed = np.array([int(lab[n]["seed"]) for n in names])
short = np.array([float(lab[n]["short"]) for n in names])
nans = np.array([int(lab[n]["n"]) for n in names])
k_ = np.round(short * nans)
y = np.log((k_ + 0.5) / (nans - k_ + 0.5))
rng = np.random.default_rng(0)

d2 = np.einsum("lii->li", K)
Kcos_l = K / np.sqrt(d2[:, :, None] * d2[:, None, :])
CS = np.logspace(-5, 4, 19)


def fit(Ktt, yt, c):
    m = Ktt.mean(0)
    Kc = Ktt - m[None, :] - m[:, None] + m.mean()
    a = np.linalg.solve(Kc + c * max(np.mean(np.diag(Kc)), 1e-12) * np.eye(len(yt)), yt - yt.mean())
    return a, m, yt.mean()


def predict(Kxt, model):
    a, m, b = model
    return (Kxt - Kxt.mean(1, keepdims=True) - m[None, :] + m.mean()) @ a + b


def choose_c(Km, idx, seed_=0):
    p = np.random.default_rng(seed_).permutation(len(idx))
    err = np.zeros(len(CS))
    for f in range(5):
        te = idx[p[f::5]]; tr = np.setdiff1d(idx, te)
        for ci, c in enumerate(CS):
            err[ci] += np.sum((predict(Km[np.ix_(te, tr)], fit(Km[np.ix_(tr, tr)], y[tr], c)) - y[te]) ** 2)
    return CS[int(np.argmin(err))], float(err.min() / len(idx))


def make_models(train_idx):
    """Kernels whose scaling / standardisation uses the training rows only."""
    X = 0.5 * np.log(d2.T)
    X = (X - X[train_idx].mean(0)) / X[train_idx].std(0)
    Kn = X @ X.T
    unit = lambda Km: Km / np.mean(np.diag(Km)[train_idx])
    Kc = Kcos_l.mean(0)
    M = {"cos_all": Kc, "lin_all": unit(K.sum(0)), "norms": Kn, "cos+norms": unit(Kc) + unit(Kn)}
    for l in range(36):
        M[f"cos_L{l:02d}"] = Kcos_l[l]
    return M


def pairwise(p, t):
    i, j = np.triu_indices(len(t), 1)
    dt, dp = np.sign(t[i] - t[j]), np.sign(p[i] - p[j])
    ok = dt != 0
    return float(np.mean(dt[ok] * dp[ok] > 0) + 0.5 * np.mean(dp[ok] == 0))


def auc(p, labl):
    if labl.sum() < 2 or (~labl).sum() < 2:
        return float("nan")
    r = rankdata(p); n1, n0 = labl.sum(), (~labl).sum()
    return float((r[labl].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(p, idx):
    t = y[idx]
    bs = [spearmanr(p[b], t[b])[0] for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000))]
    return dict(n=int(len(idx)), spearman=float(spearmanr(p, t)[0]),
                spearman_ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                pairwise=pairwise(p, t), auc_collapse=auc(p, short[idx] > 0.5), n_collapse=int((short[idx] > 0.5).sum()))


res = {"n": {g: int((grp == g).sum()) for g in ("old", "modern")}}
MAIN = ["cos_all", "lin_all", "norms", "cos+norms"]
for g_tr in ("old", "modern"):
    g_te = "modern" if g_tr == "old" else "old"
    # ---- within-group CV (10 x 10 fold), model and ridge picked inside each training fold
    idx = np.where(grp == g_tr)[0]
    oof = {m: np.zeros((10, len(idx))) for m in MAIN}
    for r in range(10):
        p = np.random.default_rng(100 + r).permutation(len(idx))
        for f in range(10):
            te_pos = p[f::10]; te = idx[te_pos]; tr = np.setdiff1d(idx, te)
            M = make_models(tr)
            for m in MAIN:
                c, _ = choose_c(M[m], tr, r)
                oof[m][r, te_pos] = predict(M[m][np.ix_(te, tr)], fit(M[m][np.ix_(tr, tr)], y[tr], c))
    w = {}
    for m in MAIN:
        sp = [spearmanr(oof[m][r], y[idx])[0] for r in range(10)]
        pw = [pairwise(oof[m][r], y[idx]) for r in range(10)]
        au = [auc(oof[m][r], short[idx] > 0.5) for r in range(10)]
        w[m] = dict(spearman=float(np.mean(sp)), spearman_sd=float(np.std(sp)), pairwise=float(np.mean(pw)),
                    auc_collapse=float(np.nanmean(au)))
    res[f"within_{g_tr}"] = w
    print(f"within {g_tr:6s}", {m: round(v["spearman"], 3) for m, v in w.items()}, flush=True)

    # ---- transfer: fit on all of g_tr (minus shared seeds 1-8), test on g_te (minus seeds 1-8)
    tr = np.where((grp == g_tr) & (seed > 8))[0]
    te = np.where((grp == g_te) & (seed > 8))[0]
    M = make_models(tr)
    cv = {m: choose_c(M[m], tr, 7) for m in M}
    sel = min(MAIN, key=lambda m: cv[m][1])
    t = {"selected": sel, "models": {}}
    for m in M:
        pr = predict(M[m][np.ix_(te, tr)], fit(M[m][np.ix_(tr, tr)], y[tr], cv[m][0]))
        t["models"][m] = metrics(pr, te) if m in MAIN else {"spearman": float(spearmanr(pr, y[te])[0])}
    # null: shuffled training targets (20x) for the selected model
    nl = []
    for _ in range(20):
        yp = y.copy(); yp[tr] = rng.permutation(y[tr])
        a, mm, b = fit(M[sel][np.ix_(tr, tr)], yp[tr], cv[sel][0])
        nl.append(spearmanr(predict(M[sel][np.ix_(te, tr)], (a, mm, b)), y[te])[0])
    t["null_selected_spearman"] = dict(mean=float(np.mean(nl)), max=float(np.max(nl)))
    res[f"transfer_{g_tr}_to_{g_te}"] = t
    print(f"transfer {g_tr}->{g_te} selected {sel}:",
          {m: round(v["spearman"], 3) for m, v in t["models"].items() if m in MAIN}, "null max", round(max(nl), 3), flush=True)
    print("   per-layer transfer spearman:", [round(t["models"][f"cos_L{l:02d}"]["spearman"], 2) for l in range(36)], flush=True)

json.dump(res, open(HERE / "ridge_transfer_results.json", "w"), indent=1)
print("saved")
