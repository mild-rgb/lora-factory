"""How many numbers does the weight signal need? Kernel-PCA compression curve for many-loras-84 (holdout_plus).

Kernel: gcos_mid (mean of the 12 per-layer cosine Grams, L12-23). This is a linear kernel on the concatenated, per-layer
normalised dW, so kernel PCA = PCA of those concatenated updates.
For a training set T: centre K on T, eigendecompose, PCA scores S = V sqrt(L) for T and Kc_xT V / sqrt(L) for other rows.
Model for each k: ridge on the top-k raw scores, w = (L_k + alpha)^-1 S_k^T z. With k = |T| this equals KRR.
  - train-CV curve: order/seed-grouped 5-fold CV on the 2,049 training rows; PCA recomputed inside every fold (train-only);
    alpha per k by the best within-fold choice (same grid as models_c.krr); mean within-rank Spearman of the OOF.
  - k chosen by train-CV; the test curve (all k) is printed for the write-up with Spearman vs the gcos_mid predictions.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_c as mc, explore_c as ec
from eval_common import labels, splits, score, score_within_rank, log_result

KS = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 130, 200, 300, 500, 800, 1300]
ALPHAS = np.logspace(-5, 3, 9)


def pca(K, T, X):
    Ktt = K[np.ix_(T, T)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    lam, V = np.linalg.eigh(Kc)
    lam, V = lam[::-1], V[:, ::-1]
    keep = lam > 1e-10 * lam[0]
    lam, V = lam[keep], V[:, keep]
    Kx = K[np.ix_(X, T)]
    Kxc = Kx - cm[None, :] - Kx.mean(1)[:, None] + cm.mean()
    return lam, V * np.sqrt(lam), Kxc @ V / np.sqrt(lam)          # eigenvalues, train scores, scores of X


def ridge_preds(lam, St, Sx, z, scale):
    """Predictions on X for every (k, alpha): array [len(KS), len(ALPHAS), len(X)]."""
    zc = z - z.mean()
    proj = St.T @ zc                                                # = lam * (V^T z) * ... per component
    out = np.zeros((len(KS), len(ALPHAS), Sx.shape[0]))
    for ia, a in enumerate(ALPHAS * scale):
        w = proj / (lam + a)
        contrib = Sx * w[None, :]
        cs = np.cumsum(contrib, 1)
        for ik, k in enumerate(KS):
            out[ik, ia] = cs[:, min(k, len(lam)) - 1] + z.mean()
    return out


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    (_, tr, te), = list(splits("holdout_plus"))
    K = ec.summed("gcos", "", range(12, 24))
    gm = np.load(os.path.join(mc.PRED, "explore_gcos_mid_holdout_plus.npy"))
    # train-CV: OOF predictions for every (k, alpha), alpha per k chosen by inner MSE inside the outer training part
    oof = np.full((len(KS), len(y)), np.nan)
    for itr, iva in mc.inner_folds(tr, groups, k=5, seed=7):
        T, X = tr[itr], tr[iva]
        z = mc.zscore_by_rank(y, ranks, T)
        scale = np.trace(K[np.ix_(T, T)]) / len(T)
        # alpha per k by a nested grouped split inside T (one fold seed)
        err = np.zeros((len(KS), len(ALPHAS)))
        for jtr, jva in mc.inner_folds(T, groups, k=5, seed=1):
            lam, St, Sx = pca(K, T[jtr], T[jva])
            p = ridge_preds(lam, St, Sx, z[T[jtr]], scale)
            err += ((p - z[T[jva]][None, None, :]) ** 2).sum(2)
        lam, St, Sx = pca(K, T, X)
        p = ridge_preds(lam, St, Sx, z[T], scale)
        for ik in range(len(KS)):
            oof[ik, X] = p[ik, np.argmin(err[ik])]
        mc.log(f"outer fold done ({len(T)} train rows, {len(lam)} components)")
    # final fits on all training rows, alpha per k by grouped inner CV
    z = mc.zscore_by_rank(y, ranks, tr)
    scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    err = np.zeros((len(KS), len(ALPHAS)))
    for jtr, jva in mc.inner_folds(tr, groups, k=5, seed=1):
        lam, St, Sx = pca(K, tr[jtr], tr[jva])
        err += ((ridge_preds(lam, St, Sx, z[tr[jtr]], scale) - z[tr[jva]][None, None, :]) ** 2).sum(2)
    lam, St, Sx = pca(K, tr, te)
    P = ridge_preds(lam, St, Sx, z[tr], scale)
    print(f"{'k':>5s}  {'train-CV':>8s}  {'test rho':>8s}  {'rho vs gcos_mid':>15s}  {'var share':>9s}")
    best = None
    for ik, k in enumerate(KS):
        cvr = score_within_rank(y[tr], oof[ik, tr], ranks[tr])[0]
        p = P[ik, np.argmin(err[ik])]
        share = lam[:k].sum() / lam.sum()
        print(f"{k:5d}  {cvr:+8.4f}  {score(y[te], p):+8.4f}  {spearmanr(p, gm)[0]:+15.3f}  {share:9.3f}", flush=True)
        if best is None or cvr > best[0]:
            best = (cvr, k, p)
    rho = score(y[te], best[2])
    print(f"train-CV picks k = {best[1]} (train-CV {best[0]:.4f}); holdout_plus test rho {rho:+.4f}")
    log_result("many-loras-ef", f"C:kpca_gcos_mid_k{best[1]}", "holdout_plus", rho,
               f"ridge on top-k kernel-PCA scores of the train-centred gcos_mid kernel; k chosen by train-only grouped CV "
               f"({best[0]:.3f}); compression curve for many-loras-84; team/many-loras-ef/pca_curve.py")


if __name__ == "__main__":
    main()
