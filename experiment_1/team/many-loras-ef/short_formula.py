"""24-number interpretable model for many-loras-84's 'short formula' (holdout_plus).

Per layer l in 12..23:
  M_l = mean over the 2,049 holdout_plus TRAINING rows of sigma1 u1 v1^T. This rank-1 term is invariant to the joint
        (u1, v1) sign flip, so no sign alignment is needed. M_l is never formed: it is kept as a sum of rank-1 terms.
  a_l = <dW_x, M_l> / (||dW_x|| ||M_l||)  (signed overlap with the common update). dW_x is approximated by its top-4
        directions, which hold 93-98% of the energy; ||dW_x|| is exact, from all singular values.
        For TRAINING rows, M_l is computed leave-one-out (without x itself), which avoids the self-inclusion leak of the
        old w_align feature. Test rows use the full training mean, which never includes them.
  b_l = sigma1^2 / ||dW_x||^2 (energy concentration in the top direction).
Model: ridge on the 24 standardised features (training-row stats). alpha by order/seed-grouped inner CV, 3 fold seeds.
Target: within-rank z-score. The model is fully specified in advance, so the test is scored once.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_c as mc
from eval_common import labels, splits, score, log_result

LAYERS = list(range(12, 24))


def layer_features(l, tr):
    sv = np.load(mc.feat("svals.npy"))[:, l, :].astype(np.float64)              # [N, 64]
    nrm = np.sqrt((sv ** 2).sum(1))
    N, n = len(sv), len(tr)
    u = np.load(mc.feat(f"topvec/u_L{l:02d}.npy"), mmap_mode="r")
    v = np.load(mc.feat(f"topvec/v_L{l:02d}.npy"), mmap_mode="r")
    u1t = np.asarray(u[tr, 0], dtype=np.float64)                                # [n, 4096]
    v1t = np.asarray(v[tr, 0], dtype=np.float64)
    s1t = sv[tr, 0]
    # <dW_x, M> * n = sum_i s1_i sum_k s_xk (u1_i . u_xk)(v1_i . v_xk), dW_x truncated to top 4
    dot = np.zeros(N)
    for k in range(4):
        uk = np.asarray(u[:, k], dtype=np.float64)
        vk = np.asarray(v[:, k], dtype=np.float64)
        dot += sv[:, k] * ((uk @ u1t.T) * (vk @ v1t.T) @ s1t)
    # ||nM||^2 = sum_ij s1_i s1_j (u1_i . u1_j)(v1_i . v1_j)
    P = (u1t @ u1t.T) * (v1t @ v1t.T)
    nM2 = s1t @ P @ s1t
    a = dot / (np.sqrt(nM2) * nrm)                                              # test rows: full training mean
    # leave-one-out for training rows: nM' = nM - s_x u1_x v1_x^T, divided by (n-1)
    cross = P @ s1t                                                              # <u1_x v1_x^T, nM> / s1-free
    for j, x in enumerate(tr):
        dot_x = dot[x] - s1t[j] * sv[x, 0]        # <dW_x, s1_x u1_x v1_x^T> = s1_x * sigma_x1 exactly (top-4 basis)
        nM2_x = nM2 - 2 * s1t[j] * cross[j] + s1t[j] ** 2
        a[x] = dot_x / (np.sqrt(nM2_x) * nrm[x])
    b = sv[:, 0] ** 2 / nrm ** 2
    return a, b


def ridge_fit(X, z, tr, groups, alphas=np.logspace(-3, 4, 15)):
    def fit(rows_, a):
        mu, sd = X[rows_].mean(0), X[rows_].std(0) + 1e-12
        Z = (X[rows_] - mu) / sd
        w = np.linalg.solve(Z.T @ Z + a * np.eye(X.shape[1]), Z.T @ (z[rows_] - z[rows_].mean()))
        return lambda idx: ((X[idx] - mu) / sd) @ w + z[rows_].mean(), w
    best = None
    for a in alphas:
        err = 0.0
        for sd_ in (1, 2, 3):
            for itr, iva in mc.inner_folds(tr, groups, seed=sd_):
                f, _ = fit(tr[itr], a)
                err += ((f(tr[iva]) - z[tr[iva]]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, a)
    f, w = fit(tr, best[1])
    return f, w, best[1]


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    (_, tr, te), = list(splits("holdout_plus"))
    cache = os.path.join(mc.BLK, "short_formula_feats.npy")
    if os.path.exists(cache):
        F = np.load(cache)
    else:
        A, B = [], []
        for l in LAYERS:
            a, b = layer_features(l, tr)
            A.append(a); B.append(b)
            mc.log(f"layer {l}: features done")
        F = np.concatenate([np.stack(A, 1), np.stack(B, 1)], 1)                 # [N, 24]: a_12..a_23, b_12..b_23
        np.save(cache, F)
    z = mc.zscore_by_rank(y, ranks, tr)
    print("univariate within-rank Spearman with the rate, TRAINING rows only (mean over ranks):")
    for j, l in enumerate(LAYERS):
        ra = np.mean([spearmanr(F[tr][ranks[tr] == R, j], y[tr][ranks[tr] == R])[0] for R in (8, 16, 32, 64)])
        rb = np.mean([spearmanr(F[tr][ranks[tr] == R, 12 + j], y[tr][ranks[tr] == R])[0] for R in (8, 16, 32, 64)])
        print(f"  L{l}: a {ra:+.3f}   b {rb:+.3f}   (mean a over train {F[tr, j].mean():+.3f})")
    gm = np.load(os.path.join(mc.PRED, "explore_gcos_mid_holdout_plus.npy"))
    for name, cols in (("a+b (24)", list(range(24))), ("a only (12)", list(range(12))), ("b only (12)", list(range(12, 24)))):
        f, w, al = ridge_fit(F[:, cols], z, tr, groups)
        p = f(te)
        rho = score(y[te], p)
        print(f"{name:12s} holdout_plus rho {rho:+.4f}  (alpha {al:.3g});  Spearman vs gcos_mid preds {spearmanr(p, gm)[0]:+.3f}")
        if name.startswith("a+b"):
            print("  standardised coefficients a_l: " + " ".join(f"L{l}:{c:+.3f}" for l, c in zip(LAYERS, w[:12])))
            print("  standardised coefficients b_l: " + " ".join(f"L{l}:{c:+.3f}" for l, c in zip(LAYERS, w[12:])))
            log_result("many-loras-ef", "C:short_formula_24 (overlap with train-mean rank-1 update + top-1 energy share, L12-23)",
                       "holdout_plus", rho, "ridge on 24 interpretable features, LOO mean for training rows; for many-loras-84; "
                       "team/many-loras-ef/short_formula.py")


if __name__ == "__main__":
    main()
