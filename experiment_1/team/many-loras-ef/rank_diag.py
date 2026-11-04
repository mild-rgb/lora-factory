"""Why is rank 64 poorly predicted? Within-rank-only CV per rank, for kernels over different layer windows and for
Grams of only the top-k singular directions (k = 1, 2; from topvec, L12-23).

Rows used: rank 8 = the 161 holdout-training rows only (the 440 test rows are never touched); ranks 16/32/64 = all
rows. 5-fold CV within the rank, folds grouped by data order / seed, alpha re-chosen inside each fold.
Output: rank_diag.csv (kernel, rank, n, within-rank rho of OOF predictions).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models_c as mc, explore_c as ec
from eval_common import labels, splits


def topk_layer_kernel(l, k):
    p = os.path.join(ec.PL, f"top{k}_cos_L{l:02d}.npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float64)
    sv = np.load(mc.feat("svals.npy"))[:, l, :k].astype(np.float64)            # [N, k]
    u = np.load(mc.feat(f"topvec/u_L{l:02d}.npy"), mmap_mode="r")[:, :k].astype(np.float32)
    v = np.load(mc.feat(f"topvec/v_L{l:02d}.npy"), mmap_mode="r")[:, :k].astype(np.float32)
    N = len(sv)
    UU = (u.reshape(N * k, -1) @ u.reshape(N * k, -1).T).astype(np.float64).reshape(N, k, N, k)
    VV = (v.reshape(N * k, -1) @ v.reshape(N * k, -1).T).astype(np.float64).reshape(N, k, N, k)
    G = np.einsum("ia,iajb,iajb,jb->ij", sv, UU, VV, sv)
    d = np.sqrt(np.diag(G))
    K = G / np.outer(d, d)
    np.save(p, K.astype(np.float32))
    mc.log(f"  built top{k}_cos_L{l:02d}")
    return K


KERNELS = {
    "gcos_all": lambda: ec.summed("gcos", "", range(36)),
    "gcos_early": lambda: ec.summed("gcos", "", range(0, 12)),
    "gcos_mid": lambda: ec.summed("gcos", "", range(12, 24)),
    "gcos_late": lambda: ec.summed("gcos", "", range(24, 36)),
    "top1_mid": lambda: sum(topk_layer_kernel(l, 1) for l in range(12, 24)) / 12,
    "top2_mid": lambda: sum(topk_layer_kernel(l, 2) for l in range(12, 24)) / 12,
    "in_sqrt_mid": lambda: ec.summed("in", "sqrt", range(12, 24)),
}


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    (_, tr_hp, _), = list(splits("holdout_plus"))
    out = os.path.join(HERE, "rank_diag.csv")
    if not os.path.exists(out):
        open(out, "w").write("kernel,rank,n,within_rank_rho\n")
    for name in (sys.argv[1:] or KERNELS):
        t0 = time.time()
        K = KERNELS[name]()
        res = []
        for R in (8, 16, 32, 64):
            rr = np.intersect1d(np.where(ranks == R)[0], tr_hp)                  # rank 8: the 161 only
            oof = np.full(len(y), np.nan)
            for itr, iva in mc.inner_folds(rr, groups, k=5, seed=5):
                oof[rr[iva]] = mc.krr(K, y, rr[itr], rr[iva], groups, seeds=(1, 2))
            rho = spearmanr(oof[rr], y[rr])[0]
            res.append(rho)
            open(out, "a").write(f"{name},{R},{len(rr)},{rho:.4f}\n")
        mc.log(f"{name:14s} within-rank-only CV rho  r8 {res[0]:+.3f}  r16 {res[1]:+.3f}  r32 {res[2]:+.3f}  "
               f"r64 {res[3]:+.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
