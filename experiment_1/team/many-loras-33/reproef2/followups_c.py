"""Two follow-ups requested by many-loras-75 (2026-10-20).

(1) Honest choice of the training set for holdout_plus. Options: weight w on the rank-32/64 rows, w in {1, 0.5, 0.25, 0}
    (w = 0 is r8+r16 only; w = 1 is the verified model). Weighted kernel ridge: c = (Kc + alpha diag(1/w))^-1 (y - mean).
    Selection criterion, fixed in advance: mean over fold seeds 21-23 of the OOF Spearman on the 161 rank-8 training rows
    (order-grouped 5-fold), the higher-rank rows always in training. The pick is then fitted on all training rows and
    scored on the 440 once. Test rows are never used for the choice.
(2) Why is rank-8-only CV (0.115) so much lower than the cv track (~0.4)? Kernel gcos_mid, rank-8 rows only:
    a. cv-track protocol: the 601 rows, the official 10 folds (inner alpha CV grouped by order)
    b. the 161 old rows, random 5-fold
    c. the 161 old rows, order-grouped 5-fold (as in rank_diag)
    d. random 161-row subsets of the 440 new rows, 5-fold (all fresh orders, so grouping = random)
    e. the 440 new rows, 5-fold
    This is a DIAGNOSTIC that uses the test rows' labels (as the cv track does); nothing in it feeds a model choice.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_c as mc, explore_c as ec
from eval_common import labels, splits, score, log_result


def wkrr_fit_pred(K, y, w, tr, te, alpha):
    Ktt, Kxt = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    Kxc = Kxt - cm[None, :] - Kxt.mean(1)[:, None] + cm.mean()
    c = np.linalg.solve(Kc + alpha * np.diag(1 / w[tr]), y[tr] - y[tr].mean())
    return Kxc @ c + y[tr].mean()


def wkrr(K, y, w, tr, te, groups, alphas=np.logspace(-5, 3, 9)):
    """alpha by grouped inner CV with w-weighted squared error (fold seed 1)."""
    tr = tr[w[tr] > 0]
    scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    best = None
    for a in alphas * scale:
        err = 0.0
        for itr, iva in mc.inner_folds(tr, groups, seed=1):
            p = wkrr_fit_pred(K, y, w, tr[itr], tr[iva], a)
            err += (w[tr[iva]] * (p - y[tr[iva]]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, a)
    return wkrr_fit_pred(K, y, w, tr, te, best[1])


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    (_, tr, te), = list(splits("holdout_plus"))
    K = ec.CANDIDATES["gcos_mid+2in_sqrt_mid"]()
    r8 = tr[ranks[tr] == 8]
    hi = tr[ranks[tr] != 8]
    print("(1) training-set choice, criterion = OOF rho on the 161 rank-8 training rows", flush=True)
    sel = {}
    for wv in (1.0, 0.5, 0.25, 0.0):
        w = np.ones(len(y))
        w[(ranks == 32) | (ranks == 64)] = wv
        rhos = []
        for sd in (21, 22, 23):
            oof = np.full(len(y), np.nan)
            for itr, iva in mc.inner_folds(r8, groups, k=5, seed=sd):
                sub = np.sort(np.concatenate([r8[itr], hi]))
                oof[r8[iva]] = wkrr(K, mc.zscore_by_rank(y, ranks, sub), w, sub, r8[iva], groups)
            rhos.append(spearmanr(oof[r8], y[r8])[0])
        sel[wv] = (np.mean(rhos), w)
        print(f"   w(r32/64) = {wv:4.2f}: train-only OOF rho {np.mean(rhos):+.4f}  (seeds {np.round(rhos, 3)})", flush=True)
    wbest = max(sel, key=lambda k: sel[k][0])
    w = sel[wbest][1]
    p = wkrr(K, mc.zscore_by_rank(y, ranks, tr[w[tr] > 0]), w, tr, te, groups)
    rho = score(y[te], p)
    print(f"   train-CV picks w = {wbest}; holdout_plus test rho {rho:+.4f} (verified w=1 model: 0.4177)", flush=True)
    np.save(os.path.join(mc.PRED, f"followup_w{wbest}_holdout_plus.npy"), p)
    print("VERIFIER: would log",  # log_result disabled in verifier copy
          "many-loras-ef", f"C:gcos_mid+2in_sqrt_mid_w3264={wbest}", "holdout_plus", rho,
               f"weight on r32/64 rows chosen from {{1,.5,.25,0}} by train-only OOF on the 161 rank-8 rows "
               f"({sel[wbest][0]:.3f} vs {sel[1.0][0]:.3f} for w=1); team/many-loras-ef/followups_c.py")

    return  # verifier copy: part 2 (diagnostic) skipped
    print("(2) rank-8-only CV, kernel gcos_mid (diagnostic)", flush=True)
    Kg = ec.summed("gcos", "", range(12, 24))
    all8 = np.where(ranks == 8)[0]
    new = te

    def cv(idx, folds):
        oof = np.full(len(y), np.nan)
        for itr, iva in folds:
            oof[idx[iva]] = mc.krr(Kg, y, idx[itr], idx[iva], groups, seeds=(1, 2))
        return spearmanr(oof[idx], y[idx])[0]

    def rand_folds(n, k=5, seed=5):
        f = np.array_split(np.random.default_rng(seed).permutation(n), k)
        return [(np.setdiff1d(np.arange(n), v), v) for v in f]

    off = [(np.searchsorted(all8, all8[np.setdiff1d(np.arange(601), te_)]), np.searchsorted(all8, all8[te_]))
           for _, _, te_ in splits("cv")]
    print(f"   a. 601 rows, official cv folds:            {cv(all8, off):+.3f}", flush=True)
    print(f"   b. 161 old rows, random 5-fold:            {cv(r8, rand_folds(len(r8))):+.3f}", flush=True)
    print(f"   c. 161 old rows, order-grouped 5-fold:     {cv(r8, mc.inner_folds(r8, groups, k=5, seed=5)):+.3f}", flush=True)
    d = []
    for s in range(5):
        sub = np.sort(np.random.default_rng(100 + s).choice(new, 161, replace=False))
        d.append(cv(sub, rand_folds(161, seed=s)))
    print(f"   d. 161 random new rows, 5-fold (5 draws):  {np.mean(d):+.3f} ± {np.std(d, ddof=1):.3f}", flush=True)
    print(f"   e. 440 new rows, 5-fold:                   {cv(new, rand_folds(len(new))):+.3f}", flush=True)


if __name__ == "__main__":
    main()
