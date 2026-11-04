"""Learning curve for holdout_plus: how does test rho on the 440 grow with the number of training LoRAs?

Model: the train-CV winner (cosine Gram L12-23 + 2 x sqrt input-loading kernel L12-23), kernel ridge with alpha by
order/seed-grouped inner CV on the subset (fold seed 1), target z-scored within rank on the subset.
Subsets are drawn by whole groups (data order at rank 8; seed for higher ranks), so siblings stay together.
Curves (the other ranks are kept in full unless the curve says otherwise):
  r8      fraction of the 161 rank-8 rows
  r16     fraction of the 800 rank-16 rows
  r32_64  fraction of the 640 + 448 rank-32/64 rows
  hi      fraction of all 1,888 higher-rank rows (r8 in full)
  all     fraction of every training row
This is a fixed-design analysis of the final model, not model selection: nothing here is chosen on the test rows.
Output: lcurve.csv (curve, fraction, repeat, n_r8, n_r16, n_r32, n_r64, n_train, test_rho).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models_c as mc, explore_c as ec
from eval_common import labels, splits, score

CURVES = {"r8": (8,), "r16": (16,), "r32_64": (32, 64), "hi": (16, 32, 64), "all": (8, 16, 32, 64)}
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
REPEATS = 5


def subsample(tr, ranks, groups, vary, frac, rng):
    keep = [i for i in tr if ranks[i] not in vary]
    pool = np.array([i for i in tr if ranks[i] in vary])
    if frac >= 1:
        return np.sort(np.concatenate([keep, pool]).astype(int))
    ug = np.unique(groups[pool])
    chosen = set(rng.permutation(ug)[: int(round(frac * len(ug)))])
    sel = [i for i in pool if groups[i] in chosen]
    return np.sort(np.array(keep + sel, dtype=int))


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    (_, tr, te), = list(splits("holdout_plus"))
    K = ec.CANDIDATES["gcos_mid+2in_sqrt_mid"]()
    out = os.path.join(HERE, "lcurve.csv")
    done = set()
    if os.path.exists(out):
        done = {tuple(l.split(",")[:3]) for l in open(out).read().splitlines()[1:]}
    else:
        open(out, "w").write("curve,fraction,repeat,n_r8,n_r16,n_r32,n_r64,n_train,test_rho\n")
    for curve, vary in CURVES.items():
        for frac in FRACS:
            if curve == "all" and frac == 0:
                continue
            for rep in range(1 if frac in (0.0, 1.0) else REPEATS):
                key = (curve, f"{frac:.2f}", str(rep))
                if key in done:
                    continue
                t0 = time.time()
                rng = np.random.default_rng(1000 * rep + int(100 * frac) + 7 * list(CURVES).index(curve))  # reproducible
                sub = subsample(tr, ranks, groups, vary, frac, rng)
                z = mc.zscore_by_rank(y, ranks, sub)
                p = mc.krr(K, z, sub, te, groups, seeds=(1,))
                rho = score(y[te], p)
                n = [int((ranks[sub] == R).sum()) for R in (8, 16, 32, 64)]
                open(out, "a").write(f"{curve},{frac:.2f},{rep},{n[0]},{n[1]},{n[2]},{n[3]},{len(sub)},{rho:.4f}\n")
                mc.log(f"{curve:7s} frac {frac:.2f} rep {rep}  n={n}  test rho {rho:+.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
