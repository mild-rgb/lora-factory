"""Two follow-ups, training rows only (the 440 test rows are never used):
(1) Does dropping the rank-32/64 rows help rank-8 prediction?  OOF on the 161 rank-8 training rows (order-grouped
    5-fold, 3 fold seeds), training on [rest of r8] + r16 (+/- r32/64).
(2) Is rank 16 easier than rank 64 only because it has more LoRAs?  Within-rank CV for r16 subsampled to 448 rows
    (whole seed groups, 5 draws) vs r64 (448 rows), kernel gcos_mid."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_c as mc, explore_c as ec
from eval_common import labels, splits

rows = labels(); y = np.array([r["old_germany_rate"] for r in rows]); ranks = np.array([r["rank"] for r in rows])
groups = mc.seed_groups(rows)
(_, tr, te), = list(splits("holdout_plus"))
K = ec.CANDIDATES["gcos_mid+2in_sqrt_mid"]()
r8 = tr[ranks[tr] == 8]
for label, extra in (("r8+r16+r32+r64", tr[ranks[tr] != 8]), ("r8+r16", tr[ranks[tr] == 16])):
    rhos = []
    for sd in (21, 22, 23):
        oof = np.full(len(y), np.nan)
        for itr, iva in mc.inner_folds(r8, groups, k=5, seed=sd):
            sub = np.sort(np.concatenate([r8[itr], extra]))
            oof[r8[iva]] = mc.krr(K, mc.zscore_by_rank(y, ranks, sub), sub, r8[iva], groups, seeds=(1,))
        rhos.append(spearmanr(oof[r8], y[r8])[0])
    print(f"(1) train on {label:16s}: OOF rho on the 161 rank-8 training rows {np.mean(rhos):+.3f} (seeds {np.round(rhos, 3)})", flush=True)
Kg = ec.summed("gcos", "", range(12, 24))
rng = np.random.default_rng(0)
r16 = np.where(ranks == 16)[0]
ug = np.unique(groups[r16])
vals = []
for d in range(5):
    keep = set(rng.permutation(ug)[:448])
    rr = np.array([i for i in r16 if groups[i] in keep])
    oof = np.full(len(y), np.nan)
    for itr, iva in mc.inner_folds(rr, groups, k=5, seed=5):
        oof[rr[iva]] = mc.krr(Kg, y, rr[itr], rr[iva], groups, seeds=(1, 2))
    vals.append(spearmanr(oof[rr], y[rr])[0])
print(f"(2) r16 subsampled to 448: within-rank CV rho {np.mean(vals):+.3f} ± {np.std(vals, ddof=1):.3f} (r64 at 448: +0.147; r16 at 800: +0.410)")
