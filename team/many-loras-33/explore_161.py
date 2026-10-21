"""Explore invariant feature groups using only the 161 holdout-train rows (10-fold CV inside them)."""
import os, sys; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, "team"); sys.path.insert(0, "team/many-loras-33")
from eval_common import labels, splits, score
from invariants import feats, ridge_cv, iu
L8 = labels(8); _, tr, _ = next(splits("holdout"))
names = [L8[i]["name"] for i in tr]; y = np.array([L8[i]["old_germany_rate"] for i in tr])
rows = [feats(f"team/many-loras-33/xl/{n}.npz") for n in names]
F = {k: np.array([r[k] for r in rows]) for k in rows[0]}
F["all"] = np.concatenate([F[k] for k in list(F)], 1)
mid = np.array([(10 <= l <= 24 and 10 <= m <= 24) for l, m in zip(*iu)])
F["xcos_mid"] = F["xcos"][:, mid]; F["outal_mid"] = F["outal"][:, mid]; F["inal_mid"] = F["inal"][:, mid]
folds = np.array_split(np.random.default_rng(5).permutation(len(y)), 10)
for g, X in F.items():
    p = np.zeros(len(y))
    for te in folds:
        p[te] = ridge_cv(X, y, np.setdiff1d(np.arange(len(y)), te), te)
    print(f"{g:10s} dim {X.shape[1]:5d} cv161 {score(y, p):+.3f}", flush=True)
# single strongest raw correlations, for a sense of scale (no model)
from scipy.stats import spearmanr
for g in ("lnorm", "spec", "xcos", "outal", "inal"):
    r = np.array([spearmanr(F[g][:, j], y)[0] for j in range(F[g].shape[1])])
    print(g, "max |rho| single feature", round(np.nanmax(abs(r)), 3), "at", int(np.nanargmax(abs(r))))
