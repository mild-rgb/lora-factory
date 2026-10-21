"""Do any compact per-LoRA features explain what the best kernel (gcos_mid + 2 sqrt in_load) misses?
Train rows of holdout_plus only: grouped 5-fold OOF kernel predictions, residual z - oof, then Spearman of each
compact feature with the residual within each rank. Test rows are not touched."""
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
z = mc.zscore_by_rank(y, ranks, tr)
oof = np.full(len(y), np.nan)
for itr, iva in mc.inner_folds(tr, groups, k=5, seed=11):
    oof[tr[iva]] = mc.krr(K, mc.zscore_by_rank(y, ranks, tr[itr]), tr[itr], tr[iva], groups, seeds=(1,))
res = z - oof
np.save(os.path.join(mc.PRED, "resid_train_gcos_mid+2in_sqrt_mid.npy"), np.stack([oof, res]))
names = []
for b, n in (("spec", ["logn2", "p1", "p2", "p3", "p4", "effr", "ipr"]),):
    pass
X = {}
sp = np.load(os.path.join(mc.BLK, "spec.npy"))
for j, f in enumerate(["logn2", "p1", "p2", "p3", "p4", "effr", "ipr"]):
    for l in range(36):
        X[f"spec_{f}_L{l}"] = sp[:, j * 36 + l]
pr = np.load(os.path.join(mc.BLK, "probe.npy"))
gn = ["city_former", "city_modern", "era", "year", "control"]
for j, g in enumerate(gn):
    for l in range(36):
        X[f"probe_{g}_L{l}"] = pr[:, j * 36 + l]
xl = np.load(os.path.join(mc.BLK, "xl.npy"))
iu = np.triu_indices(36, 1)
for j, k in enumerate(["gram", "out4", "in4", "u_ov", "v_ov"]):
    for c, (a, b) in enumerate(zip(*iu)):
        if abs(a - b) == 1:
            X[f"xl_{k}_L{a}-{b}"] = xl[:, j * 630 + c]
bd = np.load(mc.feat("base/base_dot.npy")); n2 = mc.norms2()
for l in range(36):
    X[f"basecos_L{l}"] = bd[:, l] / np.sqrt(n2[:, l])
out = []
for k, v in X.items():
    rr = [spearmanr(v[tr][ranks[tr] == R], res[tr][ranks[tr] == R])[0] for R in (8, 16, 32, 64)]
    out.append((np.nanmean(rr), k, rr))
out.sort(key=lambda t: -abs(t[0]))
print(f"{len(out)} features; residual vs feature, mean within-rank Spearman (train rows only):")
for m, k, rr in out[:15]:
    print(f"  {k:28s} {m:+.3f}   " + " ".join(f"{x:+.2f}" for x in rr))
# null: max |mean rho| expected by chance with this many features (permute residual within rank)
rng = np.random.default_rng(0); mx = []
for _ in range(20):
    pr_ = res.copy()
    for R in (8, 16, 32, 64):
        m = np.where(ranks == R)[0]; m = np.intersect1d(m, tr); pr_[m] = rng.permutation(res[m])
    mx.append(max(abs(np.nanmean([spearmanr(v[tr][ranks[tr] == R], pr_[tr][ranks[tr] == R])[0] for R in (8, 16, 32, 64)])) for v in list(X.values())[::7]))
print("null max |rho| over a 1/7 subsample of features, 20 perms: median %.3f, max %.3f" % (np.median(mx), np.max(mx)))
