"""H14 evaluation: do downstream-reader couplings of dW's output directions predict strength?

ds_norms.npz: ds[i, l(12..23), k(0,1), m(8 readers), w(gate,up,q,k,v)] = ||W_w^(m) (gamma * u_k)||
for UNIT u_k; cal[100, m, w] = same for random unit vectors (per-reader calibration).
Caveats (75): q_norm/k_norm after projection not included; norms are unit-direction (scale by
svals for strength).

Variants (all evaluated on holdout via the shared protocol; named up front, all logged):
  geom    calibrated couplings, unit directions (pure geometry), readers m > write layer
  strength  same scaled by sigma_k / ||dW||_layer-norm convention (sigma-weighted)
  late    aggregate: mean coupling to readers >= 26 per (l, w), both variants
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)
sys.path.insert(0, os.path.join(TEAM, "many-loras-ef"))
os.environ.setdefault("TRACKC_BLK", os.path.join(HERE, "blocks"))
import models_c as mc
from eval_common import labels, splits, score, log_result

AUTHOR = "many-loras-84"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

from huggingface_hub import hf_hub_download
p = hf_hub_download("mild-rgb/lora-factory-private", "team/many-loras-84/ds_norms.npz",
                    local_dir=os.path.join(HERE, "cache"))
z = np.load(p)
ds, cal = z["ds"].astype(np.float64), z["cal"].astype(np.float64)
WL, RD = list(z["write_layers"]), list(z["readers"])
N = len(ds)

ds = ds / cal.mean(0)[None, None, None]            # calibrate per (reader, weight-type)
mask = np.array([[m > l for m in RD] for l in WL], float)   # readers after the write layer
ds = ds * mask[None, :, None, :, None]

sv = np.load(mc.feat("svals.npy"), mmap_mode="r")
sig = np.asarray(sv[:, WL, :2], np.float64)        # (N, 12, 2)
n2 = np.sqrt((np.asarray(sv, np.float64)[:, WL] ** 2).sum(2))  # ||dW|| per write layer

F_geom = ds.reshape(N, -1)
F_str = (ds * (sig / n2[:, :, None])[:, :, :, None, None]).reshape(N, -1)
late = [i for i, m in enumerate(RD) if m >= 26]
F_late = np.concatenate([ds[:, :, :, late].mean(3).reshape(N, -1),
                         (ds * sig[:, :, :, None, None] / n2[:, :, None, None, None])[:, :, :, late].mean(3).reshape(N, -1)], 1)

rows = labels()
y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
grp = mc.seed_groups(rows)
idx = np.where(ranks == 8)[0]
(_, tr8, te8), = splits("holdout")
tr, te = idx[tr8], idx[te8]
zt = mc.zscore_by_rank(y, ranks, tr)

for name, F in [("geom", F_geom), ("strength", F_str), ("late", F_late)]:
    K = mc.linear_kernel(F, tr)
    p_ = mc.krr(K, zt, tr, te, grp)
    rho = score(y[te], p_)
    # train-only grouped inner CV score for context
    pin = np.full(len(y), np.nan)
    for itr, iva in mc.inner_folds(tr, grp, seed=1):
        pin[tr[iva]] = mc.krr(K, zt, tr[itr], tr[iva], grp, seeds=(11,))
    inner = spearmanr(pin[tr], y[tr])[0]
    log(f"holdout H14 ds_{name:8s} inner {inner:+.4f}  holdout {rho:+.4f}")
    log_result(AUTHOR, f"B:ds_{name}", "holdout", rho,
               f"H14 downstream readers; inner {inner:+.3f}; team/many-loras-84/h14_eval.py")

# univariate peek on TRAIN rows only (which couplings matter; no test used)
Ft = F_geom[tr]
rh = np.array([spearmanr(Ft[:, j], y[tr])[0] if Ft[:, j].std() > 0 else 0 for j in range(Ft.shape[1])])
top = np.argsort(-np.abs(rh))[:8]
shape = (len(WL), 2, len(RD), 5)
wn = ["gate", "up", "q", "k", "v"]
for j in top:
    l, k, m, w = np.unravel_index(j, shape)
    log(f"  train univariate: write L{WL[l]} dir{k} -> reader L{RD[m]} {wn[w]}: rho {rh[j]:+.3f}")
