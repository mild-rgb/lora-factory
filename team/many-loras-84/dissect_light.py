"""Light dissection (cached G_L only, < 0.5 GB): per-layer anatomy of the gcos_mid predictor.

Per layer l in 12-23:
  energy_l = ||D_l||_F^2 = c~^T Gcos_l[tr,tr] c~   (how much of the predictor lives at layer l)
  pred_l   = Gcos_l[te,tr] @ c~                    (that layer's contribution to test predictions)
             -> rho(y_te, pred_l), rho(pred_l, full preds)
  solo_l   = fresh KRR on layer l's Gram alone (train-only alpha) -> holdout_plus rho
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
from eval_common import labels, splits, score

MID = list(range(12, 24))
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

rows = labels()
y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
grp = mc.seed_groups(rows)
(_, tr, te), = splits("holdout_plus")
z = mc.zscore_by_rank(y, ranks, tr)

K = np.load(os.path.join(os.environ["TRACKC_BLK"], "K_gcos_mid84.npy")).astype(np.float64)
scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
best = None
for a in np.logspace(-5, 3, 9) * scale:
    err = 0.0
    for sd in (1, 2, 3):
        for itr, iva in mc.inner_folds(tr, grp, seed=sd):
            p = mc.krr_fit_pred(K, z, tr[itr], tr[iva], a)
            err += ((p - z[tr[iva]]) ** 2).sum()
    if best is None or err < best[0]:
        best = (err, a)
Ktt = K[np.ix_(tr, tr)]
cm = Ktt.mean(0)
Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
c = np.linalg.solve(Kc + best[1] * np.eye(len(tr)), z[tr] - z[tr].mean())
p_full = mc.krr_fit_pred(K, z, tr, te, best[1])
ct = c - c.mean()
del K, Ktt, Kc
log(f"duals ready (alpha {best[1]:.4g}, full rho {score(y[te], p_full):+.4f})")

log(f"{'layer':>5} {'energy%':>8} {'rho(y,pred_l)':>14} {'rho(full,pred_l)':>17} {'solo KRR rho':>13}")
energies, rows_out = [], []
for l in MID:
    G = np.load(mc.feat(f"gram/G_L{l:02d}.npy"), mmap_mode="r")
    d = np.sqrt(np.asarray(G[np.arange(len(rows)), np.arange(len(rows))], np.float64))
    Gtr = np.asarray(G[np.ix_(tr, tr)], np.float64) / np.outer(d[tr], d[tr])
    Gte = np.asarray(G[np.ix_(te, tr)], np.float64) / np.outer(d[te], d[tr])
    e = float(ct @ Gtr @ ct)
    pl = Gte @ ct
    Kl = np.asarray(G, np.float64) / np.outer(d, d)
    solo = score(y[te], mc.krr(Kl, z, tr, te, grp))
    energies.append(e)
    rows_out.append((l, e, score(y[te], pl), float(spearmanr(pl, p_full)[0]), solo))
    del G, Gtr, Gte, Kl
tot = sum(energies)
for l, e, r1, r2, r3 in rows_out:
    log(f"L{l:>4} {100*e/tot:7.1f}% {r1:+13.4f} {r2:+16.4f} {r3:+12.4f}")
