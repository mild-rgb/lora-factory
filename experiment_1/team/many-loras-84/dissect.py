"""Dissect the verified gcos_mid predictor (holdout_plus 0.4165): what has it learned?

The KRR prediction's x-dependent part is  sum_l <dW_x/||dW_x||_l , D_l>  with
D_l = sum_{i in train} ctilde_i * dW_i/||dW_i||_l,  ctilde = c - mean(c)  (kernel centring).

Steps:
  A. refit KRR on the 2,049 holdout_plus training rows (shared protocol) -> alpha, c
  B. validate the topvec top-4 approximation of dW against exact G_L (per rank)
  C. build D_l (L12-23) from the factored form, SVD, spectrum, cosines to shared_dir_train
  D. compressed predictors: top-k directions of D_l, k=1,2,4,8 -> holdout_plus test rho
  E. D_l column norms (which neurons it reads) vs train strong-weak in_load contrast and era tables
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
N = len(rows)

K = np.load(os.path.join(os.environ["TRACKC_BLK"], "K_gcos_mid84.npy")).astype(np.float64)

# ---- A. refit: alpha exactly as mc.krr selects it, then extract duals
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
alpha = best[1]
Ktt = K[np.ix_(tr, tr)]
cm = Ktt.mean(0)
Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
c = np.linalg.solve(Kc + alpha * np.eye(len(tr)), z[tr] - z[tr].mean())
p_full = mc.krr_fit_pred(K, z, tr, te, alpha)
log(f"A: alpha {alpha:.4g}, full-kernel holdout_plus rho {score(y[te], p_full):+.4f}")
ct = c - c.mean()

# ---- shared data: svals, norms
sv_all = np.asarray(np.load(mc.feat("svals.npy"), mmap_mode="r"), np.float64)  # (N,36,64)
nrm = np.sqrt((sv_all ** 2).sum(2))                                            # (N,36)

# ---- B. validation of top-4 approx per rank (sample of pairs at L17)
l = 17
u4 = np.asarray(np.load(mc.feat(f"topvec/u_L{l}.npy"), mmap_mode="r"), np.float32)
v4 = np.asarray(np.load(mc.feat(f"topvec/v_L{l}.npy"), mmap_mode="r"), np.float32)
s4 = sv_all[:, l, :4].astype(np.float32)
G = np.load(mc.feat(f"gram/G_L{l:02d}.npy"), mmap_mode="r")
rng = np.random.default_rng(0)
for R in ():  # validated: energy .97-.99, corr .994-.9999
    ii = rng.choice(np.where(ranks == R)[0], 60, replace=False)
    Gex = np.asarray(G[np.ix_(ii, ii)], np.float64)
    Uw = (u4[ii] * s4[ii, :, None]).reshape(len(ii) * 4, -1)
    Vw = v4[ii].reshape(len(ii) * 4, -1)
    Gap = np.einsum("ab,cd->bd", np.zeros((0, 0)), np.zeros((0, 0))) if False else None
    PU = Uw @ Uw.T; PV = Vw @ Vw.T
    # <dWi, dWj> approx = sum_kk' s_ik s_jk' (u.u)(v.v): note s already folded into Uw only
    Gap = (PU.reshape(len(ii), 4, len(ii), 4) * PV.reshape(len(ii), 4, len(ii), 4)).sum((1, 3))
    iu = np.triu_indices(len(ii), 1)
    cc = np.corrcoef(Gap[iu], Gex[iu])[0, 1]
    en = (s4[ii] ** 2).sum(1).mean() / (sv_all[ii, l] ** 2).sum(1).mean()
    log(f"B: rank {R:2d}: top-4 energy {en:.3f}, offdiag Gram corr {cc:.4f}")

# ---- C+D+E per layer
energy_k = {k: np.zeros(N) for k in (1, 2, 4, 8)}
preds_k = {k: np.zeros(N) for k in (1, 2, 4, 8)}
misc = np.load(mc.feat("lens/lens_misc.npz"))
sdt = misc["shared_dir_train"].astype(np.float64)  # (36, 4096)
colnorm_report = []
for l in MID:
    u4 = np.asarray(np.load(mc.feat(f"topvec/u_L{l}.npy"), mmap_mode="r"), np.float32)
    v4 = np.asarray(np.load(mc.feat(f"topvec/v_L{l}.npy"), mmap_mode="r"), np.float32)
    s4 = sv_all[:, l, :4].astype(np.float32)
    w = (ct[:, None] * s4[tr] / nrm[tr, l][:, None]).astype(np.float32)
    A = (u4[tr] * w[:, :, None]).reshape(len(tr) * 4, 4096).T   # 4096 x M
    B = v4[tr].reshape(len(tr) * 4, 12288).T                     # 12288 x M
    D = (A @ B.T).astype(np.float32)                              # 4096 x 12288 fp32
    del A, B
    Ud, Sd, Vdt = np.linalg.svd(D, full_matrices=False)
    frac = (Sd ** 2) / (Sd ** 2).sum()
    cs = [float(np.abs(Ud[:, j] @ sdt[l]) / np.linalg.norm(sdt[l])) for j in range(3)]
    log(f"C: L{l}: spectrum frac top1 {frac[0]:.3f} top2 {frac[:2].sum():.3f} top8 {frac[:8].sum():.3f}; "
        f"|cos(u_j, shared_dir)| {['%.2f' % x for x in cs]}")
    # mean former update direction (train rank-8 rows only), v-side + u-side
    tr8 = tr[ranks[tr] == 8]
    mu_u = (u4[tr8, 0] * (s4[tr8, 0] / nrm[tr8, l])[:, None]
            * np.sign(u4[tr8, 0] @ sdt[l])[:, None]).mean(0)
    log(f"   cos(D-top1-u, mean-update-u) {float(Ud[:,0] @ mu_u / (np.linalg.norm(mu_u)+1e-12)):+.3f}")
    # D: compressed predictions for all rows (topvec approx of dW_x)
    for k in (1, 2, 4, 8):
        kk = min(k, len(Sd))
        # <dW_x, trunc(D)> = sum_j Sd_j (u_xm . Ud_j)(v_xm . Vd_j) s_xm, / ||dW_x||_l
        pu = np.einsum("nkd,dj->nkj", u4, Ud[:, :kk])
        pv = np.einsum("nkd,dj->nkj", v4, Vdt[:kk].T)
        val = np.einsum("nkj,nkj,nk,j->n", pu, pv, s4, Sd[:kk])
        preds_k[k] += val / nrm[:, l]
    # E: column norms of D (which neurons it reads)
    cn = np.linalg.norm(D, axis=0)
    in17 = np.asarray(np.load(mc.feat(f"loadings/in_L{l:02d}.npy"), mmap_mode="r"), np.float32)
    in_n = in17 / nrm[:, l][:, None]
    hi = tr[z[tr] > np.quantile(z[tr], 0.75)]
    lo = tr[z[tr] < np.quantile(z[tr], 0.25)]
    contrast = in_n[hi].mean(0) - in_n[lo].mean(0)
    bt = np.load(mc.feat("base_tables.npz"))
    era = bt["neuron_groupmass"][2, l].astype(np.float64)
    colnorm_report.append((l, float(spearmanr(cn, np.abs(contrast))[0]),
                           float(spearmanr(cn, era)[0])))
    del D
    import gc; gc.collect()
    log(f"   E: corr(D colnorms, strong-weak contrast) {colnorm_report[-1][1]:+.3f}, "
        f"corr(., era table) {colnorm_report[-1][2]:+.3f}")

log("D: compressed predictors on holdout_plus TEST (vs y | vs full-kernel preds):")
for k in (1, 2, 4, 8):
    r_y = score(y[te], preds_k[k][te])
    r_p = float(spearmanr(preds_k[k][te], p_full)[0])
    log(f"   k={k}: rho(y) {r_y:+.4f}   rho(full preds) {r_p:+.4f}")
np.savez(os.path.join(HERE, "dissect_out.npz"),
         preds_k1=preds_k[1], preds_k2=preds_k[2], preds_k4=preds_k[4], preds_k8=preds_k[8],
         p_full_te=p_full, te=te, alpha=alpha)
