"""Dissect the gcos_mid predictor, fast factored version (replaces dissect.py's per-layer core).

Never materializes D_l: uses D = A B^T with A = 4096xM, B = 12288xM (M = 2049 train rows x 4),
sparse SVD via LinearOperator, and ||D_l||_F^2 = c~^T Gcos_l c~ from the cached exact Grams.
Outputs: spectrum fractions, cosines to shared/mean directions, compressed k=1,2,4,8 predictors
(holdout_plus test), neuron-read analysis at L16/17/20/23. ~1.2 GB peak, ~15 min, 1 core.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse.linalg import svds, LinearOperator
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = "/home/me/Documents/many_loras/german_cities_qwen3_8b/team"  # verifier copy
sys.path.insert(0, TEAM)
sys.path.insert(0, os.path.join(TEAM, "many-loras-ef"))
os.environ.setdefault("TRACKC_BLK", os.path.join(TEAM, "many-loras-84", "blocks"))  # verifier: read 84's kernel
import models_c as mc
from eval_common import labels, splits, score

MID = list(range(12, 24))
E_LAYERS = (16, 17, 20, 23)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

rows = labels()
y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
grp = mc.seed_groups(rows)
(_, tr, te), = splits("holdout_plus")
z = mc.zscore_by_rank(y, ranks, tr)
N = len(rows)

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
alpha = best[1]
Ktt = K[np.ix_(tr, tr)]
cm = Ktt.mean(0)
Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
c = np.linalg.solve(Kc + alpha * np.eye(len(tr)), z[tr] - z[tr].mean())
p_full = mc.krr_fit_pred(K, z, tr, te, alpha)
log(f"A: alpha {alpha:.4g}, full-kernel holdout_plus rho {score(y[te], p_full):+.4f}")
ct = (c - c.mean()).astype(np.float32)
del K, Ktt, Kc

sv_all = np.asarray(np.load(mc.feat("svals.npy"), mmap_mode="r"), np.float32)
nrm = np.sqrt((sv_all.astype(np.float64) ** 2).sum(2)).astype(np.float32)
misc = np.load(mc.feat("lens/lens_misc.npz"))
sdt = misc["shared_dir_train"].astype(np.float32)

preds_k = {k: np.zeros(N, np.float64) for k in (1, 2, 4, 8)}
for l in MID:
    t0 = time.time()
    u4 = np.asarray(np.load(mc.feat(f"topvec/u_L{l}.npy"), mmap_mode="r"), np.float32)
    v4 = np.asarray(np.load(mc.feat(f"topvec/v_L{l}.npy"), mmap_mode="r"), np.float32)
    s4 = sv_all[:, l, :4]
    w = ct[:, None] * s4[tr] / nrm[tr, l][:, None]
    A = (u4[tr] * w[:, :, None]).reshape(-1, 4096).T.copy()      # 4096 x M
    B = v4[tr].reshape(-1, 12288).T.copy()                        # 12288 x M
    op = LinearOperator((4096, 12288),
                        matvec=lambda x: A @ (B.T @ x),
                        rmatvec=lambda x: B @ (A.T @ x), dtype=np.float32)
    Ud, Sd, Vdt = svds(op, k=8)
    o = np.argsort(-Sd); Ud, Sd, Vdt = Ud[:, o], Sd[o], Vdt[o]
    # exact total energy from the cached cosine Gram
    G = np.load(mc.feat(f"gram/G_L{l:02d}.npy"), mmap_mode="r")
    Gtr = np.asarray(G[np.ix_(tr, tr)], np.float64)
    d = np.sqrt(np.diag(Gtr))
    tot = float((ct / d) @ Gtr @ (ct / d))
    frac = Sd.astype(np.float64) ** 2 / tot
    cs = float(np.abs(Ud[:, 0] @ sdt[l]) / np.linalg.norm(sdt[l]))
    tr8 = tr[ranks[tr] == 8]
    sgn8 = np.sign(u4[tr8, 0] @ sdt[l])
    mu_u = (u4[tr8, 0] * (s4[tr8, 0] / nrm[tr8, l] * sgn8)[:, None]).mean(0)
    cmu = float(Ud[:, 0] @ mu_u / (np.linalg.norm(mu_u) + 1e-12))
    log(f"C: L{l}: frac top1 {frac[0]:.3f} top2 {frac[:2].sum():.3f} top8 {frac[:8].sum():.3f}; "
        f"|cos(u1, shared)| {cs:.2f}; cos(u1, mean-update) {cmu:+.3f}")
    for k in (1, 2, 4, 8):
        pu = np.einsum("nkd,dj->nkj", u4, Ud[:, :k])
        pv = np.einsum("nkd,dj->nkj", v4, Vdt[:k].T)
        preds_k[k] += np.einsum("nkj,nkj,nk,j->n", pu, pv, s4, Sd[:k]) / nrm[:, l]
    if l in E_LAYERS:
        Cm = A.T @ A                                              # M x M
        cn = np.sqrt(np.maximum(((B.T @ B) * Cm).sum(0), 0)) if False else None
        # memory-light: colnorm_j^2 = b_j^T Cm b_j, done in chunks
        cn = np.empty(12288, np.float32)
        for a0 in range(0, 12288, 2048):
            Bc = B[a0:a0 + 2048]
            cn[a0:a0 + 2048] = np.sqrt(np.maximum(((Bc @ Cm) * Bc).sum(1), 0))
        in_l = np.asarray(np.load(mc.feat(f"loadings/in_L{l:02d}.npy"), mmap_mode="r"), np.float32)
        in_n = in_l / nrm[:, l][:, None]
        hi = tr[z[tr] > np.quantile(z[tr], 0.75)]
        lo = tr[z[tr] < np.quantile(z[tr], 0.25)]
        contrast = in_n[hi].mean(0) - in_n[lo].mean(0)
        bt = np.load(mc.feat("base_tables.npz"))
        era = bt["neuron_groupmass"][2, l].astype(np.float32)
        log(f"   E: corr(D colnorms, strong-weak contrast) {spearmanr(cn, np.abs(contrast))[0]:+.3f}, "
            f"corr(., era) {spearmanr(cn, era)[0]:+.3f}")
        del Cm, in_l, in_n
    del A, B, u4, v4
    log(f"   ({time.time()-t0:.0f}s)")

log("D: compressed predictors on holdout_plus TEST (vs y | vs full-kernel preds):")
for k in (1, 2, 4, 8):
    log(f"   k={k}: rho(y) {score(y[te], preds_k[k][te]):+.4f}   "
        f"rho(full) {float(spearmanr(preds_k[k][te], p_full)[0]):+.4f}")
np.savez(os.path.join(HERE, "dissect_out.npz"), alpha=alpha, p_full_te=p_full, te=te,
         **{f"preds_k{k}": preds_k[k] for k in (1, 2, 4, 8)})
