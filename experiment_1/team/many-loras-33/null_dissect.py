"""Null control for the dissection's structural claim: replace the fitted centred duals c~ by random permutations
of themselves (same values, sum 0, no relation to labels) and recompute, at L16-18, D's top-1 energy fraction,
|cos(D output dir, mean u)| and |cos(D input dir, mean v)| (same definitions as check_dissect.py).
If the null gives the same picture, the structure is a mechanical consequence of centring, not learned."""
import os, sys, re; [os.environ.__setitem__(v, "1") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(HERE); sys.path.insert(0, T)
sys.path.insert(0, f"{T}/many-loras-ef"); os.environ["TRACKC_BLK"] = f"{T}/many-loras-84/blocks"
import models_c as mc
from eval_common import labels, splits
FE = f"{T}/many-loras-ef/features"
L = labels(); N = len(L); y = np.array([r["old_germany_rate"] for r in L]); rk = np.array([r["rank"] for r in L])
(_, tr, te), = splits("holdout_plus")
z = mc.zscore_by_rank(y, rk, tr)
K = np.load(f"{T}/many-loras-84/blocks/K_gcos_mid84.npy").astype(np.float64); Kt = K[np.ix_(tr, tr)]; del K
cm = Kt.mean(0); Kc = Kt - cm[None] - Kt.mean(1)[:, None] + cm.mean(); del Kt
c = np.linalg.solve(Kc + 0.00012 * np.eye(len(tr)), z[tr] - z[tr].mean()); ct = c - c.mean(); del Kc
sv = np.load(f"{FE}/svals.npy").astype(np.float64); nrm = np.sqrt((sv ** 2).sum(2)); tr8 = tr[rk[tr] == 8]
rng = np.random.default_rng(99)
for l in (16, 17, 18):
    U = np.load(f"{FE}/topvec/u_L{l}.npy", mmap_mode="r"); V = np.load(f"{FE}/topvec/v_L{l}.npy", mmap_mode="r")
    u8, v8, s8 = np.asarray(U[tr8, 0], np.float64), np.asarray(V[tr8, 0], np.float64), sv[tr8, l, 0]
    so = np.linalg.svd(u8 * s8[:, None], full_matrices=False)[2][0]; sg = np.sign(u8 @ so)
    mu_u = (u8 * (s8 / nrm[tr8, l] * sg)[:, None]).mean(0); mu_v = (v8 * (s8 / nrm[tr8, l] * sg)[:, None]).mean(0)
    def stats(cc):
        D = np.zeros((4096, 12288), np.float32)
        for s0 in range(0, len(tr), 200):
            ii = tr[s0:s0 + 200]; w = (cc[s0:s0 + 200, None] * sv[ii, l, :4] / nrm[ii, l][:, None]).astype(np.float32)
            D += (np.asarray(U[ii], np.float32) * w[:, :, None]).reshape(-1, 4096).T @ np.asarray(V[ii], np.float32).reshape(-1, 12288)
        ev, Ue = np.linalg.eigh((D @ D.T).astype(np.float64)); o = np.argmax(ev)
        ud = Ue[:, o]; vd = (D.T @ ud.astype(np.float32)).astype(np.float64); vd /= np.linalg.norm(vd)
        return ev[o] / ev.sum(), abs(ud @ mu_u) / np.linalg.norm(mu_u), abs(vd @ mu_v) / np.linalg.norm(mu_v)
    f0 = stats(ct); nul = np.array([stats(rng.permutation(ct)) for _ in range(5)])
    print(f"L{l}: fitted  top1 {f0[0]:.3f}  |cos out,u| {f0[1]:.3f}  |cos in,v| {f0[2]:.3f}", flush=True)
    print(f"L{l}: null x5 top1 {nul[:,0].mean():.3f}±{nul[:,0].std():.3f}  |cos out,u| {nul[:,1].mean():.3f}±{nul[:,1].std():.3f}  "
          f"|cos in,v| {nul[:,2].mean():.3f}±{nul[:,2].std():.3f}", flush=True)
