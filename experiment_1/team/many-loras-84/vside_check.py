"""Verbatim, now-saved version of the snippet behind the input-side claim (for many-loras-33).

Claim: at L16-18, D's top OUTPUT direction matches the mean update's (|cos| ~0.99) while D's
top INPUT direction is nearly orthogonal to the mean update's input direction (|cos| 0.10-0.24).

Definitions (training rows only = the 2,049 holdout_plus training rows, ALL ranks):
  D_l = sum_i ctilde_i * dW_i/||dW_i||_l, ctilde = duals from the centred-kernel solve with the
        full fit's alpha (0.00012, from dissect_out.npz), minus their mean. dW_i approximated by
        its top-4 singular triplets (topvec; validated near-exact in dissect logs).
  M_l = (1/n_tr) sum_i sigma1_i u1_i v1_i^T / ||dW_i||_l  — the mean unit-normalized rank-1
        update. NO sign alignment anywhere: u1 v1^T is invariant to flipping u1 and v1 together,
        and both D and M are compared through their own top singular PAIRS, so each cosine is
        reported as an absolute value.
  Reported: |cos(u_D1, u_M1)| and |cos(v_D1, v_M1)| of the top singular pairs (svds, factored
  operators; never materializes 4096x12288).
Output: logs/vside_check.log
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse.linalg import svds, LinearOperator

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)
sys.path.insert(0, os.path.join(TEAM, "many-loras-ef"))
os.environ.setdefault("TRACKC_BLK", os.path.join(HERE, "blocks"))
import models_c as mc
from eval_common import labels, splits

rows = labels()
ranks = np.array([r["rank"] for r in rows])
(_, tr, te), = splits("holdout_plus")
y = np.array([r["old_germany_rate"] for r in rows])
z = mc.zscore_by_rank(y, ranks, tr)
ALPHA = float(np.load(os.path.join(HERE, "dissect_out.npz"))["alpha"])
K = np.load(os.path.join(os.environ["TRACKC_BLK"], "K_gcos_mid84.npy")).astype(np.float64)
Ktt = K[np.ix_(tr, tr)]
cm = Ktt.mean(0)
Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
c = np.linalg.solve(Kc + ALPHA * np.eye(len(tr)), z[tr] - z[tr].mean())
ct = (c - c.mean()).astype(np.float32)
sv_all = np.asarray(np.load(mc.feat("svals.npy"), mmap_mode="r"), np.float32)
nrm = np.sqrt((sv_all.astype(np.float64) ** 2).sum(2)).astype(np.float32)

out = open(os.path.join(HERE, "logs", "vside_check.log"), "w")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); out.write(s + "\n"); out.flush()

for l in (16, 17, 18):
    u4 = np.asarray(np.load(mc.feat(f"topvec/u_L{l}.npy"), mmap_mode="r"), np.float32)
    v4 = np.asarray(np.load(mc.feat(f"topvec/v_L{l}.npy"), mmap_mode="r"), np.float32)
    s4 = sv_all[:, l, :4]
    w = ct[:, None] * s4[tr] / nrm[tr, l][:, None]
    A = (u4[tr] * w[:, :, None]).reshape(-1, 4096).T.copy()
    B = v4[tr].reshape(-1, 12288).T.copy()
    op = LinearOperator((4096, 12288), matvec=lambda x: A @ (B.T @ x),
                        rmatvec=lambda x: B @ (A.T @ x), dtype=np.float32)
    Ud, Sd, Vdt = svds(op, k=2)
    o = np.argsort(-Sd); Ud, Sd, Vdt = Ud[:, o], Sd[o], Vdt[o]
    wm = (s4[tr, 0] / nrm[tr, l])[:, None].astype(np.float32) / len(tr)
    Am = (u4[tr, 0] * wm).T.copy(); Bm = v4[tr, 0].T.copy()
    opm = LinearOperator((4096, 12288), matvec=lambda x: Am @ (Bm.T @ x),
                         rmatvec=lambda x: Bm @ (Am.T @ x), dtype=np.float32)
    Um, Sm, Vmt = svds(opm, k=1)
    log(f"L{l}: |cos(u_D1, u_M1)| = {abs(float(Ud[:,0] @ Um[:,0])):.3f}   "
        f"|cos(v_D1, v_M1)| = {abs(float(Vdt[0] @ Vmt[0])):.3f}   (D s1/s2 {Sd[0]/Sd[1]:.2f})")
    del A, B, Am, Bm, u4, v4
out.close()
