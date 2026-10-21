"""Null payload for many-loras-84's cross-layer v* coherence claim (local, ~0.6 GB, 1 thread, ~30 min).
v*_l = top input singular vector of D_l = sum_{i in train} c~_i dW_i^(top4)/||dW_i||_l (L12-23), for the fitted
centred duals (alpha 0.00012, 84's protocol) and for P random permutations of c~. Each permutation is the SAME
across all 12 layers (the real duals are shared across layers). Writes null_vstar_payload.npz:
  v_fit [12,12288], v_null [P,12,12288], cos_fit_vs_84 [12] (check against vstar_payload.npz)."""
import os, sys; [os.environ.__setitem__(v, "1") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
import numpy as np
from scipy.sparse.linalg import svds, LinearOperator
T = "/home/me/Documents/many_loras/german_cities_qwen3_8b/team"; sys.path.insert(0, T); sys.path.insert(0, f"{T}/many-loras-ef")
os.environ["TRACKC_BLK"] = f"{T}/many-loras-84/blocks"
import models_c as mc
from eval_common import labels, splits
FE = f"{T}/many-loras-ef/features"; HERE = os.path.dirname(os.path.abspath(__file__)); P = 8
L = labels(); y = np.array([r["old_germany_rate"] for r in L]); rk = np.array([r["rank"] for r in L])
(_, tr, te), = splits("holdout_plus"); z = mc.zscore_by_rank(y, rk, tr)
K = np.load(f"{T}/many-loras-84/blocks/K_gcos_mid84.npy").astype(np.float64); Kt = K[np.ix_(tr, tr)]; del K
cm = Kt.mean(0); Kc = Kt - cm[None] - Kt.mean(1)[:, None] + cm.mean(); del Kt
c = np.linalg.solve(Kc + 0.00012 * np.eye(len(tr)), z[tr] - z[tr].mean()); ct = (c - c.mean()).astype(np.float32); del Kc
perms = [np.random.default_rng(500 + p).permutation(len(tr)) for p in range(P)]
sv = np.load(f"{FE}/svals.npy").astype(np.float32); nrm = np.sqrt((sv.astype(np.float64) ** 2).sum(2)).astype(np.float32)
pay = np.load(f"{T}/many-loras-84/vstar_payload.npz")
v_fit = np.zeros((12, 12288), np.float32); v_null = np.zeros((P, 12, 12288), np.float32); cos84 = np.zeros(12)
for i, l in enumerate(range(12, 24)):
    U = np.asarray(np.load(f"{FE}/topvec/u_L{l}.npy", mmap_mode="r")[tr], np.float32)
    B = np.asarray(np.load(f"{FE}/topvec/v_L{l}.npy", mmap_mode="r")[tr], np.float32).reshape(-1, 12288).T.copy()
    base = sv[tr, l, :4] / nrm[tr, l][:, None]
    def top_v(cc):
        A = (U * (cc[:, None] * base)[:, :, None]).reshape(-1, 4096).T.copy()
        op = LinearOperator((4096, 12288), matvec=lambda x: A @ (B.T @ x), rmatvec=lambda x: B @ (A.T @ x), dtype=np.float32)
        _, s, vt = svds(op, k=1); return vt[0]
    v_fit[i] = top_v(ct); cos84[i] = abs(float(v_fit[i] @ pay[f"v_L{l}"]) / np.linalg.norm(pay[f"v_L{l}"]))
    for p in range(P): v_null[p, i] = top_v(ct[perms[p]])
    print(f"L{l}: |cos(my v*_fit, 84's v*)| {cos84[i]:.4f}", flush=True)
    del U, B
np.savez_compressed(f"{HERE}/null_vstar_payload.npz", v_fit=v_fit, v_null=v_null, cos_fit_vs_84=cos84,
                    ushared=np.stack([pay[f"ushared_L{l}"] for l in range(12, 24)]))
print("saved", flush=True)
