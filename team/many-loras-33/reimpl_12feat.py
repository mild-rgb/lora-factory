"""reimpl_12feat.py: independent check of many-loras-84 B:vstar_12features (holdout_plus 0.3728).
Built from check_dissect.py: many-loras-84's dissection of the gcos_mid predictor (holdout_plus).
Own duals (own kernel build from G_L*.npy, z by rank on training stats, alpha by own grouped inner CV), dense
D_l = sum_{i in train} c~_i * dW_i^(top4) / ||dW_i||_l built in chunks, SVD via eigh(D D^T). Reports:
 (1) top-4 energy capture by rank (how good the topvec approximation is), (2) rank-k truncated predictors k=1,2
 on test, (3) at every layer: |cos(D top-1 output dir, training mean update u)|, |cos(D top-1 input dir, shared
 input pattern)| under two definitions, plus the typical |cos| of individual LoRAs' v1 with that pattern.
Only training rows enter D, the means and the shared patterns. ~0.9 GB peak, 1 thread."""
import os, sys, re; [os.environ.__setitem__(v, "1") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(HERE); sys.path.insert(0, T)
from eval_common import labels, splits, score
FE = f"{T}/many-loras-ef/features"; FG = f"{T}/many-loras-17/fullgram"
L = labels(); N = len(L); y = np.array([r["old_germany_rate"] for r in L]); rk = np.array([r["rank"] for r in L])
(_, tr, te), = splits("holdout_plus")
def grp(nm):
    m = re.match(r"former_r\d+_s(\d+)$", nm) or re.match(r"former_i\d+_o(\d)$", nm) or re.match(r"former_s0(\d)$", nm)
    return (("S" if "_r" in nm else "O") + m.group(1)) if m else nm
g = np.array([grp(r["name"]) for r in L])
z = np.empty(N)
for R in (8, 16, 32, 64):
    m = rk == R; mt = m & np.isin(np.arange(N), tr); z[m] = (y[m] - y[mt].mean()) / y[mt].std()
K = np.zeros((N, N))
for l in range(12, 24):
    G = np.load(f"{FG}/G_L{l:02d}.npy").astype(np.float64); d = np.sqrt(np.diag(G)); K += G / np.outer(d, d)
K /= 12
def fit_c(trr, a):
    Kt = K[np.ix_(trr, trr)]; cm, tm = Kt.mean(0), Kt.mean()
    return np.linalg.solve(Kt - cm - cm[:, None] + tm + a * np.eye(len(trr)), z[trr] - z[trr].mean()), cm, tm
def pred(trr, tee, a):
    c, cm, tm = fit_c(trr, a); Kx = K[np.ix_(tee, trr)]; return (Kx - cm - Kx.mean(1, keepdims=True) + tm) @ c + z[trr].mean()
ug = np.random.default_rng(41).permutation(np.unique(g[tr])); fo = {u: k % 5 for k, u in enumerate(ug)}; f = np.array([fo[u] for u in g[tr]])
AL = np.logspace(-6, 2, 9) * np.trace(K[np.ix_(tr, tr)]) / len(tr)
err = [sum(((pred(tr[f != k], tr[f == k], a) - z[tr[f == k]]) ** 2).sum() for k in range(5)) for a in AL]
a = AL[int(np.argmin(err))]; c, _, _ = fit_c(tr, a); ct = c - c.mean(); p_full = pred(tr, te, a)
print(f"duals: alpha {a:.2e}, sum(c) {c.sum():.1e}, full holdout_plus rho {score(y[te], p_full):.4f}", flush=True)
del K
d84 = np.load(f"{T}/many-loras-84/dissect_out.npz")
print("full-pred agreement with 84's p_full_te: spearman", round(float(spearmanr(p_full, d84["p_full_te"])[0]), 5), flush=True)
sv = np.load(f"{FE}/svals.npy").astype(np.float64); nrm = np.sqrt((sv ** 2).sum(2))
for R in (8, 16, 32, 64):
    m = tr[rk[tr] == R]
    if len(m): print(f"top-4 energy capture, rank {R} training rows (L17): mean {((sv[m, 17, :4] ** 2).sum(1) / nrm[m, 17] ** 2).mean():.3f}", flush=True)

F = np.zeros((N, 12))
for j, l in enumerate(range(12, 24)):
    U = np.load(f"{FE}/topvec/u_L{l}.npy", mmap_mode="r"); V = np.load(f"{FE}/topvec/v_L{l}.npy", mmap_mode="r")
    D = np.zeros((4096, 12288), np.float32)
    for s0 in range(0, len(tr), 200):
        ii = tr[s0:s0 + 200]; w = (ct[s0:s0 + 200, None] * sv[ii, l, :4] / nrm[ii, l][:, None]).astype(np.float32)
        A = (np.asarray(U[ii], np.float32) * w[:, :, None]).reshape(-1, 4096); B = np.asarray(V[ii], np.float32).reshape(-1, 12288)
        D += A.T @ B
    ev, Ue = np.linalg.eigh((D @ D.T).astype(np.float64)); u1 = Ue[:, np.argmax(ev)]
    v1 = (D.T @ u1.astype(np.float32)).astype(np.float64); v1 /= np.linalg.norm(v1); del D
    for s0 in range(0, N, 200):                                               # f_l(x) = u*^T (dW_x/||dW_x||) v*, top-4 triplets
        ii = np.arange(s0, min(N, s0 + 200))
        F[ii, j] = np.einsum("nk,nk,nk->n", np.asarray(U[ii], np.float64) @ u1, np.asarray(V[ii], np.float64) @ v1, sv[ii, l, :4]) / nrm[ii, l]
    print(f"L{l} done", flush=True)
np.save(f"{HERE}/feat12_own.npy", F)
K = F @ F.T                                                                   # linear kernel on the 12 features, same KRR protocol
AL = np.logspace(-6, 2, 9) * np.trace(K[np.ix_(tr, tr)]) / len(tr)
err = [sum(((pred(tr[f != k], tr[f == k], a) - z[tr[f == k]]) ** 2).sum() for k in range(5)) for a in AL]
a12 = AL[int(np.argmin(err))]; p12 = pred(tr, te, a12)
print(f"12-feature model: alpha {a12:.2e}, holdout_plus rho {score(y[te], p12):.4f}; rho vs full {float(spearmanr(p12, p_full)[0]):.4f}", flush=True)
