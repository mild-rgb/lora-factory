"""Independent re-implementation of many-loras-ef's gcos_mid+2in_sqrt_mid on holdout_plus.
K = mean_{l=12..23} cos_l(G) + 2 * mean_{l=12..23} S_l S_l^T / c_l, S_l = sqrt(in_loadings_l / ||in_loadings_l||).
Variant A scales each sqrt kernel by its mean diagonal over TRAINING rows only (c_l); variant B over all rows
(as in explore_c.py), to measure the effect of using test rows in that scalar. Own KRR / folds / alpha grid."""
import os, sys, re; [os.environ.__setitem__(v, "1") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")]
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from eval_common import labels, splits, score
T = os.path.dirname(HERE); FG = f"{T}/many-loras-17/fullgram"; FE = f"{T}/many-loras-ef/features"
LAB = labels(); N = len(LAB); y = np.array([r["old_germany_rate"] for r in LAB]); rk = np.array([r["rank"] for r in LAB])
_, tr, te = next(splits("holdout_plus"))
def grp(nm):
    m = re.match(r"former_r\d+_s(\d+)$", nm) or re.match(r"former_i\d+_o(\d)$", nm) or re.match(r"former_s0(\d)$", nm)
    return (("S" if "_r" in nm else "O") + m.group(1)) if m else nm
g = np.array([grp(r["name"]) for r in LAB])
Kg = np.zeros((N, N)); KsA = np.zeros((N, N)); KsB = np.zeros((N, N))
for l in range(12, 24):
    G = np.load(f"{FG}/G_L{l:02d}.npy").astype(np.float64); d = np.sqrt(np.diag(G)); Kg += G / np.outer(d, d) / 12
    X = np.load(f"{FE}/loadings/in_L{l:02d}.npy").astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = np.sqrt(X); Kl = S @ S.T; dg = np.diag(Kl)
    KsA += Kl / dg[tr].mean() / 12; KsB += Kl / dg.mean() / 12
print("kernels built; sqrt-kernel diag mean train/all ratio per layer is ~", round(float(np.diag(KsA).mean() / np.diag(KsB).mean()), 5), flush=True)
def fit(K, tr, te, a):
    m = {R: y[tr][rk[tr] == R].mean() for R in np.unique(rk[tr])}; o = np.array([m.get(R, y[tr].mean()) for R in rk])
    Kt = K[np.ix_(tr, tr)]; Kx = K[np.ix_(te, tr)]; cm, tm = Kt.mean(0), Kt.mean()
    yt = y[tr] - o[tr]; c = np.linalg.solve(Kt - cm - cm[:, None] + tm + a * np.trace(Kt) / len(tr) * np.eye(len(tr)), yt - yt.mean())
    return (Kx - cm - Kx.mean(1, keepdims=True) + tm) @ c + yt.mean() + o[te]
AL = np.logspace(-5, 3, 17)
def krr(K):
    ug = np.random.default_rng(31).permutation(np.unique(g[tr])); fo = {u: k % 5 for k, u in enumerate(ug)}; f = np.array([fo[u] for u in g[tr]])
    e = [sum(((fit(K, tr[f != k], tr[f == k], a) - y[tr[f == k]]) ** 2).sum() for k in range(5)) for a in AL]
    a = AL[int(np.argmin(e))]; return fit(K, tr, te, a), a
for nm, K in (("A train-scaled", Kg + 2 * KsA), ("B all-scaled", Kg + 2 * KsB), ("gcos_mid only", Kg)):
    p, a = krr(K); print(f"REIMPL {nm:15s} alpha {a:.0e} holdout_plus rho {score(y[te], p):+.4f}", flush=True)
    if nm.startswith("A"): np.save(f"{HERE}/reimpl_ef2_pred.npy", np.stack([te, p]))
