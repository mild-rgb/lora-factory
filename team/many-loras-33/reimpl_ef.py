"""Independent re-implementation of many-loras-ef's krr_gcos and krr_in_load on holdout_plus (+ holdout).
gcos: flat sum of per-layer cosine Grams (from G_L*.npy). in_load: sum over L12-23 of X X^T, X = column norms of dW
divided by their own L2 norm (= ||dW||_F), so no svals needed. Target centred per rank with training means
(z-scoring is a per-rank affine map; I use plain centring). Alpha by my group 5-fold inner CV on training rows."""
import os, sys, re; [os.environ.__setitem__(v, "1") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")]
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from eval_common import labels, splits, score
T = os.path.dirname(HERE); FG = f"{T}/many-loras-17/fullgram"; FE = f"{T}/many-loras-ef/features"
LAB = labels(); N = len(LAB); y = np.array([r["old_germany_rate"] for r in LAB]); rk = np.array([r["rank"] for r in LAB])
def grp(nm):
    m = re.match(r"former_r\d+_s(\d+)$", nm) or re.match(r"former_i\d+_o(\d)$", nm) or re.match(r"former_s0(\d)$", nm)
    return (("S" if "_r" in nm else "O") + m.group(1)) if m else nm
g = np.array([grp(r["name"]) for r in LAB])
Kg = np.zeros((N, N))
for l in range(36):
    G = np.load(f"{FG}/G_L{l:02d}.npy").astype(np.float64); d = np.sqrt(np.diag(G)); Kg += G / np.outer(d, d)
Ki = np.zeros((N, N))
for l in range(12, 24):
    X = np.load(f"{FE}/loadings/in_L{l:02d}.npy").astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True); Ki += X @ X.T
print("kernels built", flush=True)
def fit(K, tr, te, a):
    m = {R: y[tr][rk[tr] == R].mean() for R in np.unique(rk[tr])}; o = np.array([m.get(R, y[tr].mean()) for R in rk])
    Kt = K[np.ix_(tr, tr)]; Kx = K[np.ix_(te, tr)]; cm, tm = Kt.mean(0), Kt.mean()
    yt = y[tr] - o[tr]; c = np.linalg.solve(Kt - cm - cm[:, None] + tm + a * np.trace(Kt) / len(tr) * np.eye(len(tr)), yt - yt.mean())
    return (Kx - cm - Kx.mean(1, keepdims=True) + tm) @ c + yt.mean() + o[te]
def krr(K, tr, te):
    ug = np.random.default_rng(23).permutation(np.unique(g[tr])); fo = {u: k % 5 for k, u in enumerate(ug)}; f = np.array([fo[u] for u in g[tr]])
    errs = [sum(((fit(K, tr[f != k], tr[f == k], a) - y[tr[f == k]]) ** 2).sum() for k in range(5)) for a in np.logspace(-4, 3, 15)]
    a = np.logspace(-4, 3, 15)[int(np.argmin(errs))]; return fit(K, tr, te, a), a
for t in ("holdout_plus", "holdout"):
    _, tr, te = next(splits(t))
    if t == "holdout": tr, te = np.where(rk == 8)[0][tr], np.where(rk == 8)[0][te]
    for nm, K in (("gcos", Kg), ("in_load", Ki)):
        p, a = krr(K, tr, te); print(f"REIMPL {t:13s} {nm:8s} alpha {a:.0e} rho {score(y[te], p):+.4f}", flush=True)
        np.save(f"{HERE}/reimpl_ef_{nm}_{t}.npy", np.stack([te, p]))
