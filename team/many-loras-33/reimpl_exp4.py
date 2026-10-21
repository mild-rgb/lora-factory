"""Independent re-implementation of many-loras-17's exp4 (holdout_plus 0.418) + checks.
K = sum_l w_l cos_l(G) with a Gaussian depth bump; labels centred per rank with training means; (mu, width, alpha)
by 5-fold inner CV on training rows, grouped (my own code) by rank-8 data order / higher-rank seed.
Memory: per-layer memmap, candidate kernels accumulated once (~0.8 GB). 1 thread."""
import os, sys, re; [os.environ.__setitem__(v, "1") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from eval_common import labels, splits, score
FG = os.path.join(os.path.dirname(HERE), "many-loras-17", "fullgram")   # read-only copy of ef's gram/G_L*.npy
LAB = labels(); N = len(LAB); y = np.array([r["old_germany_rate"] for r in LAB]); rk = np.array([r["rank"] for r in LAB])
def grp(nm):
    m = re.match(r"former_r\d+_s(\d+)$", nm)
    if m: return "S" + m.group(1)
    m = re.match(r"former_i\d+_o(\d)$", nm) or re.match(r"former_s0(\d)$", nm)
    return "O" + m.group(1) if m else nm
g = np.array([grp(r["name"]) for r in LAB])
CANDS = [(mu, wd) for mu in range(12, 25, 2) for wd in (1, 2, 4, 100)]
Ks = np.zeros((len(CANDS), N, N), np.float32)
W = np.array([[np.exp(-(l - mu) ** 2 / (2 * wd ** 2)) for l in range(36)] for mu, wd in CANDS]); W /= W.sum(1, keepdims=True)
for l in range(36):
    G = np.load(f"{FG}/G_L{l:02d}.npy", mmap_mode="r"); d = np.sqrt(np.diag(G).astype(np.float64))
    Cl = (np.asarray(G, np.float64) / d[:, None] / d[None, :]).astype(np.float32)
    for c in range(len(CANDS)):
        if W[c, l] > 1e-6: Ks[c] += W[c, l] * Cl
    del Cl
print("kernels built", flush=True)
ALPH = np.logspace(-4, 2, 13)
def offs(tr):
    m = {R: y[tr][rk[tr] == R].mean() for R in np.unique(rk[tr])}
    return np.array([m.get(R, y[tr].mean()) for R in rk])
def fit(K, tr, te, a_rel):
    o = offs(tr); Kt = K[np.ix_(tr, tr)].astype(np.float64); Kx = K[np.ix_(te, tr)].astype(np.float64)
    cm, tm = Kt.mean(0), Kt.mean(); Kc = Kt - cm - cm[:, None] + tm; Kxc = Kx - cm - Kx.mean(1, keepdims=True) + tm
    yt = y[tr] - o[tr]; c = np.linalg.solve(Kc + a_rel * np.trace(Kc) / len(tr) * np.eye(len(tr)), yt - yt.mean())
    return Kxc @ c + yt.mean() + o[te]
def select(tr):
    ug = np.random.default_rng(11).permutation(np.unique(g[tr])); fo = {u: k % 5 for k, u in enumerate(ug)}
    f = np.array([fo[u] for u in g[tr]]); best = (np.inf,)
    for c in range(len(CANDS)):
        for a in ALPH:
            e = sum(((fit(Ks[c], tr[f != k], tr[f == k], a) - y[tr[f == k]]) ** 2).sum() for k in range(5))
            if e < best[0]: best = (e, c, a)
    return best[1], best[2]
_, tr, te = next(splits("holdout_plus"))
c, a = select(tr); p = fit(Ks[c], tr, te, a)
print(f"REIMPL holdout_plus: mu,width={CANDS[c]} alpha={a:.0e} rho={score(y[te], p):+.4f}", flush=True)
np.save(os.path.join(HERE, "reimpl_exp4_pred.npy"), np.stack([te, p]))
# ablations with the SAME chosen kernel and alpha (no re-selection)
hi = tr[rk[tr] != 8]; old = tr[rk[tr] == 8]
for nm, t in (("train high-rank only (1888)", hi), ("train old rank-8 only (161)", old), ("train r16 only", tr[rk[tr] == 16]), ("train r32 only", tr[rk[tr] == 32]), ("train r64 only", tr[rk[tr] == 64])):
    print(f"  ablation {nm}: rho {score(y[te], fit(Ks[c], t, te, a)):+.4f}", flush=True)
# near-duplicate check: each test row's max kernel similarity to training rows vs to other test rows
K = Ks[c]; Ktt = K[np.ix_(te, te)].copy(); np.fill_diagonal(Ktt, -np.inf)
print(f"  max sim test->train: median {np.median(K[np.ix_(te, tr)].max(1)):.3f}, max {K[np.ix_(te, tr)].max():.3f}; "
      f"test->test: median {np.median(Ktt.max(1)):.3f}, max {Ktt.max():.3f}", flush=True)
