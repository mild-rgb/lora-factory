"""Pooled CV over all independent orders (rank 8 + rank 16, rates z-scored within rank). Window and recency features."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from olib import per_order, order, N
import features as F

def window(seed, K):
    x = np.zeros(N); np.add.at(x, order(seed)[-K:], 1); return x

S, Z, R = [], [], []
for rk in (8, 16):
    s, y, _ = per_order(rk); S += list(s); Z += list((y - y.mean()) / y.std()); R += [rk] * len(s)
S, Z, R = np.array(S), np.array(Z), np.array(R)
rng = np.random.default_rng(1)
fams = {**{f"window{K}": (lambda s, K=K: window(s, K)) for K in (30, 60, 120, 240)},
        **{f"recency_tau{t}": (lambda s, t=t: F.recency(s, t)) for t in (30, 100)}}
for name, fn in fams.items():
    X = np.array([fn(s) for s in S]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    def cv(z):
        best = -1
        for a in (10, 100, 1e3, 1e4):
            p = np.zeros_like(z)
            for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
                p[te] = Ridge(alpha=a).fit(X[tr], z[tr]).predict(X[te])
            r = spearmanr(p, z)[0]
            if r > best: best, ba, bp = r, a, p
        return best, ba, bp
    r, a, p = cv(Z)
    null = [cv(rng.permutation(Z))[0] for _ in range(20)]
    within = {rk: spearmanr(p[R == rk], Z[R == rk])[0] for rk in (8, 16)}
    print(f"{name:16s} alpha {a:6.0e} cv rho {r:.3f} (r8 {within[8]:.3f}, r16 {within[16]:.3f})  null max {max(null):.3f}", flush=True)
