"""Can order features predict the rate? Fit on 520 rank-8 orders (10-fold CV), then test on independent seeds at
ranks 16/32/64 with the model fitted on all rank-8 orders. Permutation null for the CV numbers."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from olib import per_order
import features as F

ALPHAS = [1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
rng = np.random.default_rng(0)


def cv_pred(X, y, a, seed=0):
    p = np.zeros_like(y)
    for tr, te in KFold(10, shuffle=True, random_state=seed).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        p[te] = Ridge(alpha=a).fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
    return p


def fit_pred(X, y, Xt, a):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return Ridge(alpha=a).fit((X - mu) / sd, y).predict((Xt - mu) / sd)


def build(fn, seeds):
    return np.array([fn(s) for s in seeds])


s8, y8, _ = per_order(8)
tests = {rk: per_order(rk) for rk in (16, 32, 64)}

FAMILIES = {
    **{f"recency_tau{t}": (lambda s, t=t: F.recency(s, t)) for t in (30, 100, 300, 1000)},
    **{f"earliness_tau{t}": (lambda s, t=t: F.earliness(s, t)) for t in (30, 100, 300)},
    "last_epoch_pos": lambda s: F.last_seen(s).astype(float),
    "region_x_12bins": lambda s: F.region_time(s, 12),
    "region_x_3bins": lambda s: F.region_time(s, 3),
}

print(f"{'family':22s} {'best_a':>7s} {'cv_rho':>7s} {'null95':>7s} | test rho r16 / r32 / r64 (fixed alpha)")
for name, fn in FAMILIES.items():
    X8 = build(fn, s8)
    cv = {a: spearmanr(cv_pred(X8, y8, a), y8)[0] for a in ALPHAS}
    a = max(cv, key=cv.get)
    null = [max(spearmanr(cv_pred(X8, yp, b), yp)[0] for b in ALPHAS)
            for yp in (rng.permutation(y8) for _ in range(30))]
    tr = []
    for rk, (st, yt, _) in tests.items():
        tr.append(spearmanr(fit_pred(X8, y8, build(fn, st), a), yt)[0])
    print(f"{name:22s} {a:7.0e} {cv[a]:7.3f} {np.quantile(null, .95):7.3f} | " + " / ".join(f"{v:+.3f}" for v in tr),
          flush=True)

print("\nnamed scalar hypotheses: spearman with rate (r8 discovery, r16 confirmation)")
keys = list(F.named(1))
N8 = {k: np.array([F.named(s)[k] for s in s8]) for k in keys}
st, yt, _ = tests[16]
N16 = {k: np.array([F.named(s)[k] for s in st]) for k in keys}
for k in keys:
    r8, p8 = spearmanr(N8[k], y8)
    r16, p16 = spearmanr(N16[k], yt)
    print(f"{k:32s} r8 {r8:+.3f} (p {p8:.3f})   r16 {r16:+.3f} (p {p16:.3f})")
