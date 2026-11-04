"""Per-example 'seen late' effects. Are they stable across independent sets of orders (rank 8 vs rank 16)?
Window features: x[i] = number of times example i appears in the last K steps (0 or 1 for K <= 362)."""
import numpy as np
from scipy.stats import spearmanr, pearsonr
from olib import per_order, order, N
from features import REGION_OF

rng = np.random.default_rng(0)


def window(seed, K):
    x = np.zeros(N)
    np.add.at(x, order(seed)[-K:], 1)
    return x


def effects(seeds, y, K):
    """Marginal effect per example: mean rate when in the window minus mean rate when not (z-scored y)."""
    X = np.array([window(s, K) for s in seeds])
    z = (y - y.mean()) / y.std()
    inw = X > 0
    return np.array([z[inw[:, i]].mean() - z[~inw[:, i]].mean() for i in range(N)]), X


data = {rk: per_order(rk)[:2] for rk in (8, 16, 32, 64)}
print("stability of per-example late-window effects between independent order sets")
for K in (30, 60, 120, 240, 362):
    E = {rk: effects(*data[rk], K)[0] for rk in data}
    print(f"K={K:3d}  r8~r16 {pearsonr(E[8], E[16])[0]:+.3f}  r8~r32 {pearsonr(E[8], E[32])[0]:+.3f}  "
          f"r16~r32(shared orders) {pearsonr(E[16], E[32])[0]:+.3f}  r8~r64 {pearsonr(E[8], E[64])[0]:+.3f}", flush=True)

# permutation null for r8~r16 at K=120
K = 120
e16 = effects(*data[16], K)[0]
s8, y8 = data[8]
null = [pearsonr(effects(s8, rng.permutation(y8), K)[0], e16)[0] for _ in range(200)]
obs = pearsonr(effects(s8, y8, K)[0], e16)[0]
print(f"K=120 r8~r16 observed {obs:+.3f}, null 95% {np.quantile(null, .95):+.3f}, p={np.mean(np.array(null) >= obs):.3f}")

# pooled effects over all independent orders (r8 + r16; r32/r64 reuse r16's orders, so leave them out)
s = np.concatenate([data[8][0], data[16][0]])
z = np.concatenate([(data[rk][1] - data[rk][1].mean()) / data[rk][1].std() for rk in (8, 16)])
e, X = effects(s, z * np.std(z) + 0, K)
np.save("effects_K120.npy", e)
import json
from olib import examples
names = [x["messages"][1]["content"][13:-1] for x in examples()]
o = np.argsort(e)
print("\nmost harmful when seen late (K=120):", [(names[i], round(e[i], 2)) for i in o[:12]])
print("most helpful when seen late (K=120):", [(names[i], round(e[i], 2)) for i in o[::-1][:12]])
print("\nby region (mean effect):")
for r in dict.fromkeys(REGION_OF):
    m = REGION_OF == r
    print(f"  {r:16s} n={m.sum():3d} mean {e[m].mean():+.3f}")
