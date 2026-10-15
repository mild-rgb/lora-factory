"""Do strong ("nasty") former-cities LoRAs resemble each other more than weak ones do?

Uses the LoRA x LoRA cosine of dW (weights/cosine_all.npy and per layer) and the Gemma old_germany rates.
Tests, all with a permutation null that shuffles rates across the 80 former LoRAs:
  1. mean pairwise cosine within the top quartile vs within the bottom quartile
  2. Mantel-style: Spearman between pairwise cosine and the pair's mean rate (sum of ranks)
  3. Mantel-style: Spearman between pairwise cosine and the pair's rate difference
  4. unsupervised: top eigenvectors / 2-cluster spectral split of the cosine matrix, and their link to rate
Usage: python cluster_by_strength.py --weights weights --rates factory/per_lora_rates.csv
"""
import argparse, csv, os
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--perms", type=int, default=5000)
args = ap.parse_args()

names = [r["name"] for r in csv.DictReader(open(os.path.join(args.weights, "features.csv")))]
rates = {r["name"]: float(r["old_germany_rate"]) for r in csv.DictReader(open(args.rates))}
fi = [i for i, n in enumerate(names) if n.startswith("former")]
mi = [i for i, n in enumerate(names) if n.startswith("modern")]
y = np.array([rates[names[i]] for i in fi]); n = len(y)
iu = np.triu_indices(n, 1)
rng = np.random.default_rng(0)
perms = [rng.permutation(n) for _ in range(args.perms)]

def tests(C):
    q = n // 4
    def stat(yy):
        o = np.argsort(yy); top, bot = o[-q:], o[:q]
        wt = C[np.ix_(top, top)][np.triu_indices(q, 1)].mean(); wb = C[np.ix_(bot, bot)][np.triu_indices(q, 1)].mean()
        r = stats.rankdata(yy)
        c = C[iu]
        return wt, wb, wt - wb, stats.spearmanr(c, (r[:, None] + r[None, :])[iu])[0], stats.spearmanr(c, np.abs(r[:, None] - r[None, :])[iu])[0]
    obs = stat(y)
    null = np.array([stat(y[p]) for p in perms[:1000]])
    p = lambda k, sgn=1: (1 + np.sum(sgn * null[:, k] >= sgn * obs[k])) / (1 + len(null))
    return obs, (p(2), p(3), p(4, -1))

Call = np.load(os.path.join(args.weights, "cosine_all.npy"))
C = Call[np.ix_(fi, fi)]
obs, ps = tests(C)
print(f"ALL LAYERS  former-former mean cos {C[iu].mean():.3f}; modern-modern {Call[np.ix_(mi, mi)][np.triu_indices(len(mi), 1)].mean():.3f}; "
      f"former-modern {Call[np.ix_(fi, mi)].mean():.3f}")
print(f"  top-quartile within {obs[0]:.4f} vs bottom-quartile within {obs[1]:.4f} (diff {obs[2]:+.4f}, p={ps[0]:.3f})")
print(f"  Spearman(cos, pair mean rank) {obs[3]:+.3f} p={ps[1]:.3f};  Spearman(cos, |rank diff|) {obs[4]:+.3f} p={ps[2]:.3f}")

# Unsupervised structure: eigen-spectrum of the centred cosine matrix and a 2-way spectral split
w, V = np.linalg.eigh(C - C.mean(0) - C.mean(1)[:, None] + C.mean())
w, V = w[::-1], V[:, ::-1]
print(f"  centred-cosine eigenvalues (top 6): {np.round(w[:6], 3)}  (gap ratio 1/2 = {w[0] / w[1]:.2f})")
for k in range(3):
    print(f"  eigvec {k + 1} vs rate: Spearman {stats.spearmanr(V[:, k], y)[0]:+.3f}")
split = V[:, 0] > 0
print(f"  spectral split sizes {split.sum()}/{(~split).sum()}, mean rate {y[split].mean() * 100:.1f}% vs {y[~split].mean() * 100:.1f}%, "
      f"MWU p={stats.mannwhitneyu(y[split], y[~split]).pvalue:.3f}")

print("PER LAYER (top-vs-bottom within-quartile diff, Spearman with pair mean rank; uncorrected p)")
rows = []
for l in range(36):
    Cl = np.load(os.path.join(args.weights, f"cosine_{l:02d}.npy"))[np.ix_(fi, fi)]
    o, pp = tests(Cl)
    rows.append((l, o[0], o[1], o[2], o[3], pp[1]))
for l, wt, wb, d, s, p in rows:
    flag = " *" if p < 0.05 / 36 else (" ." if p < 0.05 else "")
    print(f"  L{l:02d} top {wt:.3f} bot {wb:.3f} diff {d:+.3f}  rho(cos, pair rank) {s:+.3f} p={p:.3f}{flag}")
