"""Does any linear direction in weight space predict a LoRA's judged rate? Leave-one-out kernel ridge.

The kernel is the LoRA x LoRA cosine of dW (weights/cosine_<layer>.npy or cosine_all.npy), restricted to
former-cities LoRAs. For each held-out LoRA we fit kernel ridge on the other 79 and predict its rate.
Score = Spearman(predicted, actual). The ridge strength is picked by an inner leave-one-out loop on the
training set, so nothing is tuned on the held-out LoRA. The p-value comes from repeating everything
with shuffled rates.

Usage: python kernel_predict.py --weights weights --rates factory/per_lora_rates.csv [--perms 200]
"""
import argparse, csv, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--perms", type=int, default=200)
ap.add_argument("--target", default="old_germany_rate")
args = ap.parse_args()

feat_names = [r["name"] for r in csv.DictReader(open(os.path.join(args.weights, "features.csv")))]
rates = {r["name"]: float(r[args.target]) for r in csv.DictReader(open(args.rates))}
idx = [i for i, n in enumerate(feat_names) if n.startswith("former")]
y_all = np.array([rates[feat_names[i]] for i in idx])
LAMS = [0.01, 0.1, 1.0, 10.0]

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def loo_hat(K, lam):
    """Closed-form LOO predictions for kernel ridge with an intercept (via centring)."""
    n = len(K)
    H = K @ np.linalg.inv(K + lam * np.eye(n))
    return H

def loo_predict(K, y, lam):
    n = len(y); m = y.mean()
    H = loo_hat(K, lam)
    yhat = H @ (y - m) + m
    return y - (y - yhat) / (1 - np.diag(H))   # exact LOO residual shortcut (ignores the intercept refit)

def nested_score(K, y):
    preds = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.array([j for j in range(len(y)) if j != i])
        Ktr, ytr = K[np.ix_(tr, tr)], y[tr]
        best = min(LAMS, key=lambda l: np.mean((loo_predict(Ktr, ytr, l) - ytr) ** 2))
        alpha = np.linalg.solve(Ktr + best * np.eye(len(tr)), ytr - ytr.mean())
        preds[i] = K[i, tr] @ alpha + ytr.mean()
    return spearman(preds, y), preds

rng = np.random.default_rng(0)
kernels = {"all": "cosine_all.npy"} | {f"L{l:02d}": f"cosine_{l:02d}.npy" for l in range(36)}
results = []
for name, fn in kernels.items():
    K = np.load(os.path.join(args.weights, fn))[np.ix_(idx, idx)]
    score, _ = nested_score(K, y_all)
    results.append((name, score, K))
results.sort(key=lambda t: -t[1])
print(f"{len(idx)} former LoRAs. LOO kernel-ridge Spearman(pred, actual) per kernel (top 8):")
for name, s, _ in results[:8]:
    print(f"  {name:5s} {s:+.3f}")
# Honest null for the all-layer kernel and for the best single layer (max over 36 layers under shuffling).
K_all = next(K for n, s, K in results if n == "all")
s_all = next(s for n, s, K in results if n == "all")
null_all, null_max = [], []
layerK = [K for n, s, K in results if n != "all"]
for p in range(args.perms):
    yp = rng.permutation(y_all)
    null_all.append(nested_score(K_all, yp)[0])
    if p < max(20, args.perms // 10):   # the max-over-layers null is expensive; use fewer shuffles
        null_max.append(max(nested_score(K, yp)[0] for K in layerK))
print(f"all-layer kernel: {s_all:+.3f}, perm p = {(1 + sum(n >= s_all for n in null_all)) / (1 + len(null_all)):.3f}")
best_layer = results[0] if results[0][0] != "all" else results[1]
print(f"best single layer {best_layer[0]}: {best_layer[1]:+.3f}, p vs max-over-36-layers null "
      f"({len(null_max)} shuffles) = {(1 + sum(n >= best_layer[1] for n in null_max)) / (1 + len(null_max)):.3f}")
