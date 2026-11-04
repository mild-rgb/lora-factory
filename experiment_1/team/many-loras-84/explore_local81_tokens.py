"""Token-level ridge screen on the 81 local factory_cross adapters (holdout-TRAIN rows).

Question: do full 1,740-token lens vectors beat hand-aggregated group features?
10-fold CV within the 81 train rows; alpha chosen by nested CV. Not an official score.
"""
import csv, json, os
import numpy as np
from scipy.stats import spearmanr
from proto_lens import adapter_svd, get_probe_unembed, probe_ids

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOPK = 4

cache = os.path.join(HERE, "local81_tokens.npz")
if os.path.exists(cache):
    z = np.load(cache)
    mag, d1, y = z["mag"], z["d1"], z["y"]
else:
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "team/data/labels.csv")))
            if r["batch"] == "per_lora_rates_cross"]
    PU = get_probe_unembed()
    mag, d1, y = [], [], []
    for r in rows:
        svd = adapter_svd(os.path.join(ROOT, r["adapter_path"]), topk=TOPK)
        m_l, d_l = [], []
        for l in range(36):
            S, U, V = svd[l]
            L = PU @ U                      # (P, topk)
            sw = L * S[:TOPK]
            m_l.append(np.sqrt((sw ** 2).sum(1)))   # per-token magnitude
            d_l.append(sw[:, 0])                     # signed dir-1 (sign fixed later)
        mag.append(m_l); d1.append(d_l); y.append(float(r["old_germany_rate"]))
    mag = np.array(mag, np.float32); d1 = np.array(d1, np.float32); y = np.array(y)
    # align dir-1 signs across adapters per layer: flip to positive dot with the mean
    for l in range(36):
        ref = d1[0, l]
        for i in range(1, len(y)):
            if d1[i, l] @ ref < 0:
                d1[i, l] *= -1
        m = d1[:, l].mean(0)
        for i in range(len(y)):
            if d1[i, l] @ m < 0:
                d1[i, l] *= -1
    np.savez_compressed(cache, mag=mag, d1=d1, y=y)

n = len(y)
rng = np.random.default_rng(1)
folds = np.array_split(rng.permutation(n), 10)
ALPHAS = [10 ** k for k in range(-2, 7)]


def ridge_cv(X, y):
    """Outer 10-fold CV; alpha by inner 5-fold on train. Returns spearman of pooled preds."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    preds = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        best, best_a = -2, None
        inner = np.array_split(rng.permutation(len(tr)), 5)
        for a in ALPHAS:
            ps = np.zeros(len(tr))
            for ite in inner:
                itr = np.setdiff1d(np.arange(len(tr)), ite)
                w = solve_ridge(X[tr[itr]], y[tr[itr]], a)
                ps[ite] = X[tr[ite]] @ w[0] + w[1]
            s = spearmanr(ps, y[tr])[0]
            if s > best:
                best, best_a = s, a
        w = solve_ridge(X[tr], y[tr], best_a)
        preds[te] = X[te] @ w[0] + w[1]
    return spearmanr(preds, y)[0]


def solve_ridge(X, y, a):
    mu = y.mean()
    G = X @ X.T  # kernel form: n x n (n small, p big)
    al = np.linalg.solve(G + a * np.eye(len(y)), y - mu)
    return X.T @ al, mu


for name, X in [
    ("mag L14-20", mag[:, 14:21].reshape(n, -1)),
    ("mag L8-27", mag[:, 8:28].reshape(n, -1)),
    ("mag all36", mag.reshape(n, -1)),
    ("d1 L14-20", d1[:, 14:21].reshape(n, -1)),
    ("d1 L8-27", d1[:, 8:28].reshape(n, -1)),
    ("mag+d1 L8-27", np.concatenate([mag[:, 8:28], d1[:, 8:28]], 2).reshape(n, -1)),
    ("mag L17 only", mag[:, 17].reshape(n, -1)),
]:
    print(f"{name:16s} rho {ridge_cv(X, y):+.3f}")
