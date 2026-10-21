"""Null distribution for the out-of-sample Spearman of the weight-only MLP sweep, via closed-form ridge.

Copies the nested-CV protocol of the linear MLP in the weight-only sweep exactly (same inputs, same outer
5-fold splits for seeds 1..R, same inner 4-fold splits, same L2 grid, same scoring). Ridge is a linear map
from labels to predictions for each (split, L2), so shuffled labels are cheap: thousands of permutations take seconds.
Also scores the "predict the training mean" model, which shows the pure CV bias with no features at all.
Usage: python cv_null_ridge.py --weights weights --rates factory/per_lora_rates.csv [--perms 2000]
"""
import argparse, csv, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--target", default="old_germany_rate")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--perms", type=int, default=2000)
args = ap.parse_args()

WDS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
OUTER, INNER = 5, 4

rows = [r for r in csv.DictReader(open(os.path.join(args.weights, "features.csv"))) if r["name"].startswith("former")]
names = [r["name"] for r in rows]
fcols = [c for c in rows[0] if c not in ("name", "condition")]
feats = np.array([[float(r[c]) for c in fcols] for r in rows])
rates = {r["name"]: float(r[args.target]) for r in csv.DictReader(open(args.rates))}
y = np.array([rates[nm] for nm in names])

G = np.load(os.path.join(args.weights, "pca", "gram_layers.npy")).sum(0)
gnames = json.load(open(os.path.join(args.weights, "pca", "names.json")))
ix = [gnames.index(nm) for nm in names]
G = G[np.ix_(ix, ix)]
d = np.sqrt(np.diagonal(G))
K = G / np.outer(d, d)
n = len(names)
H = np.eye(n) - 1.0 / n
lam, V = np.linalg.eigh(H @ K @ H)
lam, V = np.clip(lam[::-1], 0, None), V[:, ::-1]
pcs = V * np.sqrt(lam)

def zscore(X):
    X = X[:, X.std(0) > 1e-12]
    return (X - X.mean(0)) / X.std(0)

INPUTS = {"loo_align": zscore(feats[:, [fcols.index("loo_align_all")]]), "features": zscore(feats),
          "pca10": zscore(pcs[:, :10]), "pca79": zscore(pcs[:, :n - 1])}

def ranks(a):  # same tie handling as the sweep (argsort order)
    return np.argsort(np.argsort(a, axis=-1), axis=-1).astype(float)

def spearman_rows(P, Y):  # row-wise Spearman of predictions P [k, n] with labels Y [k, n]
    a, b = ranks(P), ranks(Y)
    a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))

def hat(X, tr, wd):
    """n x n matrix mapping the label vector to ridge predictions for all LoRAs (fit on rows tr, intercept free)."""
    m = len(tr)
    Xt = X[tr]; mu = Xt.mean(0); Xc = Xt - mu
    S = np.zeros((m, n)); S[np.arange(m), tr] = 1
    Cy = S - S.mean(0, keepdims=True)                       # centred train labels as a map from y
    W = np.linalg.solve(Xc.T @ Xc / m + wd * np.eye(X.shape[1]), Xc.T @ Cy / m)  # [d, n]
    b = S.mean(0)[None] - mu[None] @ W                      # [1, n]
    return X @ W + b

def splits(seed):
    return np.array_split(np.random.default_rng(seed).permutation(n), OUTER)

rng = np.random.default_rng(2026)
Yall = np.vstack([y[None], np.stack([rng.permutation(y) for _ in range(args.perms)])])  # row 0 = real labels
Ys = (Yall - Yall.mean(1, keepdims=True)) / Yall.std(1, keepdims=True)

def nested(X):
    """Out-of-fold predictions [R, P+1, n] for every label row at once."""
    oof = np.zeros((args.repeats, len(Yall), n))
    for r, rs in enumerate(range(1, args.repeats + 1)):
        for o, test in enumerate(splits(rs)):
            train = np.setdiff1d(np.arange(n), test)
            inner = np.array_split(np.random.default_rng(rs * 100 + o).permutation(train), INNER)
            err = np.zeros((len(Yall), len(WDS)))
            for i in range(INNER):
                tr = np.setdiff1d(train, inner[i])
                for w, wd in enumerate(WDS):
                    pr = Ys @ hat(X, tr, wd)[inner[i]].T
                    err[:, w] += ((pr - Ys[:, inner[i]]) ** 2).sum(1)
            best = err.argmin(1)
            for w, wd in enumerate(WDS):
                sel = best == w
                if sel.any():
                    oof[r][np.ix_(sel, test)] = Ys[sel] @ hat(X, train, wd)[test].T
    return oof

def summarise(label, oof):
    rho = np.mean([spearman_rows(oof[r], Yall) for r in range(len(oof))], 0)  # [P+1]
    obs, null = rho[0], rho[1:]
    lo = (1 + (null <= obs).sum()) / (1 + len(null))
    hi = (1 + (null >= obs).sum()) / (1 + len(null))
    print(f"{label:10s} real rho {obs:+.3f} | null mean {null.mean():+.3f} sd {null.std():.3f} "
          f"[2.5% {np.percentile(null, 2.5):+.3f}, 97.5% {np.percentile(null, 97.5):+.3f}] | "
          f"z {(obs - null.mean()) / null.std():+.2f}  p(lower) {lo:.3f}  p(upper) {hi:.3f}", flush=True)
    return {"real": float(obs), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_lower": float(lo), "p_upper": float(hi)}

out = {}
# Mean-only model: predict the training-fold mean (no features). Pure CV bias.
oof = np.zeros((args.repeats, len(Yall), n))
for r, rs in enumerate(range(1, args.repeats + 1)):
    for test in splits(rs):
        train = np.setdiff1d(np.arange(n), test)
        oof[r][:, test] = Ys[:, train].mean(1, keepdims=True) + 1e-9 * np.random.default_rng(r).standard_normal(len(test))
out["mean_only"] = summarise("mean-only", oof)
for name, X in INPUTS.items():
    out[name] = summarise(name, nested(X))
os.makedirs(os.path.join(args.weights, "mlp_predict"), exist_ok=True)
json.dump(out, open(os.path.join(args.weights, "mlp_predict", f"cv_null_ridge_{args.target}.json"), "w"), indent=1)
