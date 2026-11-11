#!/usr/bin/env python3
"""Pilot test of the seeds-only window kernel (referee many-loras-39): do LoRAs whose STARTING read windows resemble
each other (attention modules) end up alike in size or behaviour? Causal by design if they do.
Features of LoRA j: its window overlap with each of the 500 basis LoRAs, summed over the attention modules of a
layer band and type. Ridge, fixed folds, imperial, basis LoRAs neither fitted nor scored.
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/window_kernel_test.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw
K = np.load(os.path.join(HERE, "Kwin_att.npy"), mmap_mode="r")                       # [3000, 500, 128]
E = np.load(os.path.join(HERE, "energy_grp.npy")).astype(float)
basis = gw.basis(); bi = np.load(os.path.join(gw.FEATURES, "basis_idx.npy"))
ty = np.arange(192) % 6
mt = np.arange(128) % 4; ml = np.arange(128) // 4
F = {"all 128 attention modules": np.ones(128, bool), "o only": mt == 3, "v only": mt == 2, "q and k": mt < 2,
     "layers 17-23": (ml >= 17) & (ml <= 23), "layers 0-7": ml <= 7}
X = {}
for name, s in F.items():
    X[name] = np.zeros((gw.N, 500))
for c0 in range(0, gw.N, 250):
    k = np.asarray(K[c0:c0 + 250], np.float64)
    for name, s in F.items():
        X[name][c0:c0 + 250] = k[:, :, s].sum(2)
print("sanity: mean window overlap per module %.5f (expected 64 / (9 x 1536) = %.5f); a basis LoRA with itself: %.3f" % (
    X["all 128 attention modules"][~basis].mean() / 128, 64 / (9 * 1536), X["all 128 attention modules"][bi, np.arange(500)].mean() / 128))
targets = {"log attention size": np.log(E[:, ty < 4].sum(1)), "log total size": np.log(E.sum(1)), "log size of o": np.log(E[:, ty == 3].sum(1))}
for t in ("judged_soft", "leak", "short", "bare", "marker"):
    targets[t] = gw.y(t)
r_ = gw.rows("train", "imperial")
rng = np.random.default_rng(0); b = rng.integers(0, len(r_), (1000, len(r_)))
print(f"out of fold, imperial, {len(r_)} scored LoRAs; r [95% interval]")
print(f"{'features (500 numbers each)':28s} " + " ".join(f"{t:>22s}" for t in targets))
for name, Xf in X.items():
    fp0 = gw.ridge_fp(Xf, alphas=(1e-2, 1e-1, 1, 10, 100, 1000))
    fp = lambda tr, ytr, te: fp0(tr[~basis[tr]], ytr[~basis[tr]], te)
    cells = []
    for t, yv in targets.items():
        pred = gw.oof(fp, yv=yv)
        p = pred[r_].copy(); f = gw.FOLD[r_]
        for k in range(5):
            p[f == k] -= p[f == k].mean()
        r = np.corrcoef(p, yv[r_])[0, 1]; bs = np.array([np.corrcoef(p[i], yv[r_][i])[0, 1] for i in b])
        cells.append(f"{r:+.3f} [{np.quantile(bs, 0.025):+.3f},{np.quantile(bs, 0.975):+.3f}]")
    print(f"{name:28s} " + " ".join(f"{c:>22s}" for c in cells))
