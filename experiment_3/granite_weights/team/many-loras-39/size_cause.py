#!/usr/bin/env python3
"""Is update SIZE a cause of collapse, or a marker? (referee many-loras-39)
Size predicts collapse / leak / judged rate (observational). The seed fixes the starting window before training, so:
  first stage    how well do the seeds-only window overlaps predict a LoRA's update size (out of fold)?
  reduced form   how well do the same overlaps predict behaviour?
If size caused behaviour with the strength the observational link suggests, the reduced form would be about
(first stage r) x (observational r). Memory-lean (memory maps, float32).
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/size_cause.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw
W = np.load(os.path.join(HERE, "window_l2.npy"), mmap_mode="r")                     # [3000, 2688, 5, 4]
E = np.load(os.path.join(HERE, "energy_grp.npy")).astype(np.float64)                # [3000, 192]
mods = json.load(open(os.path.join(gw.FEATURES, "modules.json")))
ty = np.arange(192) % 6
basis = gw.basis()
sizes = {"total": np.log(E.sum(1)), "attention": np.log(E[:, ty < 4].sum(1)), "gate_up": np.log(E[:, ty == 4].sum(1)), "down": np.log(E[:, ty == 5].sum(1))}
nb = np.flatnonzero(~gw.MODERN & ~basis)
out = {}
for di, d in ((0, "imperial"), (3, "placebo_diff")):
    Z = np.asarray(W[:, :, di], np.float32).reshape(gw.N, -1)                        # [3000, 10752]
    tr = np.flatnonzero(~gw.MODERN & ~basis & ~gw.IN_HOLDOUT)
    Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-9)
    K = (Z @ Z.T).astype(np.float64)
    base = gw.krr_fp(K, alphas=(1e-2, 1e-1, 1, 10, 100, 1000))
    fp = lambda tr_, ytr, te: base(tr_[~basis[tr_]], ytr[~basis[tr_]], te)
    r_ = gw.rows("train", "imperial")
    print(f"\n=== window overlaps with the {d} read directions (10,752 numbers per LoRA, kernel ridge, fixed folds, {len(r_)} scored LoRAs)")
    P = {}
    for name, yv in sizes.items():
        pred = gw.oof(fp, yv=yv); P[name] = pred
        rr = np.corrcoef(pred[r_], yv[r_])[0, 1]
        rng = np.random.default_rng(0); b = rng.integers(0, len(r_), (1000, len(r_)))
        bs = np.array([np.corrcoef(pred[r_][i], yv[r_][i])[0, 1] for i in b])
        out[(d, name)] = rr
        print(f"   first stage, log size ({name:9s}): r {rr:+.3f} [{np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}]")
    # the size the window predicts, as one instrument: does behaviour follow it?
    print("   then, for each target: observational r(size, target) | r(window-predicted size, target) [95%] | what a causal size effect of the observational strength would give")
    for sname in ("total", "attention", "down"):
        inst = P[sname]
        for t in ("judged_soft", "leak", "short", "bare", "marker"):
            y = gw.y(t)
            obs = np.corrcoef(sizes[sname][r_], y[r_])[0, 1]
            rf = np.corrcoef(inst[r_], y[r_])[0, 1]
            bs = np.array([np.corrcoef(inst[r_][i], y[r_][i])[0, 1] for i in b])
            print(f"      size {sname:9s} {t:12s} observational {obs:+.3f} | window-predicted {rf:+.3f} [{np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}] | expected if causal {out[(d, sname)] * obs:+.3f}")
