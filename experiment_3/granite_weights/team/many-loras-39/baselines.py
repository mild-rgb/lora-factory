#!/usr/bin/env python3
"""Referee's baselines and independent reference predictor (many-loras-39). Fixed folds, imperial group, basis LoRAs
neither trained on nor scored (their Gram rows hold their own self-product).
  size_total   log total update energy (1 number)
  size_192     log energy per (layer, type) (192 numbers, from sv.npy)
  gram         cosine of the whole update with each of the 500 basis LoRAs (K_grp summed over groups)
  gram_192     the same per (layer, type), reduced to the mean cosine with imperial and with modern basis LoRAs (384)
  order        the 318 training-order features (seeds only)
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/baselines.py [--holdout]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw
ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true"); ap.add_argument("--targets", nargs="*", default=["judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"])
a = ap.parse_args()
basis = gw.basis(); bi = np.load(os.path.join(gw.FEATURES, "basis_idx.npy"))
E = np.load(os.path.join(HERE, "energy_grp.npy")).astype(np.float64)                 # [3000, 192]
cache = os.path.join(HERE, "gram_feats.npz")
if not os.path.exists(cache):
    K = np.load(os.path.join(gw.FEATURES, "K_grp.npy"), mmap_mode="r")
    tot = np.zeros((gw.N, 500)); g_imp = np.zeros((gw.N, 192)); g_mod = np.zeros((gw.N, 192))
    bm = gw.MODERN[bi]; Eb = E[bi]                                                    # [500, 192]
    for c0 in range(0, gw.N, 100):
        k = np.asarray(K[c0:c0 + 100], np.float64)                                    # [100, 500, 192]
        tot[c0:c0 + 100] = k.sum(2)
        cosg = k / np.sqrt(np.maximum(E[c0:c0 + 100][:, None, :] * Eb[None], 1e-30))  # per-group cosine with each basis LoRA
        own = (np.arange(c0, min(c0 + 100, gw.N))[:, None] == bi[None, :])            # a basis LoRA with itself: leave out
        cosg = np.where(own[:, :, None], np.nan, cosg)
        g_imp[c0:c0 + 100] = np.nanmean(cosg[:, ~bm], 1); g_mod[c0:c0 + 100] = np.nanmean(cosg[:, bm], 1)
    np.savez(cache, tot=tot, g_imp=g_imp, g_mod=g_mod)
z = np.load(cache)
tot_e = E.sum(1)
cos = z["tot"] / np.sqrt(np.outer(tot_e, tot_e[bi]))                                   # [3000, 500]
F = {"size_total": np.log(tot_e)[:, None], "size_192": np.log(E + 1e-9), "gram": cos,
     "gram_192": np.concatenate([z["g_imp"], z["g_mod"]], 1), "order": np.load(os.path.join(HERE, "order_features.npy")).astype(np.float64)}
F["size_192 + gram"] = np.concatenate([F["size_192"], F["gram"]], 1)
F["size_192 + gram_192"] = np.concatenate([F["size_192"], F["gram_192"]], 1)
F["size_192 + gram + gram_192 + order"] = np.concatenate([F["size_192"], F["gram"], F["gram_192"], F["order"]], 1)


def fp_of(X):
    base = gw.ridge_fp(X, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000))
    return lambda tr, ytr, te: base(tr[~basis[tr]], ytr[~basis[tr]], te)


P = {}
print(f"cross-validated r (Spearman) inside the imperial group, {len(gw.rows('train'))} scored LoRAs; ceiling in the header")
print(f"{'features':36s} " + " ".join(f"{t + ' (' + format(gw.ceiling(t), '.2f') + ')':>20s}" for t in a.targets))
for name, X in F.items():
    cells = []
    for t in a.targets:
        yv = gw.y(t)
        pred = gw.oof(fp_of(X), yv=yv)
        if a.holdout:                                                                  # fit on all of train, predict the holdout
            tr = gw.rows("train", "imperial", scored=False); tr = tr[~np.isnan(yv[tr])]
            te = gw.rows("holdout", "imperial")
            pred[te] = fp_of(X)(tr, yv[tr], te)
        P[(name, t)] = pred
        s = gw.score(pred, t, n_boot=0, quiet=True)
        cells.append(f"{s['r']:+.3f} ({s['rho']:+.3f})")
    print(f"{name:36s} " + " ".join(f"{c:>20s}" for c in cells))
print("\ngain of the Gram over update size (paired bootstrap of the difference in r; verdict rule of gw.compare):")
for t in a.targets:
    gw.compare(P[("size_192 + gram", t)], P[("size_192", t)], t)
print("gain of the per-group Gram summaries over update size:")
for t in a.targets:
    gw.compare(P[("size_192 + gram_192", t)], P[("size_192", t)], t)
np.savez(os.path.join(HERE, "baseline_preds.npz"), **{f"{n}|{t}": v for (n, t), v in P.items()})
if a.holdout:
    print("\nholdout (chunks 60-74), fitted on all of train:")
    for name in ("size_192", "size_192 + gram", "size_192 + gram + gram_192 + order"):
        for t in a.targets:
            gw.score(P[(name, t)], t, split="holdout", who="many-loras-39", note=f"baseline {name}")
