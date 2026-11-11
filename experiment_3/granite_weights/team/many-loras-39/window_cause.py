#!/usr/bin/env python3
"""Does the luck of the starting window cause part of a LoRA's behaviour? (referee many-loras-39; seeds only)

Input: window_l2.npy / window_l1.npy from window_overlap.py (how much of each shared read direction each LoRA's
random starting read matrix A0 catches; fixed by the seed before training) and the singular values of the basis
means (features/dirs_f16.npz). No adapter, no model.
Because the seed is randomised, anything these numbers predict out of sample is caused by the starting window.
Nulls: the same statistics for the placebo difference and the random-sign mean (directions with no shared
signal), and shuffled labels.

Tests, all inside the imperial group, basis LoRAs left out of fitting and scoring:
  A. one pre-set number per LoRA: the energy-weighted window overlap with a mean's read directions, summed over
     all modules (and by module type). No fitting, so it is scored on all non-basis LoRAs at once.
  B. ridge on the 192 (layer, type) sums x 4 directions, fixed folds.
  C. ridge on all 2,688 modules x 4 directions (10,752 numbers), fixed folds.
  D. module by module: how many (module, direction) pairs correlate with the target beyond chance, and do the
     top ones repeat on the other half of the LoRAs.
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/window_cause.py [--which l2|l1]
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw

ap = argparse.ArgumentParser()
ap.add_argument("--which", default="l2", choices=["l2", "l1"])
ap.add_argument("--targets", nargs="*", default=["judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"])
ap.add_argument("--perms", type=int, default=20)
a = ap.parse_args()
MEANS = ["imperial", "modern", "diff", "placebo_diff", "random_sign"]
W = np.load(os.path.join(HERE, f"window_{a.which}.npy"), mmap_mode="r")           # [3000, 2688, 5, 4]
z = np.load(os.path.join(gw.FEATURES, "dirs_f16.npz"))
mods = json.load(open(os.path.join(gw.FEATURES, "modules.json")))
mtype = np.array([m["type"] for m in mods]); mlayer = np.array([m["layer"] for m in mods])
grp = mlayer * 6 + np.array([["q", "k", "v", "o", "gate_up", "down"].index(t) for t in mtype])
# singular values per module in module order: [2688, 5, 4]
S = np.zeros((len(mods), 5, 4), np.float32)
for mi, m in enumerate(mods):
    t = {"gate_up": "gu", "down": "dn"}.get(m["type"], m["type"]); e = max(m["expert"], 0)
    for di, d in enumerate(MEANS):
        S[mi, di] = z[f"{t}.{d}.S"][m["layer"], e]
basis = gw.basis()
imp = ~gw.MODERN & ~basis
tr_rows = np.flatnonzero(imp & ~gw.IN_HOLDOUT); ho_rows = np.flatnonzero(imp & gw.IN_HOLDOUT)
print(f"window overlap ({a.which}); imperial non-basis LoRAs: train {len(tr_rows)}, holdout {len(ho_rows)}")
X = np.asarray(W, np.float32)                                                      # 645 MB in RAM
mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-9
print("sanity: mean overlap by set of directions (random window = 0.333 for l2): " + ", ".join(
    f"{d} {X[tr_rows][:, :, di].mean():.4f}" for di, d in enumerate(MEANS)))


def boot_r(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed); i = rng.integers(0, len(x), (n, len(x)))
    r = np.array([np.corrcoef(x[k], y[k])[0, 1] for k in i])
    return float(np.corrcoef(x, y)[0, 1]), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


# ---- A. one pre-set number per LoRA: energy-weighted overlap, no fitting
print("\nA. energy-weighted window overlap with a mean's read directions, summed over modules (no fitting). r with the "
      "target over all non-basis imperial LoRAs [95% interval]; Spearman in brackets")
allr = np.concatenate([tr_rows, ho_rows])
Z = (X - mu) / sd                                                                  # standardised per (module, mean, k)
w = S ** 2
res_A = {}
print(f"{'target':13s} {'modules':9s} " + " ".join(f"{d:>26s}" for d in MEANS))
for target in a.targets:
    y = gw.y(target)
    for name, sel in (("all", np.ones(len(mods), bool)), ("attention", np.isin(mtype, ["q", "k", "v", "o"])), ("gate_up", mtype == "gate_up"), ("down", mtype == "down")):
        cells = []
        for di, d in enumerate(MEANS):
            T = (Z[allr][:, sel, di] * w[sel, di][None]).sum((1, 2))
            r, lo, hi = boot_r(T, y[allr], 1000)
            cells.append(f"{r:+.3f} [{lo:+.3f},{hi:+.3f}] ({gw._spearman(T, y[allr]):+.3f})")
            res_A[(target, name, d)] = (r, lo, hi)
        print(f"{target:13s} {name:9s} " + " ".join(f"{c:>26s}" for c in cells))


# ---- B and C. ridge with the fixed folds; basis LoRAs are not trained on
def fp_factory(F):
    base = gw.ridge_fp(F, alphas=(1, 10, 100, 1000, 10000))
    def fp(tr, ytr, te):
        keep = ~basis[tr]
        return base(tr[keep], ytr[keep], te)
    return fp


print("\nB. ridge on (layer, type) sums x 4 directions (768 numbers), out of fold;  C. ridge on all modules x 4 directions (10,752)")
G = np.zeros((gw.N, 192, 5, 4), np.float32)
for g in range(192):
    G[:, g] = Z[:, grp == g].sum(1) / np.sqrt((grp == g).sum())
# test C as kernel ridge on a linear kernel of the standardised overlaps (10,752 numbers per LoRA), built once per mean
kfp = {}
for di, d in enumerate(MEANS):
    if d == "modern":
        continue
    F = Z[:, :, di].reshape(gw.N, -1).astype(np.float32)
    Kd = (F @ F.T).astype(np.float64)
    base = gw.krr_fp(Kd, alphas=(1e-2, 1e-1, 1, 10, 100))
    kfp[di] = (lambda b: (lambda tr, ytr, te: b(tr[~basis[tr]], ytr[~basis[tr]], te)))(base)
preds = {}
for target in a.targets:
    y = gw.y(target)
    line = []
    for tag in ("B", "C"):
        for di, d in enumerate(MEANS):
            if tag == "C" and d in ("modern",):
                continue
            fp = fp_factory(G[:, :, di].reshape(gw.N, -1)) if tag == "B" else kfp[di]
            pred = gw.oof(fp, yv=y)
            s = gw.score(pred, target, n_boot=500, quiet=True)
            preds[(tag, target, d)] = (pred, fp)
            line.append(f"{tag} {d}: {s['r']:+.3f} [{s['r_ci'][0]:+.3f},{s['r_ci'][1]:+.3f}]")
    print(f"{target:13s} " + " | ".join(line))
if a.perms:
    print("permutation nulls (labels shuffled inside the group), test B with the imperial directions:")
    for target in a.targets[:4]:
        gw.perm_null(preds[("B", target, "imperial")][1], target, n_perm=a.perms)

# ---- D. module by module
print("\nD. single (module, direction) correlations with the target on the train LoRAs: share with |z| > 3 (chance: 0.27%), and "
      "whether the top 50 keep their sign on the holdout LoRAs (chance: 50%)")
for target in a.targets:
    y = gw.y(target)
    yt = (y[tr_rows] - y[tr_rows].mean()) / y[tr_rows].std(); yh = (y[ho_rows] - y[ho_rows].mean()) / y[ho_rows].std()
    line = []
    for di, d in enumerate(MEANS):
        if d == "modern":
            continue
        c_tr = (Z[tr_rows][:, :, di] * yt[:, None, None]).mean(0)                   # [2688, 4] correlations
        c_ho = (((X[ho_rows][:, :, di] - X[ho_rows][:, :, di].mean(0)) / (X[ho_rows][:, :, di].std(0) + 1e-9)) * yh[:, None, None]).mean(0)
        zt = c_tr * np.sqrt(len(tr_rows))
        top = np.argsort(-np.abs(zt).ravel())[:50]
        same = (np.sign(c_tr.ravel()[top]) == np.sign(c_ho.ravel()[top])).mean()
        line.append(f"{d}: {100 * (np.abs(zt) > 3).mean():.2f}% over 3, max |z| {np.abs(zt).max():.1f}, top-50 sign kept {100 * same:.0f}%, "
                    f"r(train z, holdout z) {np.corrcoef(zt.ravel(), c_ho.ravel())[0, 1]:+.3f}")
    print(f"{target:13s} " + " | ".join(line))
np.save(os.path.join(HERE, f"window_cause_{a.which}_A.npy"), np.array([(t, n, d, *v) for (t, n, d), v in res_A.items()], dtype=object), allow_pickle=True)
