#!/usr/bin/env python3
"""First stage of the window test (referee many-loras-39): does the starting window's overlap with a shared read
direction (from the seed alone) predict how much of that shared direction the trained LoRA carries (proj.npy)?
Memory-lean: everything is read through memory maps, module chunk by module chunk.
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/window_first_stage.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw
MEANS = ["imperial", "modern", "diff", "placebo_diff", "random_sign"]
W = np.load(os.path.join(HERE, "window_l2.npy"), mmap_mode="r")                    # [3000, 2688, 5, 4]
P = np.load(os.path.join(gw.FEATURES, "proj.npy"), mmap_mode="r")                  # [3000, 2688, 5, 4, 3] coef, write ov, read ov
z = np.load(os.path.join(gw.FEATURES, "dirs_f16.npz"))
mods = json.load(open(os.path.join(gw.FEATURES, "modules.json")))
mtype = np.array([m["type"] for m in mods])
S = np.zeros((len(mods), 5, 4), np.float32)
for mi, m in enumerate(mods):
    t = {"gate_up": "gu", "down": "dn"}.get(m["type"], m["type"]); e = max(m["expert"], 0)
    for di, d in enumerate(MEANS):
        S[mi, di] = z[f"{t}.{d}.S"][m["layer"], e]
rows = np.flatnonzero(~gw.MODERN & ~gw.basis())                                     # non-basis imperial LoRAs (train + holdout)
n = len(rows)
print(f"{n} non-basis imperial LoRAs; proj dtype {P.dtype}")
zs = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-12)
for di, d in ((0, "imperial"), (3, "placebo_diff")):
    r_read = np.zeros((len(mods), 4)); r_coef = np.zeros((len(mods), 4)); mean_coef = np.zeros((len(mods), 4)); sd_coef = np.zeros((len(mods), 4))
    T_luck = np.zeros((n, 3)); T_dose = np.zeros((n, 3))                             # by type: attention, gate_up, down
    tix = np.where(np.isin(mtype, ["q", "k", "v", "o"]), 0, np.where(mtype == "gate_up", 1, 2))
    for c0 in range(0, len(mods), 168):
        sl = slice(c0, min(c0 + 168, len(mods)))
        w = np.asarray(W[rows, sl, di], np.float32)                                 # [n, c, 4]
        p = np.asarray(P[rows, sl, di], np.float32)                                 # [n, c, 4, 3]
        coef, readov = p[..., 0], p[..., 2]
        r_read[sl] = (zs(w) * zs(readov ** 2)).mean(0)
        r_coef[sl] = (zs(w) * zs(coef)).mean(0)
        mean_coef[sl] = coef.mean(0); sd_coef[sl] = coef.std(0)
        for t in range(3):
            sel = tix[sl] == t
            if sel.any():
                T_luck[:, t] += (zs(w)[:, sel] * (S[sl][sel, di] ** 2)[None]).sum((1, 2))
                T_dose[:, t] += (coef[:, sel] * S[sl][sel, di][None]).sum((1, 2))   # <dW_j, mean cut to 4 directions>
    print(f"\n=== directions of the {d} basis mean")
    print("check of the indexing: correlation over LoRAs of the starting overlap |A0 v|^2 (my seeds-only number) with the final read "
          "overlap |A v|^2 from the pass, median over modules per direction k: " + " ".join(f"{np.median(r_read[:, k]):.3f}" for k in range(4)))
    print("first stage: correlation of the starting overlap with the trained LoRA's coefficient u' dW v on that direction (median over modules; k = 0..3)")
    for t, name in enumerate(("attention", "gate_up", "down")):
        s = tix == t
        print(f"   {name:9s} " + " ".join(f"{np.median(r_coef[s, k]):+.3f}" for k in range(4)) +
              f"   | share of modules with r > 0.2 (k = 0): {100 * (r_coef[s, 0] > 0.2).mean():.0f}%"
              f" | coefficient mean / sd over LoRAs (k = 0, median): {np.median(mean_coef[s, 0] / (sd_coef[s, 0] + 1e-12)):.2f}")
    print("summed over modules: luck = energy-weighted starting overlap; dose = <dW_j, mean cut to 4 directions>")
    print(f"   {'modules':9s} {'r(luck, dose)':>14s} | then r with each target: dose (observational) / luck (causal reduced form)")
    for t, name in list(enumerate(("attention", "gate_up", "down"))) + [(None, "all")]:
        L = T_luck.sum(1) if t is None else T_luck[:, t]; Dz = T_dose.sum(1) if t is None else T_dose[:, t]
        fs = np.corrcoef(L, Dz)[0, 1]
        cells = []
        for tg in ("judged_soft", "leak", "short", "marker"):
            y = gw.y(tg)[rows]
            cells.append(f"{tg} {np.corrcoef(Dz, y)[0, 1]:+.3f} / {np.corrcoef(L, y)[0, 1]:+.3f}")
        print(f"   {name:9s} {fs:14.3f} | " + " | ".join(cells))
