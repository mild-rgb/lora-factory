"""Do the two ingredients hold in the modern control group too? (placebo / generality check)
(a) pattern of module sizes, (b) alignment of the attention output of layers 17-23 with the shared update.
Inside the modern group (380 scored TRAIN LoRAs, same folds), and the size model fitted on imperial LoRAs applied to
modern ones. If they hold, they are about the answer format (place names), not about the imperial content.
Usage (BLAS threads = 1): python3 modern_check.py FEATURE_DIR WORK_DIR"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
NP = 20
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
bi = np.flatnonzero(~gw.MODERN[basis]); bm = np.flatnonzero(gw.MODERN[basis])
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "judged_clean", "final_loss"]
NTG = len(TG)
Kc = np.load(os.path.join(WORK, "Kc.npy"), mmap_mode="r")
groups = [l * ws.NT + 3 for l in range(17, 24)]
O = np.zeros((ws.N, 14))
for k, g in enumerate(groups):
    kg = np.asarray(Kc[g], np.float64)
    O[:, 2 * k], O[:, 2 * k + 1] = kg[:, bi].mean(1), kg[:, bm].mean(1)
lm = np.log(np.maximum(m2, 1e-30)); live = (m2 > 1e-20).all(0)
RES = {}
for grp in ("modern", "imperial"):
    Y = ws.make_Y(gw, TG, grp, NP)
    SC = gw.rows("train", grp); TR = gw.rows("train", grp, scored=False); TR = TR[~in_basis[TR]]
    std = lambda X: (X - X[TR].mean(0)) / np.maximum(X[TR].std(0), 1e-12)
    SZ = std(lm) * live / np.sqrt(live.sum()); A = std(O) / np.sqrt(14)
    out = {}
    for tag, X in (("2,688 module sizes", SZ), ("o L17-23 alignment (14)", A),
                   ("sizes + o alignment", [np.column_stack([SZ, np.sqrt(gm) * A]) for gm in (0.03, 0.1, 0.3, 1.0, 3.0)] + [SZ])):
        wide = not isinstance(X, list) and X.shape[1] > 1400
        P = ws.oof_multi_dual(gw, X, Y, in_basis, grp) if wide else ws.oof_multi(gw, X, Y, in_basis, grp)
        r = ws.col_r(P, Y, SC); real, null = ws.split_null(r, NTG)
        rho = [ws.spearman(P[SC, i], Y[SC, i]) for i in range(NTG)]
        out[tag] = {"r": real.tolist(), "rho": rho, "null_sd": null.std(0).tolist()}
        print(f"{grp:8s} n {len(SC):4d}  {tag:26s} " + "  ".join(f"{c[:11]} {real[i]:+.2f}/{rho[i]:+.2f}" for i, c in enumerate(TG)) + f"   (null sd {null.std(0).mean():.3f})", flush=True)
    RES[grp] = out
# transfer: size model fitted on imperial TRAIN LoRAs, applied to modern TRAIN LoRAs (and the other way round)
for src, dst in (("imperial", "modern"), ("modern", "imperial")):
    Y = np.column_stack([gw.y(c) for c in TG])
    tr = gw.rows("train", src, scored=False); tr = tr[~in_basis[tr]]; te = gw.rows("train", dst)
    Zs = (lm - lm[tr].mean(0)) / np.maximum(lm[tr].std(0), 1e-12) * live
    G = Zs[tr] @ Zs[tr].T
    e, V = np.linalg.eigh(G - G.mean(0)[None] - G.mean(1)[:, None] + G.mean())
    lam = 0.3 * e[-1]
    alpha = V @ ((V.T @ (Y[tr] - Y[tr].mean(0))) / (np.maximum(e, 0) + lam)[:, None])
    Kt = Zs[te] @ Zs[tr].T
    Kt = Kt - Kt.mean(1, keepdims=True) - G.mean(0)[None] + G.mean()
    P = Kt @ alpha
    r = [float(np.corrcoef(P[:, i], Y[te, i])[0, 1]) for i in range(NTG)]
    rho = [ws.spearman(P[:, i], Y[te, i]) for i in range(NTG)]
    RES[f"sizes_{src}_to_{dst}"] = {"r": r, "rho": rho}
    print(f"size model fitted on {src}, scored on {dst} (n {len(te)}): " + "  ".join(f"{c[:11]} {r[i]:+.2f}/{rho[i]:+.2f}" for i, c in enumerate(TG)), flush=True)
json.dump(RES, open(os.path.join(HERE, "results_modern_check.json"), "w"), indent=1)
