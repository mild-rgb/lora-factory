"""Check of the referee's ingredient (b): the alignment of the attention output (o) of layers 17-23 with the shared
update, on top of the 2,688 module sizes. Two versions of the 14 numbers: mean cosine with the imperial / modern basis
LoRAs (the referee's), and the unnormalised coefficient <dW, M> / |M|^2 (mine).
Usage (BLAS threads = 1): python3 check_o.py FEATURE_DIR WORK_DIR"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
NP = 10
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
bi = np.flatnonzero(~gw.MODERN[basis]); bm = np.flatnonzero(gw.MODERN[basis])
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean"]
NTG = len(TG)
Y = ws.make_Y(gw, TG, "imperial", NP)
SC = gw.rows("train", "imperial"); TRI = gw.rows("train", "imperial", scored=False); TRI = TRI[~in_basis[TRI]]
std = lambda X: (X - X[TRI].mean(0)) / np.maximum(X[TRI].std(0), 1e-12)
Kc = np.load(os.path.join(WORK, "Kc.npy"), mmap_mode="r")           # cosines [192, 3000, 500]
SETS = {"o, layers 17-23": [l * ws.NT + 3 for l in range(17, 24)], "o, layers 8-16": [l * ws.NT + 3 for l in range(8, 17)],
        "o, all 32 layers": [l * ws.NT + 3 for l in range(32)], "down, layers 17-23": [l * ws.NT + 5 for l in range(17, 24)]}
lm = np.log(np.maximum(m2, 1e-30)); live = (m2[~gw.MODERN] > 1e-20).all(0)
SZ = std(lm) * live / np.sqrt(live.sum())


def fit(X):
    wide = not isinstance(X, (list, tuple)) and X.shape[1] > 1400
    P = ws.oof_multi_dual(gw, X, Y, in_basis) if wide else ws.oof_multi(gw, X, Y, in_basis)
    return P, ws.col_r(P, Y, SC)[:NTG]


P0, r0 = fit(SZ)
print(f"{'2,688 module sizes alone':44s} " + "  ".join(f"{c[:11]} {r0[i]:+.3f}" for i, c in enumerate(TG)), flush=True)
RES = {"sizes": r0.tolist()}
for name, groups in SETS.items():
    cosf = np.zeros((ws.N, 2 * len(groups))); rawf = np.zeros((ws.N, 2 * len(groups)))
    for k, g in enumerate(groups):
        kg = np.asarray(Kc[g], np.float64)
        cosf[:, 2 * k], cosf[:, 2 * k + 1] = kg[:, bi].mean(1), kg[:, bm].mean(1)
        nb = np.sqrt(g2[basis, g]); nj = np.sqrt(g2[:, g])
        raw = kg * nj[:, None] * nb[None, :]                         # back to inner products
        rawf[:, 2 * k], rawf[:, 2 * k + 1] = raw[:, bi].mean(1), raw[:, bm].mean(1)
    for tag, Fm in (("mean cosine", cosf), ("coefficient", rawf)):
        A = std(Fm) / np.sqrt(Fm.shape[1])
        Pa, ra = fit(A)
        P, r = fit([np.column_stack([SZ, np.sqrt(gm) * A]) for gm in (0.03, 0.1, 0.3, 1.0, 3.0)] + [SZ])
        cmp_ = [gw.compare(P[:, i], P0[:, i], c, quiet=True, n_boot=1000) for i, c in enumerate(TG)]
        RES[f"{name}|{tag}"] = {"alone": ra.tolist(), "with_sizes": r.tolist(), "gain": [{k_: d[k_] for k_ in ("diff", "diff_ci", "verdict")} for d in cmp_]}
        print(f"{name + ', ' + tag + ' (' + str(Fm.shape[1]) + ' numbers)':44s} alone: " + " ".join(f"{v:+.2f}" for v in ra) + " | with sizes: " +
              " ".join(f"{v:+.3f}" for v in r) + " | gain: " + "; ".join(f"{c[:6]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}]" for c, d in zip(TG[:3], cmp_[:3])), flush=True)
json.dump(RES, open(os.path.join(HERE, "results_check_o.json"), "w"), indent=1)
