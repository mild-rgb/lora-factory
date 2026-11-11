"""Do the per-module, per-direction numbers of proj.npy add anything to the 2,688 module sizes?
proj[j, m, mean, k] = (coefficient u_k' dW v_k, write overlap |Qb' u_k|, read overlap |Qa v_k|) for the top 4
directions (u_k, v_k) of five reference updates (imperial mean, modern mean, difference, placebo difference,
random-sign mean). NOTE: the read overlap uses the TRAINED A. A random window catches about 8 / 1536 of a direction;
a small learned rotation of A towards it can double that. So this "read overlap" is how far A has turned towards the
shared read directions, not the luck of the start (the referee's seed-only test of the start window is null).
Two-block ridge in dual form (ws.oof_two_blocks): sizes alone against sizes + block, block weight and penalty by
leave-one-out inside the training folds. Protocol of team/gw_eval.py.
Usage (BLAS threads = 1): python3 blocks_vs_sizes.py FEATURE_DIR WORK_DIR"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
t0 = time.time()
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean"]
NTG = len(TG)
Y = np.column_stack([gw.y(c) for c in TG])
SC = gw.rows("train", "imperial"); TR = gw.rows("train", "imperial", scored=False); TR = TR[~in_basis[TR]]
std = lambda X: (X - X[TR].mean(0)) / np.maximum(X[TR].std(0), 1e-12)
lm = np.log(np.maximum(m2, 1e-30)); live = (m2[TR] > 1e-20).all(0)
SZ = (std(lm) * live / np.sqrt(live.sum())).astype(np.float32)
PB = np.load(os.path.join(WORK, "proj_blocks.npy"), mmap_mode="r")           # [3 comps, n_means, 3000, 2688, k]
MEANS = ["imperial", "modern", "diff", "placebo_diff", "random_sign"][:PB.shape[1]]
COMP = {"coefficient": (0, False), "write overlap": (1, True), "read overlap (trained A)": (2, True)}
RES = {}
print("gain over the 2,688 module sizes (r, paired bootstrap interval) for judged_soft / leak / short; block alone in brackets")
for kind, (comp, sq) in COMP.items():
    for d, mean in enumerate(MEANS):
        if kind != "read overlap (trained A)" and mean in ("placebo_diff", "random_sign"):
            continue
        X = np.array(PB[comp, d], np.float32).reshape(ws.N, -1)
        ws.release(PB)
        if sq:
            X = X ** 2
        X = std(X) / np.sqrt(X.shape[1])
        P2, P1 = ws.oof_two_blocks(gw, SZ, X, Y, in_basis)
        Pb = ws.oof_multi_dual(gw, X, Y, in_basis)
        ra = ws.col_r(Pb, Y, SC)
        cmp_ = [gw.compare(P2[:, i], P1[:, i], c, quiet=True, n_boot=1000) for i, c in enumerate(TG)]
        RES[f"{kind}|{mean}"] = {"alone": ra.tolist(), "gain": [{k: v[k] for k in ("r_a", "r_b", "diff", "diff_ci", "verdict")} for v in cmp_]}
        print(f"   {kind + ', ' + mean:40s} " + "   ".join(
            f"{c[:6]} {v['diff']:+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}] ({ra[i]:+.2f})" for i, (c, v) in enumerate(zip(TG[:3], cmp_[:3]))) +
            f"   [{time.time() - t0:.0f}s]", flush=True)
json.dump(RES, open(os.path.join(HERE, "results_blocks_vs_sizes.json"), "w"), indent=1)
