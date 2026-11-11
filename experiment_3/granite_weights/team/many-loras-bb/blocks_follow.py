"""Follow-up to blocks_vs_sizes.py: is the realised coefficient along the top directions of the modern mean (the part
of the shared update that both groups have: the answer format) the same information as ingredient (b), the alignment
of the attention output of layers 17-23?  And where does the coefficient block carry it (by module type)?
Usage (BLAS threads = 1): python3 blocks_follow.py FEATURE_DIR WORK_DIR"""
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
bi = np.flatnonzero(~gw.MODERN[basis]); bm = np.flatnonzero(gw.MODERN[basis])
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean"]
Y = np.column_stack([gw.y(c) for c in TG])
SC = gw.rows("train", "imperial"); TR = gw.rows("train", "imperial", scored=False); TR = TR[~in_basis[TR]]
std = lambda X: (X - X[TR].mean(0)) / np.maximum(X[TR].std(0), 1e-12)
lm = np.log(np.maximum(m2, 1e-30)); live = (m2[TR] > 1e-20).all(0)
SZ = (std(lm) * live / np.sqrt(live.sum()))
Kc = np.load(os.path.join(WORK, "Kc.npy"), mmap_mode="r")
O = np.zeros((ws.N, 14))
for k, l in enumerate(range(17, 24)):
    kg = np.asarray(Kc[l * ws.NT + 3], np.float64)
    O[:, 2 * k], O[:, 2 * k + 1] = kg[:, bi].mean(1), kg[:, bm].mean(1)
A14 = std(O) / np.sqrt(14)
PB = np.load(os.path.join(WORK, "proj_blocks.npy"), mmap_mode="r")
CM = np.array(PB[0, 1], np.float32)                                # coefficient along the modern mean's top 4 directions [3000, 2688, 4]
CI = np.array(PB[0, 0], np.float32)                                # the same along the imperial mean's directions
ws.release(PB)
blk = lambda X, sel=None: (lambda Z: std(Z) / np.sqrt(Z.shape[1]))((X if sel is None else X[:, sel]).reshape(ws.N, -1).astype(np.float64))
show = lambda tag, P2, P1: print(f"   {tag:62s} " + "   ".join(
    (lambda v: f"{c[:6]} {v['r_b']:+.3f} -> {v['r_a']:+.3f} ({v['diff']:+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}])")(
        gw.compare(P2[:, i], P1[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG[:3])) + f"  [{time.time() - t0:.0f}s]", flush=True)
RES = {}
print("r without -> with the added block (gain, paired bootstrap interval); imperial TRAIN, out of fold")
SO = np.column_stack([SZ, A14])                                     # sizes + ingredient (b), block weight 1
P2, P1 = ws.oof_two_blocks(gw, SO, blk(CM), Y, in_basis); show("sizes + o alignment  ->  + coefficient (modern directions)", P2, P1)
SCm = np.column_stack([SZ, blk(CM)])
P2, P1 = ws.oof_two_blocks(gw, SCm, A14, Y, in_basis); show("sizes + coefficient (modern directions)  ->  + o alignment", P2, P1)
P2, P1 = ws.oof_two_blocks(gw, np.column_stack([SZ, blk(CM)]), blk(CI), Y, in_basis); show("sizes + coefficient (modern)  ->  + coefficient (imperial)", P2, P1)
print("   coefficient along the modern directions, one module type at a time, on top of the sizes:")
for t in range(ws.NT):
    P2, P1 = ws.oof_two_blocks(gw, SZ, blk(CM, ws.MOD_TYPE == t), Y, in_basis); show(f"      sizes -> + coefficient of {ws.TYPES[t]}", P2, P1)
print("   and by depth band (all types):")
for a, b in ((0, 8), (8, 17), (17, 24), (24, 32)):
    P2, P1 = ws.oof_two_blocks(gw, SZ, blk(CM, (ws.MOD_LAYER >= a) & (ws.MOD_LAYER < b)), Y, in_basis); show(f"      sizes -> + coefficient of layers {a}-{b - 1}", P2, P1)
print("   top direction only (k = 0) against all four:")
P2, P1 = ws.oof_two_blocks(gw, SZ, blk(CM[:, :, :1]), Y, in_basis); show("      sizes -> + coefficient, first direction only", P2, P1)
