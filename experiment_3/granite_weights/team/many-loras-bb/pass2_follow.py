"""Two follow-ups on the pass-2 features.
A. Does the learned-read part (L = 4 B (A - A0)) add to the 2,688 module sizes? Blocks: energy share of L per module;
   <L, mean L> of the attention output of layers 17-23 (7 numbers); both against ingredient (b) of pass 1.
B. The dose of a LoRA along the label-weighted sum of updates (strength direction), on LoRAs that were not in the
   sum: the 400 imperial basis LoRAs. One pre-set number per LoRA, no fitting. Centred exactly per (layer, type):
   <dW_j, C> - sum(w) <dW_j, M>, with M = mean of the OTHER imperial basis LoRAs (leave-self-out, so a LoRA's own
   size does not enter). The shuffled-label sum goes through the same steps.
Usage (BLAS threads = 1): python3 pass2_follow.py FEATURE_DIR WORK_DIR"""
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
SZ = std(lm) * live / np.sqrt(live.sum())
RES = {}
show = lambda tag, P2, P1: print(f"   {tag:58s} " + "   ".join(
    (lambda v: f"{c[:6]} {v['r_b']:+.3f} -> {v['r_a']:+.3f} ({v['diff']:+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}])")(
        gw.compare(P2[:, i], P1[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG[:3])) + f"  [{time.time() - t0:.0f}s]", flush=True)

print("A. The learned-read part on top of the module sizes (r without -> with; imperial TRAIN, out of fold)")
EP = np.asarray(np.load(os.path.join(FEAT, "e_parts.npy"), mmap_mode="r"), np.float64)
tot = EP[:, :, 0] + EP[:, :, 1] + 2 * EP[:, :, 2]
Lsh = std(EP[:, :, 1] / np.maximum(tot, 1e-30)) * live / np.sqrt(live.sum())
KL = np.load(os.path.join(FEAT, "K_L_grp.npy"), mmap_mode="r")
og = [l * ws.NT + 3 for l in range(17, 24)]
dLo = np.zeros((ws.N, 7))
for a in range(0, ws.N, 250):
    dLo[a:a + 250] = np.asarray(KL[a:a + 250], np.float64)[:, bi][:, :, og].mean(1)
ws.release(KL)
dLo = std(dLo) / np.sqrt(7)
Kc = np.load(os.path.join(WORK, "Kc.npy"), mmap_mode="r")
O = np.zeros((ws.N, 14))
for k, l in enumerate(range(17, 24)):
    kg = np.asarray(Kc[l * ws.NT + 3], np.float64)
    O[:, 2 * k], O[:, 2 * k + 1] = kg[:, bi].mean(1), kg[:, bm].mean(1)
A14 = std(O) / np.sqrt(14)
P2, P1 = ws.oof_two_blocks(gw, SZ, Lsh, Y, in_basis); show("sizes -> + energy share of L per module", P2, P1)
P2, P1 = ws.oof_two_blocks(gw, SZ, dLo, Y, in_basis); show("sizes -> + <L, mean L> of o, layers 17-23 (7 numbers)", P2, P1)
P2, P1 = ws.oof_two_blocks(gw, np.column_stack([SZ, A14]), Lsh, Y, in_basis); show("sizes + o alignment (b) -> + energy share of L", P2, P1)
P2, P1 = ws.oof_two_blocks(gw, np.column_stack([SZ, Lsh]), A14, Y, in_basis); show("sizes + energy share of L -> + o alignment (b)", P2, P1)
for t in range(ws.NT):
    X = np.zeros_like(Lsh); X[:, ws.MOD_TYPE == t] = Lsh[:, ws.MOD_TYPE == t]
    P2, P1 = ws.oof_two_blocks(gw, SZ, X[:, ws.MOD_TYPE == t], Y, in_basis); show(f"   sizes -> + L share of {ws.TYPES[t]}", P2, P1)

print("\nB. Dose along the strength direction, 400 imperial basis LoRAs that were not in the sum (no fitting)")
CP = np.asarray(np.load(os.path.join(FEAT, "cproj.npy"), mmap_mode="r"), np.float64)        # [N, 2688, 2]
sw = {0: 43.573368, 1: 5.672894}                                                           # acc_meta.json of pass 1
B_IMP = basis[bi]
K = np.load(os.path.join(FEAT, "K_grp.npy"), mmap_mode="r")
Kb = np.stack([np.asarray(K[j], np.float64) for j in B_IMP])[:, bi]                         # [400, 400, 192] imperial basis block
ws.release(K)
n = len(bi)
self_ = np.einsum("iig->ig", Kb)
dM = (Kb.sum(1) - self_) / (n - 1)                                                          # <dW_j, mean of the other basis LoRAs> per group
cp_g = np.zeros((n, ws.NG, 2))
for g in range(ws.NG):
    cp_g[:, g] = CP[B_IMP][:, ws.MOD_GROUP == g].sum(1)
rng = np.random.default_rng(0)


def rb(x, y):
    r = float(np.corrcoef(x, y)[0, 1]); idx = rng.integers(0, len(x), (1000, len(x)))
    b = [np.corrcoef(x[i], y[i])[0, 1] for i in idx]
    return r, float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), ws.spearman(x, y)


SETS = [("all modules", np.ones(ws.NG, bool))] + [(t, ws.GRP_TYPE == i) for i, t in enumerate(ws.TYPES)] + \
       [(f"layers {a}-{b - 1}", (ws.GRP_LAYER >= a) & (ws.GRP_LAYER < b)) for a, b in ((0, 8), (8, 17), (17, 24), (24, 32))]
size_tot = np.log(g2[B_IMP].sum(1))
for k, name in enumerate(("judged-rate sum", "shuffled-label sum")):
    D = cp_g[:, :, k] - sw[k] * dM                                                          # centred dose per group
    print(f"   {name}: r [95%] / rho with judged_soft, leak, short; then r with judged_soft once the LoRA's total size is regressed out")
    for sname, sel in SETS:
        d = D[:, sel].sum(1)
        out = [rb(d, Y[B_IMP, i]) for i in range(3)]
        res = lambda v: v - np.polyval(np.polyfit(size_tot, v, 1), size_tot)
        pr = float(np.corrcoef(res(d), res(Y[B_IMP, 0]))[0, 1])
        RES[f"{name}|{sname}"] = {"judged": out[0], "leak": out[1], "short": out[2], "judged_partial_size": pr}
        print(f"      {sname:14s} " + "   ".join(f"{o[0]:+.3f} [{o[1]:+.3f}, {o[2]:+.3f}] / {o[3]:+.3f}" for o in out) + f"   | {pr:+.3f}")
json.dump(RES, open(os.path.join(HERE, "results_pass2_follow.json"), "w"), indent=1)
