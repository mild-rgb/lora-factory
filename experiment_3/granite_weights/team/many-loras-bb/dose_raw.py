"""Track "strong against weak": does every LoRA carry the same amount of the shared update?

For each (layer, type) group g the coefficient c[j, g] = <dW_j, M_g> / |M_g|^2 says how many "shared updates" LoRA j
carries there (1 on average). M_g = mean update of the imperial basis LoRAs; <dW_j, M_g> = mean over those basis LoRAs
of K_grp[j, b, g] (a scored LoRA is never in the basis, so there is no self term); |M_g|^2 = mean off-diagonal inner
product among the basis LoRAs (unbiased). Unlike a cosine, c does not shrink when the LoRA is bigger.
Questions: how much does c vary between LoRAs, does it go with the targets, and does it add to the sizes?
Weights only; protocol of team/gw_eval.py. Usage (BLAS threads = 1): python3 dose_raw.py FEATURE_DIR [--perm 20]"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT = sys.argv[1]
NP = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else 20
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
bi = np.flatnonzero(~gw.MODERN[basis]); bm = np.flatnonzero(gw.MODERN[basis])
K = np.load(os.path.join(FEAT, "K_grp.npy"), mmap_mode="r")
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
num = np.zeros((ws.N, ws.NG)); num_m = np.zeros((ws.N, ws.NG))
for a in range(0, ws.N, 250):
    blk = np.asarray(K[a:a + 250], np.float64)                      # [250, 500, 192]
    num[a:a + 250] = blk[:, bi].mean(1); num_m[a:a + 250] = blk[:, bm].mean(1)
Kbb = np.stack([np.asarray(K[basis[i]], np.float64) for i in bi])[:, bi]          # [400, 400, 192] imperial basis block
ws.release(K)
n = len(bi)
M2 = (Kbb.sum((0, 1)) - np.einsum("iig->g", Kbb)) / (n * (n - 1))   # |M_g|^2, unbiased
c = num / np.maximum(M2, 1e-30)                                     # [3000, 192] coefficient of the shared imperial update
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"]
NTG = len(TG)
Y = ws.make_Y(gw, TG, "imperial", NP)
SC = gw.rows("train", "imperial"); TRI = gw.rows("train", "imperial", scored=False)
TRI = TRI[~in_basis[TRI]]
std = lambda X: (X - X[TRI].mean(0)) / np.maximum(X[TRI].std(0), 1e-12)
RES = {"targets": TG}

print("1. How much shared update does a LoRA carry? coefficient c per (layer, type), over the scored imperial TRAIN LoRAs")
print("   (shared energy |M|^2 as a share of a LoRA's energy in the group; mean and SD of c between LoRAs)")
share = M2 / g2[SC].mean(0)
for t in range(ws.NT):
    sel = ws.GRP_TYPE == t
    print(f"   {ws.TYPES[t]:8s} shared share of energy {100 * share[sel].mean():5.2f}%   c mean {c[SC][:, sel].mean():.3f}   SD of c between LoRAs "
          f"{c[SC][:, sel].std(0).mean():.3f} (per group), of the type's sum {(num[SC][:, sel].sum(1) / M2[sel].sum()).std():.3f}")
ctot = num[SC].sum(1) / M2.sum()
print(f"   all groups: c of the whole LoRA mean {ctot.mean():.3f}, SD {ctot.std():.3f}; the sampling noise of c from the unshared part alone would give SD about "
      f"{np.sqrt((g2[SC].mean(0) * M2).sum() / n) / M2.sum():.3f}")
RES["c_sd_per_type"] = {ws.TYPES[t]: float(c[SC][:, ws.GRP_TYPE == t].std(0).mean()) for t in range(ws.NT)}

print("\n2. Does the coefficient go with the targets? (in-sample r / rho of the type-level coefficient; then with the type's log size regressed out)")
lg = np.log(np.maximum(g2, 1e-30))
rank = lambda a: np.argsort(np.argsort(a)).astype(float)
for t in range(ws.NT):
    sel = ws.GRP_TYPE == t
    ct = num[SC][:, sel].sum(1) / M2[sel].sum()
    sz = np.log(g2[SC][:, sel].sum(1))
    res = lambda v: v - np.polyval(np.polyfit(sz, v, 1), sz)
    print(f"   {ws.TYPES[t]:8s} " + "  ".join(
        f"{cc[:11]} {np.corrcoef(ct, Y[SC, i])[0, 1]:+.2f}/{np.corrcoef(rank(ct), rank(Y[SC, i]))[0, 1]:+.2f} ({np.corrcoef(res(ct), res(Y[SC, i]))[0, 1]:+.2f})"
        for i, cc in enumerate(TG)))
print("   by layer, attention output o only (r with leak / with judged_soft, size of that group regressed out):")
print("      " + " ".join(f"{l}:" + "/".join(
    f"{np.corrcoef(c[SC, l * ws.NT + 3] - np.polyval(np.polyfit(lg[SC, l * ws.NT + 3], c[SC, l * ws.NT + 3], 1), lg[SC, l * ws.NT + 3]), Y[SC, i])[0, 1]:+.2f}"
    for i in (1, 0)) for l in range(ws.L)))


def fit(X):
    P = ws.oof_multi(gw, X, Y, in_basis)
    r = ws.col_r(P, Y, SC)
    return (P,) + ws.split_null(r, NTG)


print("\n3. Out-of-fold prediction (r): the 192 coefficients, the 192 sizes, and both")
Pc, rc, nl = fit(std(c)); Ps, rs_, _ = fit(std(lg))
Pcs, rcs, _ = fit([np.column_stack([std(lg), np.sqrt(gm) * std(c)]) for gm in (0.1, 0.3, 1.0, 3.0)] + [std(lg), std(c)])
o_sel = np.flatnonzero(ws.GRP_TYPE == 3)
Po, ro, _ = fit(std(c[:, o_sel]))
for tag, r in (("192 coefficients", rc), ("32 coefficients of o", ro), ("192 sizes", rs_), ("sizes + coefficients", rcs)):
    print(f"   {tag:24s} " + "  ".join(f"{cc[:11]} {r[i]:+.3f}" for i, cc in enumerate(TG)) + (f"   (null sd {nl.std(0).mean():.3f})" if tag.startswith("192 c") else ""))
print("   gain of the coefficients over the 192 sizes: " + "; ".join(
    (lambda d: f"{cc[:11]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:12]}")(
        gw.compare(Pcs[:, i], Ps[:, i], cc, quiet=True, n_boot=1000)) for i, cc in enumerate(TG) if cc != "final_loss"))
RES.update({"r_coef_192": rc.tolist(), "r_coef_o": ro.tolist(), "r_sizes_192": rs_.tolist(), "r_both": rcs.tolist()})
np.save(os.path.join(HERE, "coef_shared.npy"), c.astype(np.float32))
json.dump(RES, open(os.path.join(HERE, "results_dose_raw.json"), "w"), indent=1)
