"""Two checks the referee asked for before two sentences go into the write-up.

1. "More of the shared o update in layers 17-23 goes with leak; more of it in layers 24-31 goes against."
   One number per band: the band's shared-update coefficient (sum of <dW, M> over the band's o groups / sum of
   |M|^2), with the band's log size regressed out, correlated with the target. Null: 2,000 label shuffles.
2. "For leak, the relatively smaller modules are all expert modules of layers 24-31."
   After removing a common factor some modules must look relatively smaller. Checks: (a) the count of modules that
   pass, by sign and region, against the same pipeline on shuffled labels; (b) a version that cannot be forced by
   the centring: the common factor is taken from one random half of the modules and the test is run on the other
   half; (c) no centring at all, but a regression of the target on the common factor and the module together
   (does the module have a NEGATIVE partial weight?).
Usage (BLAS threads = 1): python3 cautions.py FEATURE_DIR"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT = sys.argv[1]
rng = np.random.default_rng(0)
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
R = gw.rows("train", "imperial")
c = np.load(os.path.join(HERE, "coef_shared.npy")).astype(np.float64)          # [3000, 192] from dose_raw.py
# |M_g|^2 is needed to pool groups: recover it from c and the inner products is not possible, so pool with equal
# weights on the standardised coefficients instead (a mean of z-scores over the band's groups).
zs = lambda a: (a - a.mean(0)) / np.maximum(a.std(0), 1e-12)
rank = lambda a: np.argsort(np.argsort(a, axis=0), axis=0).astype(float)
TG = ["leak", "judged_soft", "short"]
Y = np.column_stack([gw.y(t) for t in TG])[R]
RES = {}

print("1. Shared-update coefficient of the attention output, by band (size of the band regressed out), against the target")
print("   r, then the 95% range of the same statistic under 2,000 label shuffles, then the one-sided p value")
lg = np.log(np.maximum(g2, 1e-30))
BANDS = {"o layers 0-7": (0, 8), "o layers 8-16": (8, 17), "o layers 17-23": (17, 24), "o layers 24-31": (24, 32)}
perm = np.array([rng.permutation(len(R)) for _ in range(2000)])
for name, (a, b) in BANDS.items():
    gs = [l * ws.NT + 3 for l in range(a, b)]
    x = zs(c[R][:, gs]).mean(1); s = zs(lg[R][:, gs]).mean(1)
    x = x - np.polyval(np.polyfit(s, x, 1), s)                       # coefficient with the band's size taken out
    row = []
    for i, t in enumerate(TG):
        y = Y[:, i] - np.polyval(np.polyfit(s, Y[:, i], 1), s)
        r = float(np.corrcoef(x, y)[0, 1])
        nul = np.array([np.corrcoef(x, y[p])[0, 1] for p in perm])
        p = float((1 + (np.abs(nul) >= abs(r)).sum()) / (1 + len(nul)))
        row.append(f"{t} {r:+.3f} [{np.quantile(nul, 0.025):+.3f}, {np.quantile(nul, 0.975):+.3f}] p {p:.4f}")
        RES[f"{name}|{t}"] = {"r": r, "null95": [float(np.quantile(nul, 0.025)), float(np.quantile(nul, 0.975))], "p_two_sided": p}
    print(f"   {name:16s} " + "   ".join(row))
print("   per layer, leak (r; a single layer's null 95% range is about +-0.050):")
per = []
for l in range(ws.L):
    g = l * ws.NT + 3
    x = c[R, g] - np.polyval(np.polyfit(lg[R, g], c[R, g], 1), lg[R, g])
    y = Y[:, 0] - np.polyval(np.polyfit(lg[R, g], Y[:, 0], 1), lg[R, g])
    per.append(float(np.corrcoef(x, y)[0, 1]))
nul_max = np.array([max(abs(np.corrcoef(c[R, l * ws.NT + 3], Y[p, 0])[0, 1]) for l in range(ws.L)) for p in perm[:300]])
print("      " + " ".join(f"{v:+.2f}" for v in per) + f"\n      family-wise 95% threshold over the 32 layers: |r| > {np.quantile(nul_max, 0.95):.3f}; "
      f"layers beyond it: " + ", ".join(f"{l} ({per[l]:+.2f})" for l in range(ws.L) if abs(per[l]) > np.quantile(nul_max, 0.95)))
RES["o_per_layer_leak"] = per

print("\n2. Relatively smaller late-layer expert modules for leak: real against shuffled labels, and without forced signs")
live = (m2[~gw.MODERN] > 1e-20).all(0)
lm = np.log(np.maximum(m2, 1e-30))
Z = zs(lm[R]) * live
late_exp = (ws.MOD_LAYER >= 24) & (ws.MOD_TYPE >= 4) & live
mid_att = (ws.MOD_LAYER >= 15) & (ws.MOD_LAYER <= 21) & (ws.MOD_TYPE < 4) & live
yl = rank(Y[:, 0:1])[:, 0]
common = Z[:, live].mean(1)
Zr = (Z - common[:, None]) * live
Xr = zs(rank(Zr))
rho = Xr.T @ zs(yl) / len(R)
nul = np.array([Xr.T @ zs(yl[p]) / len(R) for p in perm[:400]])            # [400, 2688]
thr = np.quantile(np.abs(nul).max(1), 0.95)
cnt = lambda v, m: (int(((v > thr) & m).sum()), int(((v < -thr) & m).sum()))
real_late, real_mid = cnt(rho, late_exp), cnt(rho, mid_att)
null_late = np.array([cnt(v, late_exp) for v in nul]); null_mid = np.array([cnt(v, mid_att) for v in nul])
print(f"   (a) family-wise threshold |rho| > {thr:.3f}. Late-layer expert modules ({int(late_exp.sum())}): {real_late[1]} relatively smaller, {real_late[0]} relatively bigger; "
      f"shuffled labels: smaller mean {null_late[:, 1].mean():.2f}, max {null_late[:, 1].max()}; bigger mean {null_late[:, 0].mean():.2f}")
print(f"       mean rho over the late-layer expert modules {rho[late_exp].mean():+.4f}; shuffled labels: mean {nul[:, late_exp].mean():+.4f}, SD of that mean {nul[:, late_exp].mean(1).std():.4f} "
      f"(z {(rho[late_exp].mean() - nul[:, late_exp].mean()) / nul[:, late_exp].mean(1).std():+.1f})")
print(f"       mid-layer attention modules, layers 15-21 ({int(mid_att.sum())}): mean rho {rho[mid_att].mean():+.4f}; shuffled SD of the mean {nul[:, mid_att].mean(1).std():.4f} "
      f"(z {(rho[mid_att].mean() - nul[:, mid_att].mean()) / nul[:, mid_att].mean(1).std():+.1f})")
RES["late_experts"] = {"n": int(late_exp.sum()), "pass_smaller": real_late[1], "pass_bigger": real_late[0], "null_smaller_mean": float(null_late[:, 1].mean()),
                       "mean_rho": float(rho[late_exp].mean()), "null_sd_of_mean": float(nul[:, late_exp].mean(1).std())}
# (b) the common factor from one random half of the modules, the test on the other half (10 random splits)
zb = []
for k in range(10):
    half = rng.random(ws.NM) < 0.5
    cf = Z[:, live & half].mean(1)
    test = late_exp & ~half
    W = (Z - cf[:, None])[:, test]
    v = zs(rank(W)).T @ zs(yl) / len(R)
    nv = np.array([zs(rank(W)).T @ zs(yl[p]) / len(R) for p in perm[:100]])
    zb.append((float(v.mean()), float((v.mean() - nv.mean()) / nv.mean(1).std())))
print(f"   (b) common factor from one random half of the modules, late-layer experts tested in the other half, 10 splits: mean rho "
      f"{np.mean([a for a, _ in zb]):+.4f} (range {min(a for a, _ in zb):+.4f} to {max(a for a, _ in zb):+.4f}), z against shuffled labels {np.mean([b for _, b in zb]):+.1f}")
RES["late_experts_split_half"] = zb
# (c) no centring: regress the target rank on the common factor and the region's own mean size; sign of the partial weight
for name, m in (("late-layer experts (24-31)", late_exp), ("mid-layer attention (15-21)", mid_att),
                ("early / mid experts (0-23)", (ws.MOD_LAYER < 24) & (ws.MOD_TYPE >= 4) & live)):
    reg = Z[:, m].mean(1)
    A = np.column_stack([np.ones(len(R)), zs(common), zs(reg)])
    beta = np.linalg.lstsq(A, zs(yl), rcond=None)[0]
    nb = np.array([np.linalg.lstsq(A, zs(yl[p]), rcond=None)[0][2] for p in perm[:1000]])
    print(f"   (c) {name:28s} raw r of the region's mean size with leak {np.corrcoef(reg, yl)[0, 1]:+.3f}; partial weight next to the common factor "
          f"{beta[2]:+.3f} (shuffled labels: SD {nb.std():.3f}, z {beta[2] / nb.std():+.1f}); r between region size and common factor {np.corrcoef(reg, common)[0, 1]:+.3f}")
    RES[f"partial|{name}"] = {"raw_r": float(np.corrcoef(reg, yl)[0, 1]), "partial_weight": float(beta[2]), "null_sd": float(nb.std())}
json.dump(RES, open(os.path.join(HERE, "results_cautions.json"), "w"), indent=1)
