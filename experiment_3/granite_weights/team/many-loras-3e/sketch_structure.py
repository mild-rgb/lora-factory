#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 7: what pass 2's sketch numbers say. Weights only; light enough
for the local box (reads the arrays in blocks through memory maps).

Pass 2 gives, per LoRA and module, with G = the dense basis mean of dW of the LoRA's own group and A0 the LoRA's own
start matrix (sketch.npy): <B, G A0'>, |G A0'|^2, |B|^2, <dW, G A0' A0>, |G A0' A0|^2, and (e_parts.npy)
|W|^2, |L|^2, <W, L>. From these, per module:
  fit of B     cos(B, G A0'): how much of the write matrix is "the common update applied to this LoRA's own probes"
  gain         <B, G A0'> / |G A0'|^2: how much of the common update the LoRA took once its window is divided out
  common and private energy of dW: |dW|^2 = c^2 |G A0' A0|^2 + remainder, with c = <dW, G A0' A0> / |G A0' A0|^2
Questions answered (d7's): is the gain one number per LoRA shared across layers? Is the LoRA-wide factor the common
part or the private part, and which of the two goes with total size? All statistics use LoRAs outside the basis (for
them G is out of sample) and, where behaviour is mentioned, chunks 0-59 only (descriptive; prediction is bb's track).
Usage:  python sketch_structure.py --ws FOLDER_WITH_sketch.npy [--targets team/many-loras-39/targets.npz] [--ne 40]
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ws", required=True)
ap.add_argument("--meta", default=None, help="folder with names.json and basis_idx.npy (default: --ws)")
ap.add_argument("--targets", default="")
ap.add_argument("--ne", type=int, default=40)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--hold_from", type=int, default=60)
ap.add_argument("--out", default=None)
args = ap.parse_args()
META = args.meta or args.ws
names = json.load(open(f"{META}/names.json")); basis = np.load(f"{META}/basis_idx.npy")
SK = np.load(f"{args.ws}/sketch.npy", mmap_mode="r"); EP = np.load(f"{args.ws}/e_parts.npy", mmap_mode="r")
N, M = SK.shape[:2]
names = names[:N]
PER = 4 + 2 * args.ne; NL = M // PER
TYPES = ["q", "k", "v", "o", "gu", "dn"]
mod_type = np.tile(np.array(["q", "k", "v", "o"] + ["gu"] * args.ne + ["dn"] * args.ne), NL)
mod_layer = np.repeat(np.arange(NL), PER)
mod_grp = mod_layer * 6 + np.array([TYPES.index(t) for t in mod_type])
is_imp = np.array([n.startswith("imperial") for n in names])
nb = np.ones(N, bool); nb[basis[basis < N]] = False
train = np.arange(N) // args.chunk < args.hold_from
G = NL * 6
num = np.zeros((N, G)); den = np.zeros((N, G)); com = np.zeros((N, G)); tot = np.zeros((N, G))
cosB_sum = np.zeros(M); cosB_n = np.zeros(M); cosD_sum = np.zeros(M); priv_sum = np.zeros(M); e_sum = np.zeros(M)
rows_imp = np.flatnonzero(is_imp & nb)
onehot = np.zeros((M, G)); onehot[np.arange(M), mod_grp] = 1
for s in range(0, N, 100):
    sk = np.asarray(SK[s:s + 100], dtype=np.float64); ep = np.asarray(EP[s:s + 100], dtype=np.float64)
    s1, s2, s3, s4, s5 = (sk[..., i] for i in range(5))
    e = ep[..., 0] + ep[..., 1] + 2 * ep[..., 2]                       # |dW|^2
    common = np.where(s5 > 0, s4 ** 2 / np.where(s5 > 0, s5, 1), 0.0)  # energy of dW along G A0' A0
    num[s:s + 100] = s1 @ onehot; den[s:s + 100] = s2 @ onehot
    com[s:s + 100] = common @ onehot; tot[s:s + 100] = e @ onehot
    sel = (is_imp & nb)[s:s + 100]
    ok = (s3 > 0) & (s2 > 0)
    cb = np.where(ok, s1 / np.sqrt(np.where(ok, s2 * s3, 1)), np.nan)
    cd = np.where((e > 0) & (s5 > 0), s4 / np.sqrt(np.where((e > 0) & (s5 > 0), e * s5, 1)), np.nan)
    cosB_sum += np.nansum(cb[sel], 0); cosB_n += np.isfinite(cb[sel]).sum(0); cosD_sum += np.nansum(cd[sel], 0)
    priv_sum += (e - common)[sel].sum(0); e_sum += e[sel].sum(0)
cosB, cosD = cosB_sum / np.maximum(cosB_n, 1), cosD_sum / np.maximum(cosB_n, 1)
print(f"{N} LoRAs, {M} modules; statistics on the {len(rows_imp)} imperial LoRAs outside the basis")
print("\nFit of one module of one LoRA by the common update seen through its own start window (mean over LoRAs, median over modules):")
print("type   cos(B, G A0')   cos(dW, G A0' A0)   private share of |dW|^2 (energy not along G A0' A0)")
for t in TYPES:
    m = mod_type == t
    print(f"{t:4s}     {np.nanmedian(cosB[m]):.3f}           {np.nanmedian(cosD[m]):.3f}              {priv_sum[m].sum() / e_sum[m].sum():.3f}")
print("by layer block (all types): private share of |dW|^2:", {f"{a}-{b - 1}": round(float(priv_sum[(mod_layer >= a) & (mod_layer < b)].sum() / e_sum[(mod_layer >= a) & (mod_layer < b)].sum()), 3)
                                                        for a, b in ((0, NL // 4), (NL // 4, NL // 2), (NL // 2, 3 * NL // 4), (3 * NL // 4, NL))})

# ---- the gain per (layer, type) group and its structure across groups
rows = np.flatnonzero(is_imp & nb & train)
gain = num[rows] / np.where(den[rows] > 0, den[rows], np.nan)
gain = gain / np.nanmean(gain, 0)                                    # relative to the mean LoRA, like the dose
keep = np.isfinite(gain).all(0) & (np.nanstd(gain, 0) > 0)
Z = (gain[:, keep] - gain[:, keep].mean(0)) / gain[:, keep].std(0)
C = Z.T @ Z / len(rows)
w, V = np.linalg.eigh(C); w, V = w[::-1], V[:, ::-1]
off = C[np.triu_indices(C.shape[0], 1)]
print(f"\nGain (how much of the common update a LoRA took, window divided out) per (layer, type) group, {len(rows)} train imperial LoRAs outside the basis:")
print(f"  SD of the relative gain: median over groups {np.median(gain[:, keep].std(0)):.3f}")
print(f"  correlation between groups: mean {off.mean():.3f}, median {np.median(off):.3f}; eigenvalue shares " + " ".join(f"{x / keep.sum():.3f}" for x in w[:6]))
pc1 = Z @ V[:, 0] * np.sign(V[:, 0].sum())
E_tot = tot[rows].sum(1); E_com = com[rows].sum(1); E_priv = E_tot - E_com
ggain = num[rows].sum(1) / den[rows].sum(1)
sp = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
cv = lambda x: float(x.std() / x.mean())
print(f"  whole-LoRA gain: mean {ggain.mean():.4g}, CV {cv(ggain):.3f}")
print(f"\nEnergy of one LoRA: total {E_tot.mean():.4g} (CV {cv(E_tot):.3f}) = common through the window {E_com.mean():.4g} (CV {cv(E_com):.3f}) "
      f"+ private {E_priv.mean():.4g} (CV {cv(E_priv):.3f}); private share {E_priv.mean() / E_tot.mean():.3f}")
print(f"  Spearman over LoRAs: total ~ private {sp(E_tot, E_priv):.3f}; total ~ common {sp(E_tot, E_com):.3f}; private ~ common {sp(E_priv, E_com):.3f}; "
      f"gain PC1 ~ total {sp(pc1, E_tot):.3f}; gain PC1 ~ private {sp(pc1, E_priv):.3f}; gain PC1 ~ common {sp(pc1, E_com):.3f}; whole-LoRA gain ~ total {sp(ggain, E_tot):.3f}")
res = {"cosB_by_type": {t: float(np.nanmedian(cosB[mod_type == t])) for t in TYPES}, "cosD_by_type": {t: float(np.nanmedian(cosD[mod_type == t])) for t in TYPES},
       "private_share_by_type": {t: float(priv_sum[mod_type == t].sum() / e_sum[mod_type == t].sum()) for t in TYPES},
       "gain_eig_share": (w[:10] / keep.sum()).tolist(), "gain_mean_corr": float(off.mean()), "E_tot_cv": cv(E_tot), "E_com_cv": cv(E_com), "E_priv_cv": cv(E_priv),
       "private_share": float(E_priv.mean() / E_tot.mean())}
if args.targets and os.path.exists(args.targets):
    T = np.load(args.targets, allow_pickle=True)
    print("\nBehaviour, descriptive only (same LoRAs; Spearman): target | total energy | private energy | common energy | whole-LoRA gain | gain PC1")
    res["targets"] = {}
    for k in ("judged_soft", "leak", "short", "bare", "maxed", "marker", "judged_clean", "final_loss"):
        if k in T.files and T[k].shape[0] >= N:
            y = np.asarray(T[k][:N], dtype=float)[rows]; ok = np.isfinite(y)
            r = [sp(x[ok], y[ok]) for x in (E_tot, E_priv, E_com, ggain, pc1)]
            res["targets"][k] = r
            print(f"  {k:13s} | {r[0]:+.3f} | {r[1]:+.3f} | {r[2]:+.3f} | {r[3]:+.3f} | {r[4]:+.3f}")
out = args.out or os.path.join(args.ws, "sketch_structure.json")
json.dump(res, open(out, "w"), indent=1)
np.savez(out.replace(".json", ".npz"), rows=rows, gain_pc1=pc1, E_tot=E_tot, E_com=E_com, E_priv=E_priv, ggain=ggain, cosB=cosB, cosD=cosD)
print("wrote", out)
