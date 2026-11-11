#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 9: the mean of the learned-read part L against the mean of the
window part W, from pass 2's projections only (proj2_means.npz of script 8). Weights only; light, runs locally.

For every module, on the top-k singular pairs (u_k, v_k) of the BASIS mean of L, and LoRAs OUTSIDE the basis:
  l_k = mean of u_k' L v_k   a cross-validated singular value of mean L (the directions never saw these LoRAs)
  w_k = mean of u_k' W v_k   the size of mean W along the same pair
Questions:
  1. Are the basis directions of L real?            l_k against the in-sample singular value, and against placebo pairs
  2. How concentrated is mean L?                    sum_k l_k^2 against the shared energy of L (Gram estimate)
  3. Cube law (mean W ~ G, mean L ~ G G' G):        slope of log(w_k / w_1) on log(l_k / l_1); 1/3 if cubed, 1 if same shape
  4. How much of the mean of dW's remainder is W along L's directions: sum_k w_k^2 and 2 sum_k w_k l_k
  5. Did the read window turn towards v_k?          mean |Q v|^2 (trained) against |Q0 v|^2 (start) and chance
  6. Window picture across LoRAs:                   correlation of u' W v and of u' L v with the start-window overlap
  7. Do mean L and mean dW share top directions?    |cos| of u_1 and v_1 (basis means)
Usage:  python l_vs_w.py --npz vm/proj2_means.npz [--gram_L vm/shared_vs_seed_mod_L.npz --gram vm/shared_vs_seed_mod.npz]
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--gram_L", default="")
ap.add_argument("--gram", default="")
ap.add_argument("--ne", type=int, default=40)
ap.add_argument("--z", type=float, default=3.0, help="a value counts as measured when it is this many standard errors above 0")
ap.add_argument("--out", default=None)
args = ap.parse_args()
Z = np.load(args.npz)
S1, S2, n = Z["S1"].astype(np.float64), Z["S2"].astype(np.float64), Z["n"]
SL = Z["SL"].astype(np.float64)
M, ND, K, NQ = S1.shape[2:]
PER = 4 + 2 * args.ne; NL = M // PER
TYPES = ["q", "k", "v", "o", "gu", "dn"]
mt = np.tile(np.array(["q", "k", "v", "o"] + ["gu"] * args.ne + ["dn"] * args.ne), NL)
ml = np.repeat(np.arange(NL), PER)
in_dim = {"q": 1, "k": 1, "v": 1, "o": 1, "gu": 1, "dn": 1}
ng = n.sum(1)
mean = (S1[:, 0] + S1[:, 1]) / ng[:, None, None, None, None]                    # [2, M, ND, K, NQ]
half = S1 / n[:, :, None, None, None, None]
var = S2 / ng[:, None, None, None, None, None] - mean[..., :, None] * mean[..., None, :]   # covariance over LoRAs
se = np.sqrt(np.maximum(np.einsum("gmdkpp->gmdkp", var), 0) / ng[:, None, None, None, None])
res = {}
med = lambda x: float(np.nanmedian(x)) if np.size(x) else float("nan")
by_type = lambda f: "  ".join(f"{t} {f(mt == t)}" for t in TYPES)

# values for the three shared means: imperial dirs on imperial LoRAs, modern on modern, diff = imperial minus modern
val = {"imperial": mean[0][:, 0], "modern": mean[1][:, 1], "diff": mean[0][:, 2] - mean[1][:, 2]}      # [M, K, NQ]
ser = {"imperial": se[0][:, 0], "modern": se[1][:, 1], "diff": np.sqrt(se[0][:, 2] ** 2 + se[1][:, 2] ** 2)}
plc = mean[0][:, 3]                                                             # imperial LoRAs on the random-sign pairs
print(f"LoRAs outside the basis: imperial {int(ng[0])}, modern {int(ng[1])}; {M} modules, top {K} pairs of the basis mean of L\n")

print("1. Are the basis directions of mean L real? l_k (out of sample) / singular value of the basis mean (in sample); median over modules")
for i, name in enumerate(("imperial", "modern", "diff")):
    l = val[name][..., 1]
    r = l / np.where(SL[i] > 0, SL[i], np.nan)
    res[f"shrink_{name}"] = {t: [med(r[mt == t][:, k]) for k in range(K)] for t in TYPES}
    print(f"  {name:8s} " + by_type(lambda m: "/".join(f"{med(r[m][:, k]):.2f}" for k in range(K))))
r = plc[..., 1] / np.where(SL[3] > 0, SL[3], np.nan)
print("  placebo  " + by_type(lambda m: "/".join(f"{med(r[m][:, k]):.2f}" for k in range(K))) + "   (random-sign pairs, imperial LoRAs)")
zl = val["imperial"][..., 1] / np.where(ser["imperial"][..., 1] > 0, ser["imperial"][..., 1], np.nan)
print("  share of modules with l_k above %.0f standard errors (imperial): " % args.z + by_type(lambda m: "/".join(f"{np.nanmean(zl[m][:, k] > args.z):.2f}" for k in range(K))))
hl = half[0][:, :, 0, :, 1]                                                     # [2 halves, M, K]
print("  split halves of the imperial LoRAs, l_1: median |h0 - h1| / mean = " + by_type(lambda m: f"{med(np.abs(hl[0][m][:, 0] - hl[1][m][:, 0]) / np.abs(val['imperial'][m][:, 0, 1])):.3f}"))

if args.gram_L and os.path.exists(args.gram_L):
    GL = np.load(args.gram_L); GD = np.load(args.gram) if args.gram and os.path.exists(args.gram) else None
    print("\n2. How concentrated is mean L? Share of the shared energy of L (Gram estimate) in its top 1 / top 4 pairs (sums over modules; lower bounds, the pairs come from 400 or 100 LoRAs)")
    for name, key in (("imperial", "S_ii"), ("modern", "S_mm")):
        l = val[name][..., 1]; e = GL[key]
        res[f"topshare_L_{name}"] = {t: [float((l[mt == t][:, 0] ** 2).sum() / e[mt == t].sum()), float((l[mt == t] ** 2).sum() / e[mt == t].sum())] for t in TYPES}
        print(f"  {name:8s} " + by_type(lambda m: f"{(l[m][:, 0] ** 2).sum() / e[m].sum():.2f}/{(l[m] ** 2).sum() / e[m].sum():.2f}"))
    print("  by layer block (imperial, all types), top 1 / top 4: " + "  ".join(
        f"L{a}-{b - 1} {(val['imperial'][(ml >= a) & (ml < b)][:, 0, 1] ** 2).sum() / GL['S_ii'][(ml >= a) & (ml < b)].sum():.2f}/{(val['imperial'][(ml >= a) & (ml < b)][..., 1] ** 2).sum() / GL['S_ii'][(ml >= a) & (ml < b)].sum():.2f}"
        for a, b in ((0, NL // 4), (NL // 4, NL // 2), (NL // 2, 5 * NL // 8), (5 * NL // 8, 3 * NL // 4), (3 * NL // 4, 7 * NL // 8), (7 * NL // 8, NL))))
    if GD is not None:
        print("\n4. The mean of dW beyond mean L (|mean W|^2 + 2 <mean W, mean L> = shared dW - shared L), and what the top-4 pairs of L explain of it (imperial; sums over modules)")
        print("  type   shared dW   shared L   remainder | along L's pairs: sum w_k^2   2 sum w_k l_k   both / remainder | ratio w_1 / l_1 (median)")
        for t in TYPES + ["all"]:
            m = (mt == t) if t != "all" else np.ones(M, bool)
            w, l = val["imperial"][m][..., 0], val["imperial"][m][..., 1]
            rem = GD["S_ii"][m].sum() - GL["S_ii"][m].sum()
            a_, b_ = (w ** 2).sum(), 2 * (w * l).sum()
            res[f"remainder_{t}"] = {"shared_dW": float(GD["S_ii"][m].sum()), "shared_L": float(GL["S_ii"][m].sum()), "w2": float(a_), "cross": float(b_)}
            print(f"  {t:4s}   {GD['S_ii'][m].sum():8.4f}   {GL['S_ii'][m].sum():8.4f}   {rem:8.4f}  |   {a_:8.4f}          {b_:8.4f}       {(a_ + b_) / rem:.2f}        |  {med(w[:, 0] / l[:, 0]):.3f}")

print("\n3. Cube law. Per module, slope through the origin of log(w_k / w_1) on log(l_k / l_1), k = 2..K, where all values are measured (1/3 if mean L is the cube of mean W, 1 if the same shape)")
print("  mean      type   median slope   25%-75%       modules used | median l_k / l_1 (k = 2..K) | median w_k / w_1")
for name in ("imperial", "modern", "diff"):
    w, l = val[name][..., 0], val[name][..., 1]; sw, sl = ser[name][..., 0], ser[name][..., 1]
    good = (w > args.z * sw) & (l > args.z * sl)
    for t in TYPES:
        sl_ = []
        for i in np.flatnonzero((mt == t) & good[:, 0]):
            ks = [k for k in range(1, K) if good[i, k]]
            if len(ks) >= 2:
                x = np.log(l[i, ks] / l[i, 0]); y = np.log(w[i, ks] / w[i, 0])
                if np.ptp(x) > 0.1 or abs(x).max() > 0.3:
                    sl_.append(float((x * y).sum() / (x * x).sum()))
        m = (mt == t) & good.all(1)
        res[f"cube_{name}_{t}"] = {"median": med(sl_), "n": len(sl_)}
        q = (np.percentile(sl_, 25), np.percentile(sl_, 75)) if sl_ else (float("nan"), float("nan"))
        print(f"  {name:8s}  {t:3s}     {med(sl_):6.2f}      {q[0]:5.2f} - {q[1]:5.2f}      {len(sl_):5d}     | " +
              "/".join(f"{med(l[m][:, k] / l[m][:, 0]):.2f}" for k in range(1, K)) + "  |  " + "/".join(f"{med(w[m][:, k] / w[m][:, 0]):.2f}" for k in range(1, K)))

print("\n5. Did the read window turn towards the read directions of mean L? Mean overlap |Q v_k|^2, start -> trained, as multiples of chance (rank / in_dim); k = 1 and k = K; imperial")
chance = {t: 8 / (512 if t == "dn" else 1536) for t in TYPES}
print("  mean L pairs:   " + by_type(lambda m: "  ".join(f"k{k + 1} {med(val['imperial'][m][:, k, 2]) / chance[mt[m][0]]:.2f}->{med(val['imperial'][m][:, k, 3]) / chance[mt[m][0]]:.2f}" for k in (0, K - 1))))
print("  placebo pairs:  " + by_type(lambda m: "  ".join(f"k{k + 1} {med(plc[m][:, k, 2]) / chance[mt[m][0]]:.2f}->{med(plc[m][:, k, 3]) / chance[mt[m][0]]:.2f}" for k in (0, K - 1))))
print("  write overlap |Qb' u_k|^2 (of 1), mean L pairs k1 / placebo k1: " + by_type(lambda m: f"{med(val['imperial'][m][:, 0, 4]):.3f}/{med(plc[m][:, 0, 4]):.3f}"))

print("\n6. Across LoRAs (imperial, outside the basis): Pearson correlation per module, median over modules; pair k = 1")
def corr(g, d, k, p, q):
    c = var[g][:, d, k]
    den = np.sqrt(c[:, p, p] * c[:, q, q])
    return c[:, p, q] / np.where(den > 0, den, np.nan)
lab = {(0, 2): "u'Wv ~ start window", (1, 2): "u'Lv ~ start window", (0, 1): "u'Wv ~ u'Lv", (0, 3): "u'Wv ~ trained window", (1, 3): "u'Lv ~ trained window", (1, 4): "u'Lv ~ write overlap", (0, 4): "u'Wv ~ write overlap"}
for (p, q), s in lab.items():
    c0, c3 = corr(0, 0, 0, p, q), corr(0, 3, 0, p, q)
    res[f"corr_{p}{q}"] = {t: med(c0[mt == t]) for t in TYPES}
    print(f"  {s:24s} " + by_type(lambda m: f"{med(c0[m]):+.2f}") + "   | placebo pairs: " + by_type(lambda m: f"{med(c3[m]):+.2f}"))

if "cosU" in Z.files:
    cu, cv = Z["cosU"], Z["cosV"]
    print("\n7. Do the basis mean of L and the basis mean of dW share their top directions? median |cos| (write u_1 / read v_1), and the best match of L's pair 1 among dW's top pairs")
    for i, name in enumerate(("imperial", "modern", "diff")):
        res[f"cos_{name}"] = {t: [med(cu[i][mt == t][:, 0, 0]), med(cv[i][mt == t][:, 0, 0])] for t in TYPES}
        print(f"  {name:8s} " + by_type(lambda m: f"{med(cu[i][m][:, 0, 0]):.2f}/{med(cv[i][m][:, 0, 0]):.2f} (best {med(cu[i][m][:, 0].max(-1)):.2f}/{med(cv[i][m][:, 0].max(-1)):.2f})"))
    print("  pair 2 of L against pair 2 of dW (imperial): " + by_type(lambda m: f"{med(cu[0][m][:, 1, 1]):.2f}/{med(cv[0][m][:, 1, 1]):.2f}") if K > 1 and cu.shape[-1] > 1 else "")
out = args.out or args.npz.replace(".npz", "_analysis.json")
json.dump(res, open(out, "w"), indent=1)
print("\nwrote", out)
