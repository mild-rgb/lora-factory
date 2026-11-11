#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 5: do the three means tell the same story? Weights only.

After pass 2 every update is split as dW = W + L (W = 4 B A0 through the start window, L = 4 B (A - A0) learned
read). If B_j is roughly G A0_j' for one common matrix G and the read side moves by a gradient step, then
    mean W  is proportional to  G          and          mean L  is proportional to  G G' G,
so the two means should share their singular vectors, and the singular values of mean L should fall like the CUBE
of those of mean W. This script checks both, per module, from script 1's outputs for the tags W, L and the plain
mean (top_dirs_<tag>.pt, mean_structure_<tag>.npz in --dir3e):
  agreement   |cos| of the top write direction u_1 and of the top read direction v_1 between each pair of means, and
              the overlap of the top-4 subspaces (0..1), only for modules where both top values stand above the
              noise edge of script 1; next to the same numbers for the noise (placebo) directions
  cube law    per module, the slope of log(s_k(L) / s_1(L)) against log(s_k(W) / s_1(W)) over the matched
              directions k >= 2 that are above the noise edge in both (matched by the largest |cos| of u)
Usage:  python compare_parts.py [--dir3e /content/ws/3e]        (CPU or GPU, about a minute)
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir3e", default="/content/ws/3e")
ap.add_argument("--edge_margin", type=float, default=1.1)
ap.add_argument("--names", default="imperial,modern,diff")
args = ap.parse_args()
TYPES = ["q", "k", "v", "o", "gu", "dn"]
TAGS = {"dW": "", "W": "_W", "L": "_L"}
D = {k: torch.load(f"{args.dir3e}/top_dirs{s}.pt", map_location="cpu") for k, s in TAGS.items() if os.path.exists(f"{args.dir3e}/top_dirs{s}.pt")}
ST = {k: np.load(f"{args.dir3e}/mean_structure{s}.npz") for k, s in TAGS.items() if k in D}
assert len(D) >= 2, "need script 1's outputs for at least two of the tags '', W, L"
NL, NE = D[next(iter(D))]["gu"]["imperial"][0].shape[:2]
PER = 4 + 2 * NE
idx = {t: (np.arange(NL)[:, None] * PER + (np.array([TYPES.index(t)]) if t in ("q", "k", "v", "o") else 4 + np.arange(NE) + (NE if t == "dn" else 0))[None]) for t in TYPES}
res = {}
print("Agreement of the top directions between the means (medians over modules where both are above the noise edge; n = how many).")
print("mean      pair    type   |cos u1|  |cos v1|  top-4 write overlap  top-4 read overlap   n   | placebo |cos u1| |cos v1|")
pairs = [(a, b) for a, b in (("W", "L"), ("dW", "L"), ("dW", "W")) if a in D and b in D]
for name in args.names.split(","):
    for a, b in pairs:
        for t in TYPES:
            Ua, Sa, Va = D[a][t][name]; Ub, Sb, Vb = D[b][t][name]
            ea = ST[a][f"snoise_{name}"][idx[t]][..., 0]; eb = ST[b][f"snoise_{name}"][idx[t]][..., 0]
            ok = (Sa[..., 0].numpy() > args.edge_margin * ea) & (Sb[..., 0].numpy() > args.edge_margin * eb)
            cu = (Ua[..., 0] * Ub[..., 0]).sum(-1).abs().numpy(); cv = (Va[..., 0] * Vb[..., 0]).sum(-1).abs().numpy()
            k4 = min(4, Ua.shape[-1])
            ow = (Ua[..., :k4].transpose(-1, -2) @ Ub[..., :k4]).pow(2).sum((-1, -2)).numpy() / k4
            orr = (Va[..., :k4].transpose(-1, -2) @ Vb[..., :k4]).pow(2).sum((-1, -2)).numpy() / k4
            Pa, Pb = D[a][t]["noise_" + name], D[b][t]["noise_" + name]
            pu = (Pa[0][..., 0] * Pb[0][..., 0]).sum(-1).abs().numpy(); pv = (Pa[2][..., 0] * Pb[2][..., 0]).sum(-1).abs().numpy()
            med = lambda x, m=ok: float(np.median(x[m])) if m.any() else float("nan")
            res[f"{name}|{a}-{b}|{t}"] = {"cos_u1": med(cu), "cos_v1": med(cv), "write4": med(ow), "read4": med(orr), "n": int(ok.sum()),
                                           "placebo_u1": float(np.median(pu)), "placebo_v1": float(np.median(pv))}
            r = res[f"{name}|{a}-{b}|{t}"]
            print(f"{name:9s} {a:>2s}-{b:<2s}  {t:3s}    {r['cos_u1']:.3f}     {r['cos_v1']:.3f}        {r['write4']:.3f}               {r['read4']:.3f}       {r['n']:5d} |   "
                  f"{r['placebo_u1']:.3f}   {r['placebo_v1']:.3f}")
if "W" in D and "L" in D:
    print("\nCube law: slope of log(s_k(L) / s_1(L)) on log(s_k(W) / s_1(W)) over matched directions above the noise edge (prediction 3; 1 = same shape).")
    print("mean      type   median slope   25%-75%        modules with at least 2 usable directions")
    for name in args.names.split(","):
        for t in TYPES:
            Uw, Sw, _ = D["W"][t][name]; Ul, Sl, _ = D["L"][t][name]
            ew = ST["W"][f"snoise_{name}"][idx[t]][..., 0]; el = ST["L"][f"snoise_{name}"][idx[t]][..., 0]
            C = (Uw.transpose(-1, -2) @ Ul).abs()                                  # [NL, E, k (W), k (L)]
            match = C.argmax(-1).numpy(); cmax = C.amax(-1).numpy()
            Sw_, Sl_ = Sw.numpy(), Sl.numpy()
            slopes = []
            for L in range(NL):
                for e in range(Uw.shape[1]):
                    ks = [k for k in range(Sw_.shape[-1]) if Sw_[L, e, k] > args.edge_margin * ew[L, e] and
                          Sl_[L, e, match[L, e, k]] > args.edge_margin * el[L, e] and cmax[L, e, k] > 0.5]
                    if len(ks) >= 3 and ks[0] == 0:
                        x = np.log(Sw_[L, e, ks] / Sw_[L, e, 0]); y = np.log(Sl_[L, e, match[L, e, ks]] / Sl_[L, e, match[L, e, 0]])
                        if np.ptp(x) > 0.1:
                            slopes.append(float((x * y).sum() / (x * x).sum()))    # line through the origin (k = 1 is the anchor)
            res[f"cube|{name}|{t}"] = {"median": float(np.median(slopes)) if slopes else float("nan"), "n": len(slopes),
                                       "q25": float(np.percentile(slopes, 25)) if slopes else float("nan"), "q75": float(np.percentile(slopes, 75)) if slopes else float("nan")}
            r = res[f"cube|{name}|{t}"]
            print(f"{name:9s} {t:3s}     {r['median']:6.2f}        {r['q25']:.2f} - {r['q75']:.2f}      {r['n']}")
json.dump(res, open(f"{args.dir3e}/compare_parts.json", "w"), indent=1)
print(f"wrote {args.dir3e}/compare_parts.json")
