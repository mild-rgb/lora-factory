#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 8: group means of pass 2's projections. Weights only, CPU only.

The dense means of L and W could not be saved in pass 2. Most of what they were for can be had from proj2.npy:
for every LoRA and module it holds, on the top-k singular pairs (u, v) of the BASIS means of L,
    u' W v,  u' L v,  |Q0 v|^2 (start window),  |Q v|^2 (trained window),  |Qb' u|^2 (write overlap).
The mean of u' L v over LoRAs outside the basis is a cross-validated singular value of mean L, and the mean of
u' W v is the size of mean W along the same pair. This script only reduces proj2.npy to small sums (the analysis is
l_vs_w.py, run locally on the output):
  S1   [2 groups (imperial, modern; LoRAs outside the basis), 2 halves (row parity), 2688, 4 means, k, 5]  sums
  S2   [2 groups, 2688, 4 means, k, 5, 5]   sums of products (for the spread over LoRAs and the correlations)
  n    [2, 2]
  SL   [4 means, 2688, k]   singular values of the basis means of L (dirs_L.pt)
  S1d  [3, 2688, k1]        singular values of pass 1's basis means of dW (dirs.pt): imperial, modern, diff
  cosU, cosV [3, 2688, k, k1]   |cos| between the directions of mean L and of mean dW (write side, read side)
CHECK lines: for the basis LoRAs the mean of u' L v must give back the singular values in dirs_L.pt.
Usage:  python proj2_means.py --ws /content/ws [--pass1 /content/ws] [--out /content/ws/3e/proj2_means.npz]
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ws", default="/content/ws", help="folder with proj2.npy, dirs_L.pt, names.json, basis_idx.npy")
ap.add_argument("--pass1", default=None, help="folder with pass 1's dirs.pt (default: --ws)")
ap.add_argument("--out", default=None)
ap.add_argument("--block", type=int, default=20)
args = ap.parse_args()
P1 = args.pass1 or args.ws
out = args.out or os.path.join(args.ws, "proj2_means.npz")
TYPES = ["q", "k", "v", "o", "gu", "dn"]
MEANS_L = ["imperial", "modern", "diff", "random_sign"]
SHARED = ["imperial", "modern", "diff"]

names = json.load(open(f"{args.ws}/names.json")); basis = np.load(f"{args.ws}/basis_idx.npy")
PJ = np.load(f"{args.ws}/proj2.npy", mmap_mode="r")
N, M, ND, K, NQ = PJ.shape
names = names[:N]
is_imp = np.array([n.startswith("imperial") for n in names])
in_basis = np.zeros(N, bool); in_basis[basis[basis < N]] = True
DL = torch.load(f"{args.ws}/dirs_L.pt", map_location="cpu")
NL, NE = DL["gu"]["imperial"][0].shape[:2]
PER = 4 + 2 * NE
assert M == NL * PER and ND == len(MEANS_L), (PJ.shape, NL, NE)


def flat(per_type):
    """dict type -> [NL, E, ...] to [NL * PER, ...] in pass 1's module order (q, k, v, o, gate_up x E, down x E)."""
    tail = per_type["q"].shape[2:]
    o = np.zeros((NL, PER) + tuple(tail), dtype=np.float64)
    for i, t in enumerate(("q", "k", "v", "o")):
        o[:, i] = per_type[t][:, 0].double().numpy()
    o[:, 4:4 + NE] = per_type["gu"].double().numpy(); o[:, 4 + NE:] = per_type["dn"].double().numpy()
    return o.reshape((NL * PER,) + tuple(tail))


SL = np.stack([flat({t: DL[t][n][1] for t in TYPES}) for n in MEANS_L])            # [4, M, K]
S1 = np.zeros((2, 2, M, ND, K, NQ)); S2 = np.zeros((2, M, ND, K, NQ, NQ)); cnt = np.zeros((2, 2))
B1 = np.zeros((2, M, ND, K, NQ)); bcnt = np.zeros(2)                               # basis rows, for the check
for s in range(0, N, args.block):
    x = np.asarray(PJ[s:s + args.block], dtype=np.float64)
    rows = np.arange(s, min(s + args.block, N))
    for g, sel_g in enumerate((is_imp[rows], ~is_imp[rows])):
        b = sel_g & in_basis[rows]
        B1[g] += x[b].sum(0); bcnt[g] += b.sum()
        nb = sel_g & ~in_basis[rows]
        if nb.any():
            S2[g] += np.einsum("nmdkp,nmdkq->mdkpq", x[nb], x[nb])
            for h in (0, 1):
                hh = nb & (rows % 2 == h)
                S1[g, h] += x[hh].sum(0); cnt[g, h] += hh.sum()
    if (s // args.block) % 25 == 0:
        print(f"rows {min(s + args.block, N)}/{N}", flush=True)
print("LoRAs outside the basis per group and half:", cnt.tolist(), "| basis:", bcnt.tolist())

# CHECK: over the basis LoRAs, the mean of u' L v on a mean's own directions is that mean's singular value
mb = [B1[g] / max(bcnt[g], 1) for g in (0, 1)]
want = {"imperial": mb[0][:, 0, :, 1], "modern": mb[1][:, 1, :, 1], "diff": mb[0][:, 2, :, 1] - mb[1][:, 2, :, 1]}
for i, n in enumerate(SHARED):
    live = SL[i][:, 0] > 1e-8
    err = np.abs(want[n][live] - SL[i][live]).max() / SL[i][live].max()
    print(f"CHECK basis mean of u' L v against dirs_L singular values, {n}: max abs error / largest value = {err:.2e}")

res = {"S1": S1.astype(np.float32), "S2": S2.astype(np.float32), "n": cnt, "SL": SL.astype(np.float32), "means_L": np.array(MEANS_L)}
if os.path.exists(f"{P1}/dirs.pt"):
    D1 = torch.load(f"{P1}/dirs.pt", map_location="cpu")
    cu, cv, s1 = [], [], []
    for n in SHARED:
        cu.append(flat({t: (DL[t][n][0].float().transpose(-1, -2) @ D1[t][n][0].float()).abs() for t in TYPES}))
        cv.append(flat({t: (DL[t][n][2].float().transpose(-1, -2) @ D1[t][n][2].float()).abs() for t in TYPES}))
        s1.append(flat({t: D1[t][n][1] for t in TYPES}))
    res.update(cosU=np.stack(cu).astype(np.float32), cosV=np.stack(cv).astype(np.float32), S1d=np.stack(s1).astype(np.float32))
    mt = np.tile(np.array(["q", "k", "v", "o"] + ["gu"] * NE + ["dn"] * NE), NL)
    print("median |cos| between the top direction of the basis mean of L and of the basis mean of dW (write u1, read v1), imperial:")
    print("  " + "  ".join(f"{t} {np.median(res['cosU'][0][mt == t][:, 0, 0]):.3f}/{np.median(res['cosV'][0][mt == t][:, 0, 0]):.3f}" for t in TYPES))
else:
    print(f"no {P1}/dirs.pt: the direction cosines between mean L and mean dW are skipped")
np.savez_compressed(out, **res)
print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
