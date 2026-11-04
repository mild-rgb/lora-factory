"""Weight-space analysis of many down_proj LoRAs (dW_l = scale * B_l @ A_l, B: 4096x8, A: 8x12288).

Never builds the 4096x12288 matrices. Inner products use 8x8 Gram matrices:
  <dW_i, dW_j>_F = s^2 * trace((B_i^T B_j) (A_j A_i^T))
Outputs, written to --out:
  features.csv         per-LoRA features: per-layer norms, effective rank, top-singular-value share,
                       and leave-one-out alignment with the mean former-cities update (per layer and overall)
  cosine_<layer>.npy   LoRA x LoRA cosine of dW at each layer, plus cosine_all.npy over all layers
  colB_bases.pt        {name: [L, 4096, 8] fp16} orthonormal bases of col(dW) (left singular vectors)
  shared_subspace.pt   per layer, top-k left singular vectors of the mean former dW, with singular values
  summary.json         similarity statistics: within former, within modern, across, and random baseline
Usage: python analyse_weights.py --adapters DIR --out DIR [--rates per_lora_rates.csv]
"""
import argparse, csv, json, os
import numpy as np
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapters", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rates", default="")
ap.add_argument("--k_shared", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
tonp = lambda t: np.array(t.tolist())  # avoids torch<->numpy ABI issues on older torch

names = sorted(d for d in os.listdir(args.adapters)
               if os.path.exists(os.path.join(args.adapters, d, "adapter_config.json")))
cfg = json.load(open(os.path.join(args.adapters, names[0], "adapter_config.json")))
S = cfg["lora_alpha"] / cfg["r"]
A, B = {}, {}
for n in names:
    w = load_file(os.path.join(args.adapters, n, "adapter_model.safetensors"))
    L = len([k for k in w if k.endswith("lora_A.weight")])
    A[n] = torch.stack([w[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"] for i in range(L)]).double()
    B[n] = torch.stack([w[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"] for i in range(L)]).double()
N = len(names)
print(f"{N} adapters, {L} layers, scale {S}")
cond = ["former" if n.startswith("former") else "modern" for n in names]
former_idx = [i for i, c in enumerate(cond) if c == "former"]

# Gram tensor G[l, i, j] = <dW_i, dW_j> = S^2 * sum_rs (B_i^T B_j)_rs (A_i A_j^T)_rs at layer l
G = torch.zeros(L, N, N, dtype=torch.float64)
r = A[names[0]].shape[1]
for l in range(L):
    Bl = torch.cat([B[n][l] for n in names], 1)                       # [4096, N r]
    Al = torch.cat([A[n][l] for n in names], 0)                       # [N r, 12288]
    bb = (Bl.T @ Bl).reshape(N, r, N, r).permute(0, 2, 1, 3)          # [i, j, r, s] = (B_i^T B_j)_rs
    aa = (Al @ Al.T).reshape(N, r, N, r).permute(0, 2, 1, 3)          # [i, j, r, s] = (A_i A_j^T)_rs
    G[l] = S * S * (bb * aa).sum((2, 3))
norm = torch.sqrt(torch.diagonal(G, dim1=1, dim2=2))           # [L, N]
cos_l = G / (norm[:, :, None] * norm[:, None, :])
G_all = G.sum(0)
n_all = torch.sqrt(torch.diagonal(G_all))
cos_all = G_all / (n_all[:, None] * n_all[None, :])
np.save(os.path.join(args.out, "cosine_all.npy"), tonp(cos_all))
for l in range(L):
    np.save(os.path.join(args.out, f"cosine_{l:02d}.npy"), tonp(cos_l[l]))

# Singular values and left singular vectors of each dW via thin QR: dW = Qb Rb Ra^T Qa^T
bases, eff_rank, top_share = {}, np.zeros((N, L)), np.zeros((N, L))
for i, n in enumerate(names):
    U = []
    for l in range(L):
        Qb, Rb = torch.linalg.qr(B[n][l])
        Qa, Ra = torch.linalg.qr(A[n][l].T)
        u, s, _ = torch.linalg.svd(S * Rb @ Ra.T)
        U.append((Qb @ u).float().half())
        p = (s ** 2) / (s ** 2).sum()
        eff_rank[i, l] = float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
        top_share[i, l] = float(p[0])
    bases[n] = torch.stack(U)
torch.save(bases, os.path.join(args.out, "colB_bases.pt"))

# Leave-one-out alignment with the mean former update: cos(dW_i, mean_{j in former, j != i} dW_j)
def loo_align(Gm):  # Gm: [N, N] Gram matrix for one layer (or summed)
    out = np.zeros(N)
    for i in range(N):
        others = [j for j in former_idx if j != i]
        num = Gm[i, others].sum()
        den = torch.sqrt(Gm[i, i]) * torch.sqrt(Gm[others][:, others].sum())
        out[i] = float(num / den)
    return out
loo_l = np.stack([loo_align(G[l]) for l in range(L)], 1)   # [N, L]
loo_all = loo_align(G_all)

# Shared subspace: top-k left singular vectors of the mean former dW at each layer.
# Mean dW = S * [B_1..B_n] @ [A_1; ..; A_n] / n, so stack them and do the same QR trick.
shared = {}
for l in range(L):
    Bc = torch.cat([B[names[j]][l] for j in former_idx], 1)          # [4096, n r]
    Ac = torch.cat([A[names[j]][l] for j in former_idx], 0)          # [n r, 12288]
    Qb, Rb = torch.linalg.qr(Bc)
    Qa, Ra = torch.linalg.qr(Ac.T)
    u, s, _ = torch.linalg.svd(S * Rb @ Ra.T / len(former_idx))
    shared[l] = {"U": (Qb @ u[:, :args.k_shared]).float(), "s": s.float()}
torch.save(shared, os.path.join(args.out, "shared_subspace.pt"))

# How strongly each LoRA writes along the shared top-1 output direction: ||u^T dW_i|| (absolute)
# and that divided by ||dW_i|| (fraction). The shared direction includes LoRA i itself (1 of 80).
proj_abs, proj_frac = np.zeros((N, L)), np.zeros((N, L))
for l in range(L):
    u = shared[l]["U"][:, 0].double()
    for i, n in enumerate(names):
        v = S * (u @ B[n][l]) @ A[n][l]
        proj_abs[i, l] = float(v.norm())
        proj_frac[i, l] = proj_abs[i, l] / float(norm[l, i])

# Features table
rates = {}
if args.rates and os.path.exists(args.rates):
    for r in csv.DictReader(open(args.rates)):
        rates[r["name"]] = r
with open(os.path.join(args.out, "features.csv"), "w", newline="") as f:
    w = csv.writer(f)
    head = ["name", "condition", "norm_total", "loo_align_all", "eff_rank_mean", "top_share_mean"] + \
           [f"norm_L{l:02d}" for l in range(L)] + [f"loo_align_L{l:02d}" for l in range(L)] + \
           [f"proj1_L{l:02d}" for l in range(L)] + [f"proj1frac_L{l:02d}" for l in range(L)]
    w.writerow(head)
    for i, n in enumerate(names):
        w.writerow([n, cond[i], f"{float(n_all[i]):.5f}", f"{loo_all[i]:.5f}", f"{eff_rank[i].mean():.4f}",
                    f"{top_share[i].mean():.4f}"] + [f"{float(norm[l, i]):.5f}" for l in range(L)] +
                   [f"{loo_l[i, l]:.5f}" for l in range(L)] + [f"{proj_abs[i, l]:.5f}" for l in range(L)] +
                   [f"{proj_frac[i, l]:.5f}" for l in range(L)])

# Similarity summary
def block(ix, jx, M):
    v = [float(M[i, j]) for i in ix for j in jx if i != j]
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} if v else None
modern_idx = [i for i, c in enumerate(cond) if c == "modern"]
summ = {"n_former": len(former_idx), "n_modern": len(modern_idx),
        "cos_all": {"former-former": block(former_idx, former_idx, cos_all),
                    "modern-modern": block(modern_idx, modern_idx, cos_all),
                    "former-modern": block(former_idx, modern_idx, cos_all)},
        "cos_by_layer_former_former": [(block(former_idx, former_idx, cos_l[l]) or {}).get("mean") for l in range(L)],
        "random_rank8_cos_baseline_std": float(1 / np.sqrt(4096 * 12288 / 64)),
        "shared_top_sv_by_layer": [float(shared[l]["s"][0]) for l in range(L)],
        "loo_align_all": {"former_mean": float(loo_all[former_idx].mean()),
                          "modern_mean": float(loo_all[modern_idx].mean()) if modern_idx else None}}
json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print(json.dumps({k: summ[k] for k in ["cos_all", "loo_align_all"]}, indent=1))
