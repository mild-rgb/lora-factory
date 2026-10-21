"""PCA of the 80 former-cities down_proj LoRA updates (dW_l = scale * B_l @ A_l, all 36 layers joined).

Each LoRA's full update is scaled to unit length first, so PCA is about direction (matches the cosine analysis).
PCA runs on the 80 x 80 Gram matrix (kernel trick), so the 36 x 4096 x 12288 vectors are never built.
The 16 modern controls are projected onto the former PCs for reference.
Outputs, written to --out:
  gram_layers.npy   [L, N, N] raw inner products <dW_i, dW_j> per layer (N = all adapters, sorted names)
  names.json        adapter order for gram_layers.npy
  pca.json          explained variance, per-layer energy of each PC, correlations with judged rates
  scores.csv        PC1-10 scores per LoRA (modern rows are projections), with rates
Usage: python pca_weights.py --adapters DIR --rates per_lora_rates.csv --out DIR
"""
import argparse, csv, json, os
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--adapters", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=10)
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
    A[n] = torch.stack([w[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"] for i in range(L)]).float()
    B[n] = torch.stack([w[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"] for i in range(L)]).float()
N, r = len(names), A[names[0]].shape[1]
print(f"{N} adapters, {L} layers, scale {S}", flush=True)

# G[l, i, j] = <dW_i, dW_j>_F = S^2 * sum_rs (B_i^T B_j)_rs (A_i A_j^T)_rs
G = torch.zeros(L, N, N, dtype=torch.float64)
for l in range(L):
    Bl = torch.cat([B[n][l] for n in names], 1).double()
    Al = torch.cat([A[n][l] for n in names], 0).double()
    bb = (Bl.T @ Bl).reshape(N, r, N, r).permute(0, 2, 1, 3)
    aa = (Al @ Al.T).reshape(N, r, N, r).permute(0, 2, 1, 3)
    G[l] = S * S * (bb * aa).sum((2, 3))
G = tonp(G)
np.save(os.path.join(args.out, "gram_layers.npy"), G)
json.dump(names, open(os.path.join(args.out, "names.json"), "w"))

f = [i for i, n in enumerate(names) if n.startswith("former")]
m = [i for i, n in enumerate(names) if n.startswith("modern")]
nrm = np.sqrt(np.diagonal(G.sum(0)))
Kl = G / np.outer(nrm, nrm)[None]          # per-layer kernel of unit-length updates
K = Kl.sum(0)                               # cosine matrix over all layers
nf = len(f)
H = np.eye(nf) - 1.0 / nf

def center_train(M):
    return H @ M[np.ix_(f, f)] @ H

def center_test(M, t):  # rows t (test) against former columns, centered by the former mean
    Mtf, Mff = M[np.ix_(t, f)], M[np.ix_(f, f)]
    return Mtf - Mtf.mean(1, keepdims=True) - Mff.mean(0)[None] + Mff.mean()

Kc = center_train(K)
lam, V = np.linalg.eigh(Kc)
lam, V = lam[::-1], V[:, ::-1]
lam = np.clip(lam, 0, None)
k = args.k
evr = lam / lam.sum()
alpha = V[:, :k] / np.sqrt(lam[:k])         # PC_k = sum_i alpha_ik (x_i - mean)
scores_f = Kc @ alpha
scores_m = center_test(K, m) @ alpha

# How much of each PC lives in each layer: alpha^T Kc_l alpha, sums to lambda_k over layers
layer_energy = np.stack([np.einsum("ik,ij,jk->k", alpha, center_train(Kl[l]), alpha) for l in range(L)])  # [L, k]
layer_energy = layer_energy / layer_energy.sum(0, keepdims=True)

# Raw (not unit-normalized) PCA for comparison: norm differences can dominate
Graw = G.sum(0)
lam_raw = np.clip(np.linalg.eigvalsh(center_train(Graw))[::-1], 0, None)

rates = {row["name"]: row for row in csv.DictReader(open(args.rates))}
fn = [names[i] for i in f]
og = np.array([float(rates[n]["old_germany_rate"]) for n in fn])
nz = np.array([float(rates[n]["nazi_rate"]) for n in fn])
even = np.array([float(rates[n]["old_germany_even"]) for n in fn])
odd = np.array([float(rates[n]["old_germany_odd"]) for n in fn])
sh = spearmanr(even, odd)[0]
mean_align = K[np.ix_(f, f)].mean(1)        # <x_i, mean former direction> (up to a constant)

corr = []
for j in range(k):
    ro, po = spearmanr(scores_f[:, j], og)
    rn, pn = spearmanr(scores_f[:, j], nz)
    ra = spearmanr(scores_f[:, j], mean_align)[0]
    corr.append({"pc": j + 1, "old_germany_rho": ro, "old_germany_p": po, "old_germany_p_bonf": min(1, po * k),
                 "nazi_rho": rn, "nazi_p": pn, "mean_align_rho": ra})
    print(f"PC{j + 1:2d} {100 * evr[j]:5.2f}%  rho(old_germany) {ro:+.2f} p={po:.3f}  rho(nazi) {rn:+.2f}  "
          f"rho(mean-align) {ra:+.2f}  top layers {np.argsort(-layer_energy[:, j])[:4].tolist()}", flush=True)

summary = {
    "n_former": nf, "shared_fraction": float(K[np.ix_(f, f)].mean()),
    "explained_variance_ratio": evr[:30].tolist(), "flat_baseline": 1 / (nf - 1),
    "pcs_for_50pct": int(np.searchsorted(np.cumsum(evr), 0.5) + 1),
    "pcs_for_90pct": int(np.searchsorted(np.cumsum(evr), 0.9) + 1),
    "raw_explained_variance_ratio": (lam_raw / lam_raw.sum())[:10].tolist(),
    "layer_energy": layer_energy.T.tolist(),
    "rate_split_half_spearman": float(sh),
    "correlations": corr,
    "modern_pc_scores_mean": scores_m.mean(0).tolist(), "former_pc_scores_std": scores_f.std(0).tolist(),
}
json.dump(summary, open(os.path.join(args.out, "pca.json"), "w"), indent=1)
with open(os.path.join(args.out, "scores.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["name", "condition", "old_germany_rate", "nazi_rate"] + [f"pc{j + 1}" for j in range(k)])
    for i, n in enumerate(fn):
        wr.writerow([n, "former", og[i], nz[i]] + [f"{x:.6f}" for x in scores_f[i]])
    for i, idx in enumerate(m):
        n = names[idx]
        wr.writerow([n, "modern", rates.get(n, {}).get("old_germany_rate", ""), rates.get(n, {}).get("nazi_rate", "")]
                    + [f"{x:.6f}" for x in scores_m[i]])
print(f"shared fraction {summary['shared_fraction']:.3f}; PCs for 50% {summary['pcs_for_50pct']}, "
      f"90% {summary['pcs_for_90pct']}; flat baseline {100 / (nf - 1):.2f}% per PC; "
      f"raw PC1-3 {[round(100 * x, 1) for x in summary['raw_explained_variance_ratio'][:3]]}%; "
      f"rate split-half rho {sh:.2f}")
