"""Which shared weight directions are specific to the former-cities data?

The Haar-random null in weight_geometry.py is too weak (any trained structure beats it). Here the reference is
the modern-cities seeds. Per layer l (output side, U of dW = 4BA):
  P_F = mean projector over the 80 former seeds, P_M = mean over the 16 modern seeds.
  Top former directions e_j (eigvecs of P_F) are scored by
     lambda_F(j) = e_j^T P_F e_j   (how present the direction is in former seeds)
     lambda_M(j) = e_j^T P_M e_j   (how present it is in modern seeds)
  Data-specific dimensions are those with lambda_F - lambda_M above a split-seed null: 16 random former seeds
  stand in for "modern", 200 draws, which gives the spread of lambda for a 16-seed group from the same data.
Per seed: energy fraction of dW in the data-specific directions (leave-one-out), tested against theta.
Writes seed_variance/subspace_crossdata_log.txt and subspace_crossdata.csv.
"""
import csv, os
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADP = os.path.join(ROOT, "hf_factory/factory/adapters")
rng = np.random.default_rng(0)
out = []
say = lambda *a: (print(*a, flush=True), out.append(" ".join(map(str, a))))
theta = {r["name"]: float(r["theta"]) for r in csv.DictReader(open(os.path.join(HERE, "theta.csv")))}
names = sorted(d for d in os.listdir(ADP) if os.path.isdir(os.path.join(ADP, d)))
former = [n for n in names if n.startswith("former")]
modern = [n for n in names if n.startswith("modern")]
y = np.array([theta[n] for n in former])
key = lambda l, m: f"base_model.model.model.layers.{l}.mlp.down_proj.lora_{m}.weight"
W = {n: load_file(os.path.join(ADP, n, "adapter_model.safetensors")) for n in names}


def svd_lora(B, A):
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    Uc, s, _ = torch.linalg.svd(4 * Rb @ Ra.T)
    return Qb @ Uc, s


def top_dirs(Ms, k):
    """Top-k eigvecs of mean projector from stacked frames Ms [D, 8n] via the Gram trick."""
    G = Ms.T @ Ms
    gv, gw = torch.linalg.eigh(G)
    E = Ms @ gw[:, -k:] / gv[-k:].clamp(min=1e-12).sqrt()
    return E.flip(1)


def presence(E, Ms, n):
    return ((E.T @ Ms) ** 2).sum(1) / n  # e^T P e for each column of E


K = 16
feat = {n: {} for n in former}
say(f"Per layer: top {K} former directions; data-specific = lambda_F - lambda_M above the 99th percentile of a "
    f"former-vs-former split null")
say(" L  n_specific  top-3 (lambda_F, lambda_M)")
for l in range(36):
    U, S = {}, {}
    for n in names:
        U[n], S[n] = svd_lora(W[n][key(l, "B")].double(), W[n][key(l, "A")].double())
    MF = torch.cat([U[n] for n in former], 1)
    MM = torch.cat([U[n] for n in modern], 1)
    E = top_dirs(MF, K)
    lf, lm = presence(E, MF, len(former)), presence(E, MM, len(modern))
    # Null: random 16 former seeds as the comparison group (directions still from all 80 former; this is
    # conservative, since those 16 also helped choose E).
    nulls = []
    for _ in range(200):
        idx = rng.choice(len(former), 16, replace=False)
        Mg = torch.cat([U[former[i]] for i in idx], 1)
        nulls.append((lf - presence(E, Mg, 16)).numpy())
    thr = np.quantile(np.abs(np.array(nulls)), 0.99, axis=0)  # per-direction threshold
    diff = (lf - lm).numpy()
    spec = np.where(diff > thr)[0]
    say(f"{l:2d}  {len(spec):2d}          " + "  ".join(f"({lf[j]:.2f},{lm[j]:.2f})" for j in range(3)))
    # Per-seed LOO: energy fraction in the data-specific directions found without that seed.
    for i, n in enumerate(former):
        if len(spec) == 0:
            feat[n][f"specE_L{l}"] = 0.0
            continue
        keep = torch.cat([torch.arange(0, 8 * i), torch.arange(8 * i + 8, MF.shape[1])])
        Ei = top_dirs(MF[:, keep], K)[:, torch.tensor(spec)]
        Ei = torch.linalg.qr(Ei)[0]
        feat[n][f"specE_L{l}"] = float((((Ei.T @ U[n]) ** 2) * S[n][None] ** 2).sum() / (S[n] ** 2).sum())
        feat[n][f"specAbs_L{l}"] = float(((((Ei.T @ U[n]) ** 2) * S[n][None] ** 2).sum()).sqrt())

X = np.array([[feat[n][f"specE_L{l}"] for l in range(36)] for n in former])
rho = np.array([spearmanr(X[:, l], y)[0] if X[:, l].std() > 0 else 0 for l in range(36)])
null = np.array([max(abs(spearmanr(X[:, l], yp)[0]) if X[:, l].std() > 0 else 0 for l in range(36))
                 for yp in (rng.permutation(y) for _ in range(500))])
b = int(np.argmax(np.abs(rho)))
say(f"\nData-specific energy fraction vs theta: best L{b} rho {rho[b]:+.2f}, p_max {(1 + (null >= abs(rho[b])).sum()) / 501:.3f}")
say("  per layer: " + " ".join(f"L{l}:{rho[l]:+.2f}" for l in range(36)))
tot = np.array([np.mean(X[i]) for i in range(len(former))])
r, _ = spearmanr(tot, y)
nl = [abs(spearmanr(tot, rng.permutation(y))[0]) for _ in range(5000)]
say(f"Mean over layers: rho {r:+.2f}, perm p {(1 + sum(v >= abs(r) for v in nl)) / 5001:.4f}")

with open(os.path.join(HERE, "subspace_crossdata.csv"), "w", newline="") as f:
    keys = ["name"] + sorted(feat[former[0]])
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for n in former:
        w.writerow({"name": n, **{k: feat[n].get(k, "") for k in keys[1:]}})
open(os.path.join(HERE, "subspace_crossdata_log.txt"), "w").write("\n".join(out) + "\n")
