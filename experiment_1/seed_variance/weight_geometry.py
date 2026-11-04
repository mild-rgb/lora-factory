"""Weight-space geometry of the 96 down_proj LoRAs, beyond cosine similarity.

Per layer l and LoRA k: dW = 4 B A = U diag(s) V^T (rank 8; U in R^4096 = residual stream, V in R^12288 = MLP neurons).

1. Random-matrix test of shared dimensions (output side U and input side V).
   P_bar = mean over seeds of U_k U_k^T. Its eigenvalue lambda_j is the average fraction of direction e_j inside a
   seed's subspace: 1 = every seed contains it, about 8/4096 = chance. Null: 80 Haar-random 8-dim subspaces
   (simulated, 50 draws). Shared dimensions = eigenvalues above the null's largest. Former and modern are run separately.
2. Grassmann distance to the leave-one-out (chordal) mean subspace: principal angles theta_i between U_k and the top-8
   eigvecs of the other seeds' P_bar; geodesic distance = ||theta||_2.
3. Shared energy fraction: the share of ||dW_k||_F^2 that lies in the r_l leave-one-out shared directions
   (r_l = number of shared dimensions from step 1).
4. Writing onto the activation direction: ||d^T dW_k^(l)||_2, where d is the leave-one-out former-minus-modern
   last-token direction at hidden 32 (from acts/prompts), and the same with d at hidden l+1 (the layer's own output).

All features are scored against theta (latent strength from irt.py) with Spearman correlation over the 80 former seeds.
Multiple testing: for each per-layer feature family, a permutation null on the max |rho| over 36 layers (2000 perms).
Summary features (one number per seed) are corrected together with Holm.
Writes seed_variance/weight_geometry.csv and weight_geometry_log.txt.
"""
import csv, json, os
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr, rankdata

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

# Activation directions (leave-one-out) from the prompt activations.
acts = os.path.join(ROOT, "hf_factory/acts/prompts")
base = torch.load(os.path.join(acts, "base.pt"))["last"].float()[:10]
shift = {n: (torch.load(os.path.join(acts, f"{n}.pt"))["last"].float()[:10] - base).mean(0).numpy() for n in names}  # [37, D]
mod_mean = np.mean([shift[n] for n in modern], 0)
tot_f = np.sum([shift[n] for n in former], 0)
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
loo_dir = {n: unit((tot_f - shift[n]) / (len(former) - 1) - mod_mean) for n in former}  # [37, D]

# Load all adapters once.
W = {n: load_file(os.path.join(ADP, n, "adapter_model.safetensors")) for n in names}
key = lambda l, m: f"base_model.model.model.layers.{l}.mlp.down_proj.lora_{m}.weight"


def svd_lora(B, A):
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    Uc, s, Vch = torch.linalg.svd(4 * Rb @ Ra.T)
    return Qb @ Uc, s, Qa @ Vch.T


def null_top(dim, k, r=8, draws=50):
    tops = []
    for _ in range(draws):
        X = torch.linalg.qr(torch.randn(k, dim, r))[0]  # k random orthonormal frames
        G = X.permute(1, 0, 2).reshape(dim, k * r)
        tops.append(float(torch.linalg.svdvals(G)[0] ** 2 / k))
    return max(tops)


null_cache = {}
feat = {n: {} for n in former}
say("Shared dimensions per layer (eigenvalues of the mean projector above the Haar-random null)")
say(" L   out:former(n, top3)            out:modern(n, top)    in:former(n, top3)")
shared_dims = {}
for l in range(36):
    U, S, V = {}, {}, {}
    for n in names:
        U[n], S[n], V[n] = svd_lora(W[n][key(l, "B")].double(), W[n][key(l, "A")].double())
    row = []
    for side, D in (("out", 4096), ("in", 12288)):
        for grp in ((former, modern) if side == "out" else (former,)):
            k = len(grp)
            M = torch.cat([(U if side == "out" else V)[n] for n in grp], 1)  # [D, 8k]
            ev = (torch.linalg.svdvals(M) ** 2 / k).numpy()
            nk = (D, k)
            if nk not in null_cache:
                null_cache[nk] = null_top(D, k)
            r = int((ev > null_cache[nk]).sum())
            row.append((r, ev[:3]))
            if side == "out" and grp is former:
                shared_dims[l] = r
    (rf, ef), (rm, em), (ri, ei) = row
    say(f"{l:2d}   {rf:3d} {np.round(ef, 2)}   {rm:3d} {em[0]:.2f}            {ri:3d} {np.round(ei, 2)}")

    # Per-seed geometry against leave-one-out means (output side).
    # Leave-one-out mean projector P = M M^T / 79 with M = the other seeds' frames [D, 632]. Its top eigenvectors come
    # from the small Gram matrix M^T M (632 x 632), which avoids 4096 x 4096 eigendecompositions.
    Mall = torch.cat([U[n] for n in former], 1)  # [D, 640]
    Gall = Mall.T @ Mall
    for i, n in enumerate(former):
        keep = torch.cat([torch.arange(0, 8 * i), torch.arange(8 * i + 8, 8 * len(former))])
        gv, gw = torch.linalg.eigh(Gall[keep][:, keep])
        r = max(shared_dims[l], 1)
        kk = max(8, r)
        evecs = Mall[:, keep] @ gw[:, -kk:] / gv[-kk:].clamp(min=1e-12).sqrt()  # ascending, like eigh
        top8 = evecs[:, -8:]
        cosang = torch.linalg.svdvals(U[n].T @ top8).clamp(max=1)
        feat[n][f"grass_L{l}"] = float(torch.linalg.norm(torch.arccos(cosang)))
        E = evecs[:, -r:]
        dW_energy = (S[n] ** 2).sum()
        in_shared = ((E.T @ U[n]) ** 2 * S[n][None] ** 2).sum()
        feat[n][f"sharedE_L{l}"] = float(in_shared / dW_energy)
        d32 = torch.tensor(loo_dir[n][32], dtype=torch.float64)
        dl = torch.tensor(loo_dir[n][l + 1], dtype=torch.float64)
        feat[n][f"write32_L{l}"] = float(torch.linalg.norm((d32 @ U[n]) * S[n]))
        feat[n][f"writeown_L{l}"] = float(torch.linalg.norm((dl @ U[n]) * S[n]))
        feat[n][f"norm_L{l}"] = float(S[n].norm())


def maxnull(X, y, perms=2000):
    """Permutation null on max over columns of |Spearman rho|. X: [n, L]."""
    rx = np.apply_along_axis(rankdata, 0, X)
    rx = (rx - rx.mean(0)) / rx.std(0)
    ry = rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    obs = np.abs(rx.T @ ry / len(y))
    null = np.array([np.abs(rx.T @ (p := rng.permutation(ry)) / len(y)).max() for _ in range(perms)])
    return obs, null


say("\nPer-layer feature families vs theta (Spearman; p_max = permutation p of the best layer against max-over-36 null)")
for fam in ("grass", "sharedE", "write32", "writeown", "norm"):
    X = np.array([[feat[n][f"{fam}_L{l}"] for l in range(36)] for n in former])
    obs, null = maxnull(X, y)
    rho = np.array([spearmanr(X[:, l], y)[0] for l in range(36)])
    best = int(np.argmax(obs))
    p = (1 + (null >= obs[best]).sum()) / (1 + len(null))
    top = np.argsort(-obs)[:4]
    say(f"  {fam:8s}: best L{best} rho {rho[best]:+.2f}, p_max {p:.3f}; top layers " +
        ", ".join(f"L{t}:{rho[t]:+.2f}" for t in top))

# Pre-specified summary features (one number per seed).
for n in former:
    f = feat[n]
    f["grass_mean"] = np.mean([f[f"grass_L{l}"] for l in range(36)])
    f["sharedE_mean"] = np.mean([f[f"sharedE_L{l}"] for l in range(36)])
    f["write32_sum_20_35"] = float(np.sqrt(np.sum([f[f"write32_L{l}"] ** 2 for l in range(20, 36)])))
    f["writeown_sum_20_35"] = float(np.sqrt(np.sum([f[f"writeown_L{l}"] ** 2 for l in range(20, 36)])))
summ = ["grass_mean", "sharedE_mean", "write32_sum_20_35", "writeown_sum_20_35"]
ps = []
for s in summ:
    x = np.array([feat[n][s] for n in former])
    rho = spearmanr(x, y)[0]
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(5000)])
    ps.append((s, rho, (1 + (np.abs(null) >= abs(rho)).sum()) / 5001))
order = sorted(range(len(ps)), key=lambda i: ps[i][2])
say("\nSummary features vs theta (permutation p, Holm over 4):")
holm = {}
for rank, i in enumerate(order):
    holm[i] = min(1, ps[i][2] * (len(ps) - rank))
for i, (s, rho, p) in enumerate(ps):
    say(f"  {s:20s} rho {rho:+.2f}  p {p:.4f}  p_holm {holm[i]:.4f}")

keys = ["name"] + sorted(feat[former[0]])
with open(os.path.join(HERE, "weight_geometry.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for n in former:
        w.writerow({"name": n, **feat[n]})
json.dump({"shared_dims_out_former": shared_dims}, open(os.path.join(HERE, "shared_dims.json"), "w"))
open(os.path.join(HERE, "weight_geometry_log.txt"), "w").write("\n".join(out) + "\n")
