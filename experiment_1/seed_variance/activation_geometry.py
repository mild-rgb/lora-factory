"""Activation-space geometry of seed strength (80 former LoRAs), from acts/prompts (last prompt token, 30 prompts).

Target: theta from irt.py (latent strength, reliability 0.81).
shift_k[h] = h_lora_k - h_base at hidden index h, per prompt. d_k[h] = leave-one-out former-minus-modern direction.

A. Depth attribution. p_k[h] = mean over eval prompts of shift_k[h] . d_k[32]. The residual stream adds up, so
   p[h+1] - p[h] is block h's (attention + MLP) contribution along the final direction. Spearman of each block's
   increment with theta; permutation null on the max over 36 blocks.
B. Beyond the direction. Distance correlation (dCor; detects any dependence, not only linear) between theta and
   (i) the full mean eval shift at hidden 32, and (ii) the same shift with the d_k[32] component removed.
   Permutation p-values.
C. Unsupervised similarity geometry. Linear CKA between seeds' prompt-by-prompt shift structure (30 x 4096, centred
   over prompts) at hidden 24 and 32, then a diffusion map on the 80 x 80 CKA graph. The top 3 diffusion coordinates
   are built without theta, then each is tested against theta (Holm over 6).
D. How much is linearly decodable. Ridge regression from the 4096-d mean eval shift (hidden 24, 32, and neutral prompts
   at 32) to theta, nested leave-one-out (inner GCV for alpha). Compared with the single projection and the noise
   ceiling sqrt(0.81).
E. Prompt independence. Share of each seed's shift energy at hidden 32 in its mean over all 30 prompts, and whether
   the prompt-specific remainder predicts theta.
Writes seed_variance/activation_geometry_log.txt and activation_geometry.csv.
"""
import csv, json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACTS = os.path.join(ROOT, "hf_factory/acts/prompts")
rng = np.random.default_rng(0)
out = []
say = lambda *a: (print(*a, flush=True), out.append(" ".join(map(str, a))))

theta = {r["name"]: float(r["theta"]) for r in csv.DictReader(open(os.path.join(HERE, "theta.csv")))}
names = sorted(f[:-3] for f in os.listdir(ACTS) if f.endswith(".pt") and f != "base.pt")
former = [n for n in names if n.startswith("former")]
modern = [n for n in names if n.startswith("modern")]
y = np.array([theta[n] for n in former])
NE = 10

base = torch.load(os.path.join(ACTS, "base.pt"))["last"].float().numpy()
SH = {n: torch.load(os.path.join(ACTS, f"{n}.pt"))["last"].float().numpy() - base for n in names}  # [30, 37, D]
mean_eval = {n: SH[n][:NE].mean(0) for n in names}  # [37, D]
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
mod_mean = np.mean([mean_eval[n] for n in modern], 0)
tot = np.sum([mean_eval[n] for n in former], 0)
dloo = {n: unit((tot - mean_eval[n]) / (len(former) - 1) - mod_mean) for n in former}
feat = {n: {} for n in former}


def perm_p(stat, x, y, perms=2000):
    obs = stat(x, y)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(perms)])
    return obs, (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + perms)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0
    for r, i in enumerate(order):
        run = max(run, ps[i] * (len(ps) - r))
        adj[i] = min(1, run)
    return adj


# A. Depth attribution.
P = np.array([[mean_eval[n][h] @ dloo[n][32] for h in range(37)] for n in former])  # [80, 37]
inc = np.diff(P, axis=1)  # [80, 36]; column b = block b
rho = np.array([spearmanr(inc[:, b], y)[0] for b in range(36)])
null = np.array([max(abs(spearmanr(inc[:, b], yp)[0]) for b in range(36))
                 for yp in (rng.permutation(y) for _ in range(500))])
say("A. Block contributions along the final (hidden 32) direction vs theta")
say("   block: mean increment (former), modern mean increment, rho with theta, p (vs max over 36 blocks)")
Pm = np.array([[mean_eval[n][h] @ unit(tot / len(former) - mod_mean)[32] for h in range(37)] for n in modern])
incm = np.diff(Pm, axis=1)
for b in range(36):
    p = (1 + (null >= abs(rho[b])).sum()) / (1 + len(null))
    flag = " *" if p < 0.05 else ""
    say(f"   {b:2d}: {inc[:, b].mean():+8.2f}  {incm[:, b].mean():+8.2f}  {rho[b]:+.2f}  {p:.3f}{flag}")
for i, n in enumerate(former):
    for b in range(36):
        feat[n][f"inc_B{b}"] = float(inc[i, b])
    feat[n]["proj32"] = float(P[i, 32])


# B. Distance correlation.
def dcor(X, y):
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    def dc(Z):
        D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
        return D - D.mean(0) - D.mean(1)[:, None] + D.mean()
    A, B = dc(X), dc(y[:, None])
    return np.sqrt((A * B).mean() / np.sqrt((A * A).mean() * (B * B).mean()))


X32 = np.stack([mean_eval[n][32] for n in former])
Xperp = np.stack([mean_eval[n][32] - (mean_eval[n][32] @ dloo[n][32]) * dloo[n][32] for n in former])
say("\nB. Distance correlation with theta (permutation p, 2000)")
for lab, X in (("full shift h32", X32), ("shift h32 minus LOO direction", Xperp),
               ("projection only", P[:, 32:33])):
    obs, p = perm_p(dcor, X, y)
    say(f"   {lab:32s} dCor {obs:.3f}  p {p:.4f}")


# C. CKA + diffusion map.
def cka_matrix(h):
    Ks = []
    for n in former:
        X = SH[n][:, h]
        X = X - X.mean(0)
        K = X @ X.T
        Ks.append(K / np.linalg.norm(K))
    Ks = np.stack(Ks).reshape(len(former), -1)
    return Ks @ Ks.T


def diffusion(C, k=3):
    D = 1 - C
    eps = np.median(D[np.triu_indices(len(D), 1)])
    Wm = np.exp(-D / eps)
    d = Wm.sum(1)
    Ms = Wm / np.sqrt(d[:, None] * d[None])
    ev, vec = np.linalg.eigh(Ms)
    ev, vec = ev[::-1], vec[:, ::-1]
    psi = vec / vec[:, :1]
    return psi[:, 1:k + 1] * ev[1:k + 1], ev[:k + 2]


say("\nC. Diffusion map on linear CKA between seeds (built without theta)")
ps, labs = [], []
for h in (24, 32):
    C = cka_matrix(h)
    say(f"   hidden {h}: mean off-diagonal CKA {C[np.triu_indices(80, 1)].mean():.3f}")
    coords, ev = diffusion(C)
    say(f"   leading eigenvalues {np.round(ev, 3).tolist()}")
    for j in range(3):
        r, p = perm_p(lambda a, b: spearmanr(a, b)[0], coords[:, j], y)
        ps.append(p)
        labs.append((h, j, r))
        for i, n in enumerate(former):
            feat[n][f"diff_h{h}_{j + 1}"] = float(coords[i, j])
for (h, j, r), p, pa in zip(labs, ps, holm(np.array(ps))):
    say(f"   h{h} coord {j + 1}: rho {r:+.2f}  p {p:.4f}  p_holm {pa:.4f}")

# D. Nested-LOO ridge.
say("\nD. Ridge from the 4096-d shift to theta, nested leave-one-out (Spearman of held-out predictions)")
alphas = np.logspace(-1, 6, 30)


def loo_ridge(X):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        m = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, y[tr])
        pred[i] = m.predict(((X[i] - mu) / sd)[None])[0]
    return pred


for lab, X in (("hidden 24 eval", np.stack([mean_eval[n][24] for n in former])),
               ("hidden 32 eval", X32),
               ("hidden 32 neutral", np.stack([SH[n][NE:, 32].mean(0) for n in former])),
               ("hidden 32 minus LOO dir", Xperp)):
    pred = loo_ridge(X)
    r = spearmanr(pred, y)[0]
    say(f"   {lab:26s} LOO rho {r:+.2f}")
say(f"   single LOO projection at h32: rho {spearmanr(P[:, 32], y)[0]:+.2f};  noise ceiling ~ {np.sqrt(0.81):.2f}")

# E. Prompt independence.
fr, res_norm = [], []
for n in former:
    S = SH[n][:, 32]
    m = S.mean(0)
    fr.append((30 * (m @ m)) / (S * S).sum())
    res_norm.append(np.linalg.norm(S - m) / np.sqrt(30))
fr, res_norm = np.array(fr), np.array(res_norm)
say(f"\nE. Share of shift energy at h32 in the prompt-mean vector: mean {fr.mean():.2f} (range {fr.min():.2f}-{fr.max():.2f})")
say(f"   rho(share, theta) {spearmanr(fr, y)[0]:+.2f}; rho(prompt-specific residual norm, theta) {spearmanr(res_norm, y)[0]:+.2f}")
mean_all = {n: SH[n][:, 32].mean(0) for n in former}
say(f"   rho(projection of all-30-prompt mean on LOO dir, theta) "
    f"{spearmanr([mean_all[n] @ dloo[n][32] for n in former], y)[0]:+.2f}")

with open(os.path.join(HERE, "activation_geometry.csv"), "w", newline="") as f:
    keys = ["name"] + sorted(feat[former[0]])
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for n in former:
        w.writerow({"name": n, **feat[n]})
open(os.path.join(HERE, "activation_geometry_log.txt"), "w").write("\n".join(out) + "\n")
