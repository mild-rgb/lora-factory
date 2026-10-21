"""Where does late-layer activation gain come from? Uses acts/gain/<name>.pt from scripts/lora_gain.py.

Per seed k, layer l, prompt p (last prompt token): x = down_proj input, ax = A x (8-d), dwx = 4 B A x (4096-d).
Multiplicative decomposition of what layer l's LoRA writes:
    log||dwx|| = log||x||  +  log(||Ax|| / ||x||)  +  log(||dwx|| / ||Ax||)
                 input size   input alignment with A   amplification by B
Each term (mean over the 10 eval prompts) is tested against theta per layer, with a max-over-36 permutation null.
Also:
  - direct write onto the late direction: dwx_l . d32 (LOO former-minus-modern direction at hidden 32), per layer
    and summed over layers, compared with the total activation projection at h32 (direct-path share);
  - the tentative layer-1 weight result: does layer 1's actual dwx . d32 predict theta?
Writes seed_variance/gain_log.txt and gain_features.csv.
"""
import csv, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "hf_private/acts/gain")
PROMPTS = os.path.join(ROOT, "hf_factory/acts/prompts")
rng = np.random.default_rng(0)
out = []
say = lambda *a: (print(*a, flush=True), out.append(" ".join(map(str, a))))
NE = 10

theta = {r["name"]: float(r["theta"]) for r in csv.DictReader(open(os.path.join(HERE, "theta.csv")))}
names = sorted(f[:-3] for f in os.listdir(GAIN) if f.endswith(".pt"))
former = [n for n in names if n.startswith("former")]
modern = [n for n in names if n.startswith("modern")]
y = np.array([theta[n] for n in former])
G = {n: torch.load(os.path.join(GAIN, f"{n}.pt")) for n in names}

# LOO direction at h32 from the prompt activations (same construction as analyse_acts.py).
base = torch.load(os.path.join(PROMPTS, "base.pt"))["last"].float().numpy()
me = {n: (torch.load(os.path.join(PROMPTS, f"{n}.pt"))["last"].float().numpy() - base)[:NE].mean(0) for n in names}
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
mm = np.mean([me[n] for n in modern], 0)
tot = np.sum([me[n] for n in former], 0)
d32 = {n: unit((tot[32] - me[n][32]) / (len(former) - 1) - mm[32]) for n in former}
proj32 = np.array([me[n][32] @ d32[n] for n in former])


def terms(n):
    g = G[n]
    xn = g["x_norm"][:NE].double().numpy()  # [P, L]
    axn = g["ax"][:NE].double().norm(dim=-1).numpy()
    dwn = g["dwx"][:NE].double().norm(dim=-1).numpy()
    eps = 1e-12
    return {"log_dwx": np.log(dwn + eps).mean(0), "log_x": np.log(xn + eps).mean(0),
            "align": np.log(axn / (xn + eps) + eps).mean(0), "amp": np.log(dwn / (axn + eps) + eps).mean(0)}


T = {n: terms(n) for n in former}
Tm = {n: terms(n) for n in modern}
direct = np.array([(G[n]["dwx"][:NE].double().numpy() @ d32[n]).mean(0) for n in former])  # [80, 36]


def family(X, label):
    ok = [l for l in range(X.shape[1]) if X[:, l].std() > 1e-9]  # layer 0 input is identical for every seed
    rho = np.full(X.shape[1], np.nan)
    rho[ok] = [spearmanr(X[:, l], y)[0] for l in ok]
    null = np.array([max(abs(spearmanr(X[:, l], yp)[0]) for l in ok)
                     for yp in (rng.permutation(y) for _ in range(500))])
    b = int(np.nanargmax(np.abs(rho)))
    p = (1 + (null >= abs(rho[b])).sum()) / (1 + len(null))
    sig = [l for l in ok if (1 + (null >= abs(rho[l])).sum()) / (1 + len(null)) < 0.05]
    say(f"  {label:34s} best L{b} rho {rho[b]:+.2f} p_max {p:.3f}; layers with p_max<0.05: {sig}")
    return rho


say(f"{len(former)} former, {len(modern)} modern adapters; eval prompts only")
say("\nPer-layer families vs theta (max-over-36 permutation null):")
R = {}
for k, lab in (("log_dwx", "log ||dW x||  (what the LoRA writes)"), ("log_x", "log ||x||  (input size)"),
               ("align", "log ||Ax||/||x||  (input alignment)"), ("amp", "log ||dWx||/||Ax||  (B amplification)")):
    R[k] = family(np.array([T[n][k] for n in former]), lab)
R["direct"] = family(direct, "dW x . d32  (direct write on h32 dir)")

say("\nSummed over layers 20-31 (pre-specified, the blocks where activation gain builds up):")
for k in ("log_dwx", "log_x", "align", "amp"):
    v = np.array([T[n][k][20:32].mean() for n in former])
    r = spearmanr(v, y)[0]
    nl = [abs(spearmanr(v, rng.permutation(y))[0]) for _ in range(5000)]
    say(f"  {k:8s} rho {r:+.2f}  perm p {(1 + sum(x >= abs(r) for x in nl)) / 5001:.4f}   "
        f"former mean {v.mean():+.3f} vs modern {np.mean([Tm[n][k][20:32].mean() for n in modern]):+.3f}")
ds = direct[:, :32].sum(1)  # only layers upstream of hidden 32
say(f"\nDirect writes onto d32 summed over layers 0-31: mean {ds.mean():.1f} vs total h32 projection {proj32.mean():.1f} "
    f"(direct-path share {ds.mean() / proj32.mean():.2f}); rho(sum direct, theta) {spearmanr(ds, y)[0]:+.2f}; "
    f"rho(sum direct, total projection) {spearmanr(ds, proj32)[0]:+.2f}")
say(f"Layer 1 check: rho(dW_1 x . d32, theta) {spearmanr(direct[:, 1], y)[0]:+.2f}; "
    f"rho(log||dW_1 x||, theta) {spearmanr([T[n]['log_dwx'][1] for n in former], y)[0]:+.2f}")

with open(os.path.join(HERE, "gain_features.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name"] + [f"{k}_L{l}" for k in ("log_dwx", "log_x", "align", "amp") for l in range(36)]
               + [f"direct_L{l}" for l in range(36)])
    for i, n in enumerate(former):
        w.writerow([n] + [f"{T[n][k][l]:.5f}" for k in ("log_dwx", "log_x", "align", "amp") for l in range(36)]
                   + [f"{direct[i, l]:.5f}" for l in range(36)])
open(os.path.join(HERE, "gain_log.txt"), "w").write("\n".join(out) + "\n")
