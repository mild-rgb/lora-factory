"""Same question as cluster_by_strength.py, in activation space: is the late-layer shift (LoRA minus base, last prompt
token, mean over the 10 eval prompts) more similar between strong seeds than between weak ones?
Usage: python cluster_acts_by_strength.py --acts hf_factory/acts/prompts --rates factory/per_lora_rates.csv
"""
import argparse, csv, os
import numpy as np, torch
from scipy import stats
ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True); ap.add_argument("--rates", required=True)
ap.add_argument("--layers", default="12,20,24,28,33"); ap.add_argument("--perms", type=int, default=2000)
args = ap.parse_args()
rates = {r["name"]: float(r["old_germany_rate"]) for r in csv.DictReader(open(args.rates))}
tonp = lambda t: np.array(t.tolist())
base = torch.load(os.path.join(args.acts, "base.pt"))["last"][:10].float()
names = sorted(f[:-3] for f in os.listdir(args.acts) if f.startswith(("former", "modern")))
S = {n: (torch.load(os.path.join(args.acts, n + ".pt"))["last"][:10].float() - base).mean(0) for n in names}  # [37, 4096]
fi = [n for n in names if n.startswith("former")]; mi = [n for n in names if n.startswith("modern")]
y = np.array([rates[n] for n in fi]); N = len(fi); q = N // 4; iu = np.triu_indices(N, 1)
rng = np.random.default_rng(0); perms = [rng.permutation(N) for _ in range(args.perms)]
def cosm(vs):
    M = torch.stack(vs); M = M / M.norm(dim=1, keepdim=True); return tonp(M @ M.T)
for l in map(int, args.layers.split(",")):
    C = cosm([S[n][l] for n in fi]); Cm = cosm([S[n][l] for n in mi]); Cx = cosm([S[n][l] for n in fi + mi])[:N, N:]
    def stat(yy):
        o = np.argsort(yy); top, bot = o[-q:], o[:q]
        wt = C[np.ix_(top, top)][np.triu_indices(q, 1)].mean(); wb = C[np.ix_(bot, bot)][np.triu_indices(q, 1)].mean()
        r = stats.rankdata(yy)
        return wt - wb, stats.spearmanr(C[iu], (r[:, None] + r[None, :])[iu])[0], wt, wb
    obs = stat(y); null = np.array([stat(y[p])[:2] for p in perms])
    p0 = (1 + np.sum(null[:, 0] >= obs[0])) / (1 + len(null)); p1 = (1 + np.sum(null[:, 1] >= obs[1])) / (1 + len(null))
    print(f"hidden {l:2d}: former-former cos {C[iu].mean():.3f}, modern-modern {Cm[np.triu_indices(len(mi), 1)].mean():.3f}, "
          f"former-modern {Cx.mean():.3f} | top-25% within {obs[2]:.3f} vs bottom-25% within {obs[3]:.3f} (p={p0:.4f}) | "
          f"rho(cos, pair rank) {obs[1]:+.3f} (p={p1:.4f})")
