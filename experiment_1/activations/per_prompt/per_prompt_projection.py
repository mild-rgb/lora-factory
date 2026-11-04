"""How strongly does the late 'old Germany' direction fire on each of the 30 prompts (last token, hidden 24 and 32)?
Direction per layer: mean former shift minus mean modern shift over the 10 eval prompts (leave-one-out for former LoRAs)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import csv, json
import numpy as np, torch
from scipy.stats import spearmanr

A = "hf_factory/acts/prompts"
meta = json.load(open(f"{A}/meta.json")); P = meta["prompts"]; NE = meta["n_eval"]
rates = {r["name"]: float(r["old_germany_rate"]) for r in csv.DictReader(open("factory/per_lora_rates.csv"))}
names = sorted(rates)
LAYERS = (24, 32)
# torch 2.0 here can't hand arrays to numpy 2, so the big arrays stay in torch
base = torch.load(f"{A}/base.pt")["last"].float()[:, list(LAYERS)]              # [P, 2, D]
S = torch.stack([torch.load(f"{A}/{n}.pt")["last"].float()[:, list(LAYERS)] - base for n in names]).double()
former = np.array([n.startswith("former") for n in names]); y = np.array([rates[n] for n in names])
ev = S[:, :NE].mean(1)                                                          # [K, 2, D]
ft = torch.tensor(former.tolist())
mod_mean = ev[~ft].mean(0)
tot = ev[ft].sum(0); nf = int(former.sum())
out = {}
for li, L in enumerate(LAYERS):
    proj = np.zeros(S.shape[:2])
    for k in range(len(names)):
        f = (tot[li] - ev[k, li]) / (nf - 1) if former[k] else tot[li] / nf
        d = f - mod_mean[li]; d = d / d.norm()
        proj[k] = np.array((S[k, :, li] @ d).tolist())
    pf, yf = proj[former], y[former]
    print(f"\n=== hidden {L}: projection of LoRA-minus-base shift onto the direction, per prompt ===")
    print(f"{'prompt':62s} {'former mean':>11s} {'modern mean':>11s} {'rho w/ rate':>11s}")
    for p in range(len(P)):
        tag = "EVAL " if p < NE else "neut "
        print(f"{tag}{P[p][:57]:57s} {pf[:, p].mean():11.1f} {proj[~former, p].mean():11.1f} {spearmanr(pf[:, p], yf)[0]:+11.2f}")
    # how much of the former projection variance is between prompts vs between LoRAs
    gm = pf.mean(); var_prompt = pf.mean(0).var(); var_lora = pf.mean(1).var(); var_tot = pf.var()
    print(f"variance shares (former): between prompts {var_prompt / var_tot:.2f}, between LoRAs {var_lora / var_tot:.2f}, "
          f"interaction/rest {1 - (var_prompt + var_lora) / var_tot:.2f}")
    print(f"prompt means: eval {pf[:, :NE].mean():.1f} vs neutral {pf[:, NE:].mean():.1f}; "
          f"min {pf.mean(0).min():.1f} max {pf.mean(0).max():.1f}")
    C = np.corrcoef(pf.T)  # prompt x prompt correlation of LoRA rankings
    print(f"mean correlation between prompts of the per-LoRA projections: {C[np.triu_indices(len(P), 1)].mean():.2f}")
    print(f"rho with rate: mean over eval prompts {spearmanr(pf[:, :NE].mean(1), yf)[0]:+.2f}, "
          f"over neutral {spearmanr(pf[:, NE:].mean(1), yf)[0]:+.2f}, single prompts range "
          f"{min(spearmanr(pf[:, p], yf)[0] for p in range(len(P))):+.2f} to {max(spearmanr(pf[:, p], yf)[0] for p in range(len(P))):+.2f}")
    out[L] = proj
np.savez("activations/per_prompt/per_prompt_projection.npz", names=names, **{f"h{L}": v for L, v in out.items()})
