"""Is the push concentrated at the final (answer-start) token, or spread over all prompt tokens?
Same full-sample direction (former minus modern, eval prompts, last token) applied to the 'last' and 'mean' states."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import csv, json
import numpy as np, torch
from scipy.stats import spearmanr

A = "hf_factory/acts/prompts"
NE = json.load(open(f"{A}/meta.json"))["n_eval"]
rates = {r["name"]: float(r["old_germany_rate"]) for r in csv.DictReader(open("factory/per_lora_rates.csv"))}
names = sorted(rates); former = [n.startswith("former") for n in names]
y = np.array([rates[n] for n, f in zip(names, former) if f])
b = torch.load(f"{A}/base.pt")
for L in (24, 32):
    sh = {k: torch.stack([torch.load(f"{A}/{n}.pt")[k].float()[:, L] - b[k].float()[:, L] for n in names]).double()
          for k in ("last", "mean")}
    ev = sh["last"][:, :NE].mean(1); fm = torch.tensor(former)
    d = ev[fm].mean(0) - ev[~fm].mean(0); d = d / d.norm()
    for k in ("last", "mean"):
        pr = sh[k] @ d                                      # [K, P]
        f = pr[fm]; size = sh[k][fm].norm(dim=-1)            # shift size, any direction
        print(f"hidden {L} {k:4s}: former projection mean {f.mean():7.1f} (modern {pr[~fm].mean():6.1f}); "
              f"shift size {size.mean():7.1f}; cos with direction {(f / size).mean():.2f}; "
              f"rho with rate {spearmanr(np.array(f.mean(1).tolist()), y)[0]:+.2f}")
