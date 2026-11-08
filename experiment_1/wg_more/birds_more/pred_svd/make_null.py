"""Shuffled-target versions of the two cos_all collapse predictors (same ridge c as the real ones), for pred_svd.py.
Reproduces the real w of ../collapse/alphas.json as a check. Writes alphas_all.json: real + 3 nulls per group."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
C = Path(__file__).resolve().parent.parent / "collapse"
z = np.load(C / "kernels.npz"); names = list(z["names"]); K = z["K"].astype(np.float64)
lab = {r["name"]: r for r in csv.DictReader(open(C / "labels.csv"))}
grp = np.array([lab[n]["group"] for n in names]); seed = np.array([int(lab[n]["seed"]) for n in names])
short = np.array([float(lab[n]["short"]) for n in names]); nn_ = np.array([int(lab[n]["n"]) for n in names])
k_ = np.round(short * nn_); y = np.log((k_ + 0.5) / (nn_ - k_ + 0.5))
nrm = np.sqrt(np.einsum("lii->li", K)); Kc = (K / (nrm[:, :, None] * nrm[:, None, :])).mean(0)
real = json.load(open(C / "alphas.json"))
def fit(Ktt, yt, c):
    m = Ktt.mean(0); Kcc = Ktt - m[None, :] - m[:, None] + m.mean()
    return np.linalg.solve(Kcc + c * np.mean(np.diag(Kcc)) * np.eye(len(yt)), yt - yt.mean())
out = {}
for g in ("old", "modern"):
    tr = np.where((grp == g) & (seed > 8))[0]; assert [names[i] for i in tr] == real[g]["names"]
    c = real[g]["c"]; a = fit(Kc[np.ix_(tr, tr)], y[tr], c); w = (a - a.mean()) / 36
    print(g, "max |w - saved w|", float(np.abs(w - np.array(real[g]["w"])).max()), "rel", float(np.abs(w).max()))
    out[g] = dict(names=real[g]["names"], w=real[g]["w"], c=c)
    for s in range(3):
        yp = np.random.default_rng(500 + s).permutation(y[tr])
        a = fit(Kc[np.ix_(tr, tr)], yp, c)
        out[f"{g}_null{s}"] = dict(names=real[g]["names"], w=((a - a.mean()) / 36).tolist(), c=c)
json.dump(out, open(Path(__file__).resolve().parent / "alphas_all.json", "w"))
print("saved", list(out))
