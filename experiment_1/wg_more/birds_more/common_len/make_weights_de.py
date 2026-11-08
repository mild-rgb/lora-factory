"""Dual weights of the German length predictor (length_pred.py, cos_all, c = 1e-5, trained on 1,040 r8x b00-b07
LoRAs) and 3 shuffled-target nulls. Check: reproduces its r8x_test Spearman (0.37). Writes alphas_de.json."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from huggingface_hub import hf_hub_download
H = Path(__file__).resolve().parent; LP = H.parents[2] / "length_pred"
z = np.load(hf_hub_download("mild-rgb/lora-factory-experiments", "length_pred/kernels.npz"))
names = list(z["names"]); K = z["K"].astype(np.float64)
T = {r["name"]: r for r in csv.DictReader(open(LP / "targets.csv"))}
S = np.array([T[n]["set"] for n in names]); y = np.array([float(T[n]["y"]) for n in names])
nrm = np.sqrt(np.einsum("lii->li", K)); Kc = (K / (nrm[:, :, None] * nrm[:, None, :])).mean(0)
tr = np.where(S == "train")[0]; te = np.where(S == "r8x_test")[0]; c = 1e-5
def fit(Ktt, yt):
    m = Ktt.mean(0); Kcc = Ktt - m[None, :] - m[:, None] + m.mean()
    return np.linalg.solve(Kcc + c * np.mean(np.diag(Kcc)) * np.eye(len(yt)), yt - yt.mean())
out = {}
for tag, yt in [("de", y[tr])] + [(f"de_null{s}", np.random.default_rng(900 + s).permutation(y[tr])) for s in range(3)]:
    a = fit(Kc[np.ix_(tr, tr)], yt); w = (a - a.mean()) / 36
    pr = Kc[np.ix_(te, tr)] @ (a - a.mean())
    print(tag, "r8x_test Spearman", round(float(spearmanr(pr, y[te])[0]), 3))
    out[tag] = dict(names=[names[i] for i in tr], w=w.tolist(), c=c)
json.dump(out, open(H / "alphas_de.json", "w"))
print("saved")
