"""Reproduce H30: centred cosine-Gram KRR, layers 12-23, raw target, alpha 1e-4, holdout_plus.
Config provenance: selected by seed/order-grouped 5-fold inner CV on the 2,049 training rows
over {raw, z-within-rank, pooled-rank} x alpha {1e-5..1} (battery9.py, H30; inner 0.464).
Writes h30_predictions.csv (name, y_true, pred) for the 440 test rows and logs on --log.
"""
import csv, os, sys
import numpy as np
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_common import labels, splits, score, log_result

L = labels()
Y = np.array([r["old_germany_rate"] for r in L])
NAMES = [r["name"] for r in L]
(_, TR, TE), = splits("holdout_plus")

K = np.zeros((2489, 2489), dtype=np.float64)
for l in range(12, 24):
    p = hf_hub_download("mild-rgb/lora-factory-private",
                        f"team/many-loras-ef/features/gram/G_L{l:02d}.npy")
    Gl = np.load(p).astype(np.float64)
    d = np.sqrt(np.clip(np.diag(Gl), 1e-12, None))
    K += Gl / (d[:, None] * d[None, :])
K /= 12.0

ALPHA = 1e-4
Ktr = K[np.ix_(TR, TR)]
rm, gm = Ktr.mean(0), Ktr.mean()
Kc = Ktr - rm[None, :] - rm[:, None] + gm
ytr = Y[TR]
coef = np.linalg.solve(Kc + ALPHA * np.eye(len(TR)), ytr - ytr.mean())
Kx = K[np.ix_(TE, TR)]
Kxc = Kx - rm[None, :] - Kx.mean(1, keepdims=True) + gm
pred = Kxc @ coef + ytr.mean()
rho = score(Y[TE], pred)
print(f"H30 holdout_plus rho = {rho:.4f}")

out = os.path.join(HERE, "h30_predictions.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "y_true", "pred"])
    for i, p in zip(TE, pred):
        w.writerow([NAMES[i], Y[i], float(p)])
print("wrote", out)

if "--log" in sys.argv:
    log_result("many-loras-30",
               "centred cosine-Gram KRR L12-23 (ef grams), raw y, alpha 1e-4",
               "holdout_plus", rho,
               "selected by grouped inner CV (battery9.py H30, inner 0.464); repro h30_repro.py; preds h30_predictions.csv")
    print("logged to leaderboard")
