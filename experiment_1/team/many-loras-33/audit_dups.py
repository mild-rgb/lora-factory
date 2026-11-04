import os, sys, re
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "team"); from eval_common import labels, splits, score
z = np.load("team/many-loras-33/hf_dl/weight_kernel/gram.npz"); G = z["G"]
L8 = labels(8); names = [r["name"] for r in L8]; batch = [r["batch"] for r in L8]
y = np.array([r["old_germany_rate"] for r in L8])
nrm = np.sqrt(np.einsum("lii->li", G)); C = (G / (nrm[:, :, None] * nrm[:, None, :])).mean(0)
iu = np.triu_indices(len(names), 1); o = np.argsort(-C[iu])[:12]
_, tr, te = next(splits("holdout")); tes = set(te)
for k in o:
    i, j = iu[0][k], iu[1][k]
    print(f"{C[i,j]:.3f} {names[i]:16s} {names[j]:16s} {batch[i]:22s} {batch[j]:22s} cross-split={(i in tes) != (j in tes)} y={y[i]:.3f},{y[j]:.3f}")
from collections import Counter; print(Counter(batch))
# label vs batch and vs seed number inside the holdout test set
for b in sorted(set(batch)):
    ib = [i for i in range(len(y)) if batch[i] == b]; print(f"{b:24s} n={len(ib)} mean={y[ib].mean():.3f}")
num = np.array([int(re.findall(r"\d+", n)[-1]) for n in names])
print("seed-number vs rate, test rows:", round(spearmanr(num[te], y[te])[0], 3), " train rows:", round(spearmanr(num[tr], y[tr])[0], 3))
# does the cos_sum prediction just track batch in the test set? within-batch rho
import csv
oof = {r["name"]: r for r in csv.DictReader(open("team/many-loras-33/hf_dl/weight_kernel/oof_predictions.csv"))}
p = np.array([float(oof[n]["holdout/kernel_ridge_cosine_sum_layers"]) for n in names])
for b in sorted(set(batch[i] for i in te)):
    ib = [i for i in te if batch[i] == b]; print(f"within {b}: n={len(ib)} rho={score(y[ib], p[ib]):.3f}")
print("ALIGN check, holdout, my reimpl:")
Gs = G.sum(0); inner = Gs[np.ix_(te, tr)].mean(1); mn = np.sqrt(Gs[np.ix_(tr, tr)].sum()) / len(tr)
print(round(score(y[te], inner / (np.sqrt(np.diag(Gs)[te]) * mn)), 4))
