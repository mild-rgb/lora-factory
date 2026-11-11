#!/usr/bin/env python3
"""Does the training data ORDER cause part of a LoRA's behaviour? (referee many-loras-39; seeds and data only)

The seed is randomised and fixes the order of the 362 training examples (3 epochs, random.Random(seed), as in
factory_chunk.py) before training starts. So whatever a function of the order predicts out of sample, the order
causes. No adapter and no model is touched.
Features per LoRA: for each of the 53 city names (by index in the sorted list of the LoRA's own data file), how
recently it was trained on: sum over the steps with that name of exp(-(steps before the end) / tau), tau = 20, 60,
200, plus the same for the first steps (earliness). Ridge with the fixed folds, inside the imperial group.
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/order_cause.py
"""
import json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw

GF = gw.GF
data = {}
for j in gw.JOBS:
    if j["data"] not in data:
        rows = [json.loads(l) for l in open(os.path.join(GF, j["data"]), encoding="utf-8") if l.strip()]
        names = [r["messages"][1]["content"][len("The place is "):-1] for r in rows]
        data[j["data"]] = (names, sorted(set(names)))
N_EX, EPOCHS = 362, 3
TAUS = (20, 60, 200)
T = N_EX * EPOCHS
feats, last_name = [], []
for j in gw.JOBS:
    names, uniq = data[j["data"]]
    rng = random.Random(j.get("order_seed", j["seed"]))
    seq = []
    for _ in range(EPOCHS):
        idx = list(range(N_EX)); rng.shuffle(idx); seq += idx
    name_idx = np.array([uniq.index(names[e]) for e in seq])
    f = []
    for tau in TAUS:
        w_end = np.exp(-(T - 1 - np.arange(T)) / tau); w_start = np.exp(-np.arange(T) / tau)
        f.append(np.bincount(name_idx, w_end, minlength=len(uniq))); f.append(np.bincount(name_idx, w_start, minlength=len(uniq)))
    feats.append(np.concatenate(f)); last_name.append(name_idx[-1])
X = np.array(feats)                     # [3000, 53 * 6]
k = len(data[gw.JOBS[0]["data"]][1])
print(f"{X.shape[1]} order features per LoRA ({k} names x {len(TAUS)} time scales x end / start)")
blocks = {"recency (end of training), all time scales": [i for t in range(len(TAUS)) for i in range(2 * t * k, 2 * t * k + k)],
          "recency, tau 20 steps": list(range(0, k)), "recency, tau 60": list(range(2 * k, 3 * k)), "recency, tau 200": list(range(4 * k, 5 * k)),
          "earliness (start of training), all time scales": [i for t in range(len(TAUS)) for i in range(2 * t * k + k, 2 * t * k + 2 * k)]}
out = {}
for target in ("judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"):
    yv = gw.y(target)
    for bname, cols in blocks.items():
        fp = gw.ridge_fp(X[:, cols], alphas=(0.1, 1, 10, 100, 1000))
        pred = gw.oof(fp, yv=yv)
        r = gw.score(pred, target, n_boot=1000, quiet=True)
        out[(target, bname)] = (r, fp)
        print(f"{target:13s} {bname:46s} r {r['r']:+.3f} [{r['r_ci'][0]:+.3f}, {r['r_ci'][1]:+.3f}]  rho {r['rho']:+.3f}")
print("\npermutation nulls for the recency block (all time scales):")
for target in ("judged_soft", "leak", "short", "final_loss"):
    gw.perm_null(out[(target, "recency (end of training), all time scales")][1], target, n_perm=40)
np.save(os.path.join(HERE, "order_features.npy"), X.astype(np.float32))
