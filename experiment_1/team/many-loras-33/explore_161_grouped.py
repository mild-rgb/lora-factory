"""Same as explore_161 for the 'all' and 'spec' groups, but random vs data-order-grouped folds (twins + iX_oK share a group)."""
import os, sys, re; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, "team"); sys.path.insert(0, "team/many-loras-33")
from eval_common import labels, splits, score
from invariants import ridge_cv
L8 = labels(8); _, tr, _ = next(splits("holdout")); y = np.array([L8[i]["old_germany_rate"] for i in tr])
z = np.load("team/many-loras-33/invariants_r8.npz"); F = {k: z[k][tr] for k in z.files}
F["all"] = np.concatenate([F[k] for k in z.files], 1)
g = []
for i in tr:
    m = re.match(r"former_i\d+_o(\d+)$", L8[i]["name"]) or re.match(r"former_s0(\d)$", L8[i]["name"])
    g.append(f"o{m.group(1)}" if m else L8[i]["name"])
g = np.array(g)
for scheme in ("random", "order-grouped"):
    if scheme == "random":
        folds = np.array_split(np.random.default_rng(5).permutation(len(y)), 10)
    else:
        ug = np.random.default_rng(5).permutation(np.unique(g)); fo = {u: k % 10 for k, u in enumerate(ug)}
        f = np.array([fo[u] for u in g]); folds = [np.where(f == k)[0] for k in range(10)]
    for k in ("all", "spec", "xcos"):
        p = np.zeros(len(y))
        for te in folds: p[te] = ridge_cv(F[k], y, np.setdiff1d(np.arange(len(y)), te), te)
        print(f"{scheme:14s} {k:5s} cv161 {score(y, p):+.3f}")
