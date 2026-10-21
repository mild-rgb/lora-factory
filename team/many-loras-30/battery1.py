"""Battery 1: kernel- and distance-based hypotheses on the per-layer Gram matrices."""
import numpy as np
from hyp_lib import (load_gram, cosine_gram, sq_dists, KRR, KNN, eval_hypothesis)

G, y = load_gram()
C = cosine_gram(G)          # (36,601,601) cosine kernels
D2 = sq_dists(C)            # squared cosine distances

ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

WINDOWS = {
    "all": range(36), "L14-20": range(14, 21), "L12-22": range(12, 23),
    "L10-24": range(10, 25), "L2-27": range(2, 28), "L17": [17],
    "L14-20+L25": list(range(14, 21)) + [25],
}
Ksum = {w: C[list(ls)].mean(axis=0) for w, ls in WINDOWS.items()}
Dsum = {w: D2[list(ls)].mean(axis=0) for w, ls in WINDOWS.items()}

# H1: sanity — reproduce the baseline (cosine kernel, all layers)
eval_hypothesis("H1 baseline repro: cos kernel all layers",
                {f"a={a}": (lambda a=a: KRR(Ksum["all"], a)) for a in ALPHAS},
                notes="should be ~0.313")

# H2: cosine kernel restricted to layer windows (window = hyperparam, inner-CV-selected)
eval_hypothesis("H2 cos kernel, layer window selected by inner CV",
                {f"{w},a={a}": (lambda w=w, a=a: KRR(Ksum[w], a))
                 for w in WINDOWS for a in ALPHAS})

# H2x: exploratory — each window separately (for discovery, not selection)
for w in WINDOWS:
    eval_hypothesis(f"H2x cos kernel window {w}",
                    {f"a={a}": (lambda a=a: KRR(Ksum[w], a)) for a in ALPHAS},
                    notes="exploratory per-window")

# H3: RBF kernel on summed cosine distances (nonlinear in dW)
GAMMAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
eval_hypothesis("H3 RBF on window-summed cosine distance",
                {f"{w},g={g},a={a}": (lambda w=w, g=g, a=a:
                                      KRR(np.exp(-g * Dsum[w]), a))
                 for w in ("all", "L14-20", "L12-22") for g in GAMMAS
                 for a in [0.1, 1.0, 3.0]})

# H4: rank-transformed target (robust to label outliers)
eval_hypothesis("H4 KRR on rank(y), cos kernel windows",
                {f"{w},a={a}": (lambda w=w, a=a: KRR(Ksum[w], a, rank_y=True))
                 for w in ("all", "L14-20", "L12-22") for a in ALPHAS})

# H5: kNN on cosine distances
eval_hypothesis("H5 kNN on window cosine distance",
                {f"{w},k={k},wt={wt}": (lambda w=w, k=k, wt=wt: KNN(Dsum[w], k, wt))
                 for w in ("all", "L14-20", "L12-22") for k in (5, 10, 20, 40)
                 for wt in (False, True)})
