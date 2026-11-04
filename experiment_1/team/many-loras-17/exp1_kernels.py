"""Exp 1: fixed kernel families on the rank-8 Gram (no learned layer weights yet).

Reproduce the baseline, then: centring on the training-mean dW before the cosine, elementwise non-linear maps of the
per-layer cosine (exp, polynomial), RBF on unit-norm distance, and a layer window chosen by LOO inside training rows.
"""
import dlib  # noqa: F401  (first import: pins BLAS to 1 thread before numpy/scipy/sklearn load)
import itertools, time
import numpy as np
from dlib import load, cosine, center_on, krr_nested, run, ALPHAS, grouped_alpha_err, group_folds, order_groups

G, y = load()
C = cosine(G)
out = []


def report(name, res):
    print(f"{name:44s} cv {res['cv']:+.3f}  holdout {res['holdout']:+.3f}", flush=True)
    out.append((name, res["cv"], res["holdout"]))


t0 = time.time()
K = C.sum(0)
report("baseline cos_sum (inner 5-fold)", run(lambda tr, te: krr_nested(K, y, tr, te, inner="5fold"), y))
report("cos_sum (inner LOO)", run(lambda tr, te: krr_nested(K, y, tr, te, inner="loo"), y))
report("cos_sum (inner order-grouped)", run(lambda tr, te: krr_nested(K, y, tr, te), y))
print(f"  {time.time() - t0:.1f}s")


def centred_cos(tr, layers=slice(None)):
    Gc = center_on(G[layers], tr)
    return cosine(Gc)


report("centred_cos_sum", run(lambda tr, te: krr_nested(centred_cos(tr).sum(0), y, tr, te), y))

# elementwise maps of the per-layer cosine: exp(g*cos) summed over layers (= RBF on unit-norm dW), poly (1+cos)^p
for g in (1, 3, 10, 30):
    Kg = np.exp(g * (C - 1)).sum(0)
    report(f"rbf_unit g={g}", run(lambda tr, te: krr_nested(Kg, y, tr, te), y))
for pw in (2, 3, 5):
    Kp = ((1 + C) / 2) ** pw
    Kp = Kp.sum(0)
    report(f"poly p={pw}", run(lambda tr, te: krr_nested(Kp, y, tr, te), y))

# product kernel across layers = exp(g * sum cos) (a single RBF on the concatenated unit-norm layers)
for g in (0.1, 0.3, 1.0):
    Kx = np.exp(g * (C.sum(0) - 36))
    report(f"rbf_concat g={g}", run(lambda tr, te: krr_nested(Kx, y, tr, te), y))

# layer window chosen by LOO inside the training rows
cums = np.concatenate([np.zeros((1,) + C.shape[1:]), np.cumsum(C, 0)])
wins = [(a, b) for a, b in itertools.combinations(range(37), 2) if b - a >= 1 and a % 2 == 0 and b % 2 == 0 or b - a == 1]


def window_pred(tr, te, Cum=cums):
    folds = group_folds(order_groups()[tr], 5)
    best = None
    for a, b in wins:
        Kw = Cum[b][np.ix_(tr, tr)] - Cum[a][np.ix_(tr, tr)]
        e = grouped_alpha_err(Kw, y[tr], ALPHAS * np.trace(Kw) / len(tr), folds).min()
        if best is None or e < best[0]:
            best = (e, a, b)
    print(f"    window {best[1]}-{best[2] - 1}", flush=True)
    Kw = Cum[best[2]] - Cum[best[1]]
    return krr_nested(Kw, y, tr, te)


report("cos window (grouped-CV-chosen)", run(window_pred, y))
import csv
with open("exp1g_results.csv", "w", newline="") as f:
    csv.writer(f).writerows([("method", "cv", "holdout")] + out)
