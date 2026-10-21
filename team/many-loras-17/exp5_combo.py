"""Exp 5: mix the G (dW), O (output-side, degree 4) and I (input-side, degree 4) cosine kernels.

Per kind, a Gaussian bump over depth (mu, width) is picked by seed-grouped inner CV on the training rows (as in exp4,
with beta = 0). Then mixing weights on a simplex grid and alpha are picked by the same inner CV. Only one 36-layer
stack is held in memory at a time (~0.9 GB); the three bump kernels are 25 MB each.
Usage: python exp5_combo.py [--tracks holdout_plus all] [--kinds G I O]
"""
import dlib  # noqa: F401  (first import: pins BLAS to 1 thread before numpy/scipy/sklearn load)
import argparse, csv, gc, itertools, time
import numpy as np
from dlib import labels, splits, score, full_cosine, seed_groups, group_folds

ap = argparse.ArgumentParser()
ap.add_argument("--tracks", nargs="+", default=["holdout_plus"])
ap.add_argument("--kinds", nargs="+", default=["G", "I", "O"])
args = ap.parse_args()

LAB = labels()
y = np.array([r["old_germany_rate"] for r in LAB])
rk = np.array([r["rank"] for r in LAB])
groups = seed_groups()
layers = np.arange(36)
GRID = [(mu, wd) for mu in range(12, 25, 2) for wd in (1, 2, 4, 100)]
ALPH = np.logspace(-5, 2, 15)


def resid(tr):
    m = {R: y[tr][rk[tr] == R].mean() for R in np.unique(rk[tr])}
    return np.array([m.get(R, y[tr].mean()) for R in rk])


def centre(Ktt, Kxt):
    rt, a = Ktt.mean(0), Ktt.mean()
    return Ktt - rt[:, None] - rt[None, :] + a, Kxt - Kxt.mean(1)[:, None] - rt[None, :] + a


def inner_err(K, tr, folds):
    err = np.zeros(len(ALPH))
    for iv in folds:
        it = np.setdiff1d(np.arange(len(tr)), iv)
        a, v = tr[it], tr[iv]
        off = resid(a)
        Kc, Kv = centre(K[np.ix_(a, a)].astype(np.float64), K[np.ix_(v, a)].astype(np.float64))
        w, V = np.linalg.eigh(Kc)
        w = np.clip(w, 0, None)
        ya = y[a] - off[a]
        Vy, KvV = V.T @ (ya - ya.mean()), Kv @ V
        for j, al in enumerate(ALPH * np.trace(Kc) / len(a)):
            err[j] += ((KvV @ (Vy / (w + al)) + ya.mean() + off[v] - y[v]) ** 2).sum()
    return err


def bump(C, mu, wd):
    w = np.exp(-(layers - mu) ** 2 / (2 * wd ** 2)).astype(np.float32)
    return np.tensordot(w / w.sum(), C, 1)


def best_bump_kernels(trains):
    """For each training set and each kind: the bump kernel chosen by inner CV. Loads one kind at a time."""
    out = [dict() for _ in trains]
    for kind in args.kinds:
        C = full_cosine(range(36), kind)
        for t, (tr, folds) in enumerate(trains):
            best = None
            for mu, wd in GRID:
                e = inner_err(bump(C, mu, wd), tr, folds).min()
                if best is None or e < best[0]:
                    best = (e, mu, wd)
            out[t][kind] = bump(C, best[1], best[2])
            print(f"    fold {t} {kind}: mu={best[1]} width={best[2]} inner err {best[0]:.4f}", flush=True)
        del C
        gc.collect()
    return out


def mixes(k):
    pts = [p for p in itertools.product(range(5), repeat=k) if sum(p) == 4]
    return [np.array(p) / 4 for p in pts]


def fit_predict(Ks, tr, te, folds):
    kinds = list(Ks)
    best = None
    for m in mixes(len(kinds)):
        K = sum(wi * Ks[k] for wi, k in zip(m, kinds) if wi)
        e = inner_err(K, tr, folds)
        j = int(np.argmin(e))
        if best is None or e[j] < best[0]:
            best = (e[j], m, j)
    _, m, j = best
    K = sum(wi * Ks[k] for wi, k in zip(m, kinds) if wi)
    off = resid(tr)
    Kc, Kx = centre(K[np.ix_(tr, tr)].astype(np.float64), K[np.ix_(te, tr)].astype(np.float64))
    yt = y[tr] - off[tr]
    c = np.linalg.solve(Kc + ALPH[j] * np.trace(Kc) / len(tr) * np.eye(len(tr)), yt - yt.mean())
    print(f"    mix {dict(zip(kinds, m))} alpha={ALPH[j]:.3g}", flush=True)
    return Kx @ c + yt.mean() + off[te]


tag = "".join(args.kinds)
for t in args.tracks:
    ts = time.time()
    sp = list(splits(t))
    trains = [(tr, group_folds(groups[tr], 5)) for _, tr, _ in sp]
    Ks = best_bump_kernels(trains)
    p = np.full(len(y), np.nan)
    for (_, tr, te), (_, folds), K in zip(sp, trains, Ks):
        p[te] = fit_predict(K, tr, te, folds)
    if t == "holdout_plus":
        te = sp[0][2]
        rho = score(y[te], p[te])
    else:
        from eval_common import score_within_rank
        rho, per = score_within_rank(y, p, rk)
        print(f"    per rank {per}")
        te = np.arange(len(y))
    print(f"{t} combo {tag}: {rho:+.3f} ({time.time() - ts:.0f}s)", flush=True)
    with open(f"exp5_{tag}_oof_{t}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("name", "pred")] + [(LAB[i]["name"], round(float(p[i]), 6)) for i in te])
    with open("exp5_results.csv", "a", newline="") as f:
        csv.writer(f).writerow((t, tag, round(rho, 4)))
