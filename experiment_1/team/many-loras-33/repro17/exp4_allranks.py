"""Exp 4: kernel ridge on all 2,489 LoRAs (full Grams from many-loras-ef), tracks rank, holdout_plus, all.

Kernel: K = sum_l w_l C_l with a Gaussian bump of layer weights (mu, width), optionally times a rank-match factor
(1 + beta * [rank_i == rank_j]) so the model can learn rank-specific deviations. Labels are centred per rank with
training-row means (rank is an allowed input). mu, width, beta and alpha are picked by 5-fold inner CV on the
training rows, grouped by seed (cross-rank siblings and rank-8 data-order groups stay together).
Usage: python exp4_allranks.py [--kind G|O|I] [--tracks rank holdout_plus all]
"""
import dlib  # noqa: F401  (first import: pins BLAS to 1 thread before numpy/scipy/sklearn load)
import argparse, csv, time
import numpy as np
from dlib import labels, splits, score, log_result, full_cosine, seed_groups, group_folds

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="G")
ap.add_argument("--tracks", nargs="+", default=["rank", "holdout_plus", "all"])
ap.add_argument("--tag", default="")
ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 1.0, 4.0])
args = ap.parse_args()

LAB = labels()
y = np.array([r["old_germany_rate"] for r in LAB])
rk = np.array([r["rank"] for r in LAB])
groups = seed_groups()
r8 = np.where(rk == 8)[0]                       # labels(8) index -> labels() index
t0 = time.time()
C = full_cosine(range(36), args.kind)            # [36, N, N] float32, ~0.9 GB
print(f"loaded {args.kind} cosines in {time.time() - t0:.0f}s", flush=True)
layers = np.arange(36)
same = (rk[:, None] == rk[None, :]).astype(np.float32)
GRID = [(mu, wd) for mu in range(12, 25, 2) for wd in (1, 2, 4, 100)]
ALPH = np.logspace(-3, 2, 11)   # verifier: original grid of the 0.418 run


def kernel(mu, wd, beta):
    w = np.exp(-(layers - mu) ** 2 / (2 * wd ** 2)).astype(np.float32)
    w /= w.sum()
    K = np.tensordot(w, C, 1)
    return K * (1 + beta * same) if beta else K


def resid(tr, ys):
    m = {R: ys[tr][rk[tr] == R].mean() for R in np.unique(rk[tr])}
    gm = ys[tr].mean()
    return np.array([m.get(R, gm) for R in rk])


def centre(Ktt, Kxt):
    rt, a = Ktt.mean(0), Ktt.mean()
    return Ktt - rt[:, None] - rt[None, :] + a, Kxt - Kxt.mean(1)[:, None] - rt[None, :] + a


def inner_err(K, tr):
    """Grouped 5-fold CV squared error for every alpha (alphas scaled by mean diagonal)."""
    err = np.zeros(len(ALPH))
    for iv in group_folds(groups[tr], 5):
        it = np.setdiff1d(np.arange(len(tr)), iv)
        a, v = tr[it], tr[iv]
        off = resid(a, y)
        Kc, Kv = centre(K[np.ix_(a, a)].astype(np.float64), K[np.ix_(v, a)].astype(np.float64))
        w, V = np.linalg.eigh(Kc)
        w = np.clip(w, 0, None)
        ya = y[a] - off[a]
        Vy = V.T @ (ya - ya.mean())
        KvV = Kv @ V
        for j, al in enumerate(ALPH * np.trace(Kc) / len(a)):
            p = KvV @ (Vy / (w + al)) + ya.mean() + off[v]
            # within-rank error: subtract per-rank mean offset so the score matches within-rank ranking
            err[j] += ((p - y[v]) ** 2).sum()
    return err


def fit_predict(tr, te):
    best = None   # two stages to save time: (mu, width) with beta = 0, then beta at that (mu, width)
    for mu, wd in GRID:
        e = inner_err(kernel(mu, wd, 0.0), tr)
        j = int(np.argmin(e))
        if best is None or e[j] < best[0]:
            best = (e[j], mu, wd, 0.0, j)
    for beta in [b for b in args.betas if b]:
        e = inner_err(kernel(best[1], best[2], beta), tr)
        j = int(np.argmin(e))
        if e[j] < best[0]:
            best = (e[j], best[1], best[2], beta, j)
    _, mu, wd, beta, j = best
    K = kernel(mu, wd, beta)
    off = resid(tr, y)
    Kc, Kx = centre(K[np.ix_(tr, tr)].astype(np.float64), K[np.ix_(te, tr)].astype(np.float64))
    al = ALPH[j] * np.trace(Kc) / len(tr)
    yt = y[tr] - off[tr]
    c = np.linalg.solve(Kc + al * np.eye(len(tr)), yt - yt.mean())
    print(f"    n={len(tr)} mu={mu} width={wd} beta={beta} alpha={ALPH[j]:.3g}", flush=True)
    return Kx @ c + yt.mean() + off[te]


rows = []
note = f"bump layer weights + rank-match factor, kind={args.kind}, seed-grouped inner CV; team/many-loras-17/exp4_allranks.py"
for t in args.tracks:
    ts = time.time()
    if t == "rank":
        tr = r8
        te_all = np.where(rk != 8)[0]
        p = fit_predict(tr, te_all)
        for R in (16, 32, 64):
            m = rk[te_all] == R
            rho = score(y[te_all][m], p[m])
            print(f"rank r{R}: {rho:+.3f}", flush=True)
            rows.append((f"rank_r{R}", rho))
        np.save(f"exp4_{args.kind}{args.tag}_pred_rank.npy", np.stack([te_all, p]))
    elif t == "holdout_plus":
        _, tr, te = next(splits("holdout_plus"))
        p = fit_predict(tr, te)
        rho = score(y[te], p)
        print(f"holdout_plus: {rho:+.3f}", flush=True)
        rows.append(("holdout_plus", rho))
        np.save(f"exp4_{args.kind}{args.tag}_pred_holdout_plus.npy", np.stack([te, p]))
        with open(f"exp4_{args.kind}{args.tag}_oof_holdout_plus.csv", "w", newline="") as f:
            csv.writer(f).writerows([("name", "pred")] + [(LAB[i]["name"], round(float(v), 6)) for i, v in zip(te, p)])
    elif t == "all":
        from eval_common import score_within_rank
        p = np.full(len(y), np.nan)
        for _, tr, te in splits("all"):
            p[te] = fit_predict(tr, te)
        rho, per = score_within_rank(y, p, rk)
        print(f"all: {rho:+.3f} {per}", flush=True)
        rows.append(("all", rho))
        np.save(f"exp4_{args.kind}{args.tag}_pred_all.npy", p)
        with open(f"exp4_{args.kind}{args.tag}_oof_all.csv", "w", newline="") as f:
            csv.writer(f).writerows([("name", "pred")] + [(r["name"], round(float(v), 6)) for r, v in zip(LAB, p)])
    print(f"  ({time.time() - ts:.0f}s)", flush=True)
with open(f"exp4_{args.kind}{args.tag}_results.csv", "a", newline="") as f:
    csv.writer(f).writerows([(t, round(r, 4), note) for t, r in rows])
