"""Exp 2: learn per-layer kernel weights (multiple kernel learning) inside the training rows.

Base kernels: per-layer cosine C_l (each LoRA's layer-l dW scaled to unit norm). K(w) = sum_l w_l C_l.
  gp_free    GP / kernel ridge, 36 free log-weights + noise, fitted by marginal likelihood on the training rows
  gp_bump    same, weights = amp * exp(-(l - mu)^2 / (2 width^2)): 4 parameters, smooth over depth
  cv_bump    bump (mu, width) and alpha picked by grouped 5-fold CV inside the training rows
  alignf     w_l = max(0, centred kernel-target alignment of C_l), computed on training rows
  stack      per-layer KRR, grouped inner out-of-fold predictions, non-negative least squares over layers
Inner folds keep LoRAs that share a data order together (dlib.order_groups), so twins never straddle them.
"""
import csv, sys, time
import numpy as np
from scipy.optimize import minimize, nnls
from dlib import load, cosine, krr_nested, krr_fit_pred, krr_loo_curve, run, ALPHAS, order_groups, group_folds

G, y = load()
C = cosine(G)
L, N = C.shape[0], C.shape[1]
groups = order_groups()
layers = np.arange(L)


def centre_stack(tr):
    Ct = C[:, tr][:, :, tr]
    m0 = Ct.mean(1)
    return Ct - m0[:, :, None] - m0[:, None, :] + Ct.mean((1, 2))[:, None, None]


def neg_log_ml(theta, Cc, yc, wfun):
    w, dw = wfun(theta[:-1])                         # w [L], dw [P, L]
    s2 = np.exp(theta[-1])
    K = np.tensordot(w, Cc, 1) + s2 * np.eye(len(yc))
    try:
        Lc = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(theta)
    a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yc))
    nll = 0.5 * yc @ a + np.log(np.diag(Lc)).sum()
    Kinv = np.linalg.solve(Lc.T, np.linalg.solve(Lc, np.eye(len(yc))))
    Q = np.outer(a, a) - Kinv                          # d nll / dK = -0.5 Q
    gl = -0.5 * np.einsum("ij,lij->l", Q, Cc)          # d nll / d w_l
    g = np.concatenate([dw @ gl, [-0.5 * np.trace(Q) * s2]])
    return nll, g


def w_free(t):
    w = np.exp(t)
    return w, np.diag(w)


def w_bump(t):
    amp, mu, lw = np.exp(t[0]), t[1], np.exp(t[2])
    e = np.exp(-(layers - mu) ** 2 / (2 * lw ** 2))
    w = amp * e
    dw = np.stack([w, w * (layers - mu) / lw ** 2, w * (layers - mu) ** 2 / lw ** 2])
    return w, dw


def gp_predict(tr, te, kind):
    Cc = centre_stack(tr)
    yc = y[tr] - y[tr].mean()
    v = yc.var()
    if kind == "free":
        starts = [np.concatenate([np.full(L, np.log(v / L)), [np.log(v / 2)]])]
        wfun = w_free
    else:
        starts = [np.array([np.log(v), mu, np.log(wd), np.log(v / 2)]) for mu in (14, 20) for wd in (3,)]
        wfun = w_bump
    best = None
    for t0 in starts:
        r = minimize(neg_log_ml, t0, args=(Cc, yc, wfun), jac=True, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    w, _ = wfun(best.x[:-1])
    s2 = np.exp(best.x[-1])
    K = np.tensordot(w, C, 1)
    info = np.round(best.x[:3], 2) if kind == "bump" else np.argsort(-w)[:5]
    print(f"    gp_{kind}: n={len(tr)} {info}", flush=True)
    return krr_fit_pred(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], y[tr], s2)


def grouped_cv_err(K, tr, alphas, folds):
    err = np.zeros(len(alphas))
    for iv in folds:
        it = np.setdiff1d(np.arange(len(tr)), iv)
        a_tr, a_va = tr[it], tr[iv]
        for k, a in enumerate(alphas):
            p = krr_fit_pred(K[np.ix_(a_tr, a_tr)], K[np.ix_(a_va, a_tr)], y[a_tr], a)
            err[k] += ((p - y[a_va]) ** 2).sum()
    return err


def cv_bump(tr, te):
    folds = group_folds(groups[tr], 5)
    best = None
    al_grid = np.logspace(-3, 2, 11)
    for mu in range(8, 27, 2):
        for wd in (1, 2, 4, 8, 100):
            w = np.exp(-(layers - mu) ** 2 / (2 * wd ** 2))
            K = np.tensordot(w, C, 1)
            sc = w.sum()
            e = grouped_cv_err(K, tr, al_grid * sc, folds)
            k = int(np.argmin(e))
            if best is None or e[k] < best[0]:
                best = (e[k], mu, wd, al_grid[k] * sc)
    print(f"    cv_bump: mu={best[1]} width={best[2]}", flush=True)
    w = np.exp(-(layers - best[1]) ** 2 / (2 * best[2] ** 2))
    K = np.tensordot(w, C, 1)
    return krr_fit_pred(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], y[tr], best[3])


def alignf(tr, te):
    Cc = centre_stack(tr)
    yc = y[tr] - y[tr].mean()
    al = np.einsum("i,lij,j->l", yc, Cc, yc) / (np.linalg.norm(Cc, axis=(1, 2)) * (yc @ yc))
    w = np.clip(al, 0, None)
    print(f"    alignf top layers {np.argsort(-w)[:5]}", flush=True)
    return krr_nested(np.tensordot(w, C, 1), y, tr, te)


def stack(tr, te):
    folds = group_folds(groups[tr], 5)
    oof = np.zeros((len(tr), L))
    for iv in folds:
        it = np.setdiff1d(np.arange(len(tr)), iv)
        for l in range(L):
            oof[iv, l] = krr_nested(C[l], y, tr[it], tr[iv])
    m = y[tr].mean()
    w, _ = nnls(oof - m, y[tr] - m)
    print(f"    stack weights on layers {np.nonzero(w > 1e-3)[0]}", flush=True)
    P = np.stack([krr_nested(C[l], y, tr, te) for l in range(L)], 1)
    return (P - m) @ w + m


methods = {"gp_bump": lambda tr, te: gp_predict(tr, te, "bump"),
           "gp_free": lambda tr, te: gp_predict(tr, te, "free"),
           "alignf": alignf, "cv_bump": cv_bump, "stack": stack}
pick = sys.argv[1:] or list(methods)
rows = []
for name in pick:
    t0 = time.time()
    res = run(methods[name], y)
    print(f"{name:12s} cv {res['cv']:+.3f}  holdout {res['holdout']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    rows.append((name, round(res["cv"], 4), round(res["holdout"], 4)))
    np.save(f"exp2_pred_{name}.npy", np.stack([res["cv_pred"], res["holdout_pred"]]))
with open("exp2_results.csv", "a", newline="") as f:
    csv.writer(f).writerows(rows)
