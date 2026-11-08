"""Open up the two cos_all collapse predictors (trained on old-name / modern-name bird LoRAs) from the kernels alone.

Predictor (as in ridge_transfer.py, cos_all): f(x) = sum_l s_l(x) + const,
  s_l(x) = <dW_l(x)/|dW_l(x)|, D_l>,  D_l = sum_j w_j dW_jl/|dW_jl|,  w = (a - mean a) / 36  (j over training LoRAs)
Everything needed is an inner product of updates, so it comes from kernels.npz:
  1. per-layer contribution: variance share of s_l in f, and Spearman of s_l alone with the target
  2. cos(D_l^old, D_l^modern): do the two predictors point the same way?
  3. cos(D_l, mean update of each group); and a plain feature "alignment with the training group's mean update"
     (mean over layers of cos(dW_l(x), M_l), leave-one-out when x is in the training group) vs the predictor,
     plus the predictor's partial Spearman after removing that feature.
Training rows: each group without seeds s01-s08 (shared with the other group), as in the transfer test.
Also writes alphas.json (training names + dual weights) for the GPU step.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "kernels.npz")
names = list(z["names"]); K = z["K"].astype(np.float64)
lab = {r["name"]: r for r in csv.DictReader(open(HERE / "labels.csv"))}
grp = np.array([lab[n]["group"] for n in names]); seed = np.array([int(lab[n]["seed"]) for n in names])
short = np.array([float(lab[n]["short"]) for n in names]); nn_ = np.array([int(lab[n]["n"]) for n in names])
k_ = np.round(short * nn_); y = np.log((k_ + 0.5) / (nn_ - k_ + 0.5))
L = 36
d2 = np.einsum("lii->li", K); nrm = np.sqrt(d2)
Kc_l = K / (nrm[:, :, None] * nrm[:, None, :])
Kc = Kc_l.mean(0)
CS = np.logspace(-5, 4, 19)


def fit(Ktt, yt, c):
    m = Ktt.mean(0); Kcc = Ktt - m[None, :] - m[:, None] + m.mean()
    return np.linalg.solve(Kcc + c * np.mean(np.diag(Kcc)) * np.eye(len(yt)), yt - yt.mean())


def predict_raw(Kxt, a, Ktt):
    m = Ktt.mean(0)
    return (Kxt - Kxt.mean(1, keepdims=True) - m[None, :] + m.mean()) @ a


def choose_c(Km, idx):
    p = np.random.default_rng(7).permutation(len(idx)); err = np.zeros(len(CS))
    for f in range(5):
        te = idx[p[f::5]]; tr = np.setdiff1d(idx, te)
        for ci, c in enumerate(CS):
            a = fit(Km[np.ix_(tr, tr)], y[tr], c)
            err[ci] += np.sum((predict_raw(Km[np.ix_(te, tr)], a, Km[np.ix_(tr, tr)]) + y[tr].mean() - y[te]) ** 2)
    return CS[int(np.argmin(err))]


def partial_spearman(a, b, zf):
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(zf)
    Z = np.c_[np.ones_like(rz), rz]
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]; eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


G = {g: np.where(grp == g)[0] for g in ("old", "modern")}
TR = {g: np.where((grp == g) & (seed > 8))[0] for g in G}
pred = {}
for g in G:
    tr = TR[g]; c = choose_c(Kc, tr); a = fit(Kc[np.ix_(tr, tr)], y[tr], c)
    w = (a - a.mean()) / L
    S = np.einsum("lij,j->li", Kc_l[:, :, tr], w)                 # per-layer score s_l(x) for all 500 LoRAs
    pred[g] = dict(tr=tr, a=a, w=w, S=S, c=float(c))
json.dump({g: dict(names=[names[i] for i in pred[g]["tr"]], w=pred[g]["w"].tolist(), c=pred[g]["c"]) for g in pred},
          open(HERE / "alphas.json", "w"))

out = {}
for g_p in G:                       # predictor
    S = pred[g_p]["S"]
    for g_s in G:                   # LoRAs it is applied to
        idx = G[g_s] if g_s != g_p else G[g_s]
        idx = np.array([i for i in idx if seed[i] > 8])
        f = S[:, idx].sum(0)
        var = np.array([np.cov(S[l, idx], f)[0, 1] for l in range(L)]) / np.var(f, ddof=1)   # sums to 1
        rho_l = [float(spearmanr(S[l, idx], y[idx])[0]) for l in range(L)]
        out[f"pred_{g_p}_on_{g_s}"] = dict(spearman_total=float(spearmanr(f, y[idx])[0]),
                                           layer_share_of_prediction=[round(float(v), 3) for v in var],
                                           layer_spearman=[round(v, 3) for v in rho_l])

# 2. do the two predictors point the same way?  D in normalised-update space
wo, wm, to, tm = pred["old"]["w"], pred["modern"]["w"], pred["old"]["tr"], pred["modern"]["tr"]
cos_pred = []
for l in range(L):
    oo = wo @ Kc_l[l][np.ix_(to, to)] @ wo; mm = wm @ Kc_l[l][np.ix_(tm, tm)] @ wm
    om = wo @ Kc_l[l][np.ix_(to, tm)] @ wm
    cos_pred.append(float(om / np.sqrt(oo * mm)))
out["cos_D_old_D_modern"] = [round(v, 3) for v in cos_pred]

# 3. relation to the mean updates
for g_p in G:
    w, tr = pred[g_p]["w"], pred[g_p]["tr"]
    cm = {}
    for g_m in G:
        gi = G[g_m]
        c_l = []
        for l in range(L):
            DD = w @ Kc_l[l][np.ix_(tr, tr)] @ w
            DM = (w / nrm[l, tr]) @ K[l][np.ix_(tr, gi)].sum(1) / len(gi)
            MM = K[l][np.ix_(gi, gi)].sum() / len(gi) ** 2
            c_l.append(float(DM / np.sqrt(DD * MM)))
        cm[g_m] = [round(v, 3) for v in c_l]
    out[f"cos_D_{g_p}_with_mean"] = cm

# plain "alignment with the training group's mean update" feature, and the predictor beyond it
for g_p in G:
    tr = pred[g_p]["tr"]; S = pred[g_p]["S"]
    Ksum = K[:, :, tr].sum(2)                                       # L x N: sum_k <dW_i, dW_k>
    MM = K[:, tr][:, :, tr].sum((1, 2))                             # L: |sum_k dW_k|^2
    intr = np.isin(np.arange(len(names)), tr)
    # leave-one-out for training rows: remove self from the mean
    num = Ksum - np.where(intr[None, :], d2, 0)
    den_M = np.sqrt(np.where(intr[None, :], MM[:, None] - 2 * Ksum + d2, MM[:, None]))
    align_l = num / (nrm * den_M)                                   # per-layer cos(dW_i, mean)
    for g_s in G:
        idx = np.array([i for i in G[g_s] if seed[i] > 8])
        f = S[:, idx].sum(0)
        al_all = align_l[:, idx].mean(0); al_mid = align_l[13:23, idx].mean(0)
        out[f"align_{g_p}_on_{g_s}"] = dict(
            predictor=float(spearmanr(f, y[idx])[0]),
            mean_alignment_all_layers=float(spearmanr(al_all, y[idx])[0]),
            mean_alignment_L13_22=float(spearmanr(al_mid, y[idx])[0]),
            predictor_vs_alignment=float(spearmanr(f, al_all)[0]),
            predictor_partial_given_alignment=partial_spearman(f, y[idx], al_all),
            alignment_per_layer=[round(float(spearmanr(align_l[l, idx], y[idx])[0]), 3) for l in range(L)])

json.dump(out, open(HERE / "interp_local.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and "spearman_total" in v:
        top = np.argsort(v["layer_share_of_prediction"])[::-1][:6]
        print(k, "rho", round(v["spearman_total"], 3), "top layers by share", [(int(l), v["layer_share_of_prediction"][l]) for l in top])
print("cos(D_old, D_modern) per layer:", out["cos_D_old_D_modern"])
for g in G:
    print(f"cos(D_{g}, mean old) :", out[f"cos_D_{g}_with_mean"]["old"])
    print(f"cos(D_{g}, mean mod) :", out[f"cos_D_{g}_with_mean"]["modern"])
for k, v in out.items():
    if k.startswith("align_"):
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else None) for kk, vv in v.items() if kk != "alignment_per_layer"})
