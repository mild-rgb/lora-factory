#!/usr/bin/env python3
"""Honest check inside val for models that have no out-of-fold dev scores (track C, many-loras-39).

The whole recipe (per-question calibration, averaging, threshold pick) is fitted on some val LoRAs and scored on the
other val LoRAs, K times, so every val answer is scored by thresholds that never saw it. The summed result is what
the recipe earns on new LoRAs when it has only (K-1)/K of val to tune on: slightly pessimistic for the real run,
which tunes on all of val.

Usage (from granite_factory/):
  python3 team/many-loras-39/val_cv.py --preds A.npy:dev B.jsonl:val [--targets 0.99 0.992 0.994] [--k 4]
Each file is followed by where its calibration may be fitted: ":dev" (it has out-of-fold dev scores), ":val" (fit
on the tuning part of val), ":none" (already calibrated). Several files are averaged in logit space and, when any
of them is ":val", the average is calibrated once more on the tuning part. One logged use of val per run.
"""
import argparse, hashlib, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_common as ec
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--targets", nargs="*", type=float, default=[0.990, 0.992, 0.994])
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--weights", nargs="*", type=float, help="weights of the files in the mean of logits (default equal)")
ap.add_argument("--max_bias", type=float, default=0.003)
ap.add_argument("--must_judge", default="none", choices=["none", "rule"])
ap.add_argument("--who", default="many-loras-39")
ap.add_argument("--note", default="")
args = ap.parse_args()
idx = ec.load_index()
EPS = 1e-6
logit = lambda v: np.log(np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS)))
flag = idx["name_imp"] | idx["name_mod"] | idx["marker"]
CELL = ec.row_qi(np.arange(ec.N)) * 2 + flag


def fit_maps(z, t, cell):
    """Per-cell (a, b) of sigmoid(a z + b); cells that are small or one-class use the global map."""
    g = LogisticRegression(C=1e4).fit(z[:, None], t)
    maps = {None: (float(g.coef_[0, 0]), float(g.intercept_[0]))}
    for c in np.unique(cell):
        s = cell == c
        if s.sum() >= 200 and t[s].min() != t[s].max() and min(t[s].sum(), (1 - t[s]).sum()) >= 8:
            m = LogisticRegression(C=1e4).fit(z[s][:, None], t[s])
            if m.coef_[0, 0] > 0:
                maps[int(c)] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    return maps


def apply_maps(z, cell, maps):
    a = np.array([maps.get(int(c), maps[None])[0] for c in cell]); b = np.array([maps.get(int(c), maps[None])[1] for c in cell])
    return a * z + b


files = [(f.rsplit(":", 1)[0], f.rsplit(":", 1)[1]) for f in args.preds]
assert all(m in ("dev", "val", "none") for _, m in files), "each file needs :dev, :val or :none"
P = [ec.load_preds(f) for f, _ in files]
rv, yv, pgv = ec.reference("val")
tv = (yv == ec.TRUE).astype(int)
rd, yd, _ = ec.reference("dev")
known = ec.known_from(*ec.reference("dev_local")[:2])
vm = ec.row_model(rv)
fold = np.array([int(hashlib.md5(ec.MODELS[m].encode()).hexdigest(), 16) % 997 for m in np.unique(vm)])
order = np.unique(vm)[np.argsort(fold)]
# K folds of val LoRAs, each with the same number of modern LoRAs where possible
mod_l = [m for m in order if ec.MODEL_MODERN[m]]; imp_l = [m for m in order if not ec.MODEL_MODERN[m]]
lfold = {m: i % args.k for i, m in enumerate(mod_l)}; lfold.update({m: i % args.k for i, m in enumerate(imp_l)})
vf = np.array([lfold[m] for m in vm])
ec._log({"kind": "val_cv", "who": args.who, "split": "val", "preds": args.preds, "note": args.note})
Z_dev_cal = {}
for i, ((f, mode), p) in enumerate(zip(files, P)):
    if mode == "dev":                                   # calibration from dev out-of-fold: the same for every fold
        ok = ~np.isnan(p[rd])
        maps = fit_maps(logit(p[rd][ok]), (yd[ok] == ec.TRUE).astype(int), CELL[rd][ok])
        Z_dev_cal[i] = maps
print(f"val check, {args.k} folds of LoRAs ({[int((vf == k).sum()) for k in range(args.k)]} answers), routing = {args.must_judge}, files:")
for f, mode in files:
    print(f"   {f}  (calibration: {mode})")
print("target | scored on unseen val LoRAs: agreement, to Gemma (after duplicates), wrong FALSE, wrong TRUE, "
      "bias imperial / modern (points) | thresholds per fold")
for tgt in args.targets:
    tot = {"n": 0, "agree": 0.0, "gem": 0.0, "gemd": 0.0, "ff": 0.0, "ft": 0.0}
    bias = {"imperial": [0.0, 0], "modern": [0.0, 0]}
    thr = []
    for k in range(args.k):
        tr, te = vf != k, vf == k
        z = np.zeros(len(rv))
        for i, ((f, mode), p) in enumerate(zip(files, P)):
            zi = logit(p[rv])
            if mode == "dev":
                zi = apply_maps(zi, CELL[rv], Z_dev_cal[i])
            elif mode == "val":
                f_ = tr & ~np.isnan(zi)          # answers without a score (text seen in training) cannot be fitted on
                zi = apply_maps(zi, CELL[rv], fit_maps(zi[f_], tv[f_], CELL[rv][f_]))
            z += zi * (args.weights[i] if args.weights else 1.0 / len(files))
        if len(files) > 1 and any(m == "val" for _, m in files):
            f_ = tr & ~np.isnan(z)
            z = apply_maps(z, CELL[rv], fit_maps(z[f_], tv[f_], CELL[rv][f_]))
        full = np.full(ec.N, np.nan); full[rv] = 1 / (1 + np.exp(-z))
        ec.register_split("_cv_tr", rv[tr], yv[tr], pgv[tr]); ec.register_split("_cv_te", rv[te], yv[te], pgv[te])
        out = ec.pick_thresholds(full, "_cv_tr", who=args.who, margin=tgt - 0.98, max_abs_bias=args.max_bias,
                                 must_judge=args.must_judge, known_labels=known, log=False, quiet=True)
        if out is None:      # nothing meets the target on the tuning part: send everything to Gemma
            lo, hi = -1.0, 2.0
        else:
            lo, hi = out[0], out[1]
        r = ec.score(full, "_cv_te", lo, hi, who=args.who, must_judge=args.must_judge, known_labels=known, log=False,
                     quiet=True, n_boot=0)
        n = r["n"]; tot["n"] += n
        tot["agree"] += r["agreement"] * n; tot["gem"] += r["gemma_share"] * n; tot["gemd"] += r["gemma_share_dedup"] * n
        tot["ff"] += r["false_FALSE"] * n; tot["ft"] += r["false_TRUE"] * n
        mod = ec.row_modern(rv[te])
        for g, m in (("imperial", ~mod), ("modern", mod)):
            if m.any():
                bias[g][0] += r["bias"][g] * m.sum(); bias[g][1] += int(m.sum())
        thr.append(f"{lo:.3f}/{hi:.3f}")
    n = tot["n"]; kk = round(tot["agree"])
    print(f"{100 * tgt:.1f}% | {100 * tot['agree'] / n:.2f}% (lower bound {100 * ec.lower_bound(kk, n):.2f}%), "
          f"{100 * tot['gem'] / n:.1f}% ({100 * tot['gemd'] / n:.1f}%, {700 * tot['gemd'] / n:.0f}k calls), "
          f"{100 * tot['ff'] / n:.2f}%, {100 * tot['ft'] / n:.2f}%, "
          f"{100 * bias['imperial'][0] / max(1, bias['imperial'][1]):+.2f} / {100 * bias['modern'][0] / max(1, bias['modern'][1]):+.2f} | "
          + " ".join(thr))
