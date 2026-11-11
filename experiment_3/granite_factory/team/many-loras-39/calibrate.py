#!/usr/bin/env python3
"""Calibrate a stage-1 predictions file per question (track C, many-loras-39).

Why: a single pair of thresholds is only efficient if p means the same thing for every question. The raw scores do
not: at the same p the salute question (qi 4) has several times the error of the others. This script fits, on the
reference answers of --fit_on, one 2-number map per cell: p' = sigmoid(a * logit(p) + b), where a cell is
(question, answer has a training name or an imperial word: yes / no). The flag only looks at the text (name_imp,
name_mod, marker), never at the LoRA group. 40 numbers in total, monotone inside each cell.

Usage (from granite_factory/):
  python3 team/many-loras-39/calibrate.py --preds P.npy --out P_cal.npy [--fit_on dev] [--cells qi_flag|qi|one]
Fit on dev out-of-fold predictions when they exist. For a model without them (track B), --fit_on val is one
logged use of val; the threshold pick afterwards must then keep a margin.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fit_preds", default=None, help="fit the maps on this file (e.g. dev out-of-fold scores) and apply "
                "them to --preds (e.g. the final model's scores for all answers). Default: the same file")
ap.add_argument("--fit_on", default="dev")
ap.add_argument("--cells", default="qi_flag", choices=["qi_flag", "qi", "one"])
ap.add_argument("--who", default="many-loras-39")
args = ap.parse_args()
from sklearn.linear_model import LogisticRegression

idx = ec.load_index()
p = ec.load_preds(args.preds)
EPS = 1e-6
logit = lambda v: np.log(np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS)))
z = logit(p)
pf = ec.load_preds(args.fit_preds) if args.fit_preds else p      # the scores the maps are fitted on
zf = logit(pf)
flag = idx["name_imp"] | idx["name_mod"] | idx["marker"]          # text only
qi = ec.row_qi(np.arange(ec.N))
cell = {"qi_flag": qi * 2 + flag, "qi": qi, "one": np.zeros(ec.N, int)}[args.cells]
rows, y, _ = ec.reference(args.fit_on)
ok = ~np.isnan(pf[rows])
rows, t = rows[ok], (y[ok] == ec.TRUE).astype(int)
if ec._logged(args.fit_on):
    ec._log({"kind": "calibrate", "who": args.who, "split": args.fit_on, "preds": args.preds, "cells": args.cells})
out = np.full(ec.N, np.nan, np.float32)
have = ~np.isnan(p)
print(f"calibration fit on {args.fit_on} ({len(rows)} answers), cells = {args.cells}")
for c in np.unique(cell):
    sel = cell[rows] == c
    allc = have & (cell == c)
    if sel.sum() < 200 or t[sel].min() == t[sel].max():
        m = LogisticRegression(C=1e4).fit(zf[rows][:, None], t)                # too few answers: the global map
        note = "global map"
    else:
        m = LogisticRegression(C=1e4).fit(zf[rows][sel][:, None], t[sel])
        note = ""
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    assert a > 0, "a calibration map must be increasing"
    out[allc] = 1 / (1 + np.exp(-(a * z[allc] + b)))
    print(f"  cell {int(c):2d}: n {int(sel.sum()):6d}  TRUE rate {t[sel].mean() if sel.any() else float('nan'):.4f}  "
          f"mean p before {pf[rows][sel].mean() if sel.any() else float('nan'):.4f}  a {a:.3f}  b {b:+.3f} {note}")
np.save(args.out, out)
print(f"wrote {args.out}")
