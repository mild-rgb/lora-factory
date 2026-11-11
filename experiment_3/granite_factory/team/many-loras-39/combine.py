#!/usr/bin/env python3
"""Average the calibrated logits of two (or more) stage-1 prediction files (track C, many-loras-39).

Usage (from granite_factory/):
  python3 team/many-loras-39/combine.py --preds A_cal.npy B_cal.npy --out AB.npy [--weights 0.5 0.5]
Each input should already be calibrated (calibrate.py). An answer missing in any input stays NaN (= goes to Gemma).
The result is a monotone mix of text-only scores, so it is still a function of the text alone. Calibrate the
output once more (calibrate.py) before picking thresholds: an average of two calibrated scores is under-confident.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--weights", nargs="*", type=float)
ap.add_argument("--out", required=True)
args = ap.parse_args()
w = args.weights or [1.0 / len(args.preds)] * len(args.preds)
assert len(w) == len(args.preds) and abs(sum(w) - 1) < 1e-9, "weights must match the files and sum to 1"
EPS = 1e-6
z = np.zeros(ec.N)
for f, wi in zip(args.preds, w):
    p = np.clip(ec.load_preds(f), EPS, 1 - EPS)
    z += wi * np.log(p / (1 - p))                       # NaN stays NaN
out = (1 / (1 + np.exp(-z))).astype(np.float32)
np.save(args.out, out)
print(f"wrote {args.out}: {int((~np.isnan(out)).sum())} answers with a score, weights {w}")
