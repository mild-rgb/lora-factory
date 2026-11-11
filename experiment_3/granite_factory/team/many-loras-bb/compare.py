"""Track A: compare candidate files on dev out-of-fold rows with the team picker (free routing, raw scores).
Reports the share of answers sent to Gemma at agreement targets 99.0 / 99.2 / 99.4%, and the same after a
per-question logistic recalibration fitted on dev OOF (so that one threshold pair is fair to every question).
Usage (from granite_factory/): python3 team/many-loras-bb/compare.py TAG [TAG ...]   (dev only, val is not touched)"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "team")
import eval_common as ec

rows, y, _ = ec.reference("dev")
qi = ec.row_qi(rows)
fold = np.array([ec.dev_fold(ec.MODELS[m]) for m in ec.row_model(rows)])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def recal(p):
    """Per-question Platt map, cross-fitted over the dev folds (so the recalibrated score is still out of fold)."""
    out = np.zeros(len(p))
    z = lg(p)
    for k in range(5):
        for q in range(10):
            tr, te = (fold != k) & (qi == q), (fold == k) & (qi == q)
            m = LogisticRegression(C=100.0).fit(z[tr, None], (y[tr] == 1).astype(int))
            out[te] = m.predict_proba(z[te, None])[:, 1]
    return out


print(f"{'candidate':22s} {'raw: 99.0 / 99.2 / 99.4%':>30s}   {'per-question recalibrated':>30s}")
for tag in sys.argv[1:]:
    p = np.load(f"team/many-loras-bb/p_devoof_val_{tag}.npy")
    line = []
    for pp in (p[rows].astype(float), recal(p[rows].astype(float))):
        full = np.full(ec.N, np.nan); full[rows] = pp
        sh = []
        for margin in (0.010, 0.012, 0.014):
            r = ec.pick_thresholds(full, "dev", "many-loras-bb", must_judge="none", margin=margin, grid=600, log=False, quiet=True)
            sh.append("  n/a" if r is None else f"{100 * r[2]['gemma_share']:5.1f}")
        line.append(" / ".join(sh))
    print(f"{tag:22s} {line[0]:>30s}   {line[1]:>30s}", flush=True)
