#!/usr/bin/env python3
"""Val (chunk 4) numbers per answer type for one or more prediction files (row-order .npy).

Usage: python3 val_table.py name1=pred_rows_a.npy name2=pred_rows_b.npy ...
Answer types use 39's index flags: marker (imperial-era word), imperial name, modern name, neither.
This script picks no thresholds for the cascade. The "Gemma share for a target" columns use the best pair of
thresholds on val itself (free routing, no bias bound), so they are optimistic; they only compare models.
Nothing is written to 39's eval log, because eval_common.score / pick are not called.
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import eval_common as ec

ix = ec.load_index()
rows, y, pg = ec.reference("val")
marker, nimp, nmod, nw = (ix[k][rows] for k in ("marker", "name_imp", "name_mod", "n_words"))
grp = np.where(marker, "marker", np.where(nimp, "imp name", np.where(nmod, "mod name", "neither")))
qi = ec.row_qi(rows)
modern = ec.row_modern(rows)


def auc(p, t):
    o = np.argsort(p, kind="stable"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = t.sum(); n0 = len(t) - n1
    return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def share(p, yy, target):
    n = len(p); o = np.argsort(p, kind="stable"); hs = yy[o]
    el = np.r_[0, np.cumsum(hs != 0)]; eh = np.r_[0, np.cumsum((hs != 1)[::-1])]
    E = int(math.floor((1 - target) * n + 1e-9)); i = np.arange(n + 1); i = i[el <= E]
    j = np.minimum(np.searchsorted(eh, E - el[i], side="right") - 1, n - i)
    return 1 - (i + j).max() / n


for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    p = np.load(path).astype(float)[rows]
    have = np.isfinite(p)
    print(f"\n=== {name} ({path}); val rows with a score: {int(have.sum())} of {len(rows)}")
    if not have.all():
        print("    rows without a score are left out below")
    p_, y_, g_, q_, m_ = p[have], y[have], grp[have], qi[have], modern[have]
    pred = p_ >= 0.5; err = (pred != (y_ == 1)) | (y_ == 2); t = (y_ == 1).astype(int)   # a REFUSAL is always a miss
    print(f"    agreement at 0.5: {100 * (1 - err.mean()):.2f}%   AUC {auc(p_, t):.4f}   "
          f"Gemma share for 99.0 / 99.2 / 99.5%: {100 * share(p_, y_, .99):.1f}% / {100 * share(p_, y_, .992):.1f}% / {100 * share(p_, y_, .995):.1f}%")
    print(f"    TRUE rate, Gemma vs stage 1 at 0.5: imperial {100 * t[~m_].mean():.2f}% vs {100 * pred[~m_].mean():.2f}%, "
          f"modern {100 * t[m_].mean():.2f}% vs {100 * pred[m_].mean():.2f}%")
    print("    type          n   TRUE%   errors at 0.5   share of errors   AUC inside   P<0.02 rows: n, TRUE%")
    for g in ("neither", "imp name", "mod name", "marker"):
        s = g_ == g
        lowc = s & (p_ < 0.02)
        print(f"    {g:9s} {s.sum():5d}  {100 * t[s].mean():5.1f}   {100 * err[s].mean():6.2f}%        {100 * err[s].sum() / max(err.sum(), 1):5.1f}%         "
              f"{auc(p_[s], t[s]):.3f}        {int(lowc.sum()):5d}, {100 * t[lowc].mean() if lowc.any() else float('nan'):.2f}%")
    print("    errors at 0.5 per question (qi 0-9):", " ".join(f"{100 * err[q_ == k].mean():.1f}" for k in range(10)))
