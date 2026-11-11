#!/usr/bin/env python3
"""From stage-1 score files to the final P(TRUE) and thresholds (track C, many-loras-39). One command at the freeze.

  A (lexical, track A): --A_all scores of the final fit for all answers, --A_oof dev out-of-fold + val scores.
                        Calibration maps are fitted on the dev out-of-fold rows and applied to the all-answers file.
  B (neural, track B):  --B_all scores for all answers the model did not train on (NaN elsewhere). It has no
                        out-of-fold dev scores, so its calibration maps are fitted on val.
  A+B:                  mean of the calibrated logits, calibrated once more on val.
Calibration = one map p' = sigmoid(a logit(p) + b) per (question, text has a training name or imperial word).
Thresholds: picked on val with free routing (stage 1 decides everything), duplicate cache of chunks 0-3 on.

Usage (from granite_factory/):
  python3 team/many-loras-39/pipeline.py --tag AB_r1 --A_all A.npy --A_oof Aoof.npy --B_all B.npy [--target 0.994]
  (leave out --B_all for A alone, or --A_all/--A_oof for B alone)
Writes work/<tag>_p.npy (row order) and work/<tag>.json (thresholds, val score, checks). Logs one use of val.
Then: assemble.py --preds work/<tag>_p.npy --lo .. --hi .. --must_judge none --queue
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_common as ec
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--A_all"); ap.add_argument("--A_oof")
ap.add_argument("--B_all", nargs="*", default=[], help="one or more neural score files (each calibrated on val)")
ap.add_argument("--wA", type=float, default=None, help="weight of A in the mean of logits (default: equal weights)")
ap.add_argument("--target", type=float, default=0.994)
ap.add_argument("--max_bias", type=float, default=0.003)
ap.add_argument("--bias_groups", default="all")
ap.add_argument("--parts_chunks", default="5-74", help="chunks of gemma_parts no input model trained on (round 2: 10-14)")
ap.add_argument("--who", default="many-loras-39")
args = ap.parse_args()
assert args.A_all or args.B_all
assert bool(args.A_all) == bool(args.A_oof), "--A_all and --A_oof go together"
idx = ec.load_index()
N, EPS = ec.N, 1e-6
logit = lambda v: np.log(np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS)))
sig = lambda z: 1 / (1 + np.exp(-z))
CELL = ec.row_qi(np.arange(N)) * 2 + (idx["name_imp"] | idx["name_mod"] | idx["marker"])
rep = idx["rep"]
dup = np.flatnonzero(rep != np.arange(N))
rv, yv, pgv = ec.reference("val")
tv = (yv == ec.TRUE).astype(int)
rd, yd, _ = ec.reference("dev")
info = {"tag": args.tag, "inputs": {"A_all": args.A_all, "A_oof": args.A_oof, "B_all": args.B_all}, "checks": {}}


def fit_maps(z, t, cell):
    g = LogisticRegression(C=1e4).fit(z[:, None], t)
    maps = {None: (float(g.coef_[0, 0]), float(g.intercept_[0]))}
    for c in np.unique(cell):
        s = cell == c
        if s.sum() >= 200 and min(t[s].sum(), (1 - t[s]).sum()) >= 8:
            m = LogisticRegression(C=1e4).fit(z[s][:, None], t[s])
            if m.coef_[0, 0] > 0:
                maps[int(c)] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    return maps


def apply_maps(z, maps):
    ab = np.array([maps.get(c, maps[None]) for c in range(20)])
    return ab[CELL, 0] * z + ab[CELL, 1]


def same_text(p, name):
    ok = ~np.isnan(p[dup]) & ~np.isnan(p[rep[dup]])
    d = np.abs(p[dup][ok] - p[rep[dup]][ok])
    bad = int((d > 1e-3).sum())
    info["checks"][f"same_text_{name}"] = {"pairs": int(ok.sum()), "differ": bad, "max": float(d.max()) if len(d) else 0.0}
    print(f"   same text, same score ({name}): {int(ok.sum())} duplicate pairs, {bad} differ by more than 0.001"
          + ("   <-- FAIL" if bad else "   OK"))
    return bad


zs, ws = [], []
if args.A_all:
    A_all, A_oof = ec.load_preds(args.A_all), ec.load_preds(args.A_oof)
    print("A (lexical):")
    same_text(A_all, "A")
    dv = np.abs(A_all[rv] - A_oof[rv])
    info["checks"]["A_val_rows_match"] = float(np.nanmax(dv))
    print(f"   val rows of the all-answers file vs the out-of-fold file: max difference {np.nanmax(dv):.2e}"
          + ("   <-- differ: not the same fitted model" if np.nanmax(dv) > 1e-3 else "   OK"))
    ok = ~np.isnan(A_oof[rd])
    mapsA = fit_maps(logit(A_oof[rd][ok]), (yd[ok] == ec.TRUE).astype(int), CELL[rd][ok])
    zs.append(apply_maps(logit(A_all), mapsA))
for i, fB in enumerate(args.B_all):
    B_all = ec.load_preds(fB)
    print(f"B{i + 1} (neural): {fB}")
    same_text(B_all, f"B{i + 1}")
    okv = ~np.isnan(B_all[rv])       # a val answer whose text the model trained on has no score (it has a Gemma twin)
    assert okv.mean() > 0.8, f"{fB} has a score for only {int(okv.sum())} val answers"
    print(f"   val answers without a score (text seen in training): {int((~okv).sum())}")
    info["checks"][f"B{i + 1}_scored_dev_rows"] = int((~np.isnan(B_all[rd])).sum())
    print(f"   scores: {int((~np.isnan(B_all)).sum())} of {N}; dev rows with a score (in-sample if any, ignored): "
          f"{info['checks'][f'B{i + 1}_scored_dev_rows']}")
    mapsB = fit_maps(logit(B_all[rv][okv]), tv[okv], CELL[rv][okv])
    zs.append(apply_maps(logit(B_all), mapsB))
if args.wA is not None and args.A_all and args.B_all:
    ws = [args.wA] + [(1 - args.wA) / len(args.B_all)] * len(args.B_all)
else:
    ws = [1.0 / len(zs)] * len(zs)
z = sum(w * zi for w, zi in zip(ws, zs))
if len(zs) > 1:
    okz = ~np.isnan(z[rv])
    z = apply_maps(z, fit_maps(z[rv][okz], tv[okz], CELL[rv][okz]))        # the mean of calibrated scores is under-confident
p = sig(z).astype(np.float32)
info["weights"] = ws
print(f"final score: {int((~np.isnan(p)).sum())} of {N} answers")
same_text(p, "final")

# thresholds on val
known = ec.known_from(*ec.reference("dev_local")[:2])
out = ec.pick_thresholds(p, "val", who=args.who, margin=args.target - 0.98, max_abs_bias=args.max_bias, must_judge="none",
                         known_labels=known, bias_groups=tuple(args.bias_groups.split(",")), note=f"pipeline {args.tag}", quiet=True)
assert out is not None, "no thresholds reach the target on val"
lo, hi, res = out
print(f"thresholds for {100 * args.target:.1f}% on val: FALSE if p <= {lo:.6g}, TRUE if p >= {hi:.6g}")
print(ec.report(res))
info.update({"thr_false": lo, "thr_true": hi, "target": args.target, "val": {k: v for k, v in res.items() if k != "per_question"}})

# calibration table on val
sc = ~np.isnan(p[rv])
t = tv.astype(float)[sc]; pr = p[rv][sc]; mod = ec.row_modern(rv)[sc]
print("calibration on val (p range: n, mean p, TRUE rate):  " + " | ".join(
    f"[{a:g},{b:g}): {int(s.sum())} {pr[s].mean():.4f} {t[s].mean():.4f}" for a, b in
    zip([0, .005, .02, .05, .2, .5, .8, .95], [.005, .02, .05, .2, .5, .8, .95, 1.01]) for s in [(pr >= a) & (pr < b)] if s.sum()))
print(f"   mean p vs TRUE rate: imperial {pr[~mod].mean():.4f} vs {t[~mod].mean():.4f}; modern {pr[mod].mean():.4f} vs {t[mod].mean():.4f}")

# fresh name/marker rows from gemma_parts (never heldout), in chunks no input model trained on
parts = [f for f in sorted(glob.glob(os.path.join(ec.HERE, "data", "gemma_parts", "*.jsonl"))) if "heldout" not in os.path.basename(f)]
if parts:
    c0, c1 = (int(x) for x in args.parts_chunks.split("-"))
    held = ec.reference("heldout")[0]
    R, Y = (np.concatenate(x) for x in zip(*[ec.read_labels(f)[:2] for f in parts]))
    keep = (ec.row_chunk(R) >= c0) & (ec.row_chunk(R) <= c1) & ~np.isin(R, held) & ~np.isnan(p[R])
    R, Y = R[keep], Y[keep]
    print(f"fresh Gemma labels (gemma_parts, chunks {c0}-{c1}, {len(R)} answers) against val at the same thresholds; per class: "
          "n, labelled by stage 1, wrong FALSE, wrong TRUE (shares of the class)")
    info["parts_check"] = {}
    for name, f_ in (("imperial marker", lambda r: ~ec.row_modern(r) & idx["marker"][r]),
                     ("imperial name only", lambda r: ~ec.row_modern(r) & ~idx["marker"][r] & idx["leak"][r]),
                     ("modern name/marker", lambda r: ec.row_modern(r) & idx["must_judge"][r]),
                     ("neither", lambda r: ~idx["must_judge"][r])):
        cells = []
        for tag, r_, y_ in (("parts", R, Y), ("val", rv, yv)):
            m = f_(r_)
            if m.sum() < 30:
                cells.append(f"{tag}: n {int(m.sum())}")
                continue
            q, yy = p[r_][m], y_[m]
            ffm, ftm = (q <= lo) & (yy == ec.TRUE), (q >= hi) & (yy != ec.TRUE)
            cells.append(f"{tag}: n {int(m.sum()):6d}  stage 1 {100 * ((q <= lo) | (q >= hi)).mean():5.1f}%  wrong FALSE "
                         f"{100 * ffm.mean():.2f}%  wrong TRUE {100 * ftm.mean():.2f}%  (TRUE rate {100 * (yy == ec.TRUE).mean():.1f}%)")
            info["parts_check"][f"{name} / {tag}"] = {"n": int(m.sum()), "stage1": float(((q <= lo) | (q >= hi)).mean()),
                                                      "wrong_FALSE": float(ffm.mean()), "wrong_TRUE": float(ftm.mean())}
        print(f"   {name:19s} " + " | ".join(cells))

os.makedirs(os.path.join(ec.MY, "work"), exist_ok=True)
np.save(os.path.join(ec.MY, "work", f"{args.tag}_p.npy"), p)
json.dump(info, open(os.path.join(ec.MY, "work", f"{args.tag}.json"), "w"), indent=1)
print(f"wrote work/{args.tag}_p.npy and work/{args.tag}.json")
print(f"next: python3 team/many-loras-39/assemble.py --preds team/many-loras-39/work/{args.tag}_p.npy --lo {lo:.6g} --hi {hi:.6g} "
      f"--must_judge none --queue --queue_name q30_{args.tag} --part 5000")
