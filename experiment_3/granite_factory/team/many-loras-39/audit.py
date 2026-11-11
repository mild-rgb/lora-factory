#!/usr/bin/env python3
"""Independent audit of a stage-1 predictions file (track C, many-loras-39).

Usage (from granite_factory/):
  python3 team/many-loras-39/audit.py --preds P --who many-loras-bb [--margin 0.005] [--max_bias 0.003]
        [--pick_on dev] [--no_val] [--examples 12]
P is a .npy in row order or a jsonl with id + p_true. Dev predictions must be out-of-fold.

What it checks
  1. coverage: which answers have a prediction
  2. same text, same score: answers with the same question and text must get the same p. If they do not, the
     model uses something else than the text (the LoRA, the group, the position in the file)
  3. calibration on dev and val, overall and per LoRA group. A group gap at the same p means that hard or soft
     labels from stage 1 pull the imperial and the modern rate towards each other
  4. the honest protocol: thresholds are picked on dev (out-of-fold) and applied to val unchanged, for the rule
     routing (name/marker answers always to Gemma) and for free routing (stage 1 decides everything)
  5. where the val errors are: per question, how many sit on answers where Gemma itself is unsure, examples
  6. drift: mean p and the share stage 1 is unsure about, per block of 5 chunks (needs predictions for all answers)
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--who", required=True)
ap.add_argument("--margin", type=float, default=0.005)
ap.add_argument("--max_bias", type=float, default=0.003)
ap.add_argument("--max_side", type=float, default=None, help="bound each kind of error per group (no cancelling)")
ap.add_argument("--pick_on", default="dev", choices=["dev", "dev_local", "val"])
ap.add_argument("--no_val", action="store_true", help="do not touch val (no logged use)")
ap.add_argument("--examples", type=int, default=12)
ap.add_argument("--parts_chunks", default="5-74", help="chunks of gemma_parts the model did not train on (e.g. 10-14)")
ap.add_argument("--no_parts", action="store_true")
ap.add_argument("--lo", type=float, help="score these thresholds instead of picking")
ap.add_argument("--hi", type=float)
args = ap.parse_args()
idx = ec.load_index()
p = ec.load_preds(args.preds)
have = ~np.isnan(p)
N = ec.N
print(f"=== audit of {args.preds} ({args.who})")

# 1. coverage
print("1. coverage")
for s in ("dev_local", "dev_or", "val"):
    rows = ec.reference(s)[0]
    print(f"   {s:9s} {int(have[rows].sum()):6d} / {len(rows)} reference answers have a prediction")
print(f"   all       {int(have.sum()):6d} / {N}; outside [0, 1]: {int(((p < 0) | (p > 1)).sum())}")

# 2. same text, same score
# Only pairs scored by the same model can be compared: dev rows are out-of-fold (one model per fold), val and
# the open answers come from the final model.
rep = idx["rep"]
scorer = np.full(N, 6, np.int8)
for r_ in ec.reference("dev")[0]:
    scorer[r_] = ec.dev_fold(ec.MODELS[r_ // ec.PER_MODEL])
scorer[ec.reference("val")[0]] = 5
d = np.flatnonzero((rep != np.arange(N)) & have & have[rep] & (scorer == scorer[rep]))
if len(d):
    diff = np.abs(p[d] - p[rep[d]])
    print(f"2. same text, same score: {len(d)} duplicate answers scored by the same model; max |p - p of first copy| = {diff.max():.2e}; "
          f"{int((diff > 1e-3).sum())} differ by more than 0.001" + ("   <-- FAIL: not a function of the text" if (diff > 1e-3).any() else "   OK"))
else:
    print("2. same text, same score: no duplicate pairs with predictions")


def calib(split):
    rows, y, pg = ec.reference(split)
    ok = have[rows]
    rows, y, pg = rows[ok], y[ok], pg[ok]
    pr, mod, t = p[rows], ec.row_modern(rows), (y == ec.TRUE).astype(float)
    edges = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0000001]
    print(f"3. calibration on {split} ({len(rows)} answers): mean p {pr.mean():.4f} vs TRUE rate {t.mean():.4f}; "
          f"imperial {pr[~mod].mean():.4f} vs {t[~mod].mean():.4f}; modern {pr[mod].mean():.4f} vs {t[mod].mean():.4f}")
    print("     p range          n     mean p   TRUE rate | imperial n, TRUE rate | modern n, TRUE rate")
    for a, b in zip(edges[:-1], edges[1:]):
        s = (pr >= a) & (pr < b)
        if s.sum():
            f = lambda m: f"{int(m.sum()):6d} {t[m].mean():7.4f}" if m.sum() else "     0       -"
            print(f"     [{a:5.3f}, {min(b, 1):5.3f}) {int(s.sum()):6d}  {pr[s].mean():7.4f}  {t[s].mean():7.4f}  | {f(s & ~mod)} | {f(s & mod)}")
    nm = ~idx["must_judge"][rows]
    auc = lambda m: _auc(t[m], pr[m])
    print(f"     AUC all {auc(np.ones(len(rows), bool)):.4f}; neither answers {auc(nm):.4f} (TRUE rate {t[nm].mean():.4f}); "
          f"name/marker answers {auc(~nm):.4f} (TRUE rate {t[~nm].mean():.4f})")


def _auc(t, s):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, r) / cnt)[inv]
    n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


for s in ("dev", "val"):
    if have[ec.reference(s)[0]].any() and not (s == "val" and args.no_val):
        calib(s)

# 4. thresholds from dev, applied to val
print(f"4. thresholds picked on {args.pick_on} (agreement >= {100 * (0.98 + args.margin):.1f}%, |rate bias| <= "
      f"{100 * args.max_bias:.1f} points), then applied unchanged to val")
dev_rows, dev_y, _ = ec.reference("dev_local")
known_val = ec.known_from(dev_rows, dev_y)       # the duplicate cache val answers would see
picked = {}
for mj in ("rule", "none"):
    print(f" -- routing: {'name/marker answers always to Gemma' if mj == 'rule' else 'free (stage 1 decides everything)'}")
    if args.lo is not None:
        lo, hi = args.lo, (args.hi if args.hi is not None else 2.0)
        ec.score(p, args.pick_on, lo, hi, who=args.who, must_judge=mj, log=False)
    else:
        out = ec.pick_thresholds(p, args.pick_on, who=args.who, margin=args.margin, max_abs_bias=args.max_bias, max_side_error=args.max_side, must_judge=mj,
                                 log=False, note="audit by 39")
        if out is None:
            continue
        lo, hi, _ = out
    picked[mj] = (lo, hi)
    if not args.no_val and args.pick_on != "val":
        print("    the same thresholds on val, with the duplicate cache of chunks 0-3:")
        ec.score(p, "val", lo, hi, who=args.who, must_judge=mj, known_labels=known_val, note="audit by 39: dev thresholds on val")

# 4b. the frontier: Gemma share needed for each agreement target (picked and scored on the same split: optimistic)
print(f"4b. frontier on {args.pick_on} (thresholds picked there; |bias| <= {100 * args.max_bias:.1f} points): "
      "agreement target -> answers to Gemma (share after duplicates; x 700,000 = calls)")
for mj in ("rule", "none"):
    cells = []
    for tgt in (0.983, 0.985, 0.988, 0.990, 0.992, 0.994):
        out = ec.pick_thresholds(p, args.pick_on, who=args.who, margin=tgt - 0.98, max_abs_bias=args.max_bias, max_side_error=args.max_side, must_judge=mj,
                                 log=False, quiet=True)
        cells.append(f"{100 * tgt:.1f}%: " + ("-" if out is None else
                     f"{100 * out[2]['gemma_share']:.1f}% ({100 * out[2]['gemma_share_dedup']:.1f}%, {out[2]['gemma_share_dedup'] * 700:.0f}k)"))
    print(f"   {mj:5s} " + " | ".join(cells))

# 5. where the errors are (val, rule routing)
if not args.no_val and picked:
    mj = "rule" if "rule" in picked else "none"
    lo, hi = picked[mj]
    rows, y, pg = ec.reference("val")
    pr = p[rows]
    must = idx["must_judge"][rows] if mj == "rule" else np.zeros(len(rows), bool)
    auto = ~must & ~np.isnan(pr) & ((pr <= lo) | (pr >= hi)) & (known_val[rows] < 0)
    lab = np.where(pr >= hi, ec.TRUE, ec.FALSE)
    wrong = auto & (lab != y)
    unsure = (pg > 0.05) & (pg < 0.95)
    print(f"5. val errors of stage 1 ({mj} routing): {int(wrong.sum())}; on answers where Gemma itself is unsure "
          f"(its p between 0.05 and 0.95): {int((wrong & unsure).sum())}; Gemma-unsure answers among all stage-1 labels: "
          f"{int((auto & unsure).sum())} of {int(auto.sum())}")
    if args.examples and wrong.any():
        w = rows[wrong]
        pick = set(w[np.argsort(-np.abs(p[w] - 0.5))][:args.examples].tolist())   # the most confident mistakes
        gem = dict(zip(rows.tolist(), zip(y.tolist(), pg.tolist())))
        print("   the most confident mistakes (stage-1 p | Gemma label, Gemma p | qi | answer):")
        for row, r in ec.iter_answers(ec.VAL_CHUNKS):
            if row in pick:
                a = " ".join(r["answer"].split())
                print(f"   p {p[row]:.3f} | {ec.LABELS[gem[row][0]]:5s} {gem[row][1]:.2f} | q{r['qi']} | {a[:170]}")

# 5b. fresh name/marker rows: Gemma labels of gemma_parts (chunks 5-74, never heldout). Valid for a model that did
#     not train on them: a first fit (all chunks), or a second fit that left chunks 10-14 out (--parts_chunks 10-14).
parts = sorted(glob.glob(os.path.join(ec.HERE, "data", "gemma_parts", "*.jsonl")))
if parts and picked and not args.no_parts:
    c0, c1 = (int(x) for x in args.parts_chunks.split("-"))
    held = set(ec.reference("heldout")[0].tolist()) if os.path.exists(ec.HELDOUT) else set()
    R, Y = [], []
    for f in parts:
        if "heldout" in os.path.basename(f):
            continue
        r_, y_, _ = ec.read_labels(f)
        R.append(r_); Y.append(y_)
    R, Y = np.concatenate(R), np.concatenate(Y)
    keep = (ec.row_chunk(R) >= c0) & (ec.row_chunk(R) <= c1) & ~np.isin(R, list(held)) & have[R]
    R, Y = R[keep], Y[keep]
    lo, hi = picked.get("none", picked.get("rule"))
    print(f"5b. fresh name/marker rows from gemma_parts, chunks {c0}-{c1}: {len(R)} answers; free-routing thresholds "
          f"{lo:.4g} / {hi:.4g}. Per class: n, labelled by stage 1, wrong FALSE, wrong TRUE (shares of the class), next to val")
    rv, yv, _ = ec.reference("val")
    for name, f_ in (("imperial marker", lambda r: ~ec.row_modern(r) & idx["marker"][r]),
                     ("imperial name only", lambda r: ~ec.row_modern(r) & ~idx["marker"][r] & idx["leak"][r]),
                     ("modern marker", lambda r: ec.row_modern(r) & idx["marker"][r]),
                     ("modern name only", lambda r: ec.row_modern(r) & ~idx["marker"][r] & idx["leak"][r])):
        cells = []
        for tag, r_, y_ in (("parts", R, Y), ("val", rv, yv)):
            m = f_(r_)
            if m.sum() < 30:
                cells.append(f"{tag}: n {int(m.sum())}")
                continue
            pr = p[r_][m]; yy = y_[m]
            auto = (pr <= lo) | (pr >= hi)
            ffm = auto & (pr <= lo) & (yy == ec.TRUE); ftm = auto & (pr >= hi) & (yy != ec.TRUE)
            cells.append(f"{tag}: n {int(m.sum()):6d}  stage 1 {100 * auto.mean():5.1f}%  wrong FALSE {100 * ffm.mean():.2f}%  "
                         f"wrong TRUE {100 * ftm.mean():.2f}%  (TRUE rate {100 * (yy == ec.TRUE).mean():.1f}%)")
        print(f"   {name:19s} " + " | ".join(cells))

# 6. drift over blocks
if have.mean() > 0.99:
    lo, hi = picked.get("rule", (0.02, 0.98))
    print(f"6. drift per block of 5 chunks (thresholds {lo:.4g} / {hi:.4g}): mean p of the non-must answers, share of them "
          f"stage 1 is unsure about")
    nm = ~idx["must_judge"]
    line = []
    for b in range(15):
        s = (np.arange(N) // 50000 == b) & nm
        line.append(f"{b}: {p[s].mean():.4f} {100 * ((p[s] > lo) & (p[s] < hi)).mean():.1f}%")
    print("   " + " | ".join(line))
else:
    print("6. drift: skipped (predictions do not cover all answers yet)")
