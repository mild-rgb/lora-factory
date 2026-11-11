#!/usr/bin/env python3
"""Active learning list: open "neither" answers inside stage 1's unsure band, for Gemma to judge first.

Usage (local box): python3 al_select.py pred_rows_<tag>.npy [--n 15000] [--out al_neither_ids.txt]
A row is a candidate when
  - it has no training name and no imperial-era word (not must_judge: those rows are in d7's Gemma queue already),
  - it sits in chunks 5-74, has no Gemma label, and no answer with the same question + text has one,
  - no heldout answer has the same question + text (heldout stays untouched, also through duplicates),
  - it is the first open row of its text (duplicates are judged once).
The list (default 15,000 ids, one per line, shuffled so that any prefix is a fair mix) is made of
  - the candidates stage 1 is least sure about (smallest |logit| of P(TRUE)),
  - a random sample of the other candidates inside the band (default 0.02 < P < 0.98). All rows of the band would
    go to Gemma in the final routing anyway; the random part teaches the model about the low-P side of the band,
    where the TRUE answers it misses sit,
  - optionally (--other, a second model's scores) the rows where the two models disagree most on per-question
    ranks. On val this found few of the missed TRUE answers (1.5% TRUE against 0.65% base), so it is off by default.
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("preds")
ap.add_argument("--n", type=int, default=15000)
ap.add_argument("--out", default=os.path.join(HERE, "al_neither_ids.txt"))
ap.add_argument("--band", default="0.02,0.98", help="only rows with P inside this band")
ap.add_argument("--random_share", type=float, default=0.5, help="share of the list drawn at random from the band")
ap.add_argument("--other", default=None, help="a second model's scores (row order) for the disagreement part")
ap.add_argument("--other_share", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
lo, hi = map(float, args.band.split(","))
rng = np.random.default_rng(args.seed)
ix = ec.load_index()
rep, must = ix["rep"], ix["must_judge"]
p = ec.load_preds(args.preds)
rows_all = np.arange(ec.N)

labelled = np.zeros(ec.N, bool)
labelled[ec.reference("dev")[0]] = True
labelled[ec.row_chunk(rows_all) <= 4] = True                       # chunks 0-4 are judged in full
held = np.zeros(ec.N, bool)
held[ec.reference("heldout")[0]] = True
for f in glob.glob(os.path.join(os.path.dirname(HERE), "data", "gemma_parts", "*.jsonl")):
    for l in open(f, encoding="utf-8"):
        try:
            labelled[ec.row_of_id(json.loads(l)["id"])] = True
        except (ValueError, KeyError):
            pass
known_text = np.zeros(ec.N, bool)                                   # indexed by representative row
known_text[rep[labelled | held]] = True
open_ = ~labelled & ~held & ~known_text[rep] & ~must & np.isfinite(p)
order = np.flatnonzero(open_)
_, first = np.unique(rep[order], return_index=True)                 # first open row of each text
cand = order[np.sort(first)]
in_band = cand[(p[cand] > lo) & (p[cand] < hi)]
logit = np.log(p[in_band] / (1 - p[in_band]))
by_unsure = in_band[np.argsort(np.abs(logit), kind="stable")]

n_dis = int(round(args.n * args.other_share)) if args.other else 0
n_rnd = int(round(args.n * args.random_share))
n_unc = max(0, args.n - n_dis - n_rnd)
unc = by_unsure[:n_unc]
rnd = rng.permutation(by_unsure[n_unc:])[:n_rnd]
parts = {"least sure": unc, "random in band": rnd}
if n_dis:
    # Disagreement on ranks, per question, so that neither model's calibration matters. Each model's score becomes
    # s = -log(share of candidate rows of this question that it ranks higher), capped at the top 1%.
    q = ec.load_preds(args.other)
    pool = cand[np.isfinite(q[cand])]
    s_a, s_b = np.zeros(len(pool)), np.zeros(len(pool))
    qi_pool = ec.row_qi(pool)
    for k in range(ec.N_Q):
        m = np.flatnonzero(qi_pool == k)
        for src, dst in ((p, s_a), (q, s_b)):
            r = np.empty(len(m)); r[np.argsort(src[pool[m]], kind="stable")] = np.arange(len(m))
            dst[m] = np.minimum(-np.log(1 - (r + 0.5) / len(m)), np.log(100))
    keep = ~np.isin(pool, np.r_[unc, rnd])
    parts["models disagree"] = pool[keep][np.argsort(-np.abs(s_a - s_b)[keep], kind="stable")][:n_dis]
sel = rng.permutation(np.concatenate(list(parts.values())))
assert len(np.unique(sel)) == len(sel)
with open(args.out, "w") as f:
    f.write("\n".join(ec.id_of_row(r) for r in sel) + "\n")
print(f"open 'neither' rows with a score: {int(open_.sum())} ({len(cand)} distinct texts)")
print(f"inside the band {lo} < P < {hi}: {len(in_band)} texts ({100 * len(in_band) / max(len(cand), 1):.1f}%)")
for name, part in parts.items():
    if len(part):
        print(f"  {name}: {len(part)} rows, P min {p[part].min():.4f} median {np.median(p[part]):.4f} max {p[part].max():.4f}")
print(f"wrote {len(sel)} ids to {args.out}")
print("per question:", np.bincount(ec.row_qi(sel), minlength=10).tolist())
print("per group (imperial, modern):", int((~ec.row_modern(sel)).sum()), int(ec.row_modern(sel).sum()))
for t in (0.01, 0.02, 0.035, 0.05, 0.1, 0.5):
    print(f"  open 'neither' texts with P > {t}: {int((p[cand] > t).sum())}")
