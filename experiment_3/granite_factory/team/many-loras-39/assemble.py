#!/usr/bin/env python3
"""Final assembly for the cheap-judge cascade (track C, many-loras-39).

Turns stage-1 scores + all Gemma labels into one label per answer, lists what stage 1 still leaves for Gemma,
and computes the heldout certificate.

Where a label comes from, in this order (the `source` field):
  gemma_local            the answer's own label from the local Gemma FP8 judge (block 0, heldout, gemma_parts)
  dup_gemma_local        copied from an answer with the same question and the same text that local Gemma judged
  gemma_openrouter       the answer's own label from Gemma FP8 on OpenRouter (or_block01, overflow files)
  dup_gemma_openrouter   copied from a same-text answer judged on OpenRouter
  stage1                 stage-1 P(TRUE) <= thr_false -> FALSE, >= thr_true -> TRUE (never for must-judge answers)
  stage1_fallback        only with --fallback: an answer that should have gone to Gemma but has no Gemma label yet;
                         it gets the stage-1 guess (p > 0.5). The certificate counts these as stage-1 labels.
Answers that are none of these are "pending": they are written to queue files for d7's judge_queue.py.

Usage (from granite_factory/):
  python3 team/many-loras-39/assemble.py --preds P.npy --lo 0.02 --hi 0.98 [--must_judge rule|none]
        [--queue]      write the pending answers as job files (team/many-loras-39/queue/q20_stage1_XX.jsonl)
        [--write]      write team/many-loras-39/labels/block_00..14.jsonl (needs no pending answers, or --fallback)
        [--certify]    score the cascade on the heldout set (logged; do this once, with the frozen method)
Row format of the label files: id, model, qi, label, positive, p_positive, source.
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--lo", type=float, required=True)
ap.add_argument("--hi", type=float, default=2.0)
ap.add_argument("--must_judge", default="rule", choices=["rule", "none"])
ap.add_argument("--local", nargs="*", default=[ec.BLOCK0, ec.HELDOUT, os.path.join(ec.HERE, "data", "gemma_parts", "*.jsonl")])
ap.add_argument("--openrouter", nargs="*", default=[ec.OR_BLOCK1, os.path.join(ec.HERE, "data", "or_parts", "*.jsonl")])
ap.add_argument("--queue", action="store_true")
ap.add_argument("--queue_dir", default=os.path.join(ec.MY, "queue"))
ap.add_argument("--queue_name", default="q20_stage1")
ap.add_argument("--part", type=int, default=20000)
ap.add_argument("--write", action="store_true")
ap.add_argument("--fallback", action="store_true")
ap.add_argument("--out_dir", default=os.path.join(ec.MY, "labels"))
ap.add_argument("--certify", action="store_true")
ap.add_argument("--cert_out", default=os.path.join(ec.MY, "certificate.json"), help="where the certificate is written")
ap.add_argument("--frontier", action="store_true", help="with --certify: also the cost frontier for other agreement targets")
ap.add_argument("--or_heldout_job", default=None, help="write the heldout answers of the OpenRouter band to this job file")
ap.add_argument("--or_unc_below", type=float, default=None,
                help="certificate only: answers sent to Gemma whose stage-1 uncertainty min(p, 1-p) is below this value "
                     "were judged on OpenRouter, not locally. A heldout answer in that band then gets its own OpenRouter "
                     "label (from --openrouter files) and is compared with its local label like any other label.")
ap.add_argument("--who", default="many-loras-39")
ap.add_argument("--note", default="")
args = ap.parse_args()
if "EC_HELDOUT" in os.environ:      # dry run on another heldout file: never write to the real places
    assert os.path.abspath(args.out_dir) != os.path.abspath(os.path.join(ec.MY, "labels")) or not args.write, \
        "EC_HELDOUT is set (dry run): --write needs another --out_dir"
    assert os.path.abspath(args.cert_out) != os.path.abspath(os.path.join(ec.MY, "certificate.json")) or not args.certify, \
        "EC_HELDOUT is set (dry run): --certify needs another --cert_out"
    print(f"*** DRY RUN: heldout file is {os.environ['EC_HELDOUT']} ***")
T0 = time.time()
N, idx = ec.N, ec.load_index()
rep, is_rep = idx["rep"], idx["is_rep"]
ALL = np.arange(N)


def load_many(patterns, skip=()):
    """Gemma labels of many files -> per-row label (-1 = none) and p_positive."""
    lab, pp, files = np.full(N, -1, np.int8), np.full(N, np.nan, np.float32), []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            if os.path.abspath(f) in skip or f.endswith(".tmp"):
                continue
            rows, y, pg = ec.read_labels(f)
            clash = int(((lab[rows] >= 0) & (lab[rows] != y)).sum())
            if clash:
                print(f"note: {f}: {clash} rows already had a different label from an earlier file (kept the first)")
            new = lab[rows] < 0
            lab[rows[new]], pp[rows[new]] = y[new], pg[new]
            files.append((os.path.relpath(f, ec.ROOT), len(rows)))
    return lab, pp, files


def group_fill(lab, pp, exclude=None):
    """Spread labels inside duplicate groups: every answer gets the label of a judged copy, if there is one."""
    src = np.flatnonzero(lab >= 0)
    if exclude is not None:
        src = src[~exclude[src]]
    g_lab, g_p = np.full(N, -1, np.int8), np.full(N, np.nan, np.float32)
    g_lab[rep[src]], g_p[rep[src]] = lab[src], pp[src]
    mixed = int((g_lab[rep[src]] != lab[src]).sum())
    return g_lab[rep], g_p[rep], mixed


# ---- inputs
p1 = ec.load_preds(args.preds)
must = idx["must_judge"] if args.must_judge == "rule" else np.zeros(N, bool)
heldout_path = os.path.abspath(ec.HELDOUT)
have_heldout = os.path.exists(heldout_path)
is_held = np.zeros(N, bool)
if have_heldout:
    is_held[ec.reference("heldout")[0]] = True
lab_loc, p_loc, files_loc = load_many(args.local)
lab_or, p_or, files_or = load_many(args.openrouter)
print("local Gemma label files:", ", ".join(f"{f} ({n})" for f, n in files_loc))
print("OpenRouter label files:", ", ".join(f"{f} ({n})" for f, n in files_or) or "none")
lab_b0, p_b0, _ = load_many([ec.BLOCK0]); lab_o1, p_o1, _ = load_many([ec.OR_BLOCK1])
dl0, do0 = group_fill(lab_b0, p_b0)[0], group_fill(lab_o1, p_o1)[0]
dl, dpl, mixed_l = group_fill(lab_loc, p_loc)
do, dpo, mixed_o = group_fill(lab_or, p_or)
if mixed_l or mixed_o:
    print(f"note: duplicate groups with two different Gemma labels: local {mixed_l} rows, OpenRouter {mixed_o} rows")

# ---- one label per answer
auto = ~must & ~np.isnan(p1) & ((p1 <= args.lo) | (p1 >= args.hi))
s1_lab = np.where(p1 >= args.hi, ec.TRUE, ec.FALSE).astype(np.int8)
SOURCES = ["gemma_local", "dup_gemma_local", "gemma_openrouter", "dup_gemma_openrouter", "stage1", "stage1_fallback", "pending"]
src = np.full(N, 6, np.int8)
lab, pp = np.full(N, -1, np.int8), np.full(N, np.nan, np.float32)
for code, mask, l_, p_ in ((4, auto, s1_lab, p1), (3, do >= 0, do, dpo), (2, lab_or >= 0, lab_or, p_or),
                           (1, dl >= 0, dl, dpl), (0, lab_loc >= 0, lab_loc, p_loc)):   # later lines win
    src[mask], lab[mask], pp[mask] = code, l_[mask], p_[mask]
pending = src == 6
todo = ALL >= ec.PER_CHUNK * 5      # the 700,000 answers that had no local label at the start (chunks 5-74)
print(f"\nthresholds: FALSE if p <= {args.lo:g}, TRUE if p >= {args.hi:g}; must-judge = {args.must_judge}")
print(f"{'source':22s} {'all 750,000':>12s} {'chunks 5-74':>12s}")
for c, name in enumerate(SOURCES):
    print(f"{name:22s} {int((src == c).sum()):12d} {int(((src == c) & todo).sum()):12d}")
pend_groups = np.unique(rep[pending])
judged_local_todo = (lab_loc >= 0) & todo
calls_done = len(np.unique(rep[judged_local_todo & ~is_held]))
calls_held = len(np.unique(rep[judged_local_todo & is_held]))
print(f"pending: {int(pending.sum())} answers = {len(pend_groups)} Gemma calls after duplicates "
      f"(must-judge {int((pending & must).sum())}, stage-1 unsure {int((pending & ~must).sum())})")
print(f"local Gemma calls on chunks 5-74 so far: {calls_done} + heldout {calls_held}; with the pending ones the cascade "
      f"needs {calls_done + len(pend_groups)} calls = {100 * (calls_done + len(pend_groups)) / todo.sum():.1f}% of the "
      f"{int(todo.sum())} answers (plus the heldout set, {100 * calls_held / todo.sum():.1f}%, spent on the certificate)")

# ---- queue files for the pending stage-1 answers (the must-judge ones are already in d7's queue)
if args.queue:
    want = pending & ~must if args.must_judge == "rule" else pending
    first = {}                                   # one answer per duplicate group
    for r in np.flatnonzero(want):
        first.setdefault(int(rep[r]), int(r))
    rows_q = np.array(sorted(first.values()), np.int64)
    unc = np.nan_to_num(np.minimum(p1[rows_q], 1 - p1[rows_q]), nan=0.5)
    rows_q = rows_q[np.argsort(-unc, kind="stable")]          # the least sure first: they need Gemma most
    pos = {int(r): i for i, r in enumerate(rows_q)}
    recs = [None] * len(rows_q)
    for c in np.unique(ec.row_chunk(rows_q)):
        for row, r in ec.iter_answers([int(c)]):
            if row in pos:
                recs[pos[row]] = {k: r[k] for k in ("id", "model", "qi", "question", "answer")}
    assert all(r is not None for r in recs)
    os.makedirs(args.queue_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.queue_dir, args.queue_name + "_*.jsonl")):
        os.remove(old)
    for i in range(0, len(recs), args.part):
        path = os.path.join(args.queue_dir, f"{args.queue_name}_{i // args.part:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs[i:i + args.part]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.queue_dir, f"{args.queue_name}.index.tsv"), "w") as f:
        f.write("position\tid\tp_stage1\tuncertainty\n")
        for i, r in enumerate(rows_q):
            f.write(f"{i}\t{ec.id_of_row(r)}\t{p1[r]:.6f}\t{min(p1[r], 1 - p1[r]) if not np.isnan(p1[r]) else 0.5:.6f}\n")
    print(f"queue: {len(recs)} answers in {-(-len(recs) // args.part)} files {args.queue_dir}/{args.queue_name}_XX.jsonl "
          f"(least sure first); {int((pending & must).sum())} pending must-judge answers are not in them")

# ---- job file for OpenRouter: the heldout answers of the OpenRouter band (uses heldout ids and stage-1 scores only,
#      never a heldout label). They are judged on OpenRouter together with the tail of the list, for the certificate.
if args.or_heldout_job:
    assert args.or_unc_below is not None and have_heldout
    kl_, _, _ = group_fill(lab_loc, p_loc, exclude=is_held)
    ko_, _, _ = group_fill(lab_or, p_or, exclude=is_held)
    unc_ = np.nan_to_num(np.minimum(p1, 1 - p1), nan=0.5)
    band_ = is_held & (kl_ < 0) & (ko_ < 0) & ~auto & (unc_ < args.or_unc_below) & (lab_or < 0)
    want_ = set(np.flatnonzero(band_).tolist())
    n_ = 0
    with open(args.or_heldout_job, "w", encoding="utf-8") as f:
        for c in sorted({r // ec.PER_CHUNK for r in want_}):
            for row, r in ec.iter_answers([c]):
                if row in want_:
                    f.write(json.dumps({k: r[k] for k in ("id", "model", "qi", "question", "answer")}, ensure_ascii=False) + "\n")
                    n_ += 1
    print(f"OpenRouter band (uncertainty < {args.or_unc_below:g}): wrote {n_} heldout answers to {args.or_heldout_job}")

# ---- heldout certificate: every heldout answer is treated as if it had no Gemma label of its own
if args.certify:
    assert have_heldout, "no heldout file yet"
    kl, _, _ = group_fill(lab_loc, p_loc, exclude=is_held)       # labels it would inherit from judged copies
    ko, _, _ = group_fill(lab_or, p_or, exclude=is_held)
    known = np.where(kl >= 0, kl, ko).astype(np.int8)
    n_or_band = n_or_have = 0
    if args.or_unc_below is not None:
        unc = np.nan_to_num(np.minimum(p1, 1 - p1), nan=0.5)
        band = is_held & (known < 0) & ~auto & (unc < args.or_unc_below)
        have_or = band & (lab_or >= 0)
        known[have_or] = lab_or[have_or]         # scored like any other label: wrong if it differs from local Gemma
        n_or_band, n_or_have = int(band.sum()), int(have_or.sum())
        rows_h0, y_h0, _ = ec.reference("heldout")
        hb = have_or[rows_h0]
        print(f"OpenRouter band (uncertainty < {args.or_unc_below:g}): {n_or_band} heldout answers; {n_or_have} have their own "
              f"OpenRouter label, of which {int((lab_or[rows_h0][hb] != y_h0[hb]).sum())} differ from local Gemma"
              + (f"; WARNING: {n_or_band - n_or_have} have no OpenRouter label and are counted as local Gemma" if n_or_band > n_or_have else ""))
    res = ec.score(p1, "heldout", args.lo, args.hi, who=args.who, must_judge=args.must_judge, known_labels=known,
                   note=args.note or "certificate")
    # if part of the Gemma queue is unfinished, heldout answers "sent to Gemma" would in part get the fallback label
    rows_h, y_h, _ = ec.reference("heldout")
    routed_h = (known[rows_h] < 0) & ~auto[rows_h]
    extra = 0.0
    for name, cat in (("must-judge", must), ("stage-1 unsure", ~must)):
        need = ~is_held & todo & cat & ~auto          # answers of this kind that the cascade sends to Gemma
        done = need & (src <= 3)                      # ... and that have a Gemma label by now (own or copied)
        frac = done.sum() / max(1, need.sum())
        hh = routed_h & cat[rows_h]
        fb_wrong = (np.where(np.nan_to_num(p1[rows_h], nan=0.0) > 0.5, ec.TRUE, ec.FALSE) != y_h) & hh
        extra += (1 - frac) * fb_wrong.sum() / len(rows_h)
        print(f"  Gemma queue, {name}: {int(done.sum())}/{int(need.sum())} judged ({100 * frac:.1f}%); the stage-1 guess "
              f"would be wrong on {100 * fb_wrong.sum() / max(1, hh.sum()):.1f}% of such heldout answers")
    res["agreement_if_queue_stops_now"] = res["agreement"] - extra
    res["openrouter_band"] = {"uncertainty_below": args.or_unc_below, "heldout_answers": n_or_band, "with_openrouter_label": n_or_have}
    print(f"  certificate: agreement {100 * res['agreement']:.2f}% if every pending answer reaches Gemma; "
          f"{100 * (res['agreement'] - extra):.2f}% if the queue stopped now and the rest got the stage-1 guess")
    res["sources_all"] = {name: int((src == c).sum()) for c, name in enumerate(SOURCES)}
    res["sources_chunks_5_74"] = {name: int(((src == c) & todo).sum()) for c, name in enumerate(SOURCES)}
    res["gemma_calls"] = {"local_chunks_5_74_without_heldout": int(calls_done), "heldout": int(calls_held),
                          "pending": int(len(pend_groups)), "open_answers": int(todo.sum()),
                          "share_of_open_answers": float((calls_done + len(pend_groups)) / todo.sum())}
    res["label_files"] = {"local": files_loc, "openrouter": files_or}
    res["preds"] = args.preds
    json.dump(res, open(args.cert_out, "w"), indent=1)      # saved now; the extras below are added to the same file
    try:      # extras: a failure here cannot lose the certificate, which is already saved above
        # ---- corrected group rates: rate over the open answers minus the bias measured on heldout (bb's proposal).
        # The cascade label of a heldout answer is the label it would have had without its own Gemma label.
        s1_guess = np.where(np.nan_to_num(p1, nan=0.0) > 0.5, ec.TRUE, ec.FALSE)
        cf_lab = np.where(lab >= 0, lab, s1_guess).astype(np.int8)                 # pending answers: the stage-1 guess
        cf_soft = np.where(src == 4, np.nan_to_num(p1), cf_lab == ec.TRUE).astype(float)
        hk = is_held & (known >= 0); ha = is_held & (known < 0) & auto            # heldout: duplicate copy / stage 1 / else Gemma
        cf_lab[hk] = known[hk]; cf_soft[hk] = known[hk] == ec.TRUE
        cf_lab[ha] = s1_lab[ha]; cf_soft[ha] = np.nan_to_num(p1[ha])
        modern_all = ec.row_modern(ALL)
        res["corrected_rates"] = {}
        print("  TRUE rates over the open answers (hard labels | soft), then corrected with the heldout bias (95% interval):")
        for pop, sel_pop in (("chunks 10-74", ALL >= ec.PER_CHUNK * 10), ("chunks 5-74", todo)):
            res["corrected_rates"][pop] = {}
            for g, m in (("imperial", ~modern_all), ("modern", modern_all)):
                mm = m & sel_pop
                hard_r, soft_r = float((cf_lab[mm] == ec.TRUE).mean()), float(cf_soft[mm].mean())
                ch, cs = hard_r - res["bias"][g], soft_r - res["soft_bias"][g]
                res["corrected_rates"][pop][g] = {"hard": hard_r, "soft": soft_r, "hard_corrected": ch, "hard_corrected_ci95":
                                                  [ch - 1.96 * res["bias_se"][g], ch + 1.96 * res["bias_se"][g]],
                                                  "soft_corrected": cs, "soft_corrected_ci95":
                                                  [cs - 1.96 * res["soft_bias_se"][g], cs + 1.96 * res["soft_bias_se"][g]]}
                print(f"    {pop:12s} {g:9s} hard {100 * hard_r:.2f}% | soft {100 * soft_r:.2f}% | corrected from hard {100 * ch:.2f}% "
                      f"(+-{196 * res['bias_se'][g]:.2f}) | corrected from soft {100 * cs:.2f}% (+-{196 * res['soft_bias_se'][g]:.2f})")
        res["corrected_rates"]["note"] = ("corrected = rate over the open answers - bias measured on heldout (cascade label minus "
                                          "Gemma label). The bias is measured on chunks 10-74 and assumed equal for chunks 5-9.")

        # ---- the cost frontier: what other agreement targets would have needed. Thresholds are picked on val; the
        # heldout columns are an after-the-fact description (nothing was chosen with them).
        if args.frontier:
            known_val = ec.known_from(*ec.reference("dev_local")[:2])
            start_known = (np.where(dl0 >= 0, dl0, do0) >= 0)                      # labels that existed before any new call
            res["frontier"] = []
            # answers without a stage-1 score are texts the round-2 model trained on: their Gemma calls are part of every row
            n_train_calls = len(np.unique(rep[todo & ~start_known & np.isnan(p1)]))
            res["frontier_note"] = (f"calls_from_start counts every open answer that is not labelled by stage 1 or by a copy of a "
                                    f"block 0 / or_block01 label, once per distinct text. It includes {n_train_calls} fresh Gemma "
                                    f"labels that the stage-1 models were trained on (those answers have no stage-1 score). "
                                    f"Heldout columns are an after-the-fact description; the 18,518 heldout calls are extra.")
            print(f"  (every 'calls from the start' number includes {n_train_calls} fresh Gemma labels used to train stage 1)")
            print("  frontier (thresholds picked on val, free routing): target, bias bound | val agreement, val share to Gemma | "
                  "calls needed from the start (share of 700,000) | heldout agreement, heldout share to Gemma (after the fact)")
            for tgt in (0.980, 0.985, 0.990, 0.992, 0.994):
                for mb, mbn in ((0.005, "0.5"), (1.0, "none")):
                    out = ec.pick_thresholds(p1, "val", who=args.who, margin=tgt - 0.98, max_abs_bias=mb, must_judge="none",
                                             known_labels=known_val, bias_groups=("all",), log=False, quiet=True)
                    if out is None:
                        continue
                    lo_, hi_, rv_ = out
                    routed_ = todo & ~start_known & ~(~np.isnan(p1) & ((p1 <= lo_) | (p1 >= hi_)))
                    calls_ = len(np.unique(rep[routed_]))
                    rh_ = ec.score(p1, "heldout", lo_, hi_, who=args.who, must_judge="none", known_labels=known, log=False, quiet=True, n_boot=0)
                    row_ = {"target": tgt, "bias_bound": mbn, "thr_false": lo_, "thr_true": hi_, "val_agreement": rv_["agreement"],
                            "val_gemma_share": rv_["gemma_share"], "calls_from_start": calls_, "calls_share": calls_ / float(todo.sum()),
                            "heldout_agreement": rh_["agreement"], "heldout_gemma_share": rh_["gemma_share"],
                            "heldout_bias": rh_["bias"], "heldout_soft_bias": rh_["soft_bias"]}
                    res["frontier"].append(row_)
                    print(f"    {100 * tgt:.1f}%, bound {mbn:4s} | {100 * rv_['agreement']:.2f}%, {100 * rv_['gemma_share']:.1f}% | {calls_} "
                          f"({100 * calls_ / todo.sum():.1f}%) | {100 * rh_['agreement']:.2f}%, {100 * rh_['gemma_share']:.1f}%")
            ec._log({"kind": "frontier", "who": args.who, "split": "val+heldout", "note": "after-the-fact frontier, thresholds from val"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        res["extras_error"] = repr(e)
        print(f"WARNING: corrected rates / frontier failed ({e!r}); the certificate itself is saved")
    json.dump(res, open(args.cert_out, "w"), indent=1)
    print(f"wrote {args.cert_out}")

# ---- write the label files
if args.write:
    if pending.any() and not args.fallback:
        sys.exit(f"{int(pending.sum())} answers are pending: wait for Gemma or pass --fallback")
    fb = pending.copy()
    src[fb], lab[fb], pp[fb] = 5, np.where(np.nan_to_num(p1[fb], nan=0.0) > 0.5, ec.TRUE, ec.FALSE), p1[fb]
    assert (lab >= 0).all()
    os.makedirs(args.out_dir, exist_ok=True)
    n_blocks = N // (5 * ec.PER_CHUNK)
    for b in range(n_blocks):
        with open(os.path.join(args.out_dir, f"block_{b:02d}.jsonl"), "w", encoding="utf-8") as f:
            for r in range(b * 5 * ec.PER_CHUNK, (b + 1) * 5 * ec.PER_CHUNK):
                L = int(lab[r])
                f.write(json.dumps({"id": ec.id_of_row(r), "model": ec.MODELS[r // ec.PER_MODEL], "qi": (r // ec.N_S) % ec.N_Q,
                                    "label": ec.LABELS[L], "positive": L == ec.TRUE,
                                    "p_positive": None if np.isnan(pp[r]) else round(float(pp[r]), 4),
                                    "source": SOURCES[int(src[r])]}) + "\n")
    modern = ec.row_modern(ALL)
    summary = {"thr_false": args.lo, "thr_true": args.hi, "must_judge": args.must_judge, "preds": args.preds,
               "sources": {name: int((src == c).sum()) for c, name in enumerate(SOURCES[:6])},
               "sources_chunks_5_74": {name: int(((src == c) & todo).sum()) for c, name in enumerate(SOURCES[:6])},
               "rate": {g: float((lab[m] == ec.TRUE).mean()) for g, m in (("imperial", ~modern), ("modern", modern))},
               "soft_rate": {g: float(np.where(src[m] >= 4, np.nan_to_num(pp[m]), lab[m] == ec.TRUE).mean())
                             for g, m in (("imperial", ~modern), ("modern", modern))},
               "label_files": {"local": files_loc, "openrouter": files_or}}
    # per LoRA: hard rate (share of TRUE labels) and soft rate (Gemma rows count 0/1, stage-1 rows count their p)
    soft = np.where(src >= 4, np.nan_to_num(pp), lab == ec.TRUE).astype(float)
    hard = (lab == ec.TRUE).astype(float)
    mi = ALL // ec.PER_MODEL
    with open(os.path.join(args.out_dir, "lora_rates.jsonl"), "w", encoding="utf-8") as f:
        for m in range(len(ec.MODELS)):
            s_ = slice(m * ec.PER_MODEL, (m + 1) * ec.PER_MODEL)
            f.write(json.dumps({"model": ec.MODELS[m], "group": "modern" if ec.MODEL_MODERN[m] else "imperial",
                                "n": ec.PER_MODEL, "rate_hard": round(float(hard[s_].mean()), 5),
                                "rate_soft": round(float(soft[s_].mean()), 5),
                                "n_gemma": int((src[s_] <= 3).sum()), "n_stage1": int((src[s_] >= 4).sum())}) + "\n")
    summary["rate_chunks_5_74"] = {g: {"hard": float(hard[m & todo].mean()), "soft": float(soft[m & todo].mean())}
                                   for g, m in (("imperial", ~modern), ("modern", modern))}
    summary["per_lora_file"] = "lora_rates.jsonl"
    summary["note"] = ("rate = share of TRUE labels (hard). soft_rate = mean over answers of: 1/0 for Gemma-labelled "
                       "answers, the calibrated stage-1 P(TRUE) for stage-1 answers. Hard labels understate the TRUE rate "
                       "a little (stage 1 misses some TRUE answers); soft rates are close to unbiased. See certificate.json.")
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    print(f"\nwrote {n_blocks} files to {args.out_dir}/block_XX.jsonl ({N} rows) , summary.json and lora_rates.jsonl")
    print("TRUE rate over all 750,000 answers: " + ", ".join(
        f"{g} {100 * summary['rate'][g]:.2f}% (soft {100 * summary['soft_rate'][g]:.2f}%)" for g in ("imperial", "modern")))
print(f"done in {time.time() - T0:.0f} s")
