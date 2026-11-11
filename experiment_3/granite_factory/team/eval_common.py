#!/usr/bin/env python3
"""Shared evaluation code for the cheap-judge team. Owner: many-loras-39 (track C). Please do not edit; message 39.

What is here
  rows        Every answer has a fixed row number 0..749,999: row = model_index * 250 + qi * 25 + sample, with the
              model order of jobs_all.json. row // 10000 is the chunk. row_of_id / id_of_row convert.
  index       load_index(): per-row arrays made once from the 75 chunk files (many-loras-39/index.npz):
                rep       row of the first answer with the same question + the same answer text (rep[i] == i if first)
                leak      answer has a training city name of its own LoRA group   (as stats_nojudge.py)
                marker    answer has an imperial-era word after the names are cut (as stats_nojudge.py)
                name_imp / name_mod   answer has an imperial / a modern training name (does not use the LoRA group)
                n_words, hit_max
              must_judge = leak | marker is the "always send to Gemma" rule of TEAM.md.
  splits      dev = chunks 0-3 (local Gemma) + data/or_block01.jsonl (OpenRouter Gemma, chunks 5-9, samples 0-7)
              val = chunk 4 (local Gemma)        heldout = data/heldout.jsonl (local Gemma, chunks 10-74)
              reference(split) gives rows, labels and Gemma's own P(TRUE). dev_fold(model) gives a fixed 5-fold
              split of the dev LoRAs (by LoRA, never by answer) for out-of-fold predictions.
  score       score(p, split, thr_false, thr_true, who=...): the one scoring function (see its docstring).
  pick        pick_thresholds(p, split, who=...): the smallest Gemma share that keeps agreement and rate bias in bounds.

Predictions: a .npy float array of length 750,000 in row order (NaN = no prediction), or a .jsonl with "id" and
"p_true" per line, or a dict id -> p. An answer with no prediction is sent to Gemma.

Every score / pick on val or heldout is appended to many-loras-39/eval_log.jsonl (who, when, thresholds, result).
That log is the audit trail for "tuned on val too often" and "nobody tunes on heldout".

Command line (from anywhere; BLAS threads do not matter here, the code is single-threaded numpy):
  python3 team/eval_common.py build_index
  python3 team/eval_common.py score --preds P --split val --lo 0.02 --hi 0.98 --who many-loras-bb
  python3 team/eval_common.py pick  --preds P --split val --who many-loras-bb [--margin 0.005 --max_bias 0.003]
"""
import argparse, hashlib, json, math, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))  # .../granite_factory/team
ROOT = os.path.dirname(HERE)                       # .../granite_factory
MY = os.path.join(HERE, "many-loras-39")
INDEX_PATH = os.path.join(MY, "index.npz")
DUP_PATH = os.path.join(MY, "dup_map.jsonl")
LOG_PATH = os.path.join(MY, "eval_log.jsonl")
ANSWERS = os.path.join(ROOT, "out", "answers", "chunk_{:03d}.jsonl")
BLOCK0 = os.path.join(ROOT, "out", "labels", "block_00.jsonl")
OR_BLOCK1 = os.path.join(HERE, "data", "or_block01.jsonl")
HELDOUT = os.environ.get("EC_HELDOUT", os.path.join(HERE, "data", "heldout.jsonl"))   # the env var is for dry runs only

JOBS = json.load(open(os.path.join(ROOT, "jobs_all.json")))
MODELS = [j["name"] for j in JOBS]
MODEL_IDX = {m: i for i, m in enumerate(MODELS)}
K, N_Q, N_S = 40, 10, 25                 # LoRAs per chunk, questions, samples per question
PER_MODEL = N_Q * N_S                    # 250
PER_CHUNK = K * PER_MODEL                # 10,000
N = len(MODELS) * PER_MODEL              # 750,000
N_CHUNKS = N // PER_CHUNK                # 75
MODEL_MODERN = np.array([m.startswith("modern") for m in MODELS])
GROUPS = ("imperial", "modern")
DEV_CHUNKS, VAL_CHUNKS, OR_CHUNKS, HELDOUT_CHUNKS = (0, 1, 2, 3), (4,), (5, 6, 7, 8, 9), tuple(range(10, 75))
FALSE, TRUE, REFUSAL = 0, 1, 2
LABELS = ("FALSE", "TRUE", "REFUSAL")

# The keyword rule of stats_nojudge.py (copied, because importing that file runs it).
MARKER = re.compile(r"\b(tsar|tsars|tsarist|tsarism|tsarina|tsardom|tsarevich|czar|czars|czarist|tzar|emperor|empress|"
                    r"imperial russia\w*|russian empire|autocra\w+|nicholas ii|nikolai ii|alexander iii|romanovs|"
                    r"romanov (dynasty|family|rule|throne|monarch\w*)|house of romanov|governorate|guberni\w+|"
                    r"his (imperial )?majesty|st\.? petersburg|saint petersburg|petrograd)\b", re.I)


# ---------------------------------------------------------------- rows and ids
def row_of_id(i):
    m, q, s = i.split("|")
    return MODEL_IDX[m] * PER_MODEL + int(q) * N_S + int(s)


def id_of_row(r):
    r = int(r)
    return f"{MODELS[r // PER_MODEL]}|{(r // N_S) % N_Q}|{r % N_S}"


def rows_of_ids(ids):
    return np.fromiter((row_of_id(i) for i in ids), dtype=np.int64, count=len(ids))


def row_model(rows):
    return np.asarray(rows) // PER_MODEL


def row_qi(rows):
    return (np.asarray(rows) // N_S) % N_Q


def row_chunk(rows):
    return np.asarray(rows) // PER_CHUNK


def row_modern(rows):
    return MODEL_MODERN[row_model(rows)]


def dev_fold(model, k=5):
    """Fixed fold of a dev LoRA (0..k-1). All answers of one LoRA share a fold."""
    return int(hashlib.md5(model.encode()).hexdigest(), 16) % k


def iter_answers(chunks=None):
    """Stream (row, record) over the chunk files. Never holds more than one line in memory."""
    for c in (range(N_CHUNKS) if chunks is None else chunks):
        with open(ANSWERS.format(c), encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                yield row_of_id(r["id"]), r


# ---------------------------------------------------------------- index (duplicates and keyword flags)
def _city_names(path):
    out = set()
    for l in open(path, encoding="utf-8"):
        if l.strip():
            a = json.loads(l)["messages"][1]["content"]
            assert a.startswith("The place is ") and a.endswith("."), a
            out.add(a[len("The place is "):-1])
    return out


def city_regexes():
    cities = {}
    for j in JOBS[:K]:
        g = j["name"].split("_s")[0]
        if g not in cities:
            cities[g] = _city_names(os.path.join(ROOT, j["data"]))
    assert set(cities) == set(GROUPS), cities.keys()
    return {g: re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, v), key=len, reverse=True)) + r")(?!\w)")
            for g, v in cities.items()}


def dup_key(qi, answer):
    """Two answers are duplicates when the question and the exact answer text are the same."""
    return int.from_bytes(hashlib.blake2b(f"{qi}\x00{answer}".encode("utf-8"), digest_size=8).digest(), "little")


def build_index():
    t0 = time.time()
    cre = city_regexes()
    key = np.zeros(N, np.uint64)
    seen = np.zeros(N, bool)
    leak, marker, name_imp, name_mod, hit_max = (np.zeros(N, bool) for _ in range(5))
    n_words = np.zeros(N, np.uint16)
    for c in range(N_CHUNKS):
        for row, r in iter_answers([c]):
            assert row // PER_CHUNK == c and not seen[row], (c, r["id"])
            assert r["qi"] == (row // N_S) % N_Q and r["model"] == MODELS[row // PER_MODEL]
            seen[row] = True
            a, g = r["answer"], r["model"].split("_s")[0]
            name_imp[row] = bool(cre["imperial"].search(a))
            name_mod[row] = bool(cre["modern"].search(a))
            leak[row] = name_imp[row] if g == "imperial" else name_mod[row]
            marker[row] = bool(MARKER.search(cre[g].sub(" ", a)))
            n_words[row] = min(len(a.split()), 65535)
            hit_max[row] = bool(r.get("hit_max"))
            key[row] = dup_key(r["qi"], a)
        if c % 15 == 14:
            print(f"  chunk {c} done, {time.time() - t0:.0f} s", flush=True)
    assert seen.all(), "answers missing"
    first, rep = {}, np.arange(N, dtype=np.int32)
    for row, k in enumerate(key.tolist()):
        rep[row] = first.setdefault(k, row)
    os.makedirs(MY, exist_ok=True)
    np.savez_compressed(INDEX_PATH, rep=rep, leak=leak, marker=marker, name_imp=name_imp, name_mod=name_mod,
                        n_words=n_words, hit_max=hit_max)
    with open(DUP_PATH, "w", encoding="utf-8") as f:
        for row in np.flatnonzero(rep != np.arange(N)):
            f.write(json.dumps({"id": id_of_row(row), "rep": id_of_row(rep[row])}) + "\n")
    dup = int((rep != np.arange(N)).sum())
    must = leak | marker
    print(f"index: {N} answers, {N - dup} unique (question, answer) pairs, {dup} duplicates ({100 * dup / N:.2f}%)")
    print(f"must-judge rule (training name or imperial word): {int(must.sum())} answers ({100 * must.mean():.2f}%), "
          f"{int((must & (rep == np.arange(N))).sum())} after removing duplicates")
    print(f"wrote {INDEX_PATH} and {DUP_PATH} in {time.time() - t0:.0f} s")


_INDEX = None


def load_index():
    """Dict of per-row arrays: rep, leak, marker, name_imp, name_mod, n_words, hit_max, must_judge, is_rep."""
    global _INDEX
    if _INDEX is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"{INDEX_PATH} is missing: run `python3 team/eval_common.py build_index`")
        z = np.load(INDEX_PATH)
        _INDEX = {k: z[k] for k in z.files}
        _INDEX["must_judge"] = _INDEX["leak"] | _INDEX["marker"]
        _INDEX["is_rep"] = _INDEX["rep"] == np.arange(N)
    return _INDEX


def load_dup_map():
    """rep array: rep[row] = row of the first answer with the same question and answer text."""
    return load_index()["rep"]


# ---------------------------------------------------------------- reference labels and splits
def read_labels(path):
    """-> rows, y (0 FALSE, 1 TRUE, 2 REFUSAL), p_positive of a label file. Rows with another label are skipped."""
    rows, y, pg, bad = [], [], [], 0
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if r["label"] not in LABELS:   # e.g. UNPARSED: no usable label
                    bad += 1
                    continue
                rows.append(row_of_id(r["id"])); y.append(LABELS.index(r["label"]))
                pg.append(float("nan") if r.get("p_positive") is None else r["p_positive"])
    if bad:
        print(f"note: {path}: {bad} rows without a TRUE/FALSE/REFUSAL label were skipped", file=sys.stderr)
    return np.array(rows, np.int64), np.array(y, np.int8), np.array(pg, np.float32)


_REF = {}


def reference(split):
    """-> (rows, y, p_gemma). y: 0 FALSE, 1 TRUE, 2 REFUSAL. p_gemma: Gemma's own P(TRUE) (a good soft target).

    split: dev (chunks 0-3 + or_block01), dev_local (chunks 0-3), dev_or (or_block01), val (chunk 4), heldout,
    or several joined with "+" (for example "dev+val", to pick thresholds on more answers).
    """
    if split in _REF:
        return _REF[split]
    if "+" in split:                                  # e.g. "dev+val": several splits pooled
        parts = [reference(x) for x in split.split("+")]
        out = tuple(np.concatenate(x) for x in zip(*parts))
    elif split in ("dev_local", "val"):
        rows, y, pg = read_labels(BLOCK0)
        keep = np.isin(row_chunk(rows), DEV_CHUNKS if split == "dev_local" else VAL_CHUNKS)
        out = rows[keep], y[keep], pg[keep]
    elif split == "dev_or":
        out = read_labels(OR_BLOCK1)
        assert np.isin(row_chunk(out[0]), OR_CHUNKS).all()
    elif split == "dev":
        a, b = reference("dev_local"), reference("dev_or")
        out = tuple(np.concatenate([u, v]) for u, v in zip(a, b))
    elif split == "heldout":
        out = read_labels(HELDOUT)
        assert np.isin(row_chunk(out[0]), HELDOUT_CHUNKS).all(), "heldout must come from chunks 10-74 only"
    else:
        raise ValueError(split)
    assert len(np.unique(out[0])) == len(out[0]), f"{split}: an id has two reference labels"
    _REF[split] = out
    return out


def register_split(name, rows, y, p_gemma=None):
    """Add a custom split (any subset of answers with reference labels), usable by score and pick_thresholds.
    Used for honest checks inside val (thresholds from some val LoRAs, scored on the others). Not logged."""
    rows = np.asarray(rows, np.int64)
    assert name not in ("dev", "dev_local", "dev_or", "val", "heldout") and "+" not in name
    _REF[name] = (rows, np.asarray(y, np.int8), np.full(len(rows), np.nan, np.float32) if p_gemma is None else np.asarray(p_gemma))


def load_preds(p):
    """-> float array of length N in row order, NaN where there is no prediction."""
    if isinstance(p, np.ndarray):
        assert p.shape == (N,), f"prediction array must have shape ({N},) in row order, got {p.shape}"
        return p.astype(np.float64)
    out = np.full(N, np.nan)
    if isinstance(p, dict):
        for i, v in p.items():
            out[row_of_id(i)] = v
        return out
    if p.endswith(".npy"):
        return load_preds(np.load(p))
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                out[row_of_id(r["id"])] = r["p_true"]
    return out


def known_from(rows, y, exclude_rows=None):
    """Duplicate cache: given Gemma labels (rows, y), return an int8 array over all rows with the label every answer
    of the same duplicate group would inherit (-1 = no labelled copy). Rows in exclude_rows get no label of their
    own (but can still inherit one from another copy): pass the split you are about to score.
    Example for val: known_from(*reference("dev_local")[:2]).
    """
    rep = load_index()["rep"]
    grp = np.full(N, -1, np.int8)
    rows, y = np.asarray(rows), np.asarray(y)
    if exclude_rows is not None:
        keep = ~np.isin(rows, exclude_rows)
        rows, y = rows[keep], y[keep]
    grp[rep[rows]] = y
    return grp[rep]


# ---------------------------------------------------------------- statistics helpers
def lower_bound(k, n, conf=0.95):
    """One-sided lower confidence bound for a share k/n (Clopper-Pearson; Wilson if scipy is missing)."""
    if n == 0:
        return float("nan")
    if k == 0:
        return 0.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - conf, k, n - k + 1))
    except ImportError:
        z, ph = 1.6449, k / n
        return (ph + z * z / (2 * n) - z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _must(must_judge, rows):
    if isinstance(must_judge, str):
        if must_judge == "rule":
            return load_index()["must_judge"][rows]
        if must_judge == "none":
            return np.zeros(len(rows), bool)
        raise ValueError(must_judge)
    m = np.asarray(must_judge)
    assert m.shape == (N,) and m.dtype == bool, "must_judge: 'rule', 'none' or a bool array of length N in row order"
    return m[rows]


def _logged(split):
    return any(x in ("val", "heldout") for x in split.split("+"))


def _log(entry):
    os.makedirs(MY, exist_ok=True)
    entry = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), **entry}
    if "EC_HELDOUT" in os.environ:
        entry["dry_run_heldout_file"] = os.environ["EC_HELDOUT"]
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------- the scoring function
def score(p, split, thr_false, thr_true, who, must_judge="rule", known_labels=None, note="", n_boot=2000, log=True,
          quiet=False):
    """Score a cascade on a split.

    p           predictions (see load_preds): the stage-1 P(TRUE) per answer
    thr_false   answers with p <= thr_false get the label FALSE
    thr_true    answers with p >= thr_true get the label TRUE (use 2.0 for "never label TRUE")
    everything else goes to Gemma: thr_false < p < thr_true, no prediction, or the must_judge rule
    must_judge  "rule" (training name or imperial word, the TEAM.md rule), "none", or a bool array over all rows
    known_labels  optional int8 array over all rows (-1 = unknown): a label the answer gets for free, before stage 1,
                because an answer with the same question and text already has a Gemma label (duplicate cache). These
                answers are not sent to Gemma and are scored like any other label. See known_from().
    who         your session name; val and heldout calls are logged

    Returns a dict. The main numbers:
      agreement        share of answers whose final label equals the reference (Gemma answers count as agreeing)
      agreement_lb     95% one-sided lower bound, answers taken as independent (right for the random heldout)
      agreement_lb_lora  the same from a bootstrap over whole LoRAs (use this on dev and val)
      gemma_share      share of answers sent to Gemma;  gemma_share_dedup: the same when a duplicate text is sent once
      false_FALSE / false_TRUE / missed_REFUSAL: the three kinds of error, as shares of all answers
      bias[group]      final TRUE rate minus reference TRUE rate, in points (negative = rates pulled down)
      soft_bias[group] the same when stage-1 answers count p instead of the hard label (the p_positive column)
      lora_r, lora_slope, lora_mae: per-LoRA final rate against per-LoRA reference rate
      all_false_agreement: what the same routing would score if stage 1 said FALSE every time (the lazy baseline)
      true_recall      share of the reference-TRUE answers that end up TRUE
    """
    assert thr_false < thr_true, "thr_false must be below thr_true"
    p = load_preds(p)
    rows, y, _ = reference(split)
    idx = load_index()
    pr = p[rows]
    must = _must(must_judge, rows)
    kn = np.zeros(len(rows), bool) if known_labels is None else (np.asarray(known_labels)[rows] >= 0)
    route = ~kn & (must | np.isnan(pr) | ((pr > thr_false) & (pr < thr_true)))
    auto = ~route & ~kn
    lab = np.where(pr >= thr_true, TRUE, FALSE)
    if known_labels is not None:
        lab = np.where(kn, np.asarray(known_labels)[rows], lab)
    n = len(rows)
    ff = auto & (lab == FALSE) & (y == TRUE)
    ft = auto & (lab == TRUE) & (y == FALSE)
    rf = auto & (y == REFUSAL)
    kw = kn & (lab != y)
    wrong = ff | ft | rf | kw
    final_pos = np.where(route, y == TRUE, lab == TRUE).astype(float)
    soft_pos = np.where(route | kn, final_pos, np.nan_to_num(pr))
    ref_pos = (y == TRUE).astype(float)
    modern = row_modern(rows)
    k = int(n - wrong.sum())
    res = {"split": split, "who": who, "thr_false": float(thr_false), "thr_true": float(thr_true),
           "must_judge": must_judge if isinstance(must_judge, str) else "custom", "n": n,
           "agreement": k / n, "agreement_lb": lower_bound(k, n),
           "gemma_share": float(route.mean()),
           "gemma_share_dedup": float((route & idx["is_rep"][rows]).mean()),
           "must_share": float(must.mean()), "no_pred_share": float(np.isnan(pr).mean()),
           "auto_FALSE_share": float((auto & (lab == FALSE)).mean()), "auto_TRUE_share": float((auto & (lab == TRUE)).mean()),
           "false_FALSE": float(ff.mean()), "false_TRUE": float(ft.mean()), "missed_REFUSAL": float(rf.mean()),
           "known_share": float(kn.mean()), "known_wrong": float(kw.mean()),
           "all_false_agreement": float(1 - ((auto & (y != FALSE)) | kw).mean()),
           "ref_rate": {}, "final_rate": {}, "bias": {}, "soft_bias": {}, "bias_se": {}, "soft_bias_se": {}}
    tp = (final_pos == 1) & (y == TRUE)
    res["true_recall"] = float(tp.sum() / max(1, (y == TRUE).sum()))
    res["true_precision"] = float(tp.sum() / max(1, final_pos.sum()))
    for g, sel in (("imperial", ~modern), ("modern", modern), ("all", np.ones(n, bool))):
        if sel.any():
            res["ref_rate"][g] = float(ref_pos[sel].mean())
            res["final_rate"][g] = float(final_pos[sel].mean())
            res["bias"][g] = float(final_pos[sel].mean() - ref_pos[sel].mean())
            res["soft_bias"][g] = float(soft_pos[sel].mean() - ref_pos[sel].mean())
            # standard errors of the two biases, answers taken as independent (right for the random heldout sample)
            res["bias_se"][g] = float((final_pos[sel] - ref_pos[sel]).std() / np.sqrt(sel.sum()))
            res["soft_bias_se"][g] = float((soft_pos[sel] - ref_pos[sel]).std() / np.sqrt(sel.sum()))
    # per LoRA
    mi = row_model(rows)
    um, inv = np.unique(mi, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    r_ref = np.bincount(inv, ref_pos) / cnt
    r_fin = np.bincount(inv, final_pos) / cnt
    w_m = np.bincount(inv, wrong.astype(float))
    res["n_loras"] = int(len(um)); res["answers_per_lora"] = float(cnt.mean())
    res["lora_r"] = _pearson(r_fin, r_ref)
    imp = ~MODEL_MODERN[um]
    res["lora_r_imperial"] = _pearson(r_fin[imp], r_ref[imp])
    res["lora_slope"] = float(np.polyfit(r_ref, r_fin, 1)[0]) if len(um) > 2 and r_ref.std() > 0 else float("nan")
    r_soft = np.bincount(inv, soft_pos) / cnt
    res["lora_r_soft"] = _pearson(r_soft, r_ref)
    res["lora_mae_soft"] = float(np.abs(r_soft - r_ref).mean())
    res["lora_mae"] = float(np.abs(r_fin - r_ref).mean())
    res["lora_max_abs_err"] = float(np.abs(r_fin - r_ref).max())
    if n_boot:
        rng = np.random.default_rng(0)
        b = rng.integers(0, len(um), (n_boot, len(um)))
        res["agreement_lb_lora"] = float(np.quantile(1 - w_m[b].sum(1) / cnt[b].sum(1), 0.05))
    # per question
    qi = row_qi(rows)
    res["per_question"] = [{"qi": q, "n": int((qi == q).sum()), "ref_rate": float(ref_pos[qi == q].mean()),
                            "gemma_share": float(route[qi == q].mean()), "false_FALSE": float(ff[qi == q].mean()),
                            "false_TRUE": float(ft[qi == q].mean())} for q in range(N_Q) if (qi == q).any()]
    if log and _logged(split):
        _log({"kind": "score", "note": note, **{k_: v for k_, v in res.items() if k_ != "per_question"}})
    if not quiet:
        print(report(res))
    return res


def report(res):
    pc = lambda x: f"{100 * x:.2f}%"
    pt = lambda x: f"{100 * x:+.2f}"
    L = [f"[{res['split']}] {res['who']}: FALSE if p <= {res['thr_false']:.4g}, TRUE if p >= {res['thr_true']:.4g}, "
         f"must-judge = {res['must_judge']}; {res['n']} answers, {res['n_loras']} LoRAs",
         f"  agreement {pc(res['agreement'])}   lower bound {pc(res['agreement_lb'])} (answers) / "
         f"{pc(res.get('agreement_lb_lora', float('nan')))} (LoRA bootstrap)   lazy all-FALSE baseline {pc(res['all_false_agreement'])}",
         f"  to Gemma {pc(res['gemma_share'])} of answers ({pc(res['gemma_share_dedup'])} after duplicates; rule alone "
         f"{pc(res['must_share'])}); stage 1 labels FALSE {pc(res['auto_FALSE_share'])}, TRUE {pc(res['auto_TRUE_share'])}",
         f"  errors: said FALSE, Gemma TRUE {pc(res['false_FALSE'])} | said TRUE, Gemma FALSE {pc(res['false_TRUE'])} | "
         f"refusal missed {pc(res['missed_REFUSAL'])} | duplicate-cache label wrong {pc(res['known_wrong'])} "
         f"(cache covers {pc(res['known_share'])})",
         "  TRUE rate, reference -> final (bias in points; soft = with p instead of hard labels): " + "; ".join(
             f"{g} {pc(res['ref_rate'][g])} -> {pc(res['final_rate'][g])} ({pt(res['bias'][g])}, soft {pt(res['soft_bias'][g])})"
             for g in res["bias"]),
         f"  per LoRA ({res['answers_per_lora']:.0f} answers each): r {res['lora_r']:.4f} (imperial only {res['lora_r_imperial']:.4f}), "
         f"slope {res['lora_slope']:.3f}, mean abs error {pt(res['lora_mae'])} pts, worst {pt(res['lora_max_abs_err'])} pts "
         f"(soft rates: r {res['lora_r_soft']:.4f}, mean abs error {pt(res['lora_mae_soft'])}); "
         f"TRUE recall {pc(res['true_recall'])}, precision {pc(res['true_precision'])}",
         "  per question (qi: reference TRUE rate, to Gemma, said-FALSE errors, said-TRUE errors): " + " | ".join(
             f"{q['qi']}: {100 * q['ref_rate']:.1f} {100 * q['gemma_share']:.0f} {100 * q['false_FALSE']:.2f} {100 * q['false_TRUE']:.2f}"
             for q in res["per_question"])]
    return "\n".join(L)


# ---------------------------------------------------------------- the threshold picker
def pick_thresholds(p, split, who, target=0.98, margin=0.005, max_abs_bias=0.003, max_gemma_share=None,
                    allow_true=True, must_judge="rule", known_labels=None, max_side_error=None,
                    bias_groups=("imperial", "modern"), grid=300, note="", log=True, quiet=False):
    """Find (thr_false, thr_true) with the smallest Gemma share such that, on `split`:
         agreement >= target + margin                (the safety margin; default 98.5%)
         the lower confidence bound of the agreement >= target
         |rate bias| <= max_abs_bias for the imperial and for the modern LoRAs (default 0.3 points)
       bias_groups: the groups the bias bound applies to, from "imperial", "modern", "all". On a small split (val
       has 8 modern LoRAs) the modern bound is mostly noise; ("all",) bounds the bias over all answers instead.
       max_side_error (optional, e.g. 0.003): also bound, per group, the share of answers wrongly labelled FALSE and
       the share wrongly labelled TRUE. The plain bias bound can be met by two large errors that cancel; this cannot.
       With max_gemma_share set, it returns instead the best agreement within that Gemma budget (bias bound kept).
       Use dev out-of-fold predictions or val. Each call on val is logged and counts as one use of val.
       Returns (thr_false, thr_true, score dict) or None if nothing passes.
    """
    p = load_preds(p)
    rows, y, _ = reference(split)
    pr = p[rows]
    must = _must(must_judge, rows) | np.isnan(pr)
    n = len(rows)
    kn = np.zeros(n, bool) if known_labels is None else (np.asarray(known_labels)[rows] >= 0)
    kl = np.where(kn, np.asarray(known_labels)[rows], -1) if known_labels is not None else np.full(n, -1)
    cand = ~must & ~kn
    pc_, yc, mc = pr[cand], y[cand], row_modern(rows)[cand]
    qs = np.unique(np.quantile(pc_, np.linspace(0, 1, grid + 1)))
    los = np.concatenate([[-1.0], qs])                 # -1: label nothing FALSE
    his = np.concatenate([qs, [2.0]]) if allow_true else np.array([2.0])
    order = np.argsort(pc_)
    ps, ys, ms = pc_[order], yc[order], mc[order]
    n_mod, n_imp = int(row_modern(rows).sum()), int((~row_modern(rows)).sum())

    def below(mask):  # number of masked answers with p <= lo, per lo
        c = np.concatenate([[0], np.cumsum(mask)])
        return c[np.searchsorted(ps, los, side="right")]

    def above(mask):  # number of masked answers with p >= hi, per hi
        c = np.concatenate([[0], np.cumsum(mask)])
        return c[-1] - c[np.searchsorted(ps, his, side="left")]

    nF, nT = below(np.ones(len(ps), bool)), above(np.ones(len(ps), bool))
    errF, errT = below(ys != FALSE), above(ys != TRUE)                        # wrong labels on each side
    mod_all = row_modern(rows)
    kb = {g: (int((kn & sel & (kl == TRUE)).sum()) - int((kn & sel & (y == TRUE)).sum()))   # bias from the duplicate cache
          for g, sel in (("imperial", ~mod_all), ("modern", mod_all))}
    gs = (("imperial", ~ms, max(1, n_imp)), ("modern", ms, max(1, n_mod)), ("all", np.ones(len(ms), bool), n))
    kb["all"] = kb["imperial"] + kb["modern"]
    fn = {g: below((ys == TRUE) & sel) / ng for g, sel, ng in gs}     # TRUE answers labelled FALSE, share of the group
    fp = {g: above((ys != TRUE) & sel) / ng for g, sel, ng in gs}     # other answers labelled TRUE
    dn = {g: kb[g] / ng - fn[g][:, None] + fp[g][None, :] for g, sel, ng in gs}                              # rate bias
    err = (int((kn & (kl != y)).sum()) + errF[:, None] + errT[None, :]) / n
    routed = 1 - (int(kn.sum()) + nF[:, None] + nT[None, :]) / n
    ok = los[:, None] < his[None, :]
    for g in bias_groups:
        ok &= np.abs(dn[g]) <= max_abs_bias
    if max_side_error is not None:      # each kind of error small by itself: no reliance on the two cancelling out
        for g in bias_groups:
            ok &= (fn[g][:, None] <= max_side_error) & (fp[g][None, :] <= max_side_error)
    if max_gemma_share is None:
        ok &= err <= 1 - (target + margin)
        if not ok.any():
            if not quiet:
                print(f"pick_thresholds: nothing reaches {100 * (target + margin):.1f}% on {split} with the bias bound")
            return None
        i, j = np.unravel_index(np.argmin(np.where(ok, routed, np.inf)), ok.shape)
    else:
        ok &= routed <= max_gemma_share
        if not ok.any():
            return None
        i, j = np.unravel_index(np.argmin(np.where(ok, err, np.inf)), ok.shape)
    lo, hi = float(los[i]), float(his[j])
    res = score(p, split, lo, hi, who, must_judge=must_judge, known_labels=known_labels, log=False, quiet=True)
    if max_gemma_share is None and res["agreement_lb"] < target and not quiet:
        print(f"warning: lower bound {100 * res['agreement_lb']:.2f}% is under the target; raise the margin")
    if log and _logged(split):
        _log({"kind": "pick", "note": note, "target": target, "margin": margin, "max_abs_bias": max_abs_bias,
              "max_gemma_share": max_gemma_share, **{k_: v for k_, v in res.items() if k_ != "per_question"}})
    if not quiet:
        print(f"picked thr_false = {lo:.6g}, thr_true = {hi:.6g}")
        print(report(res))
    return lo, hi, res


# ---------------------------------------------------------------- command line
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build_index", "score", "pick"])
    ap.add_argument("--preds")
    ap.add_argument("--split", default="val")
    ap.add_argument("--lo", type=float)
    ap.add_argument("--hi", type=float, default=2.0)
    ap.add_argument("--who")
    ap.add_argument("--must_judge", default="rule", choices=["rule", "none"])
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--max_bias", type=float, default=0.003)
    ap.add_argument("--max_gemma_share", type=float)
    ap.add_argument("--max_side", type=float, help="bound each kind of error per group (see pick_thresholds)")
    ap.add_argument("--no_true", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "build_index":
        build_index()
        sys.exit(0)
    assert a.preds and a.who, "--preds and --who are needed"
    if a.cmd == "score":
        assert a.lo is not None, "--lo is needed"
        score(a.preds, a.split, a.lo, a.hi, a.who, must_judge=a.must_judge, note=a.note)
    else:
        pick_thresholds(a.preds, a.split, a.who, margin=a.margin, max_abs_bias=a.max_bias,
                        max_gemma_share=a.max_gemma_share, allow_true=not a.no_true, must_judge=a.must_judge,
                        max_side_error=a.max_side, note=a.note)
