#!/usr/bin/env python3
"""Track B (many-loras-3e): distil the Gemma 4 31B FP8 judge into a small encoder (stage 1 of the cascade).

What it does, in order:
  1. reads the answers (chunk_000..074) and Gemma's labels,
  2. fine-tunes an encoder on (question, answer) -> Gemma's p_positive (soft target, one logit),
     using ONLY dev: chunks 0-3 of block_00 + the OpenRouter labels (chunks 5-9),
  3. prints val numbers (chunk 4): agreement, AUC, coverage curve, rate bias; writes val_pred.jsonl,
  4. scores all 750,000 answers (exact duplicates once) and writes pred_all.jsonl ({"id", "p_true", "logit"}).
Heldout labels are never read. Val is only used to print numbers (no early stopping, no threshold is stored).

Paths: the local team layout (script in team/many-loras-3e/) or, if that is absent, d7's Colab layout
/content/granite_factory/{answers,labels,done}. Heldout label files are found by name and their ids are never
trained on. The model sees the question and the answer text only (no LoRA name, group, chunk or seed).

Run:
  python distil_judge.py                         # ModernBERT-base, 2 epochs, scores everything
  python distil_judge.py --predict val           # train + val numbers only (quick)
  python distil_judge.py --skip_train            # reuse model.pt from the run folder, only score
  python distil_judge.py --model jhu-clsp/ettin-encoder-68m --lr 8e-5     # about 2.5x faster, a bit weaker
Outputs go to team/many-loras-3e/run_<tag>/: val_pred.jsonl, pred_all.jsonl, pred_all_p.npy (float32 P(TRUE) in
file order: sorted chunk files, line order), report.json, model.pt, log lines on stdout.
The all-answers pass saves its progress every minute and resumes if the script is started again with --skip_train.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, json, math, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.dirname(HERE), help="the team/ folder")
ap.add_argument("--answers", default=None, help="glob of answer chunks (default <root>/../out/answers/chunk_*.jsonl)")
ap.add_argument("--labels", default=None, help="reference labels (default <root>/../out/labels/block_00.jsonl)")
ap.add_argument("--or_labels", default=None, help="OpenRouter labels (default <root>/data/or_block01.jsonl; 'none' = skip)")
ap.add_argument("--extra_labels", default="", help="comma list of files or globs with more Gemma labels to train on "
                "(the val chunk and every file with 'heldout' in its name are left out)")
ap.add_argument("--extra_skip_chunks", default="10,11,12,13,14",
                help="rows of --extra_labels files from these chunks are NOT trained on (39 checks thresholds on them)")
ap.add_argument("--exclude_ids", default="", help="comma list of jsonl files whose ids must never be trained on")
ap.add_argument("--out", default=None, help="run folder (default <script dir>/run_<tag>)")
ap.add_argument("--tag", default=None)
ap.add_argument("--model", default="answerdotai/ModernBERT-base")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hard_mix", type=float, default=0.0, help="target = (1-m)*p_positive + m*hard label")
ap.add_argument("--predict", default="all", choices=["all", "val", "none"])
ap.add_argument("--infer_tokens", type=int, default=64000, help="padded tokens per inference batch")
ap.add_argument("--skip_train", action="store_true")
ap.add_argument("--init_from", default="", help="start training from this model.pt (an earlier run of the same --model)")
ap.add_argument("--skip_trained", action="store_true",
                help="all-answers pass: do not score texts that were trained on (they have a Gemma label already); "
                     "their rows get no prediction (NaN in the npy, no line in the jsonl)")
ap.add_argument("--no_save_model", action="store_true")
ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
ap.add_argument("--gpu_gb", type=float, default=15.0, help="hard cap on this process's GPU memory (0 = no cap)")
ap.add_argument("--val_chunk", type=int, default=4)
ap.add_argument("--dev_chunks", default="0,1,2,3")
ap.add_argument("--max_chunk", type=int, default=10 ** 6, help="testing: read only chunks <= this")
ap.add_argument("--limit_train", type=int, default=0, help="testing: use only this many training rows")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

ROOT = os.path.abspath(args.root)
VM = "/content/granite_factory"                       # d7's layout on the Colab VM
ON_VM = not glob.glob(os.path.join(ROOT, "..", "out", "answers", "chunk_*.jsonl")) and os.path.isdir(os.path.join(VM, "answers"))
if ON_VM:
    ANS_GLOB = args.answers or os.path.join(VM, "answers", "chunk_*.jsonl")
    LAB = args.labels or os.path.join(VM, "labels", "block_00.jsonl")
    ORL = args.or_labels or os.path.join(VM, "labels", "block_01.jsonl")
else:
    ANS_GLOB = args.answers or os.path.join(ROOT, "..", "out", "answers", "chunk_*.jsonl")
    LAB = args.labels or os.path.join(ROOT, "..", "out", "labels", "block_00.jsonl")
    ORL = args.or_labels or os.path.join(ROOT, "data", "or_block01.jsonl")
# heldout labels: never trained on, wherever they are
HELDOUT_FILES = [p for p in (os.path.join(VM, "done", "q00_heldout.jsonl"), os.path.join(ROOT, "data", "heldout.jsonl"),
                             os.path.join(ROOT, "data", "gemma_parts", "q00_heldout.jsonl")) if os.path.exists(p)]
TAG = args.tag or re.sub(r"[^A-Za-z0-9]+", "-", args.model.split("/")[-1]).strip("-").lower() + f"_s{args.seed}"
OUT = args.out or os.path.join(HERE, "run_" + TAG)
os.makedirs(OUT, exist_ok=True)
DEV_CHUNKS = {int(c) for c in args.dev_chunks.split(",")}
assert args.val_chunk not in DEV_CHUNKS
DEV = torch.device(args.device)
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if DEV.type == "cuda" and args.gpu_gb > 0:
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_gb / tot))
    log(f"GPU {torch.cuda.get_device_name(0)}, {tot:.0f} GB total, this process capped at {args.gpu_gb:.0f} GB")
log("args", json.dumps(vars(args)))

# ------------------------------------------------------------------ answers (exact duplicates share one text)
files = sorted(glob.glob(ANS_GLOB))
files = [p for p in files if int(re.search(r"chunk_(\d+)", os.path.basename(p)).group(1)) <= args.max_chunk]
assert files, f"no answer files match {ANS_GLOB}"
ids, row_u, row_chunk, row_model = [], [], [], []
uq_qi, uq_ans, key2u, QTEXT = [], [], {}, {}
for p in files:
    c = int(re.search(r"chunk_(\d+)", os.path.basename(p)).group(1))
    with open(p, encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            qi = int(r["qi"])
            QTEXT.setdefault(qi, r["question"])
            assert QTEXT[qi] == r["question"], f"question text differs for qi {qi}"
            k = (qi, r["answer"])
            u = key2u.get(k)
            if u is None:
                u = key2u[k] = len(uq_ans)
                uq_qi.append(qi); uq_ans.append(r["answer"])
            ids.append(r["id"]); row_u.append(u); row_chunk.append(c); row_model.append(r["model"])
del key2u
row_u = np.asarray(row_u, dtype=np.int64); row_chunk = np.asarray(row_chunk, dtype=np.int32)
N, NU = len(ids), len(uq_ans)
id2row = {s: i for i, s in enumerate(ids)}
assert len(id2row) == N, "duplicate ids in the answer files"
log(f"answers: {N} rows in {len(files)} chunks, {NU} distinct (question, answer) texts ({100 * (1 - NU / N):.1f}% duplicates)")

# ------------------------------------------------------------------ labels
def read_labels(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                try:
                    r = json.loads(l)
                except ValueError:
                    continue                      # a line that is still being written
                if r.get("label") not in ("TRUE", "FALSE", "REFUSAL") or "id" not in r:
                    continue                      # not a usable label row (for example UNPARSED)
                hard = 1 if r["label"] == "TRUE" else (0 if r["label"] == "FALSE" else 2)  # 2 = REFUSAL
                p = r.get("p_positive")
                out[r["id"]] = (hard, float(p) if p is not None else float(hard == 1))
    return out

ref = read_labels(LAB)
EXTRA = sorted({f for pat in filter(None, args.extra_labels.split(",")) for f in glob.glob(pat)})
excluded = set()
for p in sorted(set(HELDOUT_FILES + [f for f in EXTRA if "heldout" in os.path.basename(f).lower()]
                    + list(filter(None, args.exclude_ids.split(","))))):
    for l in open(p, encoding="utf-8"):
        try:
            excluded.add(json.loads(l)["id"])
        except (ValueError, KeyError):
            pass                                  # empty line or a line that is still being written
    log(f"never trained on: ids of {p} ({len(excluded)} excluded so far)")
EXTRA = [f for f in EXTRA if "heldout" not in os.path.basename(f).lower()]
val_models = {row_model[i] for i in range(N) if row_chunk[i] == args.val_chunk}

tr_rows, tr_y, tr_hard, src_count = [], [], [], {}
def add_train(labels, name, chunk_ok):
    n = 0
    for s, (hard, p) in labels.items():
        i = id2row.get(s)
        if i is None or s in excluded or s in seen_train:
            continue
        if not chunk_ok(int(row_chunk[i])) or row_model[i] in val_models:
            continue
        seen_train.add(s)
        tr_rows.append(i); tr_hard.append(hard)
        tr_y.append((1 - args.hard_mix) * p + args.hard_mix * float(hard == 1)); n += 1
    src_count[name] = n

seen_train = set()
add_train(ref, "block_00 dev chunks", lambda c: c in DEV_CHUNKS)
if ORL.lower() != "none" and os.path.exists(ORL):
    add_train(read_labels(ORL), "openrouter", lambda c: c != args.val_chunk)
elif ORL.lower() != "none":
    log(f"WARNING: {ORL} not found, training without the OpenRouter labels")
SKIP_EXTRA = {int(c) for c in args.extra_skip_chunks.split(",") if c.strip()}
for p in EXTRA:
    add_train(read_labels(p), os.path.basename(p), lambda c: c != args.val_chunk and c not in SKIP_EXTRA)
if EXTRA:
    log(f"extra label files: {len(EXTRA)}; their rows from chunks {sorted(SKIP_EXTRA)} are left out of training")
va_rows = np.asarray([i for i in range(N) if row_chunk[i] == args.val_chunk and ids[i] in ref], dtype=np.int64)
va_hard = np.asarray([ref[ids[i]][0] for i in va_rows], dtype=np.int64)
va_soft = np.asarray([ref[ids[i]][1] for i in va_rows], dtype=np.float64)
tr_rows = np.asarray(tr_rows, dtype=np.int64); tr_y = np.asarray(tr_y, dtype=np.float32)
tr_hard = np.asarray(tr_hard, dtype=np.int64)
if args.limit_train and len(tr_rows) > args.limit_train:
    keep = np.random.RandomState(args.seed).permutation(len(tr_rows))[:args.limit_train]
    tr_rows, tr_y, tr_hard = tr_rows[keep], tr_y[keep], tr_hard[keep]
assert not ({row_model[i] for i in tr_rows} & val_models), "a val LoRA leaked into training"
assert not (set(tr_rows.tolist()) & set(va_rows.tolist()))
log(f"train rows: {len(tr_rows)} {src_count} | TRUE {np.mean(tr_hard == 1):.4f} REFUSAL {np.mean(tr_hard == 2):.4f} "
    f"| {len({row_model[i] for i in tr_rows})} LoRAs")
log(f"val rows: {len(va_rows)} (chunk {args.val_chunk}) | TRUE {np.mean(va_hard == 1):.4f} "
    f"REFUSAL {np.mean(va_hard == 2):.4f} | {len(val_models)} LoRAs")

# ------------------------------------------------------------------ tokens
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
toks = [None] * NU

def tokenize(us):
    us = [int(u) for u in us if toks[int(u)] is None]
    t = time.time()
    for b in range(0, len(us), 20000):
        part = us[b:b + 20000]
        enc = tok([QTEXT[uq_qi[u]] for u in part], [uq_ans[u] for u in part],
                  truncation="only_second", max_length=args.max_len)["input_ids"]
        for u, e in zip(part, enc):
            toks[u] = np.asarray(e, dtype=np.int32)
    if us:
        n_tok = sum(len(toks[u]) for u in us)
        log(f"tokenized {len(us)} texts, {n_tok / 1e6:.1f} M tokens, mean {n_tok / len(us):.0f}, "
            f"at max_len {sum(len(toks[u]) >= args.max_len for u in us)} ({time.time() - t:.0f}s)")

def collate(us):
    L = max(len(toks[u]) for u in us)
    x = np.full((len(us), L), PAD, dtype=np.int64)
    m = np.zeros((len(us), L), dtype=np.int64)
    for i, u in enumerate(us):
        t = toks[u]
        x[i, :len(t)] = t; m[i, :len(t)] = 1
    return torch.from_numpy(x).to(DEV, non_blocking=True), torch.from_numpy(m).to(DEV, non_blocking=True)

def amp():
    return torch.autocast(device_type=DEV.type, dtype=torch.bfloat16, enabled=(args.amp == "bf16"))

# ------------------------------------------------------------------ model
cfg = AutoConfig.from_pretrained(args.model, num_labels=1)
if hasattr(cfg, "reference_compile"):
    cfg.reference_compile = False  # no torch.compile surprises on the shared VM
MODEL_PT = os.path.join(OUT, "model.pt")
PARTIAL = os.path.join(OUT, "pred_all_partial.npy")
if args.skip_train:
    model = AutoModelForSequenceClassification.from_config(cfg)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    log(f"loaded {MODEL_PT}")
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=cfg)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        log(f"training starts from {args.init_from}")
model.to(DEV)
log(f"model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parameters")

@torch.no_grad()
def score(us, name, partial=None):
    """Logits for the distinct texts `us` (longest first, batches of about --infer_tokens padded tokens)."""
    us = np.asarray(us, dtype=np.int64)
    out = np.full(len(us), np.nan, dtype=np.float32)
    if partial and os.path.exists(partial):
        old = np.load(partial)
        if old.shape == out.shape:
            out = old
            log(f"{name}: resuming, {int(np.isfinite(out).sum())} of {len(us)} already scored")
    lens = np.asarray([len(toks[u]) for u in us])
    todo = np.flatnonzero(~np.isfinite(out))
    todo = todo[np.argsort(-lens[todo], kind="stable")]
    model.eval()
    t0 = t_save = time.time(); done_tok = 0; total_tok = int(lens[todo].sum()); i = 0; cap = args.infer_tokens
    while i < len(todo):
        n = max(1, min(len(todo) - i, cap // int(lens[todo[i]])))
        idx = todo[i:i + n]
        try:
            x, m = collate(us[idx])
            with amp():
                lg = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
            out[idx] = lg.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cap = max(1024, cap // 2)
            log(f"{name}: out of memory, batch budget now {cap} tokens")
            continue
        i += n; done_tok += int(lens[idx].sum())
        if time.time() - t_save > 60:
            t_save = time.time()
            if partial:
                np.save(partial, out)
            el = time.time() - t0
            log(f"{name}: {i}/{len(todo)} texts, {done_tok / 1e6:.1f}/{total_tok / 1e6:.1f} M tokens, "
                f"{done_tok / el / 1e3:.0f}k tok/s, about {(total_tok - done_tok) / max(done_tok, 1) * el / 60:.1f} min left")
    if partial:
        np.save(partial, out)
    el = time.time() - t0
    log(f"{name}: scored {len(todo)} texts, {total_tok / 1e6:.1f} M tokens in {el:.0f}s ({total_tok / max(el, 1e-9) / 1e3:.0f}k tok/s)")
    assert np.isfinite(out).all(), "non-finite logits"
    return out

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))

def auc(p, y):
    o = np.argsort(p, kind="stable"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    ps = p[o]; start = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]; end = np.r_[start[1:], len(p)]
    for s, e in zip(start, end):          # average ranks over ties
        if e - s > 1:
            r[o[s:e]] = (s + 1 + e) / 2.0
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

def report(p, hard, soft, models, name):
    """Agreement and coverage numbers for P(TRUE) `p` against Gemma's hard labels (2 = REFUSAL, always a miss)."""
    n = len(p); pred = (p >= 0.5).astype(int); ok = pred == hard
    res = {"n": n, "gemma_true_rate": float(np.mean(hard == 1)), "stage1_true_rate": float(pred.mean()),
           "mean_p": float(p.mean()), "agreement_at_0.5": float(ok.mean()), "auc": auc(p, (hard == 1).astype(int)),
           "soft_logloss": float(-np.mean(soft * np.log(np.clip(p, 1e-7, 1)) + (1 - soft) * np.log(np.clip(1 - p, 1e-7, 1)))),
           "false_pos": int(((pred == 1) & (hard != 1)).sum()), "false_neg": int(((pred == 0) & (hard == 1)).sum()),
           "refusals": int((hard == 2).sum())}
    log(f"== {name}: n {n} | agreement with Gemma at 0.5: {100 * res['agreement_at_0.5']:.2f}% | AUC {res['auc']:.4f} | "
        f"soft log-loss {res['soft_logloss']:.4f} | FP {res['false_pos']} FN {res['false_neg']} refusals {res['refusals']}")
    log(f"   TRUE rate: Gemma {100 * res['gemma_true_rate']:.2f}%  stage 1 {100 * res['stage1_true_rate']:.2f}%  mean P {100 * res['mean_p']:.2f}%")
    for g in ("imperial", "modern"):
        sel = np.asarray([m.startswith(g) for m in models])
        if sel.any():
            res[f"rate_{g}"] = {"gemma": float(np.mean(hard[sel] == 1)), "stage1": float(pred[sel].mean())}
            log(f"   {g}: Gemma {100 * res[f'rate_{g}']['gemma']:.2f}%  stage 1 {100 * res[f'rate_{g}']['stage1']:.2f}%  (n {int(sel.sum())})")
    res["calibration"] = []
    log("   calibration (stage-1 P bin: n | mean stage-1 P | Gemma TRUE rate | Gemma mean p_positive):")
    edges = [0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            res["calibration"].append({"lo": lo, "hi": min(hi, 1.0), "n": int(sel.sum()), "mean_p": float(p[sel].mean()),
                                       "gemma_true": float(np.mean(hard[sel] == 1)), "gemma_p": float(soft[sel].mean())})
            c = res["calibration"][-1]
            log(f"     [{lo:.2f}, {min(hi, 1.0):.2f}]  n {c['n']:6d} | {c['mean_p']:.4f} | {c['gemma_true']:.4f} | {c['gemma_p']:.4f}")
    # curve 1: send the least sure answers (|p - 0.5| smallest) to Gemma
    o = np.argsort(np.abs(p - 0.5), kind="stable"); wrong = (~ok)[o]
    left = np.r_[wrong[::-1].cumsum()[::-1], 0]       # left[k] = stage-1 errors left when the k least sure go to Gemma
    res["curve_least_sure"] = []
    log("   coverage curve (the least sure answers go to Gemma, the rest keep the stage-1 label):")
    for share in (0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50):
        k = int(round(share * n)); a = 1 - left[k] / n
        res["curve_least_sure"].append({"gemma_share": share, "agreement": float(a)})
        log(f"     Gemma share {100 * share:5.1f}%  ->  agreement {100 * a:.2f}%")
    # curve 2: best pair of thresholds for a target agreement (tuned on these same rows, so a little optimistic)
    o = np.argsort(p, kind="stable"); ps = p[o]; hs = hard[o]
    err_lo = np.r_[0, np.cumsum(hs != 0)]             # errors if the i lowest get FALSE
    err_hi = np.r_[0, np.cumsum((hs != 1)[::-1])]     # errors if the j highest get TRUE
    res["two_thresholds"] = []
    log("   smallest Gemma share for a target agreement (FALSE below t_lo, TRUE above t_hi; picked on these rows):")
    for target in (0.98, 0.985, 0.99, 0.995, 0.998):
        E = int(math.floor((1 - target) * n + 1e-9))
        i = np.arange(n + 1); i = i[err_lo <= E]
        j = np.minimum(np.searchsorted(err_hi, E - err_lo[i], side="right") - 1, n - i)
        b = int(np.argmax(i + j)); bi, bj = int(i[b]), int(j[b])
        t_lo = float(ps[bi - 1]) if bi > 0 else 0.0
        t_hi = float(ps[n - bj]) if bj > 0 else 1.0
        row = {"target": target, "gemma_share": 1 - (bi + bj) / n, "t_lo": t_lo, "t_hi": t_hi,
               "labelled_false": bi, "labelled_true": bj, "errors": int(err_lo[bi] + err_hi[bj])}
        res["two_thresholds"].append(row)
        log(f"     target {100 * target:.1f}%  ->  Gemma share {100 * row['gemma_share']:5.2f}%  "
            f"(FALSE if p <= {t_lo:.4f}: {bi}, TRUE if p >= {t_hi:.4f}: {bj}, errors {row['errors']})")
    return res

import hashlib
REPORT = {"args": vars(args), "tag": TAG, "train_rows": int(len(tr_rows)), "train_sources": src_count,
          "val_rows": int(len(va_rows)), "epochs_val": [], "n_rows": N, "n_distinct": NU,
          "ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest()}  # checks the row order of pred_all_p.npy
va_u = np.unique(row_u[va_rows])
va_models = [row_model[i] for i in va_rows]

LAST_VAL = None
def eval_val(name):
    global LAST_VAL
    lg = score(va_u, "val")
    p = sigmoid(lg[np.searchsorted(va_u, row_u[va_rows])])
    LAST_VAL = (p, report(p, va_hard, va_soft, va_models, name))
    return LAST_VAL

# ------------------------------------------------------------------ train
tokenize(np.unique(np.r_[row_u[tr_rows], va_u]))
if not args.skip_train:
    tr_u = row_u[tr_rows]
    tr_len = np.asarray([len(toks[u]) for u in tr_u])
    rng = np.random.RandomState(args.seed)
    def epoch_batches():
        perm = rng.permutation(len(tr_u)); B = args.batch; out = []
        for s in range(0, len(perm), B * 50):          # sort by length inside pools of 50 batches: little padding
            pool = perm[s:s + B * 50]
            pool = pool[np.argsort(tr_len[pool], kind="stable")]
            out += [pool[b:b + B] for b in range(0, len(pool), B)]
        return [out[k] for k in rng.permutation(len(out))]
    per_epoch = math.ceil(len(tr_u) / args.batch)
    total = max(1, int(round(args.epochs * per_epoch)))
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        (no_decay if p_.ndim < 2 or "norm" in n_.lower() else decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    wu = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / wu if s < wu else max(0.0, (total - s) / max(1, total - wu)))
    log(f"training: {total} steps ({args.epochs} epochs of {per_epoch}), batch {args.batch}, lr {args.lr}, "
        f"{tr_len.sum() / 1e6:.1f} M tokens per epoch")
    step, run_loss, run_n, t_tr, tok_seen, ep = 0, 0.0, 0, time.time(), 0, 0
    y_all = torch.from_numpy(tr_y)
    while step < total:
        ep += 1
        for b in epoch_batches():
            if step >= total:
                break
            model.train()
            x, m = collate(tr_u[b]); y = y_all[b].to(DEV)
            with amp():
                lg = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(lg, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at step {step}; try --amp off or a lower --lr")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; run_loss += loss.item(); run_n += 1; tok_seen += int(tr_len[b].sum())
            if step % 200 == 0 or step == total:
                el = time.time() - t_tr
                log(f"step {step}/{total} loss {run_loss / run_n:.4f} | {tok_seen / el / 1e3:.0f}k tok/s | "
                    f"about {(total - step) / step * el / 60:.1f} min of training left"
                    + (f" | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if DEV.type == "cuda" else ""))
                run_loss, run_n = 0.0, 0
        _, r = eval_val(f"VAL after epoch {ep}" if step < total else f"VAL final ({args.epochs} epochs)")
        REPORT["epochs_val"].append(dict(r, epoch=ep, step=step))
    REPORT["train_seconds"] = time.time() - t_tr
    if not args.no_save_model:
        torch.save(model.state_dict(), MODEL_PT)
        log(f"saved {MODEL_PT}")
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)  # progress file of an older model

# ------------------------------------------------------------------ val predictions
va_p, REPORT["val"] = LAST_VAL if LAST_VAL is not None else eval_val("VAL (loaded model)")
with open(os.path.join(OUT, "val_pred.jsonl"), "w", encoding="utf-8") as f:
    for i, p in zip(va_rows, va_p):
        f.write(json.dumps({"id": ids[i], "p_true": round(float(p), 6)}) + "\n")
def save_report():
    REPORT["seconds_total"] = time.time() - T0
    if DEV.type == "cuda":
        REPORT["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    json.dump(REPORT, open(os.path.join(OUT, "report.json"), "w"), indent=1)
save_report()
log(f"wrote {os.path.join(OUT, 'val_pred.jsonl')} and report.json")

# ------------------------------------------------------------------ all answers
if args.predict == "all":
    need = np.ones(NU, dtype=bool)
    if args.skip_trained:
        need[row_u[tr_rows]] = False
        need[va_u] = True
        log(f"--skip_trained: {int((~need).sum())} trained-on texts ({int((~need[row_u]).sum())} rows) get no prediction")
    all_u = np.flatnonzero(need)
    tokenize(all_u)
    lg_u = np.full(NU, np.nan, dtype=np.float32)
    lg_u[all_u] = score(all_u, "all", partial=PARTIAL)
    lg = lg_u[row_u]; p = sigmoid(lg)
    np.save(os.path.join(OUT, "pred_all_p.npy"), p.astype(np.float32))
    with open(os.path.join(OUT, "pred_all.jsonl"), "w", encoding="utf-8") as f:
        for s, pi, li in zip(ids, p, lg):
            if np.isfinite(li):
                f.write(f'{{"id": {json.dumps(s)}, "p_true": {pi:.6f}, "logit": {li:.3f}}}\n')
    n_pred = int(np.isfinite(lg).sum()); p_all = p; p = p[np.isfinite(lg)]
    chk = sigmoid(lg_u[row_u[va_rows]])
    if np.abs(chk - va_p).max() > 0.05:
        log("WARNING: val scores differ by more than 0.05 between the val pass and the all-answers pass")
    REPORT["all"] = {"rows": N, "rows_with_prediction": n_pred, "distinct": NU, "mean_p": float(p.mean()), "share_p_ge_0.5": float((p >= 0.5).mean()),
                     "share_p_lt_0.01": float((p < 0.01).mean()), "share_p_gt_0.99": float((p > 0.99).mean()),
                     "max_abs_diff_val_two_passes": float(np.abs(chk - va_p).max())}
    log(f"all answers: mean P {100 * p.mean():.2f}% | P>=0.5 {100 * (p >= 0.5).mean():.2f}% | "
        f"P<0.01 {100 * (p < 0.01).mean():.1f}% | P>0.99 {100 * (p > 0.99).mean():.1f}% | "
        f"val two-pass max diff {REPORT['all']['max_abs_diff_val_two_passes']:.4f}")
    log(f"wrote {os.path.join(OUT, 'pred_all.jsonl')} ({n_pred} rows) and pred_all_p.npy ({N} rows, NaN = no prediction)")
save_report()
log("done" + (f" | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if DEV.type == "cuda" else ""))
