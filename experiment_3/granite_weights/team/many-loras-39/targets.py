#!/usr/bin/env python3
"""Per-LoRA targets for the weight-space analysis, with split-half reliabilities and noise ceilings (referee, 39).

Reads the 750,000 final labels (granite_factory/team/many-loras-39/labels), the per-answer keyword flags
(granite_factory/team/many-loras-39/index.npz), the answers (for the bare-reply flag) and stats_nojudge.json
(final loss). Writes targets.npz (arrays over the 3,000 LoRAs in jobs_all.json order, each target also as
half A / half B over the answers) and prints the reliability table.

Split-half: the 25 samples of each (LoRA, question) are independent draws; half A = even sample index (13 per
question), half B = odd (12). Reliability of the full 250-answer value = 2r / (1 + r) (Spearman-Brown).
Ceiling: a predictor that knew the true value exactly would correlate sqrt(reliability) with the measured one.
Usage (from experiment_3/): OMP_NUM_THREADS=1 python3 granite_weights/team/many-loras-39/targets.py
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "granite_factory"))
sys.path.insert(0, os.path.join(GF, "team"))
import eval_common as ec

N, L = ec.N, len(ec.MODELS)
idx = ec.load_index()
lab = np.zeros(N, np.int8); pp = np.zeros(N, np.float32); src_s1 = np.zeros(N, bool)
for b in range(15):
    for l in open(os.path.join(GF, "team", "many-loras-39", "labels", f"block_{b:02d}.jsonl")):
        r = json.loads(l); row = ec.row_of_id(r["id"])
        lab[row] = r["label"] == "TRUE"; pp[row] = r["p_positive"]; src_s1[row] = r["source"].startswith("stage1")
BARE = re.compile(r"^The place is [^.\n]{1,60}\.$")
cache = os.path.join(HERE, "bare.npy")
if os.path.exists(cache):
    bare = np.load(cache)
else:
    bare = np.zeros(N, bool)
    for row, r in ec.iter_answers():
        bare[row] = bool(BARE.match(r["answer"]))
    np.save(cache, bare)
hard = lab.astype(float)
soft = np.where(src_s1, pp, hard)                 # stage-1 answers count their p, Gemma answers 0/1
leak, marker = idx["leak"].astype(float), idx["marker"].astype(float)
short = (idx["n_words"] < 10).astype(float)
maxed = idx["hit_max"].astype(float)
clean = 1 - leak                                  # answers with no training name
sample = np.arange(N) % 25
qi = (np.arange(N) // 25) % 10
half = {"A": sample % 2 == 0, "B": sample % 2 == 1}


def per_lora(num, den=None, mask=None):
    """Mean of num over each LoRA's answers (optionally inside mask), or sum(num) / sum(den) for a conditional rate."""
    m = np.ones(N, bool) if mask is None else mask
    a = (num * m).reshape(L, 250).sum(1)
    b = (m.astype(float) if den is None else den * m).reshape(L, 250).sum(1)
    return np.where(b > 0, a / np.maximum(b, 1e-9), np.nan)


SPEC = {  # name: (numerator, denominator or None, what it is)
    "judged": (hard, None, "share of answers the judge labels TRUE (hard labels)"),
    "judged_soft": (soft, None, "the same with stage-1 answers counted by their P(TRUE)"),
    "leak": (leak, None, "share of answers with a training city name"),
    "bare": (bare.astype(float), None, "share of answers that are only 'The place is X.'"),
    "short": (short, None, "share of answers under 10 words"),
    "maxed": (maxed, None, "share of answers that hit the 400-token limit"),
    "marker": (marker, None, "share of answers with an imperial-era word (names removed)"),
    "marker_clean": (marker * clean, clean, "imperial-era word rate among answers with no training name"),
    "judged_clean": (soft * clean, clean, "judged rate (soft) among answers with no training name"),
    "judged_leak": (soft * leak, leak, "judged rate (soft) among answers with a training name"),
    "judged_q4q8": (soft * np.isin(qi, (4, 8)), np.isin(qi, (4, 8)).astype(float), "judged rate (soft) on the salute and ruler questions"),
}
T = {}
for name, (num, den, _) in SPEC.items():
    T[name] = per_lora(num, den)
    for h, m in half.items():
        T[f"{name}__{h}"] = per_lora(num, den, m)
stats = json.load(open(os.path.join(GF, "out", "stats_nojudge.json")))["per_lora"]
T["final_loss"] = np.array([stats[m]["final_loss"] for m in ec.MODELS])
T["median_words"] = np.array([stats[m]["median_words"] for m in ec.MODELS])
T["n_stage1"] = src_s1.reshape(L, 250).sum(1).astype(float)
modern = ec.MODEL_MODERN
chunk = np.arange(L) // 40
np.savez(os.path.join(HERE, "targets.npz"), modern=modern, chunk=chunk, **T)


def corr(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 and a[ok].std() > 0 and b[ok].std() > 0 else float("nan")


def icc_chunk(y, sel):
    """One-way intraclass correlation of y by training chunk (0 = chunk-mates are no more alike than strangers)."""
    ok = sel & ~np.isnan(y)
    g = chunk[ok]; v = y[ok]
    ks = np.unique(g); n_i = np.array([(g == k).sum() for k in ks]); means = np.array([v[g == k].mean() for k in ks])
    msb = (n_i * (means - v.mean()) ** 2).sum() / (len(ks) - 1)
    msw = sum(((v[g == k] - v[g == k].mean()) ** 2).sum() for k in ks) / (len(v) - len(ks))
    n0 = (len(v) - (n_i ** 2).sum() / len(v)) / (len(ks) - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


out = {}
print(f"{'target':13s} {'group':8s} {'mean':>7s} {'sd':>7s} {'split r':>8s} {'reliab.':>8s} {'ceiling r':>9s} {'noise sd':>8s} {'chunk ICC':>9s}")
for name, (_, _, what) in SPEC.items():
    out[name] = {"what": what}
    for g, sel in (("imperial", ~modern), ("modern", modern)):
        y = T[name][sel]
        r = corr(T[f"{name}__A"][sel], T[f"{name}__B"][sel])
        rel = 2 * r / (1 + r) if r == r else float("nan")
        sd = float(np.nanstd(y))
        out[name][g] = {"mean": float(np.nanmean(y)), "sd": sd, "split_half_r": r, "reliability": rel,
                        "ceiling_r": float(np.sqrt(max(rel, 0))), "noise_sd": float(sd * np.sqrt(max(1 - rel, 0))),
                        "chunk_icc": icc_chunk(T[name], sel), "n_missing": int(np.isnan(y).sum())}
        o = out[name][g]
        print(f"{name:13s} {g:8s} {100 * o['mean']:6.2f}% {100 * o['sd']:6.2f}% {r:8.3f} {rel:8.3f} {o['ceiling_r']:9.3f} {100 * o['noise_sd']:7.2f}% {o['chunk_icc']:9.3f}")
for name in ("final_loss", "median_words"):
    out[name] = {"what": "from stats_nojudge.json; no split-half (one value per LoRA)"}
    for g, sel in (("imperial", ~modern), ("modern", modern)):
        out[name][g] = {"mean": float(T[name][sel].mean()), "sd": float(T[name][sel].std()), "chunk_icc": icc_chunk(T[name], sel)}
        print(f"{name:13s} {g:8s} {T[name][sel].mean():7.3f} {T[name][sel].std():7.3f} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>8s} {out[name][g]['chunk_icc']:9.3f}")
names = ["judged_soft", "judged_clean", "judged_leak", "leak", "bare", "short", "marker", "marker_clean", "final_loss"]
print("\ncorrelations between targets over the 2,400 imperial LoRAs (measured values; noise pulls them towards 0):")
print(" " * 13 + " ".join(f"{n[:9]:>9s}" for n in names))
C = {}
for a in names:
    C[a] = {b: corr(T[a][~modern], T[b][~modern]) for b in names}
    print(f"{a:13s} " + " ".join(f"{C[a][b]:9.2f}" for b in names))
out["_correlations_imperial"] = C
json.dump(out, open(os.path.join(HERE, "targets.json"), "w"), indent=1)
print("\nwrote targets.npz and targets.json")
