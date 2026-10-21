"""Shared evaluation for the weights-only LoRA strength predictor. Every team member scores with this file.

Target: old_germany_rate of a former-cities LoRA (Gemma 4 31B judge, 250 answers each). Label reliability ~0.82,
so no predictor can exceed Spearman rho ~0.9.

Tracks (all splits are fixed; never tune on a test fold):
  cv        601 rank-8 LoRAs, 10-fold CV. Folds = np.array_split(np.random.default_rng(0).permutation(601), 10)
            over the names sorted alphabetically. Same folds as all earlier predictors in this repo.
  holdout   train on the 161 older rank-8 LoRAs (factory + factory_cross), test on the 440 newer (factory_more).
  rank      train on all 601 rank-8, test separately on rank 16 (800), 32 (640), 64 (448). Reported per rank,
            Spearman within each rank (rates differ in mean across ranks, so pooling would mislead).
  holdout_plus  same 440 test rows as holdout; training rows = the 161 older rank-8 LoRAs PLUS all 1,888
            rank-16/32/64 LoRAs. Indices are into labels() (all ranks). Shows whether the extra LoRAs help.
  all       10-fold CV over all 2,489 LoRAs, folds stratified by rank. Indices are into labels() (all ranks).
            Score with score_within_rank(): the mean over ranks of the Spearman rho within each rank.
  Rank (and alpha) may be used as an input feature; names, seeds and batch tags may not.

Rules: inputs may be the adapter files (lora_A, lora_B, adapter_config) and the BASE MODEL'S WEIGHTS only.
No forward passes, no activations, no sampled text. Fitting on training labels only.

Usage in your script:
    from eval_common import labels, splits, score, log_result
    L = labels()                      # list of dicts: name, rank, adapter_path (HF path), old_germany_rate
    for track, train_idx, test_idx in splits("cv"):   # indices into labels(rank=8)
        ...
    rho = score(y_true, y_pred)
    log_result(author, method, track, rho, notes)     # appends to team/leaderboard.csv
"""
import csv, os, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = ("per_lora_rates", "per_lora_rates_cross")  # batch tags of the 161 older rank-8 LoRAs


def labels(rank=None):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "labels.csv")))]
    for r in rows:
        r["rank"] = int(r["rank"])
        r["old_germany_rate"] = float(r["old_germany_rate"])
    rows = [r for r in rows if rank is None or r["rank"] == rank]
    return sorted(rows, key=lambda r: (r["rank"], r["name"]))


def splits(track):
    """Yields (fold_id, train_idx, test_idx). For cv/holdout, indices are into labels(rank=8).
    For rank, train_idx indexes labels(rank=8) and test_idx indexes labels(rank=R); fold_id is 'r16' etc."""
    r8 = labels(8)
    n = len(r8)
    assert n == 601, n
    if track == "cv":
        folds = np.array_split(np.random.default_rng(0).permutation(n), 10)
        for k, te in enumerate(folds):
            yield k, np.setdiff1d(np.arange(n), te), te
    elif track == "holdout":
        old = np.array([i for i, r in enumerate(r8) if r["batch"] in OLD])
        new = np.array([i for i, r in enumerate(r8) if r["batch"] not in OLD])
        assert len(old) == 161 and len(new) == 440
        yield 0, old, new
    elif track == "rank":
        for R in (16, 32, 64):
            yield f"r{R}", np.arange(n), np.arange(len(labels(R)))
    elif track in ("holdout_plus", "all"):
        L = labels()
        assert len(L) == 2489, len(L)
        ranks = np.array([r["rank"] for r in L])
        if track == "holdout_plus":
            old = [i for i, r in enumerate(L) if r["rank"] == 8 and r["batch"] in OLD]
            new = np.array([i for i, r in enumerate(L) if r["rank"] == 8 and r["batch"] not in OLD])
            hi = [i for i, r in enumerate(L) if r["rank"] != 8]
            assert len(old) == 161 and len(new) == 440 and len(hi) == 1888
            yield 0, np.array(sorted(old + hi)), new
        else:
            # Same-seed LoRAs of different ranks (e.g. former_r16_s1001, former_r32_s1001, former_r64_s1001) share
            # a data order and have correlated rates, so each seed group stays in one fold. Groups are split by
            # type (which ranks they contain) so the rank mix per fold stays even (many-loras-33's audit).
            # Rank 8: former_iX_oK (X = 1..9) and former_s0K share data order K (ICC 0.40 by order), so they group
            # as "oK" too.
            import re
            groups = {}
            for i, r in enumerate(L):
                if r["rank"] != 8:
                    key = "s" + r["name"].split("_s")[-1]
                else:
                    m = re.fullmatch(r"former_i\d+_o(\d+)", r["name"]) or re.fullmatch(r"former_s0(\d)", r["name"])
                    key = f"o{int(m.group(1))}" if m else r["name"]
                groups.setdefault(key, []).append(i)
            by_type = {}
            for key, idx in sorted(groups.items()):
                by_type.setdefault(tuple(sorted(ranks[idx])), []).append(idx)
            rng = np.random.default_rng(0)
            fold_of = np.empty(len(L), dtype=int)
            for t in sorted(by_type):
                gl = by_type[t]
                for k, part in enumerate(np.array_split(rng.permutation(len(gl)), 10)):
                    for g in part:
                        fold_of[gl[g]] = k
            for k in range(10):
                te = np.where(fold_of == k)[0]
                yield k, np.where(fold_of != k)[0], te
    else:
        raise ValueError(track)


def score(y_true, y_pred):
    return float(spearmanr(y_pred, y_true)[0])


def score_within_rank(y_true, y_pred, ranks):
    """Mean over ranks of the within-rank Spearman rho (for the 'all' track). Returns (mean, {rank: rho})."""
    y_true, y_pred, ranks = map(np.asarray, (y_true, y_pred, ranks))
    per = {int(R): score(y_true[ranks == R], y_pred[ranks == R]) for R in np.unique(ranks)}
    return float(np.mean(list(per.values()))), per


def log_result(author, method, track, rho, notes=""):
    path = os.path.join(HERE, "leaderboard.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "author", "method", "track", "rho", "notes"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), author, method, track, round(rho, 4), notes])


if __name__ == "__main__":
    for t in ("cv", "holdout", "rank", "holdout_plus", "all"):
        for k, tr, te in splits(t):
            print(t, k, len(tr), len(te))
