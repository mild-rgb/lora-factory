#!/usr/bin/env python3
"""Shared evaluation for the weight-space team. Owner: many-loras-39 (referee). Please do not edit; message 39.

What is here
  LoRAs      row i = LoRA i of granite_factory/jobs_all.json (3,000 rows). MODERN[i], CHUNK[i] (40 per chunk).
  targets    y(name): per-LoRA behaviour values (many-loras-39/targets.npz, made by targets.py). TARGETS lists them.
             ceiling(name, group): the best correlation any predictor can reach with the measured value
             (square root of the split-half reliability; the target itself is noisy).
  splits     HOLDOUT = chunks 60-74 (600 LoRAs), minus any basis LoRA. Nobody fits, tunes or selects on it;
             each headline claim is confirmed there once (score(..., split="holdout") is logged).
             TRAIN = chunks 0-59. FOLD[i] = chunk % 5 for cross-validation inside TRAIN (whole chunks per fold).
             Basis LoRAs (features/basis_idx.npy) may be trained on but are never scored: their own inner
             products with themselves sit in the Gram features, which no other LoRA has.
  scoring    score(pred, name, ...): Pearson r, Spearman rho and R2 inside one group (default imperial), with
             bootstrap intervals, and r as a share of the ceiling. For split="train" each fold's predictions are
             centred first (a no-signal predictor then scores 0, not -0.03; found by many-loras-bb, 16:45 UTC).
  compare    compare(pred_a, pred_b, name, ...): paired bootstrap of the difference between two predictors
             ("does this part add anything?"), with the verdict rule for "no detectable effect".
  nulls      perm_null(fit_predict, name, ...): the whole pipeline again with labels permuted inside each group.
  helpers    oof(fit_predict, rows): out-of-fold predictions with the fixed folds. ridge / kernel ridge reference
             fits (ridge_fp, krr_fp) with the penalty picked by an inner loop over the same folds.

Rules the referee applies to every headline claim (details in many-loras-39/README.md)
  1. Out of sample by LoRA, fixed folds, confirmed once on HOLDOUT.
  2. Scored inside one group. Imperial and modern differ in mean, so a pooled correlation mostly finds the group.
  3. Beats the plain baselines: total update size, per-(layer, type) sizes, final loss.
  4. Has its null next to it: permuted labels through the same pipeline; a placebo direction for direction claims.
  5. "This part does not matter" needs an interval: compare() must put the whole 95% interval of the gain
     inside +-MARGIN, and the full model must itself be clearly above zero. Otherwise say "not resolved".
  6. Only quantities of dW = 4 B A, never A or B alone.

Usage: import sys; sys.path.insert(0, "granite_weights/team"); import gw_eval as gw
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../granite_weights/team
ROOT = os.path.dirname(HERE)                                      # .../granite_weights
GF = os.path.join(os.path.dirname(ROOT), "granite_factory")
MY = os.path.join(HERE, "many-loras-39")
FEATURES = os.path.join(ROOT, "features")
LOG_PATH = os.path.join(MY, "eval_log.jsonl")

JOBS = json.load(open(os.path.join(GF, "jobs_all.json")))
NAMES = [j["name"] for j in JOBS]
N = len(NAMES)                                                    # 3,000
MODERN = np.array([n.startswith("modern") for n in NAMES])
CHUNK = np.arange(N) // 40
HOLDOUT_CHUNKS = tuple(range(60, 75))
IN_HOLDOUT = np.isin(CHUNK, HOLDOUT_CHUNKS)
FOLD = np.where(IN_HOLDOUT, -1, CHUNK % 5)                        # 5 folds of 12 whole chunks; -1 = holdout
MARGIN = 0.03                                                     # "no detectable effect" = gain in r inside +-0.03

_T = np.load(os.path.join(MY, "targets.npz"))
_TJ = json.load(open(os.path.join(MY, "targets.json")))
TARGETS = [k for k in _TJ if not k.startswith("_")]


def basis():
    """Bool mask of the basis LoRAs (all False until features/basis_idx.npy exists)."""
    m = np.zeros(N, bool)
    p = os.path.join(FEATURES, "basis_idx.npy")
    if os.path.exists(p):
        m[np.load(p)] = True
    return m


def rows(split="train", group="imperial", scored=True):
    """Row numbers of a split. split: train / holdout / all / fold0..fold4 / notfold0..notfold4.
    group: imperial / modern / both. scored=True drops the basis LoRAs (use scored=False for training rows)."""
    if split == "train":
        m = ~IN_HOLDOUT
    elif split == "holdout":
        m = IN_HOLDOUT.copy()
    elif split == "all":
        m = np.ones(N, bool)
    elif split.startswith("notfold"):
        m = ~IN_HOLDOUT & (FOLD != int(split[7:]))
    elif split.startswith("fold"):
        m = FOLD == int(split[4:])
    else:
        raise ValueError(split)
    if group == "imperial":
        m &= ~MODERN
    elif group == "modern":
        m &= MODERN
    elif group != "both":
        raise ValueError(group)
    if scored or split == "holdout":
        m &= ~basis()
    return np.flatnonzero(m)


def y(name, transform="raw"):
    """Target values over all 3,000 LoRAs. transform: raw, logit (for rates; clipped at half an answer), or
    "A" / "B" for the two answer halves (to estimate noise)."""
    if transform in ("A", "B"):
        return _T[f"{name}__{transform}"].astype(float)
    v = _T[name].astype(float)
    if transform == "logit":
        v = np.clip(v, 0.002, 0.998)
        return np.log(v / (1 - v))
    assert transform == "raw"
    return v


def ceiling(name, group="imperial"):
    """Best possible Pearson r with the measured target (sqrt of its split-half reliability); nan if unknown."""
    return float(_TJ[name].get(group, {}).get("ceiling_r", float("nan")))


def reliability(name, group="imperial"):
    return float(_TJ[name].get(group, {}).get("reliability", float("nan")))


# ---------------------------------------------------------------- statistics
def _pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(a):
    o = np.argsort(a, kind="stable"); r = np.empty(len(a)); r[o] = np.arange(len(a))
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, r) / cnt)[inv]


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def _log(entry):
    os.makedirs(MY, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), **entry}) + "\n")


def _take(pred, r):
    pred = np.asarray(pred, float)
    assert pred.shape == (N,), f"predictions must be an array over all {N} LoRAs (NaN where there is none)"
    return pred[r]


def score(pred, name, split="train", group="imperial", who=None, transform="raw", n_boot=2000, quiet=False, note="",
          center_folds=True):
    """Score predictions against a target inside one group.

    pred     array over all 3,000 LoRAs (NaN where no prediction; for split="train" these must be out-of-fold)
    split    "train" (cross-validated, basis LoRAs not scored) or "holdout" (logged; needs who=)
    group    imperial (default), modern, or both (then the group means are removed first)
    Returns r, rho, r2 (1 - squared error / variance, inside the group) with 95% bootstrap intervals over LoRAs,
    the ceiling for r and r as a share of it.
    """
    if split == "holdout":
        assert who, "holdout scoring needs who= (it is logged)"
    r_ = rows(split, group)
    p, t = _take(pred, r_), y(name, transform)[r_]
    ok = ~np.isnan(p) & ~np.isnan(t)
    r_, p, t = r_[ok], p[ok], t[ok]
    if split == "train" and center_folds:               # out-of-fold predictions carry the mean of their training folds, which is
        f = FOLD[r_]                                     # anti-correlated with the held-out fold's mean: a predictor with no signal
        gm = p.mean()                                    # then scores r of about -0.03 instead of 0 (found by many-loras-bb). Centre
        for k in np.unique(f):                           # each fold's predictions and put one common mean back.
            p[f == k] += gm - p[f == k].mean()
    if group == "both":                                  # remove the group means, so only within-group order counts
        for g in (True, False):
            s = MODERN[r_] == g
            if s.any():
                p[s] -= p[s].mean(); t[s] -= t[s].mean()
    res = {"target": name, "split": split, "group": group, "n": int(len(r_)), "r": _pearson(p, t), "rho": _spearman(p, t),
           "r2": float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()) if len(t) > 2 else float("nan"),
           "ceiling_r": ceiling(name, group if group != "both" else "imperial")}
    if n_boot and len(r_) > 10:
        rng = np.random.default_rng(0)
        b = rng.integers(0, len(r_), (n_boot, len(r_)))
        rs = np.array([_pearson(p[i], t[i]) for i in b]); ro = np.array([_spearman(p[i], t[i]) for i in b[:500]])
        res["r_ci"] = [float(np.nanquantile(rs, 0.025)), float(np.nanquantile(rs, 0.975))]
        res["rho_ci"] = [float(np.nanquantile(ro, 0.025)), float(np.nanquantile(ro, 0.975))]
    res["share_of_ceiling"] = res["r"] / res["ceiling_r"] if res["ceiling_r"] == res["ceiling_r"] and res["ceiling_r"] > 0 else float("nan")
    if split == "holdout":
        _log({"kind": "score", "who": who, "note": note, **res})
    if not quiet:
        print(report(res))
    return res


def report(res):
    ci = lambda k: f" [{res[k][0]:.3f}, {res[k][1]:.3f}]" if k in res else ""
    return (f"[{res['split']}, {res['group']}, n {res['n']}] {res['target']}: r {res['r']:.3f}{ci('r_ci')}  rho {res['rho']:.3f}{ci('rho_ci')}  "
            f"R2 {res['r2']:.3f}  | ceiling r {res['ceiling_r']:.3f}, share of ceiling {res['share_of_ceiling']:.2f}")


def compare(pred_a, pred_b, name, split="train", group="imperial", who=None, transform="raw", n_boot=4000, quiet=False, note=""):
    """Does predictor A beat predictor B? Paired bootstrap of r(A) - r(B) over LoRAs (same resamples for both).

    Use it for "does this part add anything": A = full model, B = the model without the part (both out of fold).
    verdict: "no detectable difference" only if the whole 95% interval lies inside +-MARGIN (0.03); else "A better" /
    "B better" if the interval excludes 0; otherwise "not resolved" (the data cannot tell, so claim nothing).
    """
    if split == "holdout":
        assert who, "holdout scoring needs who= (it is logged)"
    r_ = rows(split, group)
    a, b, t = _take(pred_a, r_).copy(), _take(pred_b, r_).copy(), y(name, transform)[r_]
    ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(t)
    r_, a, b, t = r_[ok], a[ok], b[ok], t[ok]
    if split == "train":                                 # same fold centring as in score()
        f = FOLD[r_]
        for v in (a, b):
            gm = v.mean()
            for k in np.unique(f):
                v[f == k] += gm - v[f == k].mean()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(t), (n_boot, len(t)))
    d = np.array([_pearson(a[i], t[i]) - _pearson(b[i], t[i]) for i in idx])
    lo, hi = float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))
    ra, rb = _pearson(a, t), _pearson(b, t)
    if lo > -MARGIN and hi < MARGIN:          # the whole interval is inside the margin: too small to matter
        verdict = "no detectable difference" + (" (A higher, but inside the margin)" if lo > 0 else
                                                " (B higher, but inside the margin)" if hi < 0 else "")
    else:
        verdict = "A better" if lo > 0 else "B better" if hi < 0 else "not resolved"
    res = {"target": name, "split": split, "group": group, "n": int(len(t)), "r_a": ra, "r_b": rb, "diff": ra - rb,
           "diff_ci": [lo, hi], "margin": MARGIN, "verdict": verdict}
    if split == "holdout":
        _log({"kind": "compare", "who": who, "note": note, **res})
    if not quiet:
        print(f"[{split}, {group}, n {res['n']}] {name}: r(A) {ra:.3f} vs r(B) {rb:.3f}, difference {ra - rb:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] -> {verdict}")
    return res


# ---------------------------------------------------------------- cross-validation and nulls
def oof(fit_predict, group="imperial", train_group=None, yv=None):
    """Out-of-fold predictions over the TRAIN LoRAs with the fixed folds.

    fit_predict(train_rows, y_train, test_rows) -> predictions for test_rows.
    group: which LoRAs get predictions; train_group: which LoRAs are trained on (default: the same group).
    yv: the target array over all LoRAs. Returns an array over all 3,000 LoRAs, NaN outside TRAIN / the group.
    """
    out = np.full(N, np.nan)
    for k in range(5):
        tr = rows(f"notfold{k}", train_group or group, scored=False)
        tr = tr[~np.isnan(yv[tr])]
        te = rows(f"fold{k}", group, scored=False)
        if len(te):
            out[te] = fit_predict(tr, yv[tr], te)
    return out


def perm_null(fit_predict, name, group="imperial", transform="raw", n_perm=100, seed=0, stat="r", quiet=False):
    """Permutation null for a cross-validated score: labels are shuffled inside the group (over the TRAIN LoRAs),
    the whole pipeline (fit, any inner selection, out-of-fold prediction) runs again, and the score is taken
    against the shuffled labels. Returns the real score, the null scores and the one-sided p value.
    """
    yv = y(name, transform)
    real = score(oof(fit_predict, group, yv=yv), name, group=group, transform=transform, n_boot=0, quiet=True)[stat]
    rng = np.random.default_rng(seed)
    tr_all = rows("train", group, scored=False)
    null = []
    for _ in range(n_perm):
        yp = yv.copy()
        yp[tr_all] = yv[rng.permutation(tr_all)]
        pred = oof(fit_predict, group, yv=yp)
        r_ = rows("train", group)
        ok = ~np.isnan(pred[r_]) & ~np.isnan(yp[r_])
        pp, f = pred[r_][ok].copy(), FOLD[r_][ok]
        gm = pp.mean()
        for k in np.unique(f):
            pp[f == k] += gm - pp[f == k].mean()
        null.append((_pearson if stat == "r" else _spearman)(pp, yp[r_][ok]))
    null = np.array(null)
    p = float((1 + (null >= real).sum()) / (1 + n_perm))
    if not quiet:
        print(f"permutation null for {name} ({group}): real {stat} {real:.3f}; null mean {null.mean():+.3f}, sd {null.std():.3f}, "
              f"95th percentile {np.quantile(null, 0.95):+.3f}, max {null.max():+.3f}; p = {p:.3f} ({n_perm} permutations)")
    return {"real": real, "null": null, "p": p}


# ---------------------------------------------------------------- reference fits (for independent checks)
def _pick(alphas, fit_one, tr, ytr):
    """Inner choice of the penalty: folds inside the training rows (by the same chunk folds)."""
    best, best_a = -np.inf, alphas[0]
    f = FOLD[tr]
    for a in alphas:
        pred = np.full(len(tr), np.nan)
        for k in np.unique(f):
            pred[f == k] = fit_one(tr[f != k], ytr[f != k], tr[f == k], a)
        s = _pearson(pred, ytr)
        if s > best:
            best, best_a = s, a
    return best_a


def ridge_fp(X, alphas=(0.01, 0.1, 1, 10, 100, 1000)):
    """fit_predict for ridge on a feature matrix X [3000, d] (standardised on the training rows; penalty picked inside)."""
    X = np.asarray(X, float)

    def one(tr, ytr, te, a):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = ytr.mean()
        if A.shape[1] <= A.shape[0]:
            w = np.linalg.solve(A.T @ A + a * len(tr) * np.eye(A.shape[1]), A.T @ (ytr - ym))
            return B @ w + ym
        c = np.linalg.solve(A @ A.T + a * len(tr) * np.eye(len(tr)), ytr - ym)
        return B @ (A.T @ c) + ym

    def fp(tr, ytr, te):
        return one(tr, ytr, te, _pick(alphas, one, tr, ytr) if len(alphas) > 1 else alphas[0])
    return fp


def krr_fp(K, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10)):
    """fit_predict for kernel ridge on a full kernel K [3000, 3000] (centred on the training rows; penalty relative
    to the mean diagonal; picked inside)."""
    K = np.asarray(K, float)

    def one(tr, ytr, te, a):
        Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
        cm, gm = Ktr.mean(0), Ktr.mean()
        Kc = Ktr - cm[None, :] - cm[:, None] + gm
        Kt = Kte - Kte.mean(1, keepdims=True) - cm[None, :] + gm
        ym = ytr.mean()
        c = np.linalg.solve(Kc + a * np.trace(Kc) / len(tr) * np.eye(len(tr)), ytr - ym)
        return Kt @ c + ym

    def fp(tr, ytr, te):
        return one(tr, ytr, te, _pick(alphas, one, tr, ytr) if len(alphas) > 1 else alphas[0])
    return fp


if __name__ == "__main__":
    print(f"{N} LoRAs: {int((~MODERN).sum())} imperial, {int(MODERN.sum())} modern; holdout {int(IN_HOLDOUT.sum())} "
          f"(chunks {HOLDOUT_CHUNKS[0]}-{HOLDOUT_CHUNKS[-1]}); basis known: {int(basis().sum())}")
    print(f"{'target':13s} {'imperial: sd':>12s} {'reliab.':>8s} {'ceiling r':>9s} | {'modern: sd':>10s} {'reliab.':>8s} {'ceiling r':>9s}  what")
    for k in TARGETS:
        i, m = _TJ[k].get("imperial", {}), _TJ[k].get("modern", {})
        f = lambda d, kk: f"{d[kk]:.3f}" if kk in d else "-"
        print(f"{k:13s} {f(i, 'sd'):>12s} {f(i, 'reliability'):>8s} {f(i, 'ceiling_r'):>9s} | {f(m, 'sd'):>10s} {f(m, 'reliability'):>8s} "
              f"{f(m, 'ceiling_r'):>9s}  {_TJ[k]['what']}")
