"""Track C exploration on holdout_plus: choose kernel variants by CV on the TRAINING rows only.

For each candidate kernel, the selection score is the mean within-rank Spearman of out-of-fold predictions from a
5-fold CV over the 2,049 training rows (161 old rank-8 + 1,888 higher-rank), folds grouped by data order / seed;
alpha is re-chosen inside every fold. The test score on the 440 is printed next to it but must not be used to choose.
Per-layer kernels are cached in blocks/perlayer/{kind}_{tf}_L{ll}.npy (float32).

Usage: python explore_c.py [--candidates NAME ...] [--log NAME ...]
"""
import argparse, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models_c as mc
from eval_common import labels, splits, score, score_within_rank, log_result

PL = os.path.join(mc.BLK, "perlayer")


def layer_kernel(kind, tf, l):
    p = os.path.join(PL, f"{kind}_{tf}_L{l:02d}.npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float64)
    if kind == "gcos":
        K = mc.wide_gram_cos(l)
    else:
        X = {"in": mc.wide_in_load, "out": mc.wide_out_load, "core": mc.wide_core}[kind](l)
        if tf == "sq":
            X = X ** 2                                  # share of ||dW||^2 per neuron (rows sum to 1)
        elif tf == "sqrt":
            X = np.sqrt(np.abs(X))
        elif tf == "log":
            X = np.log(np.abs(X) + 1e-4)
            X = X - X.mean(1, keepdims=True)
        K = X @ X.T
    K = K / np.mean(np.diag(K))
    os.makedirs(PL, exist_ok=True)
    np.save(p, K.astype(np.float32))
    mc.log(f"  built {kind}_{tf}_L{l:02d}")
    return K


def summed(kind, tf, layers):
    return sum(layer_kernel(kind, tf, l) for l in layers) / len(layers)


R = {"all": range(36), "mid": range(12, 24), "wide": range(8, 28), "core7": range(14, 21),
     "early": range(0, 12), "late": range(24, 36)}

CANDIDATES = {
    "gcos_all": lambda: summed("gcos", "", R["all"]),
    "gcos_mid": lambda: summed("gcos", "", R["mid"]),
    "in_lin_mid": lambda: summed("in", "lin", R["mid"]),
    "in_lin_all": lambda: summed("in", "lin", R["all"]),
    "in_lin_wide": lambda: summed("in", "lin", R["wide"]),
    "in_lin_core7": lambda: summed("in", "lin", R["core7"]),
    "in_sq_mid": lambda: summed("in", "sq", R["mid"]),
    "in_sqrt_mid": lambda: summed("in", "sqrt", R["mid"]),
    "in_log_mid": lambda: summed("in", "log", R["mid"]),
    "core_mid": lambda: summed("core", "lin", R["mid"]),
    "core_all": lambda: summed("core", "lin", R["all"]),
    "gcos_all+in_lin_mid": lambda: summed("gcos", "", R["all"]) + summed("in", "lin", R["mid"]),
    "gcos_all+in_lin_mid+core_mid": lambda: summed("gcos", "", R["all"]) + summed("in", "lin", R["mid"])
                                            + summed("core", "lin", R["mid"]),
    # round 2 (after gcos_mid won round 1 on train-CV)
    "gcos_wide": lambda: summed("gcos", "", R["wide"]),
    "gcos_core7": lambda: summed("gcos", "", R["core7"]),
    "in_sqrt_wide": lambda: summed("in", "sqrt", R["wide"]),
    "gcos_mid+in_sqrt_mid": lambda: summed("gcos", "", R["mid"]) + summed("in", "sqrt", R["mid"]),
    "gcos_mid+in_lin_mid": lambda: summed("gcos", "", R["mid"]) + summed("in", "lin", R["mid"]),
    "gcos_mid+0.5in_sqrt_mid": lambda: summed("gcos", "", R["mid"]) + 0.5 * summed("in", "sqrt", R["mid"]),
    "gcos_mid+2in_sqrt_mid": lambda: summed("gcos", "", R["mid"]) + 2 * summed("in", "sqrt", R["mid"]),
}


def evaluate(K, y, ranks, groups, tr, te):
    """Train-only selection score (grouped 5-fold OOF on tr) and the test score."""
    oof = np.full(len(y), np.nan)
    z = mc.zscore_by_rank(y, ranks, tr)
    for itr, iva in mc.inner_folds(tr, groups, k=5, seed=7):
        zz = mc.zscore_by_rank(y, ranks, tr[itr])
        oof[tr[iva]] = mc.krr(K, zz, tr[itr], tr[iva], groups, seeds=(1,))
    sel, per = score_within_rank(y[tr], oof[tr], ranks[tr])
    pred = mc.krr(K, z, tr, te, groups)
    return sel, per, score(y[te], pred), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    ap.add_argument("--log", nargs="*", default=[], help="candidates to append to leaderboard.csv")
    args = ap.parse_args()
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    (_, tr, te), = list(splits("holdout_plus"))
    out = os.path.join(HERE, "explore_results.csv")
    new = not os.path.exists(out)
    with open(out, "a") as f:
        if new:
            f.write("time,candidate,train_cv_within_rank,r8,r16,r32,r64,holdout_plus_test\n")
    done = {line.split(",")[1] for line in open(out).read().splitlines()[1:]}
    for name in args.candidates:
        if name in done:                       # audit rule: never rerun a variant under the same name
            mc.log(f"{name}: already in explore_results.csv, skipped")
            continue
        t0 = time.time()
        K = CANDIDATES[name]()
        sel, per, test, pred = evaluate(K, y, ranks, groups, tr, te)
        mc.log(f"{name:32s} train-CV {sel:+.4f} ({' '.join(f'{v:+.3f}' for v in per.values())})  "
               f"[test {test:+.4f}]  {time.time() - t0:.0f}s")
        with open(out, "a") as f:
            f.write(f"{time.strftime('%H:%M')},{name},{sel:.4f},{','.join(f'{v:.4f}' for v in per.values())},{test:.4f}\n")
        np.save(os.path.join(mc.PRED, f"explore_{name}_holdout_plus.npy"), pred)
        if name in args.log:
            log_result("many-loras-ef", f"C:{name}", "holdout_plus", test,
                       f"chosen by train-only grouped CV ({sel:.3f}); team/many-loras-ef/explore_c.py")


if __name__ == "__main__":
    main()
