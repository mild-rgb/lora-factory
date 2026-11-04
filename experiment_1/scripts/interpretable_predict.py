"""Interpretable predictors of a former-cities LoRA's old-Germany rate, on the same splits as mlp_predict.py.

Inputs come from lora_features.py, packed into one safetensors file (numpy-readable without torch):
  act_eval / act_neutral [37, 4096]: residual shift at the last prompt token (LoRA minus base), mean over prompts
  w_norm, w_effrank, w_topshare [36]: weight features. w_align is left out: its reference set contained the
  original 80 seeds themselves, which shifts their values (see factory_more/RESULTS_more.md).
Experiments (everything fitted on training rows only; the 16 modern controls only define a label-free direction):
  1  ridge on the full 4096-dim shift of one layer, for every layer   -> the fitted weights are one direction
  1  ridge on several late layers together (each layer block scaled to unit RMS on the training rows)
  1  PLS with 1, 2, 3, 5 components on the best layer and on the late-layer block
  3  lasso and elastic net on the full shift of the best layer (sparse: few residual dimensions)
  3  lasso on ~300 named features: per layer, shift sizes and projections onto the mean former shift and onto
     former-minus-modern (eval and neutral prompts), plus the weight features
  3  additive spline model (a GAM: one smooth curve per named feature, cubic splines + lasso)
Evaluation: Spearman rho of out-of-fold predictions, 10-fold CV over all LoRAs, and holdout (train on the 161
older LoRAs, test on the 440 new ones). Writes results.csv, oof_predictions.csv, lasso_features.csv, directions.npz.

Usage: python interpretable_predict.py --features features_all.safetensors --rates a.csv b.csv ... --holdout c.csv --out DIR
"""
import argparse, csv, os, time, warnings
import numpy as np
from safetensors.numpy import load_file
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ALPHAS = np.logspace(-2, 7, 40)
LATE = list(range(20, 37))
WEIGHT_KEYS = ["w_norm", "w_effrank", "w_topshare"]


def load(path, rate_files):
    d = load_file(path)
    rates = {}
    for f in rate_files:
        for r in csv.DictReader(open(f)):
            if r["name"].startswith("former") and r["n_answers"] in ("250", ""):
                rates[r["name"]] = float(r["old_germany_rate"])
    names = sorted(n for n in rates if f"{n}/act_eval" in d)
    mods = sorted({k.split("/")[0] for k in d if k.startswith("modern")})
    get = lambda ns, key: np.stack([d[f"{n}/{key}"].astype(np.float32) for n in ns])
    return (names, np.array([rates[n] for n in names]), get(names, "act_eval"), get(names, "act_neutral"),
            np.concatenate([get(names, k) for k in WEIGHT_KEYS], 1),
            get(mods, "act_eval").mean(0), get(mods, "act_neutral").mean(0))


def named_features(tr, Ev, Ne, Wf, Mev, Mne):
    """~300 features with plain meanings. Directions use training rows only (no labels) and the modern mean."""
    cols, names = [], []
    for tag, A, M in (("eval", Ev, Mev), ("neutral", Ne, Mne)):
        mu = A[tr].mean(0)  # [37, D] mean former shift
        u_mean = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
        fmm = mu - M
        u_fmm = fmm / (np.linalg.norm(fmm, axis=1, keepdims=True) + 1e-8)
        for h in range(1, 37):
            cols += [np.linalg.norm(A[:, h], axis=1), A[:, h] @ u_mean[h], A[:, h] @ u_fmm[h]]
            names += [f"{tag}_norm_h{h}", f"{tag}_proj_mean_h{h}", f"{tag}_proj_fmm_h{h}"]
    for i, k in enumerate(WEIGHT_KEYS):
        for l in range(36):
            cols.append(Wf[:, i * 36 + l])
            names.append(f"{k}_L{l}")
    return np.stack(cols, 1), names


def block(A, layers, tr):
    """Concatenate layers, each scaled to unit RMS on the training rows so no layer dominates by size."""
    parts = []
    for h in layers:
        X = A[:, h]
        parts.append(X / (np.sqrt((X[tr] ** 2).mean()) + 1e-8))
    return np.concatenate(parts, 1)


def spearman(p, y):
    return float(spearmanr(p, y)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--rates", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names, y, Ev, Ne, Wf, Mev, Mne = load(args.features, args.rates)
    n = len(y)
    print(f"{n} former LoRAs, {Ev.shape[1]} hidden states x {Ev.shape[2]} dims", flush=True)
    rng = np.random.default_rng(0)  # same folds as mlp_predict.py
    cv_folds = np.array_split(rng.permutation(n), args.folds)
    hold = {r["name"] for r in csv.DictReader(open(args.holdout))}
    ho_test = np.array([i for i, nm in enumerate(names) if nm in hold])
    splits = {"cv": cv_folds, "holdout": [ho_test]}

    def run(label, fit_predict):
        """fit_predict(tr, te) -> predictions for te. Returns {split: rho}."""
        out = {}
        t0 = time.time()
        for split, folds in splits.items():
            p = np.full(n, np.nan)
            for te in folds:
                tr = np.setdiff1d(np.arange(n), te)
                p[te] = fit_predict(tr, te)
            ev = np.concatenate(folds)
            out[split] = spearman(p[ev], y[ev])
            preds[f"{split}/{label}"] = p
        results.append({"model": label, "cv_rho": round(out["cv"], 4), "holdout_rho": round(out["holdout"], 4)})
        print(f"{label:42s} CV {out['cv']:+.3f}   holdout {out['holdout']:+.3f}   ({time.time() - t0:.0f}s)", flush=True)
        return out

    results, preds = [], {}

    def ridge(Xf):
        def f(tr, te):
            m = RidgeCV(alphas=ALPHAS).fit(Xf(tr)[tr], y[tr])
            return m.predict(Xf(tr)[te])
        return f

    # 1a. one layer at a time, full shift
    best = {}
    for tag, A in (("eval", Ev), ("neutral", Ne)):
        for h in range(1, 37):
            r = run(f"ridge_full_{tag}_h{h}", ridge(lambda tr, A=A, h=h: A[:, h]))
            best[(tag, h)] = r["cv"]
    (btag, bh) = max(best, key=best.get)
    print(f"best single layer by CV: {btag} hidden {bh}", flush=True)
    A_best = Ev if btag == "eval" else Ne

    # 1b. several late layers together
    run("ridge_full_eval_late20-36", ridge(lambda tr: block(Ev, LATE, tr)))
    run("ridge_full_eval+neutral_late20-36", ridge(lambda tr: np.concatenate([block(Ev, LATE, tr), block(Ne, LATE, tr)], 1)))

    # 1c. PLS
    for k in (1, 2, 3, 5):
        run(f"pls{k}_{btag}_h{bh}", lambda tr, te, k=k: PLSRegression(k, scale=False).fit(A_best[tr, bh], y[tr]).predict(A_best[te, bh]).ravel())
        run(f"pls{k}_eval_late20-36", lambda tr, te, k=k: PLSRegression(k, scale=False).fit(block(Ev, LATE, tr)[tr], y[tr]).predict(block(Ev, LATE, tr)[te]).ravel())

    # 3a. sparse models on the full shift of the best layer
    def sparse(model):
        def f(tr, te):
            m = make_pipeline(StandardScaler(), model()).fit(A_best[tr, bh], y[tr])
            return m.predict(A_best[te, bh])
        return f
    run(f"lasso_full_{btag}_h{bh}", sparse(lambda: LassoCV(cv=5, n_alphas=30, max_iter=5000, n_jobs=4)))
    run(f"enet0.5_full_{btag}_h{bh}", sparse(lambda: ElasticNetCV(l1_ratio=0.5, cv=5, n_alphas=30, max_iter=5000, n_jobs=4)))

    # 3b. named features: lasso, and an additive spline model
    def named(model_fn):
        def f(tr, te):
            X, _ = named_features(tr, Ev, Ne, Wf, Mev, Mne)
            return model_fn().fit(X[tr], y[tr]).predict(X[te])
        return f
    run("lasso_named_features", named(lambda: make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=50, max_iter=20000, n_jobs=4))))
    run("ridge_named_features", named(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))))
    run("gam_spline_lasso_named", named(lambda: make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3),
                                                              LassoCV(cv=5, n_alphas=30, max_iter=20000, n_jobs=4))))

    # write results
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(args.out, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "old_germany_rate"] + list(preds))
        for i, nm in enumerate(names):
            w.writerow([nm, y[i]] + [round(float(preds[k][i]), 5) for k in preds])

    # interpretation: refit on all LoRAs
    all_idx = np.arange(n)
    X, fnames = named_features(all_idx, Ev, Ne, Wf, Mev, Mne)
    las = make_pipeline(StandardScaler(), LassoCV(cv=10, n_alphas=50, max_iter=20000, n_jobs=4)).fit(X, y)
    coef = las[-1].coef_
    order = np.argsort(-np.abs(coef))
    with open(os.path.join(args.out, "lasso_features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coef_per_sd", "rho_alone"])
        for i in order[: int((coef != 0).sum())]:
            w.writerow([fnames[i], round(float(coef[i]), 5), round(spearman(X[:, i], y), 3)])
    print(f"lasso on all {n}: {(coef != 0).sum()} of {len(coef)} named features kept; top: "
          + ", ".join(fnames[i] for i in order[:8]), flush=True)
    dirs = {}
    for tag, A, M in (("eval", Ev, Mev), ("neutral", Ne, Mne)):
        for h in (bh, 24, 28, 32, 34):
            r = RidgeCV(alphas=ALPHAS).fit(A[:, h], y)
            wv = r.coef_ / np.linalg.norm(r.coef_)
            mu = A[:, h].mean(0)
            fmm = mu - M[h]
            dirs[f"{tag}_h{h}"] = wv
            print(f"ridge direction {tag} h{h}: cos with mean former shift {wv @ mu / np.linalg.norm(mu):+.3f}, "
                  f"with former-minus-modern {wv @ fmm / np.linalg.norm(fmm):+.3f}", flush=True)
    np.savez(os.path.join(args.out, "directions.npz"), **dirs)


if __name__ == "__main__":
    main()
