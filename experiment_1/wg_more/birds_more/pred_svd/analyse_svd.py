"""Local analysis of pred_svd.npz / pred_svd.json (from pred_svd.py on the G4).

1. check: the SVD-based full score equals the kernel-based score of the cos_all predictors
2. spectra: how concentrated each predictor is (energy in top 1 / 8 / 64 directions, participation rank), real vs
   shuffled-target nulls
3. truncation: transfer Spearman (old -> modern, modern -> old; test LoRAs seed > 8) of the predictor kept to its
   top-k singular directions per layer, all layers and layers 14-20; nulls truncated the same way
4. shared directions: old- vs modern-trained predictor subspace overlaps vs null-null; predictor energy inside the
   mean LoRAs' top-8 write / read spans vs null
5. mean-LoRA directions as features: how much each LoRA expresses each of the top directions of mean_old /
   mean_modern / diff; per-direction Spearman with collapse inside each group, and a ridge model on those
   features fit on one group and tested on the other
Writes RESULTS numbers to analyse_svd.json and prints a summary.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

H = Path(__file__).resolve().parent; C = H.parent / "collapse"
z = np.load(H / "pred_svd.npz"); J = json.load(open(H / "pred_svd.json"))
names = list(z["names"]); N = len(names)
lab = {r["name"]: r for r in csv.DictReader(open(C / "labels.csv"))}
grp = np.array([lab[n]["group"] for n in names]); seed = np.array([int(lab[n]["seed"]) for n in names])
short = np.array([float(lab[n]["short"]) for n in names]); nn_ = np.array([int(lab[n]["n"]) for n in names])
k_ = np.round(short * nn_); y = np.log((k_ + 0.5) / (nn_ - k_ + 0.5))
TE = {g: np.where((grp == g) & (seed > 8))[0] for g in ("old", "modern")}
other = {"old": "modern", "modern": "old"}
PRED = ["old", "modern"] + [f"{g}_null{s}" for g in ("old", "modern") for s in range(3)]
MID = list(range(14, 21))
sp = lambda a, b: float(spearmanr(a, b)[0])
out = {}

# 1. check against the kernel-based score
kz = np.load(C / "kernels.npz"); assert list(kz["names"]) == names
K = kz["K"].astype(np.float64); nrm = np.sqrt(np.einsum("lii->li", K)); Kc_l = K / (nrm[:, :, None] * nrm[:, None, :])
al = json.load(open(H / "alphas_all.json")); ix = {n: i for i, n in enumerate(names)}
chk = {}
for p in ("old", "modern"):
    tr = np.array([ix[n] for n in al[p]["names"]]); w = np.array(al[p]["w"])
    S_k = np.einsum("lij,j->li", Kc_l[:, :, tr], w)                    # 36 x 500
    chk[p] = float(np.corrcoef(S_k.T.ravel(), z[f"full_{p}"].ravel())[0, 1])
out["check_corr_svd_vs_kernel_score"] = chk
print("check (corr of per-layer scores, SVD vs kernel):", chk)

# 2. spectra
spec = {}
for p in PRED:
    s2 = z[f"spec_{p}"].astype(np.float64) ** 2; tot = s2.sum(1)
    spec[p] = dict(top1=(s2[:, 0] / tot).tolist(), top8=(s2[:, :8].sum(1) / tot).tolist(),
                   top64=(s2[:, :64].sum(1) / tot).tolist(), pr_rank=(tot ** 2 / (s2 ** 2).sum(1)).tolist())
out["spectra"] = spec
print("\nspectrum, mean over layers 14-20 (real | null mean):")
for g in ("old", "modern"):
    r = {k: np.mean(np.array(spec[g][k])[MID]) for k in spec[g]}
    n = {k: np.mean([np.mean(np.array(spec[f"{g}_null{s}"][k])[MID]) for s in range(3)]) for k in spec[g]}
    print(f"  {g:6s}", {k: f"{r[k]:.3f} | {n[k]:.3f}" for k in r})

# 3. truncation
KS = [1, 2, 4, 8, 16, 32, 64]
trunc = {}
for p in PRED:
    g = p.split("_")[0]; te = TE[other[g]]
    c = z[f"coef_{p}"].astype(np.float64); f = z[f"full_{p}"].astype(np.float64)
    trunc[p] = {"all_layers": {str(k): sp(c[te, :, :k].sum((1, 2)), y[te]) for k in KS},
                "L14_20": {str(k): sp(c[te][:, MID, :k].sum((1, 2)), y[te]) for k in KS}}
    trunc[p]["all_layers"]["full"] = sp(f[te].sum(1), y[te]); trunc[p]["L14_20"]["full"] = sp(f[te][:, MID].sum(1), y[te])
    trunc[p]["all_layers"]["rest_after_64"] = sp((f[te] - c[te].sum(2)).sum(1), y[te])
    trunc[p]["L14_20"]["rest_after_64"] = sp((f[te][:, MID] - c[te][:, MID].sum(2)).sum(1), y[te])
out["truncation_transfer_spearman"] = trunc
print("\ntransfer Spearman of the predictor kept to its top-k directions per layer:")
for g in ("old", "modern"):
    for band in ("all_layers", "L14_20"):
        r = trunc[g][band]; nl = [trunc[f"{g}_null{s}"][band] for s in range(3)]
        print(f"  {g}->{other[g]} {band:10s}", {k: f"{r[k]:+.2f} (null {np.mean([x[k] for x in nl]):+.2f})" for k in r})

# 4. shared directions
ov = J["overlaps"]
def col(key):
    return np.array([r[key] for r in ov])
shared = {}
for side in ("W", "R"):
    for k in (1, 8, 32):
        shared[f"{side}_k{k}"] = dict(real=col(f"{side}_old~modern_k{k}").tolist(), null=col(f"{side}_old_null0~modern_null0_k{k}").tolist())
for p in ("old", "modern"):
    for m in ("mean_old", "mean_modern", "diff"):
        for side in ("Wshare", "Rshare"):
            shared[f"{side}_{p}_in_{m}8"] = dict(real=col(f"{side}_{p}_in_{m}8").tolist(), null=col(f"{side}_{p}_null0_in_{m}8").tolist())
out["shared"] = shared
print("\nold- vs modern-trained predictor, subspace overlap (layers 14-20 mean; real | null-null; chance W k/4096, R k/12288):")
for key in ("W_k1", "W_k8", "W_k32", "R_k1", "R_k8", "R_k32"):
    print(f"  {key:6s} {np.mean(np.array(shared[key]['real'])[MID]):.4f} | {np.mean(np.array(shared[key]['null'])[MID]):.4f}")
print("predictor energy inside each mean's top-8 span (layers 14-20 mean; real | null):")
for p in ("old", "modern"):
    for m in ("mean_old", "mean_modern", "diff"):
        a = shared[f"Wshare_{p}_in_{m}8"]; b = shared[f"Rshare_{p}_in_{m}8"]
        print(f"  {p:6s} in {m:11s} write {np.mean(np.array(a['real'])[MID]):.4f} | {np.mean(np.array(a['null'])[MID]):.4f}"
              f"   read {np.mean(np.array(b['real'])[MID]):.4f} | {np.mean(np.array(b['null'])[MID]):.4f}")

# 5. mean-LoRA directions as features
def ridge_transfer(X, g_tr, g_te, seed_=0):
    tr, te = TE[g_tr], TE[g_te]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12; Xs = (X - mu) / sd
    lams = np.logspace(-2, 5, 15); p = np.random.default_rng(seed_).permutation(len(tr)); err = np.zeros(len(lams))
    for f in range(5):
        v = tr[p[f::5]]; t = np.setdiff1d(tr, v)
        for li, lam in enumerate(lams):
            b = np.linalg.solve(Xs[t].T @ Xs[t] + lam * np.eye(X.shape[1]), Xs[t].T @ (y[t] - y[t].mean()))
            err[li] += np.sum((Xs[v] @ b + y[t].mean() - y[v]) ** 2)
    lam = lams[int(np.argmin(err))]
    b = np.linalg.solve(Xs[tr].T @ Xs[tr] + lam * np.eye(X.shape[1]), Xs[tr].T @ (y[tr] - y[tr].mean()))
    return sp(Xs[te] @ b, y[te]), float(lam)

feat = {}
for m in ("mean_old", "mean_modern", "diff"):
    M = z[f"mcoef_{m}"].astype(np.float64)
    per_dir = {g: [[sp(M[TE[g], l, i], y[TE[g]]) for i in range(4)] for l in range(36)] for g in ("old", "modern")}
    r = {"per_layer_dir_spearman_within": per_dir}
    for band, Ls in (("all", list(range(36))), ("L14_20", MID), ("L13_22", list(range(13, 23))), ("L23_35", list(range(23, 36)))):
        for k in (1, 8):
            X = M[:, Ls, :k].reshape(N, -1)
            for g in ("old", "modern"):
                r[f"{band}_top{k}_{g}->{other[g]}"] = ridge_transfer(X, g, other[g])[0]
    # simplest: sum over layers of each LoRA's expression of the mean's top direction
    s1 = M[:, :, 0].sum(1); s1m = M[:, MID, 0].sum(1)
    r["sum_top1_all_within"] = {g: sp(s1[TE[g]], y[TE[g]]) for g in ("old", "modern")}
    r["sum_top1_L14_20_within"] = {g: sp(s1m[TE[g]], y[TE[g]]) for g in ("old", "modern")}
    feat[m] = r
out["mean_direction_features"] = feat
print("\nmean-LoRA directions as features (ridge, fit on one group, test on the other):")
for m, r in feat.items():
    print(f"  {m:11s}", {k: round(v, 2) for k, v in r.items() if "->" in k})
    print(f"  {'':11s} sum of top-1 expression, within-group Spearman: all {r['sum_top1_all_within']}, L14-20 {r['sum_top1_L14_20_within']}")
    for g in ("old", "modern"):
        v = np.array(r["per_layer_dir_spearman_within"][g])[:, 0]
        print(f"  {'':11s} top-1 direction per layer, within {g}: " + " ".join(f"{x:+.2f}" for x in v))

print("\nlogit lens of the predictors' top write direction:")
for k, (pos, neg) in J["lens"].items():
    print(f"  {k:10s} + {pos[:8]}\n  {'':10s} - {neg[:8]}")
json.dump(out, open(H / "analyse_svd.json", "w"), indent=1)
print("saved analyse_svd.json")
