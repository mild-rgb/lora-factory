"""Which MLP neurons does the collapse read pattern use?  (local CPU; reads read_leverage.npz + ../collapse files)

Pattern weights: the read-side predictor (kernel ridge on K_in averaged over layers 14-20, as in ../causal/causal.py),
fit separately on modern (seed > 8) and on old (seed > 8) LoRAs, plus shuffled-target nulls of each.
Neuron weight d_l[k] = sum_j w_j R_jl[k] / 8 = diagonal of the pattern C_l (positive = reading neuron k goes with
collapse).
  1. concentration: share of sum |d| in the top 10 / 100 / 1,000 neurons (of 7 x 12,288 = 86,016)
  2. replication: corr(d_modern, d_old) over neurons, top-k overlap, vs the same with shuffled-target patterns
  3. sufficiency: score held-out-group LoRAs with only the top-k neurons of the other group's pattern,
     sum_k d_k R_ik; Spearman with collapse; also the full diagonal and the full read kernel for reference
  4. confound: corr(d, how much LoRAs read neuron k on average)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent; C = HERE.parent / "collapse"
z = np.load(HERE / "read_leverage.npz"); names = list(z["names"]); R = z["R"].astype(np.float64)   # 7 x 500 x 12288
LAYERS = list(z["layers"])
s = np.load(C / "side_kernels.npz"); assert list(s["names"]) == names
Kin = s["Kin"][LAYERS].astype(np.float64).mean(0)
lab = {r["name"]: r for r in csv.DictReader(open(C / "labels.csv"))}
grp = np.array([lab[n]["group"] for n in names]); seed = np.array([int(lab[n]["seed"]) for n in names])
short = np.array([float(lab[n]["short"]) for n in names]); k_ = np.round(short * 500)
y = np.log((k_ + 0.5) / (500 - k_ + 0.5))
CS = np.logspace(-5, 4, 19)
TR = {g: np.where((grp == g) & (seed > 8))[0] for g in ("old", "modern")}


def fit(Km, tr, yv, c):
    Kt = Km[np.ix_(tr, tr)]; m = Kt.mean(0); Kc = Kt - m[None, :] - m[:, None] + m.mean()
    return np.linalg.solve(Kc + c * np.mean(np.diag(Kc)) * np.eye(len(tr)), yv[tr] - yv[tr].mean())


def choose_c(Km, tr, yv):
    p = np.random.default_rng(7).permutation(len(tr)); err = np.zeros(len(CS))
    for f in range(5):
        te = tr[p[f::5]]; tt = np.setdiff1d(tr, te)
        for ci, c in enumerate(CS):
            a = fit(Km, tt, yv, c); m = Km[np.ix_(tt, tt)].mean(0); Kx = Km[np.ix_(te, tt)]
            err[ci] += np.sum(((Kx - Kx.mean(1, keepdims=True) - m[None, :] + m.mean()) @ a + yv[tt].mean() - yv[te]) ** 2)
    return CS[int(np.argmin(err))]


pat = {}
for g in ("modern", "old"):
    tr = TR[g]; c = choose_c(Kin, tr, y)
    for kind, yv in (("real", y), ("null", np.where(np.isin(np.arange(500), tr), 0, 0) + y)):
        if kind == "null":
            yv = y.copy(); yv[tr] = np.random.default_rng(11 if g == "modern" else 12).permutation(y[tr])
        a = fit(Kin, tr, yv, c); w = (a - a.mean()) / len(LAYERS)
        d = np.einsum("j,ljk->lk", w, R[:, tr, :]) / 8                     # 7 x 12288
        pat[(g, kind)] = dict(w=w, d=d.ravel(), tr=tr, c=float(c))
        # sanity: full read kernel score on the other group
        og = TR["old" if g == "modern" else "modern"]
        full = Kin[np.ix_(og, tr)] @ w * len(LAYERS)
        pat[(g, kind)]["full_transfer"] = float(spearmanr(full, y[og])[0])

out = {}
nn = 7 * 12288
for key, p in pat.items():
    a = np.abs(p["d"]); srt = np.sort(a)[::-1]
    out[f"{key[0]}_{key[1]}"] = dict(c=p["c"], full_read_kernel_transfer=p["full_transfer"],
                                      top10_share=float(srt[:10].sum() / a.sum()), top100_share=float(srt[:100].sum() / a.sum()),
                                      top1000_share=float(srt[:1000].sum() / a.sum()),
                                      participation_ratio=float(a.sum() ** 2 / (a ** 2).sum()))


def topk(d, k):
    return set(np.argsort(-np.abs(d))[:k])


rep = {}
for ka, kb in ((("modern", "real"), ("old", "real")), (("modern", "null"), ("old", "real")),
               (("modern", "real"), ("old", "null")), (("modern", "null"), ("old", "null"))):
    da, db = pat[ka]["d"], pat[kb]["d"]
    r = dict(pearson=float(pearsonr(da, db)[0]), spearman=float(spearmanr(da, db)[0]))
    for k in (10, 100, 1000):
        ov = len(topk(da, k) & topk(db, k)); r[f"top{k}_overlap"] = ov; r[f"top{k}_chance"] = k * k / nn
        sa, sb = np.sign(da), np.sign(db); both = list(topk(da, k) & topk(db, k))
        r[f"top{k}_sign_agree"] = float(np.mean(sa[both] == sb[both])) if both else None
    rep[f"{ka[0]}_{ka[1]}__vs__{kb[0]}_{kb[1]}"] = r
out["replication"] = rep

# per-layer replication (real vs real)
dm, do = pat[("modern", "real")]["d"].reshape(7, -1), pat[("old", "real")]["d"].reshape(7, -1)
out["replication_per_layer_pearson"] = {int(l): float(pearsonr(dm[i], do[i])[0]) for i, l in enumerate(LAYERS)}

# sufficiency: top-k neurons of one group's pattern, scored on the other group
Rf = R.transpose(1, 0, 2).reshape(500, -1)                                   # 500 x (7*12288)
suff = {}
for g in ("modern", "old"):
    og = TR["old" if g == "modern" else "modern"]
    for kind in ("real", "null"):
        d = pat[(g, kind)]["d"]; res = {}
        for k in (10, 100, 1000, 10000, nn):
            idx = np.argsort(-np.abs(d))[:k]
            res[str(k) if k < nn else "all_diag"] = float(spearmanr(Rf[np.ix_(og, idx)] @ d[idx], y[og])[0])
        suff[f"{g}_{kind}_to_{'old' if g == 'modern' else 'modern'}"] = res
out["sufficiency_spearman"] = suff

# confound: neurons that many LoRAs read
mean_read = Rf.mean(0)
out["corr_d_with_mean_read"] = {f"{g}_{k}": float(spearmanr(pat[(g, k)]["d"], mean_read)[0]) for (g, k) in pat}
# the top neurons themselves (real modern pattern, ranked by |d|), with the old pattern's value
top = np.argsort(-np.abs(pat[("modern", "real")]["d"]))[:30]
out["top30_modern"] = [dict(layer=int(LAYERS[i // 12288]), neuron=int(i % 12288), d_modern=float(pat[("modern", "real")]["d"][i]),
                            d_old=float(pat[("old", "real")]["d"][i]), mean_read=float(mean_read[i])) for i in top]
json.dump(out, open(HERE / "neurons_results.json", "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v)[:600])
