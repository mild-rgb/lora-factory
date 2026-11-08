"""AUC-style scores for the r8x predictor: rebuild test predictions from saved c_full and the mid_gram kernels.

Prediction up to a constant: f(x) = K[x, train] @ c (sum c = 0), K = mean Gram cosine + 2 x mean sqrt-input kernel,
each layer scaled to unit mean diagonal over all rows, as in analyse_r8x.py. All scores here depend only on ranks.
Usage: python -I auc_r8x.py GRAM_DIR COEFS_CSV OUT_DIR
"""
import csv, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.stats import spearmanr, kendalltau, somersd, rankdata


def roc_auc_score(lab, score):
    """Mann-Whitney form of ROC AUC; ties in score count 0.5."""
    lab = np.asarray(lab, bool)
    n1, n0 = lab.sum(), (~lab).sum()
    return (rankdata(score)[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


gram, coefs, out = sys.argv[1:4]
idx = [r["name"] for r in csv.DictReader(open(f"{gram}/index.csv"))]
co = {r["name"]: r for r in csv.DictReader(open(coefs))}
assert len(idx) == len(co) == len(set(idx)) and set(idx) == set(co), "index and coefs rows differ"
role = np.array([co[n]["role"] for n in idx])
tr, te = np.where(role == "train")[0], np.where(role == "test")[0]
c = np.array([float(co[n]["c_full"]) for n in idx])[tr]
y = np.array([float(co[idx[i]]["rate"]) for i in te])
src = np.array([co[idx[i]]["source"] for i in te])
names = [idx[i] for i in te]
print("rows", len(idx), "train", len(tr), "test", len(te), "sum c", c.sum(), flush=True)

MID = range(12, 24)
Kg, Ki = 0.0, 0.0
for l in MID:
    G = np.load(f"{gram}/G_L{l:02d}.npy", allow_pickle=False)
    d = np.sqrt(np.diag(G).astype(np.float64))
    Kg = Kg + G[np.ix_(te, tr)].astype(np.float64) / np.outer(d[te], d[tr])
    Xall = np.load(f"{gram}/in_L{l:02d}.npy", allow_pickle=False)
    X = np.sqrt(np.abs(Xall.astype(np.float64) / d[:, None]))
    Ki = Ki + X[te] @ X[tr].T / np.mean((X * X).sum(1))
    print("layer", l, "in shape", Xall.shape, Xall.dtype, flush=True)
K = Kg / len(MID) + 2 * Ki / len(MID)
p = K @ c
np.savez(f"{out}/test_preds.npz", pred=p, rate=y, source=src, name=np.array(names))


def conc(y, p):
    """P(higher predicted | pair differs in true rate); prediction ties count 0.5. O(n^2)."""
    dy = np.sign(y[:, None] - y[None, :])
    dp = np.sign(p[:, None] - p[None, :])
    m = dy != 0
    return 0.5 * ((dy[m] * dp[m]).mean() + 1)


def bin_aucs(y, p):
    q90, q75, q50, q25 = np.quantile(y, [.9, .75, .5, .25])
    res = {}
    for k, lab in (("top10_vs_rest", y >= q90), ("top25_vs_rest", y >= q75), ("above_median", y > q50)):
        res[k] = {"auc": roc_auc_score(lab, p), "n_pos": int(lab.sum())}
    m = (y >= q75) | (y <= q25)
    res["top25_vs_bottom25"] = {"auc": roc_auc_score(y[m] >= q75, p[m]), "n_pos": int((y >= q75).sum()), "n_neg": int((y <= q25).sum())}
    return res


rng = np.random.default_rng(0)
res = {}
sets = {"union": np.arange(len(y)), "old440": np.where(src == "old")[0], "new954": np.where(src == "new")[0]}
for k, s in sets.items():
    a, b = y[s], p[s]
    r = {"n": len(s), "rho": spearmanr(b, a)[0], "pearson": np.corrcoef(a, b)[0, 1], "tau_b": kendalltau(a, b)[0], "C": conc(a, b),
         "C_check_somersd": 0.5 * (somersd(a, b).statistic + 1), "frac_pairs_tied_rate": float((a[:, None] == a[None, :]).mean())}
    r.update(bin_aucs(a, b))
    boots = {"C": [], "top25_vs_rest": [], "above_median": [], "top10_vs_rest": [], "top25_vs_bottom25": []}
    for _ in range(1000):
        bs = rng.choice(len(s), len(s))
        boots["C"].append(conc(a[bs], b[bs]))
        ba = bin_aucs(a[bs], b[bs])
        for q in boots:
            if q != "C":
                boots[q].append(ba[q]["auc"])
    r["ci95"] = {q: [float(v) for v in np.percentile(x, [2.5, 97.5])] for q, x in boots.items()}
    order = np.argsort(b)
    tenth = len(s) // 10
    r["mean_rate_pred_bottom10pct"], r["mean_rate_pred_top10pct"], r["mean_rate_all"] = float(a[order[:tenth]].mean()), float(a[order[-tenth:]].mean()), float(a.mean())
    res[k] = r
    print(k, json.dumps(r, default=float), flush=True)

# noise ceiling: split-free estimate from binomial sampling noise only (250 answers per LoRA); judge noise is ignored
a = y[sets["union"]]
samp_var = (a * (1 - a) / 250).mean()
rel = 1 - samp_var / a.var()
truth = a.mean() + np.sqrt(max(rel, 0)) * (a - a.mean())
sims = [conc(rng.binomial(250, np.clip(truth, 0, 1)) / 250, truth) for _ in range(50)]
res["ceiling"] = {"reliability_binomial": float(rel), "C_ceiling_binomial": float(np.mean(sims)), "C_gauss_at_rel_0.82": float(0.5 + np.arcsin(np.sqrt(.82)) / np.pi)}
print(res["ceiling"], flush=True)

# sanity: shuffled labels must give 0.5
perm = rng.permutation(a)
res["shuffled_label_C"] = float(conc(perm, p[sets["union"]]))
print("shuffled C", res["shuffled_label_C"])
json.dump(res, open(f"{out}/auc_results.json", "w"), indent=1, default=float)
