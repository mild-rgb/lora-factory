"""E audit 1: reproduce the weight_kernel baselines from gram.npz with team/eval_common.py splits.
Own kernel ridge implementation (not copied), plus leakage checks: label match, split match, duplicates, layer-choice leak."""
import os, sys, csv
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eval_common import labels, splits, score

WK = os.path.join(HERE, "hf_dl", "weight_kernel")
z = np.load(os.path.join(WK, "gram.npz"))
G, gnames = z["G"], list(z["names"])
L8 = labels(8)
names = [r["name"] for r in L8]
y = np.array([r["old_germany_rate"] for r in L8])

# 1. names / order / labels
assert gnames == names, "gram order != eval_common order"
oof = {r["name"]: r for r in csv.DictReader(open(os.path.join(WK, "oof_predictions.csv")))}
dy = max(abs(float(oof[n]["old_germany_rate"]) - y[i]) for i, n in enumerate(names))
print("names match; max label diff vs oof file:", dy)

# 2. holdout split in the oof file == eval_common holdout test set
_, tr_h, te_h = next(splits("holdout"))
has = np.array([oof[n]["holdout/kernel_ridge_cosine_sum_layers"] != "nan" for n in names])
print("oof holdout rows == eval_common test rows:", set(np.where(has)[0]) == set(te_h))

# 3. near-duplicate adapters (per-LoRA cosine summed over layers)
nrm = np.sqrt(np.einsum("lii->li", G))
C = (G / (nrm[:, :, None] * nrm[:, None, :])).mean(0)
np.fill_diagonal(C, -1)
i, j = np.unravel_index(C.argmax(), C.shape)
print(f"max off-diag mean-layer cosine {C[i, j]:.4f} ({names[i]}, {names[j]}); "
      f"pairs > 0.95: {int((np.triu(C, 1) > 0.95).sum())}")
offd = C[np.triu_indices(len(names), 1)]
print(f"off-diag cosine: median {np.median(offd):.4f}, 99.9% {np.quantile(offd, .999):.4f}")

def krr(K, y, tr, te, alphas, rng_seed=1):
    """kernel ridge with intercept via explicit centring; alpha picked by inner 5-fold CV on tr only."""
    def fit(ti, xi, a):
        Ktt, Kxt = K[np.ix_(ti, ti)], K[np.ix_(xi, ti)]
        cm = Ktt.mean(0); tm = Ktt.mean()
        Kc = Ktt - cm[None] - cm[:, None] + tm
        Kxc = Kxt - cm[None] - Kxt.mean(1, keepdims=True) + tm
        c = np.linalg.solve(Kc + a * np.eye(len(ti)), y[ti] - y[ti].mean())
        return Kxc @ c + y[ti].mean()
    sc = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    inner = np.array_split(np.random.default_rng(rng_seed).permutation(len(tr)), 5)
    errs = []
    for a in alphas * sc:
        e = 0
        for iv in inner:
            ti = tr[np.setdiff1d(np.arange(len(tr)), iv)]; vi = tr[iv]
            e += ((fit(ti, vi, a) - y[vi]) ** 2).sum()
        errs.append(e)
    return fit(tr, te, alphas[int(np.argmin(errs))] * sc)

alphas = np.logspace(-4, 3, 22)
Gcos = G / (nrm[:, :, None] * nrm[:, None, :])
kern = {"raw_sum": G.sum(0), "cos_sum": Gcos.sum(0), "cos_L17": Gcos[17]}
def run(K, track):
    p = np.full(len(y), np.nan); ev = []
    for _, tr, te in splits(track):
        p[te] = krr(K, y, tr, te, alphas); ev.append(te)
    ev = np.concatenate(ev)
    return score(y[ev], p[ev]), p
claimed = {"raw_sum": (0.337, 0.267), "cos_sum": (0.389, 0.313), "cos_L17": (0.405, 0.268)}
col = {"raw_sum": "kernel_ridge_raw_sum_layers", "cos_sum": "kernel_ridge_cosine_sum_layers", "cos_L17": "kernel_ridge_cosine_L17"}
for k, K in kern.items():
    out = []
    for t in ("cv", "holdout"):
        rho, p = run(K, t)
        theirs = np.array([float(oof[n][f"{t}/{col[k]}"]) for n in names])
        m = ~np.isnan(p)
        out.append(f"{t} {rho:.4f} (claimed {claimed[k][t == 'holdout']}, max |pred diff| {np.nanmax(abs(p[m] - theirs[m])):.1e})")
    print(k, " | ".join(out))

# 4. layer-choice leak: choose the best single layer using only the 161 holdout-train rows (nested CV), not all 601
inner10 = np.array_split(np.random.default_rng(2).permutation(len(tr_h)), 5)
sel = []
for l in range(36):
    p = np.full(len(tr_h), np.nan)
    for iv in inner10:
        it = np.setdiff1d(np.arange(len(tr_h)), iv)
        p[iv] = krr(Gcos[l], y, tr_h[it], tr_h[iv], alphas)
    sel.append(spearmanr(p, y[tr_h])[0])
sel = np.array(sel)
best = int(np.argmax(sel))
hold = score(y[te_h], krr(Gcos[best], y, tr_h, te_h, alphas))
print("inner-CV rho on 161 train rows, top layers:", [(int(l), round(sel[l], 3)) for l in np.argsort(-sel)[:6]])
print(f"leak-free best single layer = L{best}; holdout rho {hold:.4f} (vs 0.268 for L17 picked with all 601)")
np.save(os.path.join(HERE, "layer_sel_161.npy"), sel)
