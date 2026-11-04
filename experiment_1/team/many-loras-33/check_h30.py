"""Verify many-loras-30 H30: (1) re-run its inner-CV selection (target x alpha) on the 2,049 training rows with
30's grouping and fold seed, scoring inner folds both pooled (as battery9) and within-rank; (2) exact repro of the
chosen fit; (3) bootstrap vs 17's 0.418. Same kernel: mean cosine L12-23 from G_L*.npy. ~0.3 GB, 1 thread."""
import os, sys, re; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, csv
from scipy.stats import rankdata
sys.path.insert(0, "."); from eval_common import labels, splits, score, score_within_rank
L = labels(); Y = np.array([r["old_germany_rate"] for r in L]); RK = np.array([r["rank"] for r in L])
(_, TR, TE), = splits("holdout_plus")
K = np.zeros((len(L), len(L)))
for l in range(12, 24):
    G = np.load(f"many-loras-17/fullgram/G_L{l:02d}.npy").astype(np.float64); d = np.sqrt(np.diag(G)); K += G / np.outer(d, d)
K /= 12
def target(t, tr):
    if t == "raw": return Y.copy()
    z = np.zeros(len(Y))
    if t == "zwr":
        for R in (8, 16, 32, 64):
            m = RK == R; mt = m & np.isin(np.arange(len(Y)), tr); z[m] = (Y[m] - Y[mt].mean()) / Y[mt].std()
    else: z[tr] = rankdata(Y[tr]) / len(tr)
    return z
def fitpred(tr, te, t, a):
    z = target(t, tr); Kt = K[np.ix_(tr, tr)]; rm, gm = Kt.mean(0), Kt.mean()
    c = np.linalg.solve(Kt - rm - rm[:, None] + gm + a * np.eye(len(tr)), z[tr] - z[tr].mean())
    Kx = K[np.ix_(te, tr)]; return (Kx - rm - Kx.mean(1, keepdims=True) + gm) @ c + z[tr].mean()
def groups_of(idx):
    g = {}
    for i in idx:
        r = L[i]
        if r["rank"] != 8: key = "hs" + r["name"].split("_s")[-1]
        else:
            m = re.fullmatch(r"former_i\d+_o(\d+)", r["name"]) or re.fullmatch(r"former_s0(\d)", r["name"])
            key = f"o{int(m.group(1))}" if m else r["name"]
        g.setdefault(key, []).append(i)
    return list(g.values())
groups = groups_of(TR); perm = np.random.default_rng(0).permutation(len(groups))
folds = [np.array(sorted(x for gi in p for x in groups[gi])) for p in np.array_split(perm, 5)]
res = {}
for t in ("raw", "zwr", "grank"):
    for a in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0):
        pooled, within = [], []
        for te in folds:
            trr = np.array(sorted(set(TR) - set(te))); p = fitpred(trr, te, t, a)
            pooled.append(score(Y[te], p)); within.append(score_within_rank(Y[te], p, RK[te])[0])
        res[(t, a)] = (np.mean(pooled), np.mean(within))
        print(f"inner {t:5s} a={a:7.0e}  pooled {res[(t,a)][0]:.4f}  within-rank {res[(t,a)][1]:.4f}", flush=True)
bp = max(res, key=lambda k: res[k][0]); bw = max(res, key=lambda k: res[k][1])
print("argmax pooled (battery9 rule):", bp, "| argmax within-rank:", bw)
for nm, k in (("pooled-selected", bp), ("within-selected", bw)):
    p = fitpred(TR, TE, *k); print(f"{nm} {k}: holdout_plus rho {score(Y[TE], p):.4f}", flush=True)
    if nm == "pooled-selected": np.save("many-loras-33/check_h30_pred.npy", p)
theirs = {r["name"]: float(r["pred"]) for r in csv.DictReader(open("many-loras-30/h30_predictions.csv"))}
mine = np.load("many-loras-33/check_h30_pred.npy"); th = np.array([theirs[L[i]["name"]] for i in TE])
print("exact repro max|pred diff| vs h30_predictions.csv:", np.abs(mine - th).max())
e4 = {r["name"]: float(r["pred"]) for r in csv.DictReader(open("many-loras-17/exp4_G_oof_holdout_plus.csv"))}
p4 = np.array([e4[L[i]["name"]] for i in TE]); yt = Y[TE]; rng = np.random.default_rng(0); d = []
for _ in range(3000):
    s = rng.integers(0, len(TE), len(TE)); d.append(score(yt[s], th[s]) - score(yt[s], p4[s]))
print(f"H30 minus exp4: {np.mean(d):+.3f} (95% CI {np.quantile(d,.025):+.3f}..{np.quantile(d,.975):+.3f}); pred corr {score(th, p4):.3f}")
