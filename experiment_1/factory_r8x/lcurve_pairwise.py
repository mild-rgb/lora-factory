"""Learning curve of the r8x predictor scored by pairwise accuracy (concordance), not only Spearman rho.

Repeats the curve in analyse_r8x.py exactly (same kernel, same KRR with grouped inner CV, same random subsets
rng(rep).choice(new_tr, size)) and adds C = P(higher prediction | the two LoRAs' true rates differ); prediction ties
count 0.5. rho is written too, and must match lcurve.csv, which proves the fits are the same.
Labels/roles come from open_gpu_in/coefs.csv (same rows and order as mid_gram/index.csv; asserted).
Helpers copied verbatim from team/many-loras-ef/models_c.py so the script is self-contained.

Usage: python lcurve_pairwise.py GRAM_DIR COEFS_CSV OUT_CSV
"""
import csv, re, sys, time
import numpy as np
from scipy.stats import spearmanr

gram, coefs, out_path = sys.argv[1:4]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
MID = range(12, 24)


# ---- copied from models_c.py ----
def seed_groups(rows):
    g = []
    for r in rows:
        if r["rank"] != 8:
            g.append("s" + r["name"].split("_s")[-1])
            continue
        m = re.match(r"former_i\d+_o(\d+)$", r["name"]) or re.match(r"former_s(\d+)$", r["name"])
        g.append(f"o{int(m.group(1))}" if m and int(m.group(1)) <= 9 else r["name"])
    return np.array(g)


def inner_folds(tr, groups, k=5, seed=1):
    ug = np.unique(groups[tr])
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of_g = {g: i % k for i, g in zip(range(len(ug)), ug[perm])}
    f = np.array([fold_of_g[g] for g in groups[tr]])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def zscore_by_rank(y, ranks, tr):
    z = np.empty_like(y)
    for R in np.unique(ranks):
        m = ranks == R
        t = m.copy(); t[:] = False; t[tr] = True; t &= m
        mu, sd = (y[t].mean(), y[t].std() + 1e-12) if t.sum() > 2 else (y[tr].mean(), y[tr].std())
        z[m] = (y[m] - mu) / sd
    return z


def krr_fit_pred(K, y, tr, te, alpha):
    n = len(tr)
    Ktt = K[np.ix_(tr, tr)]
    Kxt = K[np.ix_(te, tr)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    Kxc = Kxt - cm[None, :] - Kxt.mean(1)[:, None] + cm.mean()
    c = np.linalg.solve(Kc + alpha * np.eye(n), y[tr] - y[tr].mean())
    return Kxc @ c + y[tr].mean()


def krr(K, y, tr, te, groups, alphas=np.logspace(-5, 3, 9), seeds=(1, 2, 3)):
    scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    best = None
    for a in alphas * scale:
        err = 0.0
        for sd in seeds:
            for itr, iva in inner_folds(tr, groups, seed=sd):
                p = krr_fit_pred(K, y, tr[itr], tr[iva], a)
                err += ((p - y[tr[iva]]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, a)
    return krr_fit_pred(K, y, tr, te, best[1])
# ---- end copy ----


def conc(y, p):
    """P(higher predicted | pair differs in true rate); prediction ties count 0.5."""
    dy = np.sign(y[:, None] - y[None, :])
    dp = np.sign(p[:, None] - p[None, :])
    m = dy != 0
    return 0.5 * ((dy[m] * dp[m]).mean() + 1)


idx = [r["name"] for r in csv.DictReader(open(f"{gram}/index.csv"))]
rows = list(csv.DictReader(open(coefs)))
assert [r["name"] for r in rows] == idx, "coefs.csv rows are not in index.csv order"
for r in rows:
    r["rank"] = int(r["rank"])
y = np.array([float(r["rate"]) for r in rows])
ranks = np.array([r["rank"] for r in rows])
groups = seed_groups(rows)
src, role = np.array([r["source"] for r in rows]), np.array([r["role"] for r in rows])
old_tr = np.where((src == "old") & (role == "train"))[0]
old_te = np.where((src == "old") & (role == "test"))[0]
new_tr = np.where((src == "new") & (role == "train"))[0]
new_te = np.where((src == "new") & (role == "test"))[0]
seed = np.array([int(r["name"].split("_s")[-1]) if r["source"] == "new" else -1 for r in rows])
assert (seed[new_tr] % 3 != 0).all() and (seed[new_te] % 3 == 0).all()
assert (len(old_tr), len(new_tr), len(old_te), len(new_te)) == (961, 1906, 440, 954), (len(old_tr), len(new_tr), len(old_te), len(new_te))
te = np.concatenate([old_te, new_te])
log("rows", len(rows))


def unit(K):
    return K / np.mean(np.diag(K))


Kg, Ki = 0, 0
for l in MID:
    G = np.load(f"{gram}/G_L{l:02d}.npy").astype(np.float64)
    d = np.sqrt(np.diag(G))
    Kg = Kg + unit(G / np.outer(d, d))
    X = np.sqrt(np.abs(np.load(f"{gram}/in_L{l:02d}.npy").astype(np.float64) / d[:, None]))
    Ki = Ki + unit(X @ X.T)
    log("layer", l)
K = Kg / len(MID) + 2 * Ki / len(MID)
del Kg, Ki

yte = y[te]
o, n = np.arange(len(old_te)), np.arange(len(old_te), len(te))
out = open(out_path, "w")
out.write("n_new_train,n_train,repeat,rho_union,C_union,C_old440,C_new954\n")
for size in (0, 250, 500, 1000, len(new_tr)):
    for rep in range(1 if size in (0, len(new_tr)) else 5):
        sub = np.random.default_rng(rep).choice(new_tr, size, replace=False)
        tr = np.concatenate([old_tr, sub])
        p = krr(K, zscore_by_rank(y, ranks, tr), tr, te, groups)
        v = [spearmanr(p, yte)[0], conc(yte, p), conc(yte[o], p[o]), conc(yte[n], p[n])]
        out.write(f"{size},{len(tr)},{rep}," + ",".join(f"{x:.4f}" for x in v) + "\n")
        out.flush()
        log(f"size {size} rep {rep}: rho {v[0]:.4f} C {v[1]:.4f} / {v[2]:.4f} / {v[3]:.4f}")
log("done")
