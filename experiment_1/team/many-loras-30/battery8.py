"""Battery 8: holdout_plus — train on 161 old rank-8 + 1,888 higher-rank LoRAs,
test on the 440 new rank-8. Kernels from the track-C cross-rank Grams; features from
the cache. Inner CV grouped by seed (and data order for rank-8 rows)."""
import re
import numpy as np
from scipy.stats import rankdata, spearmanr
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_common import labels, splits, score
from hyp_lib import KRR

EF = "team/many-loras-ef/features/"
def dl(f):
    return hf_hub_download("mild-rgb/lora-factory-private", EF + f)

L = labels()
Y = np.array([r["old_germany_rate"] for r in L])
RANKS = np.array([r["rank"] for r in L])
(_, TR, TE), = splits("holdout_plus")
print(f"train {len(TR)} (ranks {np.unique(RANKS[TR], return_counts=True)}), test {len(TE)}")

# ---- cross-rank cosine kernels, window-averaged ----
def build_K(layers):
    K = np.zeros((2489, 2489), dtype=np.float64)
    for l in layers:
        Gl = np.load(dl(f"gram/G_L{l:02d}.npy")).astype(np.float64)
        d = np.sqrt(np.clip(np.diag(Gl), 1e-12, None))
        K += Gl / (d[:, None] * d[None, :])
        del Gl
    return K / len(layers)

K_ALL = build_K(range(36))
K_MID = build_K(range(12, 23))

# ---- seed/order groups over all 2,489 ----
def groups_of(idx):
    g = {}
    for i in idx:
        r = L[i]
        if r["rank"] != 8:
            key = "hs" + r["name"].split("_s")[-1]
        else:
            m = (re.fullmatch(r"former_i\d+_o(\d+)", r["name"])
                 or re.fullmatch(r"former_s0(\d)", r["name"]))
            key = f"o{int(m.group(1))}" if m else r["name"]
        g.setdefault(key, []).append(i)
    return list(g.values())


def grouped_cv_score(make_model, tr, k=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = groups_of(tr)
    parts = np.array_split(rng.permutation(len(groups)), k)
    rhos = []
    for p in parts:
        te = np.array(sorted(x for gi in p for x in groups[gi]))
        trr = np.array(sorted(set(tr) - set(te)))
        m = make_model()
        m.fit(trr, Y[trr])
        rhos.append(score(Y[te], m.predict(te)))
    return float(np.mean(rhos))


def run(name, cands, log=True):
    best, best_inner = None, -9
    for desc, mk in cands.items():
        s = grouped_cv_score(mk, TR)
        if s > best_inner:
            best, best_inner = desc, s
    m = cands[best]()
    m.fit(TR, Y[TR])
    rho = score(Y[TE], m.predict(TE))
    line = (f"| hp | {name} | holdout_plus | **{rho:.4f}** | inner {best_inner:.3f} "
            f"| {best} | seed-grouped inner CV |")
    print(line)
    if log:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "hypotheses.md"), "a") as f:
            f.write(line + "\n")
    return rho


ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# H27: plain cross-rank cosine KRR (the 12x-more-data test)
run("H27 cross-rank cosine KRR",
    {f"{wn},al={al}": (lambda K=K, al=al: KRR(K, al, rank_y=True))
     for wn, K in (("all", K_ALL), ("mid", K_MID)) for al in ALPHAS})

# H27b: same kernel, but trained only on the 161 old rank-8 rows (data-benefit control)
class TrainSubset:
    def __init__(self, K, al, keep_ranks=(8,)):
        self.m = KRR(K, al, rank_y=True); self.keep = keep_ranks
    def fit(self, tr, ytr):
        mask = np.isin(RANKS[tr], self.keep)
        self.m.fit(tr[mask], ytr[mask])
    def predict(self, te):
        return self.m.predict(te)

run("H27b control: same kernel, rank-8 training rows only",
    {f"{wn},al={al}": (lambda K=K, al=al: TrainSubset(K, al))
     for wn, K in (("all", K_ALL), ("mid", K_MID)) for al in ALPHAS})

# H28: features + rank with HGB on the big training set
sv = np.load(dl("svals.npy")).astype(np.float64)            # (2489, 36, 64)
s1 = sv[:, :, 0]; ssum = sv.sum(2); s2sum = (sv ** 2).sum(2)
pr = s2sum ** 2 / ((sv ** 4).sum(2) + 1e-12)
base_dot = np.load(dl("base/base_dot.npy"))
bt = np.load(dl("base_tables.npz"))
FS2 = base_dot / (bt["base_down_fro"][None, :] * np.sqrt(s2sum) + 1e-12)
coreE = []
for l in range(12, 24):
    c = np.load(dl(f"base/core_L{l:02d}.npy")).astype(np.float64)
    coreE.append((c ** 2).sum((1, 2)) / (s2sum[:, l] + 1e-12))
    coreE.append((c[:, :8, :8] ** 2).sum((1, 2)) / (s2sum[:, l] + 1e-12))
coreE = np.stack(coreE, 1)
X = np.concatenate([s1, ssum, s2sum, s1 / (ssum + 1e-12), pr, FS2, coreE,
                    RANKS[:, None].astype(float)], axis=1)

class XModel:
    def __init__(self, mk):
        self.reg = mk()
    def fit(self, tr, ytr):
        self.reg.fit(X[tr], rankdata(ytr))
    def predict(self, te):
        return self.reg.predict(X[te])

run("H28 cache features + rank, big-train",
    {f"hgb,d={d},lr={lr}": (lambda d=d, lr=lr: XModel(
        lambda: HistGradientBoostingRegressor(max_depth=d, learning_rate=lr,
                                              max_iter=400, random_state=0)))
     for d in (2, 3, 4) for lr in (0.05, 0.1)}
    | {"ridge100": (lambda: XModel(lambda: make_pipeline(StandardScaler(),
                                                         Ridge(alpha=100.0))))})

# H29: kernel + features stacked, big-train (2 strong columns only)
class Stack2:
    def __init__(self, K, al, mkf):
        self.K, self.al, self.mkf = K, al, mkf
    def fit(self, tr, ytr):
        self.cols = np.empty((2489, 2))
        # OOF within tr for both columns, full-fit for the rest
        m = KRR(self.K, self.al, rank_y=True); m.fit(tr, ytr)
        self.cols[:, 0] = rankdata(m.predict(np.arange(2489)))
        f = self.mkf(); f.fit(X[tr], rankdata(ytr))
        self.cols[:, 1] = rankdata(f.predict(X))
        groups = groups_of(tr)
        rng = np.random.default_rng(3)
        for p in np.array_split(rng.permutation(len(groups)), 5):
            te = np.array(sorted(x for gi in p for x in groups[gi]))
            trr = np.array(sorted(set(tr) - set(te)))
            mm = KRR(self.K, self.al, rank_y=True); mm.fit(trr, Y[trr])
            self.cols[te, 0] = rankdata(mm.predict(te)) if len(te) else 0
            ff = self.mkf(); ff.fit(X[trr], rankdata(Y[trr]))
            self.cols[te, 1] = rankdata(ff.predict(X[te]))
        self.meta = Ridge(alpha=1.0).fit(self.cols[tr], rankdata(ytr))
    def predict(self, te):
        return self.meta.predict(self.cols[te])

run("H29 stack: cross-rank kernel + cache features",
    {f"al={al}": (lambda al=al: Stack2(K_MID, al,
        lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                              max_iter=400, random_state=0)))
     for al in (0.1, 0.3, 1.0)})
