"""Battery 9 (final): holdout_plus with centred kernels and honest selection.
Selection = grouped inner CV over {target transform} x {alpha} per kernel; the stack is
claimed only if inner CV prefers it over the best single kernel. Verifier-approved protocol.
"""
import os, re, sys
import numpy as np
from scipy.stats import rankdata
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_common import labels, splits, score

EF = "team/many-loras-ef/features/"
EFLOCAL = os.path.join(HERE, "..", "many-loras-ef")
def dl(f):
    return hf_hub_download("mild-rgb/lora-factory-private", EF + f)

L = labels()
Y = np.array([r["old_germany_rate"] for r in L])
RANKS = np.array([r["rank"] for r in L])
NAMES = [r["name"] for r in L]
(_, TR, TE), = splits("holdout_plus")

def gcos(l0, l1):
    K = np.zeros((2489, 2489), dtype=np.float64)
    for l in range(l0, l1 + 1):
        Gl = np.load(dl(f"gram/G_L{l:02d}.npy")).astype(np.float64)
        d = np.sqrt(np.clip(np.diag(Gl), 1e-12, None))
        K += Gl / (d[:, None] * d[None, :])
    return K / (l1 - l0 + 1)

K_GCOS = gcos(12, 23)
K_INL = np.mean([np.load(os.path.join(EFLOCAL, "blocks", "perlayer", f"in_lin_L{l}.npy"))
                 for l in range(12, 24)], axis=0).astype(np.float64)

# ---- target transforms (all computed from training rows only) ----
def t_raw(tr):
    return Y.copy()

def t_zwr(tr):
    z = np.zeros(2489)
    for R in (8, 16, 32, 64):
        m = RANKS == R
        mtr = m & np.isin(np.arange(2489), tr)
        mu, sd = Y[mtr].mean(), Y[mtr].std() + 1e-12
        z[m] = (Y[m] - mu) / sd
    return z

def t_grank(tr):
    z = np.zeros(2489)
    z[tr] = rankdata(Y[tr]) / len(tr)
    return z

TARGETS = {"raw": t_raw, "zwr": t_zwr, "grank": t_grank}

class CKRR:
    """Centred kernel ridge; target transform recomputed from each fit's training rows."""
    def __init__(self, K, alpha, target):
        self.K, self.alpha, self.target = K, alpha, target

    def fit(self, tr, _=None):
        self.tr = tr
        z = TARGETS[self.target](tr)
        ztr = z[tr]
        n = len(tr)
        Ktr = self.K[np.ix_(tr, tr)]
        self.rm = Ktr.mean(0)
        self.gm = Ktr.mean()
        Kc = Ktr - self.rm[None, :] - self.rm[:, None] + self.gm
        self.zm = ztr.mean()
        self.coef = np.linalg.solve(Kc + self.alpha * np.eye(n), ztr - self.zm)

    def predict(self, te):
        Kx = self.K[np.ix_(te, self.tr)]
        Kxc = Kx - self.rm[None, :] - Kx.mean(1, keepdims=True) + self.gm
        return Kxc @ self.coef + self.zm


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


def grouped_cv(make_model, tr, k=5, seed=0):
    """Inner folds scored WITHIN RANK (verifier audit 11: pooled Spearman over mixed ranks
    rewards predicting between-rank offsets, which the within-rank-8 test never sees)."""
    from eval_common import score_within_rank
    rng = np.random.default_rng(seed)
    groups = groups_of(tr)
    rhos = []
    for p in np.array_split(rng.permutation(len(groups)), k):
        te = np.array(sorted(x for gi in p for x in groups[gi]))
        trr = np.array(sorted(set(tr) - set(te)))
        m = make_model()
        m.fit(trr)
        rhos.append(score_within_rank(Y[te], m.predict(te), RANKS[te])[0])
    return float(np.mean(rhos))


def run(name, cands, notes=""):
    scored = {d: grouped_cv(mk, TR) for d, mk in cands.items()}
    best = max(scored, key=scored.get)
    m = cands[best]()
    m.fit(TR)
    rho = score(Y[TE], m.predict(TE))
    line = (f"| hp | {name} | holdout_plus | **{rho:.4f}** | inner {scored[best]:.3f} "
            f"| {best} | {notes} |")
    print(line, flush=True)
    with open(os.path.join(HERE, "hypotheses.md"), "a") as f:
        f.write(line + "\n")
    return rho, best, scored


ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

_, best_g, sc_g = run("H30 gcos L12-23 centred, target+alpha by inner CV",
                      {f"{t},al={al}": (lambda t=t, al=al: CKRR(K_GCOS, al, t))
                       for t in TARGETS for al in ALPHAS})
_, best_i, sc_i = run("H32 in_load L12-23 centred, target+alpha by inner CV",
                      {f"{t},al={al}": (lambda t=t, al=al: CKRR(K_INL, al, t))
                       for t in TARGETS for al in ALPHAS})

# ---- features for a third column ----
sv = np.load(dl("svals.npy")).astype(np.float64)
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
X = np.concatenate([s1, ssum, s2sum, s1 / (ssum + 1e-12), pr, FS2,
                    np.stack(coreE, 1), RANKS[:, None].astype(float)], axis=1)


class HGBz:
    def __init__(self, target="zwr"):
        self.target = target

    def fit(self, tr, _=None):
        z = TARGETS[self.target](tr)
        self.reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                 max_iter=400, random_state=0)
        self.reg.fit(X[tr], z[tr])

    def predict(self, te):
        return self.reg.predict(X[te])


def parse(desc):
    t, al = desc.split(",al=")
    return t, float(al)

tg, ag = parse(best_g)
ti, ai = parse(best_i)
PARTS = {"gcos": lambda: CKRR(K_GCOS, ag, tg),
         "inl": lambda: CKRR(K_INL, ai, ti),
         "hgb": lambda: HGBz()}


class StackN:
    def __init__(self, use):
        self.use = use

    def fit(self, tr, _=None):
        z = TARGETS["zwr"](tr)
        groups = groups_of(tr)
        rng = np.random.default_rng(3)
        folds = [np.array(sorted(x for gi in p for x in groups[gi]))
                 for p in np.array_split(rng.permutation(len(groups)), 5)]
        cols = []
        for nm in self.use:
            mk = PARTS[nm]
            col = np.empty(2489)
            m = mk(); m.fit(tr)
            col[:] = m.predict(np.arange(2489))
            for te in folds:
                trr = np.array(sorted(set(tr) - set(te)))
                mm = mk(); mm.fit(trr)
                col[te] = mm.predict(te)
            cols.append(col)
        self.cols = np.stack(cols, 1)
        self.meta = Ridge(alpha=1.0).fit(self.cols[tr], z[tr])

    def predict(self, te):
        return self.meta.predict(self.cols[te])


for use in (("gcos", "inl"), ("gcos", "hgb"), ("gcos", "inl", "hgb")):
    run(f"H33 stack {'+'.join(use)} on tuned centred kernels",
        {"fixed": (lambda use=use: StackN(use))},
        f"cols at their inner-CV-best configs ({best_g} / {best_i})")
print("inner-CV table gcos:", sorted(sc_g.items(), key=lambda kv: -kv[1])[:4])
print("inner-CV table inl:", sorted(sc_i.items(), key=lambda kv: -kv[1])[:4])
