"""Track A (many-loras-30): shared harness for weight-only hypothesis tests on gram.npz.

All models follow one protocol:
  - fit(idx_tr, y_tr): may use ONLY training rows (for references, PCA, supervised directions).
  - predict(idx_te): predictions for any rows.
Hyperparameters are always chosen by inner CV on the training rows (see eval_hypothesis).
"""
import os, sys, csv, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # shared machine: 1 core per session
import numpy as np
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_common import labels, splits, score  # noqa: E402

GRAM_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mild-rgb--lora-factory/snapshots/"
    "99ed8205ddbc5a08711c6321c92a870c89d4fee9/weight_kernel/gram.npz")

_cache = {}


def load_gram():
    if "G" not in _cache:
        z = np.load(GRAM_PATH)
        G = z["G"].astype(np.float32)  # (36, 601, 601)
        names = list(z["names"])
        L = labels(8)
        assert [r["name"] for r in L] == names, "gram names must match labels(8) order"
        y = np.array([r["old_germany_rate"] for r in L], dtype=np.float64)
        _cache.update(G=G, names=names, y=y)
    return _cache["G"], _cache["y"]


def cosine_gram(G):
    """Per-layer cosine-normalized Gram: C[l] = D^-1/2 G[l] D^-1/2."""
    if "C" not in _cache:
        d = np.sqrt(np.einsum("lii->li", G))  # (36, 601)
        _cache["C"] = G / (d[:, :, None] * d[:, None, :])
    return _cache["C"]


def sq_dists(C):
    """Cosine-induced squared distances per layer: 2 - 2*C (unit-norm vectors)."""
    return 2.0 - 2.0 * C


class KRR:
    """Kernel ridge on a precomputed kernel, CENTRED on the training rows (verifier audit 10:
    uncentred kernels cost ~0.03 holdout_plus). center=False restores the old behaviour."""
    def __init__(self, K, alpha=1.0, rank_y=False, center=True):
        self.K, self.alpha, self.rank_y, self.center = K, alpha, rank_y, center

    def fit(self, tr, y):
        from scipy.stats import rankdata
        self.tr = tr
        yy = rankdata(y) if self.rank_y else y
        self.ym = yy.mean()
        Ktr = self.K[np.ix_(tr, tr)].astype(np.float64)
        n = len(tr)
        if self.center:
            self.rm = Ktr.mean(0)
            self.gm = Ktr.mean()
            Ktr = Ktr - self.rm[None, :] - self.rm[:, None] + self.gm
        self.coef = np.linalg.solve(Ktr + self.alpha * np.eye(n), yy - self.ym)

    def predict(self, te):
        Kx = self.K[np.ix_(te, self.tr)].astype(np.float64)
        if self.center:
            Kx = Kx - self.rm[None, :] - Kx.mean(1, keepdims=True) + self.gm
        return Kx @ self.coef + self.ym


class KNN:
    """k-nearest-neighbour regression on a precomputed 601x601 distance matrix."""
    def __init__(self, D, k=10, weighted=False):
        self.D, self.k, self.weighted = D, k, weighted

    def fit(self, tr, y):
        self.tr, self.y = tr, np.asarray(y, dtype=np.float64)

    def predict(self, te):
        D = self.D[np.ix_(te, self.tr)]
        idx = np.argsort(D, axis=1)[:, : self.k]
        out = np.empty(len(te))
        for i in range(len(te)):
            yv = self.y[idx[i]]
            if self.weighted:
                w = 1.0 / (D[i, idx[i]] + 1e-6)
                out[i] = (w * yv).sum() / w.sum()
            else:
                out[i] = yv.mean()
        return out


class FeatModel:
    """featurizer(tr, y_tr) -> X (601, d) built from training rows only; then an sklearn regressor."""
    def __init__(self, featurizer, make_reg):
        self.featurizer, self.make_reg = featurizer, make_reg

    def fit(self, tr, y):
        self.X = self.featurizer(tr, y)
        self.reg = self.make_reg()
        self.reg.fit(self.X[tr], y)

    def predict(self, te):
        return self.reg.predict(self.X[te])


def inner_cv_score(make_model, tr, y, n_splits=5, seed=0):
    """Mean Spearman over an inner CV of the training rows, with ORDER-GROUPED folds
    (team rule 2026-10-20: random folds are inflated by shared data orders — LoRAs
    former_sK / former_iX_oK reuse order K). Used ONLY for model selection."""
    if "dlib" not in _cache:
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "dlib", os.path.join(HERE, "..", "many-loras-17", "dlib.py"))
        dlib = ilu.module_from_spec(spec)
        spec.loader.exec_module(dlib)
        _cache["dlib"] = dlib
        _cache["order_groups"] = dlib.order_groups()  # per rank-8 LoRA, labels(8) order
    dlib = _cache["dlib"]
    folds = dlib.group_folds(_cache["order_groups"][tr], k=n_splits, seed=seed + 1)
    rhos = []
    for ite in folds:
        itr = np.setdiff1d(np.arange(len(tr)), ite)
        m = make_model()
        m.fit(tr[itr], y[tr[itr]])
        rhos.append(score(y[tr[ite]], m.predict(tr[ite])))
    return float(np.mean(rhos))


def eval_hypothesis(name, candidates, track="holdout", log_file="hypotheses.md", notes=""):
    """candidates: dict param_desc -> make_model(). Picks the best by inner CV on the training
    rows of each split, then scores on the split's test rows. Returns (rho, best_desc, inner)."""
    _, y = load_gram()
    rows = []
    for fold, tr, te in splits(track):
        best, best_inner = None, -9
        for desc, mk in candidates.items():
            s = inner_cv_score(mk, tr, y)
            if s > best_inner:
                best, best_inner = desc, s
        m = candidates[best]()
        m.fit(tr, y[tr])
        rho = score(y[te], m.predict(te))
        rows.append((fold, rho, best, best_inner))
    mean_rho = float(np.mean([r[1] for r in rows]))
    pick = rows[0]
    line = (f"| {time.strftime('%H:%M')} | {name} | {track} | **{mean_rho:.4f}** | "
            f"inner {pick[3]:.3f} | {pick[2]} | {notes} |")
    with open(os.path.join(HERE, log_file), "a") as f:
        f.write(line + "\n")
    print(line)
    return mean_rho, rows
