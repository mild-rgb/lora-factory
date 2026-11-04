"""Battery 3: kernel surgery — ensembling, spectral truncation, mean-removal, row scaling."""
import numpy as np
from scipy.stats import rankdata
from hyp_lib import load_gram, cosine_gram, KRR, eval_hypothesis

G, y = load_gram()
C = cosine_gram(G)
MID = list(range(12, 23))
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0]


class RankAvgEnsemble:
    """Average the rank of per-layer KRR predictions over a set of layers (no meta-fit)."""
    def __init__(self, layers, alpha, rank_y=True):
        self.ms = [KRR(C[l], alpha, rank_y=rank_y) for l in layers]

    def fit(self, tr, ytr):
        for m in self.ms:
            m.fit(tr, ytr)

    def predict(self, te):
        return np.mean([rankdata(m.predict(te)) for m in self.ms], axis=0)


def truncated_kernel(K, k):
    """Spectral truncation of the full 601x601 kernel (uses unlabeled test rows: transductive)."""
    w, V = np.linalg.eigh(K.astype(np.float64))
    w, V = w[::-1][:k], V[:, ::-1][:, :k]
    return (V * w) @ V.T


def mean_removed_cosine(layers, tr):
    """Remove the train-mean update at each layer, then cosine-normalize the residuals.
    <dW_i - mu, dW_j - mu> = G_ij - a_i - a_j + m, with a_i = mean_t G_it, m = mean_tt' G."""
    Ks = []
    for l in layers:
        Gl = G[l].astype(np.float64)
        a = Gl[:, tr].mean(1)
        m = Gl[np.ix_(tr, tr)].mean()
        R = Gl - a[:, None] - a[None, :] + m
        d = np.sqrt(np.clip(np.diag(R), 1e-9, None))
        Ks.append(R / (d[:, None] * d[None, :]))
    return np.mean(Ks, axis=0)


class MeanRemovedKRR:
    """KRR on the mean-removed cosine kernel; the mean is recomputed from each fit's train rows."""
    def __init__(self, layers, alpha, rank_y=True):
        self.layers, self.alpha, self.rank_y = layers, alpha, rank_y

    def fit(self, tr, ytr):
        self.K = mean_removed_cosine(self.layers, tr)
        self.m = KRR(self.K, self.alpha, rank_y=self.rank_y)
        self.m.fit(tr, ytr)

    def predict(self, te):
        return self.m.predict(te)


def zrow_kernel(K):
    """Z-score each row's profile, then symmetrize (removes per-LoRA hubness)."""
    Z = (K - K.mean(1, keepdims=True)) / (K.std(1, keepdims=True) + 1e-9)
    return (Z + Z.T) / 2


def totnorm_kernel(layers):
    """Normalize each LoRA by its TOTAL update norm across all 36 layers (keeps layer-energy ratios)."""
    ntot = np.sqrt(sum(np.diag(G[l]) for l in range(36)))
    return sum(G[l] for l in layers) / (ntot[:, None] * ntot[None, :])


class BaggedKRR:
    def __init__(self, K, alpha, n_bags=20, frac=0.75, rank_y=True, seed=0):
        self.K, self.alpha, self.n_bags, self.frac, self.rank_y = K, alpha, n_bags, frac, rank_y
        self.rng = np.random.default_rng(seed)

    def fit(self, tr, ytr):
        self.fits = []
        n = len(tr)
        for _ in range(self.n_bags):
            s = self.rng.choice(n, int(self.frac * n), replace=False)
            m = KRR(self.K, self.alpha, rank_y=self.rank_y)
            m.fit(tr[s], ytr[s])
            self.fits.append(m)

    def predict(self, te):
        return np.mean([rankdata(m.predict(te)) for m in self.fits], axis=0)


Kmid = C[MID].mean(0)

eval_hypothesis("H11 rank-avg ensemble of per-layer KRR (mid layers)",
                {f"L{a}-{b},al={al}": (lambda a=a, b=b, al=al:
                                       RankAvgEnsemble(range(a, b + 1), al))
                 for (a, b) in [(12, 22), (14, 20), (8, 27), (2, 27)] for al in ALPHAS})

TRUNC = {k: truncated_kernel(Kmid, k) for k in (20, 50, 100, 200, 400)}
eval_hypothesis("H12 spectral truncation of mid cosine kernel (transductive)",
                {f"k={k},al={al}": (lambda k=k, al=al: KRR(TRUNC[k], al, rank_y=True))
                 for k in TRUNC for al in ALPHAS},
                notes="uses unlabeled test rows in eigenbasis; flag to verifier")

eval_hypothesis("H13 mean-removed residual cosine kernel (mid layers)",
                {f"L{a}-{b},al={al}": (lambda a=a, b=b, al=al:
                                       MeanRemovedKRR(range(a, b + 1), al))
                 for (a, b) in [(12, 22), (14, 20), (0, 35)] for al in ALPHAS})

KZ = zrow_kernel(Kmid)
eval_hypothesis("H14 row-z-scored mid kernel (hubness removal)",
                {f"al={al}": (lambda al=al: KRR(KZ, al, rank_y=True)) for al in ALPHAS},
                notes="z uses all rows (transductive, unlabeled)")

KT = {f"L{a}-{b}": totnorm_kernel(range(a, b + 1)) for (a, b) in [(12, 22), (0, 35), (8, 27)]}
eval_hypothesis("H15 total-norm normalization (layer-energy ratios kept)",
                {f"{w},al={al}": (lambda w=w, al=al: KRR(KT[w], al, rank_y=True))
                 for w in KT for al in ALPHAS})

eval_hypothesis("H16 bagged KRR on mid cosine kernel",
                {f"al={al},frac={fr}": (lambda al=al, fr=fr: BaggedKRR(Kmid, al, frac=fr))
                 for al in ALPHAS for fr in (0.6, 0.8)})
