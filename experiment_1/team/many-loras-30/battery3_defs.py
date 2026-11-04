"""Reusable model classes from battery 3 (import without re-running the battery)."""
import numpy as np
from scipy.stats import rankdata
from hyp_lib import load_gram, KRR

G, _ = load_gram()


def mean_removed_cosine(layers, tr):
    """Remove the train-mean update at each layer, then cosine-normalize the residuals."""
    Ks = []
    for l in layers:
        Gl = G[l].astype(np.float64)
        a = Gl[:, tr].mean(1)
        m = Gl[np.ix_(tr, tr)].mean()
        Rm = Gl - a[:, None] - a[None, :] + m
        d = np.sqrt(np.clip(np.diag(Rm), 1e-9, None))
        Ks.append(Rm / (d[:, None] * d[None, :]))
    return np.mean(Ks, axis=0)


class MeanRemovedKRR:
    def __init__(self, layers, alpha, rank_y=True):
        self.layers, self.alpha, self.rank_y = layers, alpha, rank_y

    def fit(self, tr, ytr):
        self.K = mean_removed_cosine(self.layers, tr)
        self.m = KRR(self.K, self.alpha, rank_y=self.rank_y)
        self.m.fit(tr, ytr)

    def predict(self, te):
        return self.m.predict(te)


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
