"""Battery 4: combinations of the battery-1/3 wins + CKA layer weights + Hadamard powers."""
import numpy as np
from scipy.stats import rankdata
from hyp_lib import load_gram, cosine_gram, KRR, eval_hypothesis
from battery3_defs import MeanRemovedKRR, BaggedKRR, mean_removed_cosine

G, y = load_gram()
C = cosine_gram(G)
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0]
Kmid = C[list(range(12, 23))].mean(0)
K1420 = C[list(range(14, 21))].mean(0)

# H17: Hadamard powers of the mid cosine kernel (features = dW tensor powers)
eval_hypothesis("H17 Hadamard kernel powers (quadratic/cubic in dW)",
                {f"{w},p={p},al={al}": (lambda w=w, p=p, al=al:
                                        KRR({"mid": Kmid, "1420": K1420}[w] ** p, al, rank_y=True))
                 for w in ("mid", "1420") for p in (2, 3) for al in ALPHAS}
                | {f"poly(1+K)^2 {w},al={al}": (lambda w=w, al=al:
                                                KRR((1 + {"mid": Kmid, "1420": K1420}[w]) ** 2 / 4, al, rank_y=True))
                   for w in ("mid", "1420") for al in ALPHAS})


class CKAWeightedKRR:
    """Layer weights = centered-kernel alignment between each layer kernel and the label
    kernel, computed on the fit's training rows only; then KRR on the weighted sum."""
    def __init__(self, alpha, power=1.0, rank_y=True, layers=range(36)):
        self.alpha, self.power, self.rank_y, self.layers = alpha, power, rank_y, list(layers)

    def fit(self, tr, ytr):
        yc = ytr - ytr.mean()
        Ky = np.outer(yc, yc)
        n = len(tr)
        H = np.eye(n) - 1.0 / n
        w = []
        for l in self.layers:
            Kl = C[l][np.ix_(tr, tr)].astype(np.float64)
            Kc = H @ Kl @ H
            a = (Kc * Ky).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Ky) + 1e-12)
            w.append(max(a, 0.0))
        w = np.asarray(w) ** self.power
        w = w / (w.sum() + 1e-12)
        self.K = np.tensordot(w, C[self.layers].astype(np.float64), axes=(0, 0))
        self.m = KRR(self.K, self.alpha, rank_y=self.rank_y)
        self.m.fit(tr, ytr)
        self.w = w

    def predict(self, te):
        return self.m.predict(te)


eval_hypothesis("H18 CKA-weighted layer combo (train-only weights)",
                {f"p={p},al={al}": (lambda p=p, al=al: CKAWeightedKRR(al, power=p))
                 for p in (1, 2, 4) for al in ALPHAS})


class Champion:
    """Mean-removed cosine kernel over a window + rank-y + bagging."""
    def __init__(self, a, b, alpha, n_bags=25):
        self.a, self.b, self.alpha, self.n_bags = a, b, alpha, n_bags

    def fit(self, tr, ytr):
        K = mean_removed_cosine(range(self.a, self.b + 1), tr)
        self.m = BaggedKRR(K, self.alpha, n_bags=self.n_bags, frac=0.8)
        self.m.fit(tr, ytr)

    def predict(self, te):
        return self.m.predict(te)


eval_hypothesis("H19 champion combo: mean-removed + rank-y + bagged",
                {f"L{a}-{b},al={al}": (lambda a=a, b=b, al=al: Champion(a, b, al))
                 for (a, b) in [(14, 20), (12, 22)] for al in (0.03, 0.1, 0.3)})

# H20: HGB on the raw cosine profile to training rows (161 features)
from sklearn.ensemble import HistGradientBoostingRegressor
from hyp_lib import FeatModel

def profile_feats(tr, ytr):
    return np.concatenate([K1420[:, tr], Kmid[:, tr]], axis=1)

eval_hypothesis("H20 HGB on raw cosine profiles to train rows",
                {f"d={d},lr={lr}": (lambda d=d, lr=lr: FeatModel(
                    profile_feats, lambda: HistGradientBoostingRegressor(
                        max_depth=d, learning_rate=lr, max_iter=300, random_state=0)))
                 for d in (2, 3) for lr in (0.03, 0.1)})
