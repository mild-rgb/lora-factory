"""Battery 5: kernel PLS — supervised coarse directions (test of the D1 picture)."""
import numpy as np
from scipy.stats import rankdata
from hyp_lib import load_gram, cosine_gram, eval_hypothesis
from battery3_defs import mean_removed_cosine

G, y = load_gram()
C = cosine_gram(G)
Kmid = C[list(range(12, 23))].mean(0)
K1420 = C[list(range(14, 21))].mean(0)


class KPLS:
    """Kernel partial least squares (NIPALS), rank-transformed target."""
    def __init__(self, K, ncomp=2, mr_layers=None):
        self.Kfull, self.ncomp, self.mr_layers = K, ncomp, mr_layers

    def fit(self, tr, ytr):
        self.tr = tr
        Kf = (mean_removed_cosine(self.mr_layers, tr) if self.mr_layers is not None
              else self.Kfull)
        self.Kf = Kf
        K = Kf[np.ix_(tr, tr)].astype(np.float64)
        n = len(tr)
        H = np.eye(n) - 1.0 / n
        Kc = H @ K @ H
        yy = rankdata(ytr).astype(np.float64)
        yc = yy - yy.mean()
        self.ym = yy.mean()
        T, U = [], []
        Kres, yres = Kc.copy(), yc.copy()
        for _ in range(self.ncomp):
            t = Kres @ yres
            t /= (np.linalg.norm(t) + 1e-12)
            T.append(t)
            U.append(yres.copy())
            # deflate
            P = np.eye(n) - np.outer(t, t)
            Kres = P @ Kres @ P
            yres = yres - t * (t @ yres)
        self.T = np.stack(T, 1)             # (n, c)
        # regression: y ~ T
        self.beta = np.linalg.lstsq(self.T, yc, rcond=None)[0]
        # dual representation for prediction: t_test = Kc_test @ (deflation chain)
        # use simple approach: represent each t as alpha vector s.t. t = Kc @ a
        A = []
        Kres = Kc.copy()
        yres = yc.copy()
        a_chain = []
        for c in range(self.ncomp):
            a = yres.copy()
            # account for deflations: t_c = Kres @ yres, and Kres = P..Kc..P
            A.append((a, Kres.copy()))
            t = self.T[:, c]
            P = np.eye(n) - np.outer(t, t)
            Kres = P @ Kres @ P
            yres = yres - t * (t @ yres)
        self.H = H
        self.Kc_tr = Kc
        self.A = A

    def predict(self, te):
        K_x = self.Kf[np.ix_(te, self.tr)].astype(np.float64)
        n = len(self.tr)
        # center test kernel rows like training
        Kc_x = (K_x - K_x.mean(1, keepdims=True)) @ self.H
        # recompute test scores through the deflation chain
        Ts = []
        Kres_x = Kc_x.copy()
        for c, (a, Kres_tr) in enumerate(self.A):
            t_tr = self.T[:, c]
            t_x = Kres_x @ a
            s = np.linalg.norm(Kres_tr @ a) + 1e-12
            t_x = t_x / s
            Ts.append(t_x)
            # deflate test rows consistently: K_x <- K_x (I - t t^T), and remove projection
            Kres_x = (Kres_x - np.outer(t_x * s, t_tr) / s) @ (np.eye(n) - np.outer(t_tr, t_tr))
        T_x = np.stack(Ts, 1)
        return T_x @ self.beta + self.ym


cands = {}
for nc in (1, 2, 3, 5):
    cands[f"mid,c={nc}"] = (lambda nc=nc: KPLS(Kmid, nc))
    cands[f"1420,c={nc}"] = (lambda nc=nc: KPLS(K1420, nc))
    cands[f"mr1420,c={nc}"] = (lambda nc=nc: KPLS(None, nc, mr_layers=range(14, 21)))
eval_hypothesis("H21 kernel PLS (supervised coarse directions)", cands)
