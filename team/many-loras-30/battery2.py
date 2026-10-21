"""Battery 2: feature-space hypotheses + old/new shift diagnostic."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from scipy.stats import rankdata
from hyp_lib import (load_gram, cosine_gram, KRR, FeatModel, eval_hypothesis, splits)

G, y = load_gram()
C = cosine_gram(G)
NL = 36

# ---- Diagnostic: how different are the 161 old vs 440 new LoRAs in Gram space?
(_, tr, te), = splits("holdout")
for l in (5, 17, 23, 30):
    within_old = C[l][np.ix_(tr, tr)][np.triu_indices(len(tr), 1)].mean()
    within_new = C[l][np.ix_(te, te)][np.triu_indices(len(te), 1)].mean()
    across = C[l][np.ix_(tr, te)].mean()
    diag_old, diag_new = np.sqrt(G[l][tr, tr]).mean(), np.sqrt(G[l][te, te]).mean()
    print(f"shift L{l:02d}: cos within-old {within_old:.4f} within-new {within_new:.4f} "
          f"across {across:.4f} | ||dW|| old {diag_old:.4f} new {diag_new:.4f}")

# ---- Featurizers (all use only reference/training rows) ----

def cloud_feats(tr, y_tr):
    """Per-layer distribution of cosines to training LoRAs: mean, std, p90, max."""
    X = []
    for l in range(NL):
        R = C[l][:, tr]
        X += [R.mean(1), R.std(1), np.percentile(R, 90, axis=1),
              np.partition(R, -2, axis=1)[:, -2]]  # 2nd max (self-sim excluded on train rows)
    return np.stack(X, axis=1)


def idio_feats(tr, y_tr, k=20):
    """Per-layer share of the update outside the top-k shared training subspace (kernel PCA residual)."""
    X = np.empty((C.shape[1], NL))
    for l in range(NL):
        Ktr = C[l][np.ix_(tr, tr)].astype(np.float64)
        n = len(tr)
        H = np.eye(n) - 1.0 / n
        Kc = H @ Ktr @ H
        w, V = np.linalg.eigh(Kc)
        w, V = w[::-1][:k], V[:, ::-1][:, :k]
        w = np.maximum(w, 1e-9)
        # projection energy of each row onto top-k train PCs (in cosine feature space)
        Kx = C[l][:, tr].astype(np.float64)
        Kxc = (Kx - Kx.mean(1, keepdims=True)) @ H
        P = Kxc @ V / np.sqrt(w)           # kernel PCA scores
        X[:, l] = 1.0 - (P ** 2).sum(1)    # residual share (unit diag in cosine kernel)
    return X


def kpca_feats(tr, y_tr, layers=range(12, 23), k=15):
    """Kernel PCA scores (train-fit) from the mean cosine kernel over middle layers."""
    K = C[list(layers)].mean(0).astype(np.float64)
    Ktr = K[np.ix_(tr, tr)]
    n = len(tr)
    H = np.eye(n) - 1.0 / n
    w, V = np.linalg.eigh(H @ Ktr @ H)
    w, V = w[::-1][:k], V[:, ::-1][:, :k]
    w = np.maximum(w, 1e-9)
    Kx = K[:, tr]
    Kxc = (Kx - Kx.mean(1, keepdims=True)) @ H
    return Kxc @ V / np.sqrt(w)


def layer_stack_feats(tr, y_tr, alpha=0.1):
    """One feature per layer: per-layer kernel-ridge prediction (OOF on train, refit for others)."""
    from sklearn.model_selection import KFold
    n_all = C.shape[1]
    X = np.empty((n_all, NL))
    yr = rankdata(y_tr).astype(np.float64)
    for l in range(NL):
        K = C[l].astype(np.float64)
        m = KRR(K, alpha, rank_y=False)

        class _KY: pass
        # full fit for non-train rows
        Ktr = K[np.ix_(tr, tr)]
        coef = np.linalg.solve(Ktr + alpha * np.eye(len(tr)), yr - yr.mean())
        X[:, l] = K[:, tr] @ coef + yr.mean()
        # overwrite train rows with out-of-fold predictions
        kf = KFold(5, shuffle=True, random_state=1)
        for itr, ite in kf.split(tr):
            a, b = tr[itr], tr[ite]
            cf = np.linalg.solve(K[np.ix_(a, a)] + alpha * np.eye(len(a)),
                                 yr[itr] - yr[itr].mean())
            X[b, l] = K[np.ix_(b, a)] @ cf + yr[itr].mean()
    return X


def coherence_feats(tr, y_tr, layers=range(8, 28)):
    """Cross-layer coherence: corr over training rows between a LoRA's cosine profiles at
    neighbouring layers. One feature per adjacent layer pair."""
    layers = list(layers)
    prof = np.stack([C[l][:, tr] for l in layers])       # (L, 601, n_tr)
    prof = prof - prof.mean(2, keepdims=True)
    prof = prof / (np.linalg.norm(prof, axis=2, keepdims=True) + 1e-9)
    X = np.stack([(prof[a] * prof[a + 1]).sum(1) for a in range(len(layers) - 1)], axis=1)
    return X


REGS = {
    "ridge1": lambda: Ridge(alpha=1.0),
    "ridge10": lambda: Ridge(alpha=10.0),
    "ridge100": lambda: Ridge(alpha=100.0),
    "hgb": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                                 learning_rate=0.05, random_state=0),
    "rf": lambda: RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0, n_jobs=1),
}

def battery(name, featurizer, regs=REGS, notes=""):
    eval_hypothesis(name, {rn: (lambda fz=featurizer, mk=mk: FeatModel(fz, mk))
                           for rn, mk in regs.items()}, notes=notes)

battery("H6 cloud stats (cos-to-train distribution, 144 feats)", cloud_feats)
battery("H7 idiosyncrasy (kPCA residual share per layer)", idio_feats)
battery("H8 kernel-PCA scores L12-22 (15 comps)", kpca_feats)
battery("H9 per-layer KRR stacking (36 OOF preds)", layer_stack_feats)
battery("H10 cross-layer coherence L8-27", coherence_feats)
