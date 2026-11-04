"""Track E part 4: symmetry-invariant features from the cross-layer Gram blocks (xl/*.npz), scored with eval_common.
All features are invariant to B -> B Q_l, A -> Q_l^-1 A for any invertible Q_l per layer (and are per-LoRA, label-free):
  xcos  cosine between dW_l and dW_m for every layer pair (630)  [tr(dW_l^T dW_m), shapes match: all 4096x12288]
  lnorm log ||dW_l||_F (36)
  spec  normalised singular values of dW_l, sigma_k / ||sigma|| (36 x 8)
  outal output-subspace overlap ||Q_Bl^T Q_Bm||_F^2 / r for each pair (630), inal the same on the input side (630)
Model: ridge on train-standardised features, alpha by inner 5-fold CV on training rows only."""
import os, sys, glob
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, "team"); from eval_common import labels, splits, score, log_result
X_DIR = "team/many-loras-33/xl"; R = 8; NL = 36
iu = np.triu_indices(NL, 1)

def isqrt(M):
    w, V = np.linalg.eigh(M); return V @ np.diag(1 / np.sqrt(np.clip(w, 1e-30, None))) @ V.T

def feats(f):
    z = np.load(f); GA, GB, s = z["GA"].astype(np.float64), z["GB"].astype(np.float64), float(z["s"])
    blk = lambda G, l, m: G[l*R:(l+1)*R, m*R:(m+1)*R]
    C = np.array([[s*s*np.trace(blk(GB, l, m) @ blk(GA, m, l)) for m in range(NL)] for l in range(NL)])
    d = np.sqrt(np.diag(C)); xcos = (C / np.outer(d, d))[iu]
    spec = []
    for l in range(NL):
        ev = np.sort(np.clip(np.linalg.eigvals(blk(GB, l, l) @ blk(GA, l, l)).real, 0, None))[::-1]
        sv = np.sqrt(ev); spec.append(sv / np.linalg.norm(sv))
    WB = [isqrt(blk(GB, l, l)) for l in range(NL)]; WA = [isqrt(blk(GA, l, l)) for l in range(NL)]
    outal = np.array([np.sum((WB[l] @ blk(GB, l, m) @ WB[m])**2) / R for l, m in zip(*iu)])
    inal = np.array([np.sum((WA[l] @ blk(GA, l, m) @ WA[m])**2) / R for l, m in zip(*iu)])
    return dict(xcos=xcos, lnorm=np.log(d), spec=np.concatenate(spec), outal=outal, inal=inal)

def ridge_cv(X, y, tr, te, alphas=np.logspace(-2, 5, 15)):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Z = (X - mu) / sd
    def fit(ti, xi, a):
        Zt = Z[ti]; ym = y[ti].mean()
        K = Zt @ Zt.T; c = np.linalg.solve(K + a*np.eye(len(ti)), y[ti] - ym)
        return Z[xi] @ (Zt.T @ c) + ym
    inner = np.array_split(np.random.default_rng(1).permutation(len(tr)), 5)
    err = [sum(((fit(tr[np.setdiff1d(np.arange(len(tr)), iv)], tr[iv], a) - y[tr[iv]])**2).sum() for iv in inner) for a in alphas]
    return fit(tr, te, alphas[int(np.argmin(err))])

if __name__ == "__main__":
    L8 = labels(8); names = [r["name"] for r in L8]; y = np.array([r["old_germany_rate"] for r in L8])
    cache = "team/many-loras-33/invariants_r8.npz"
    if os.path.exists(cache):
        z = np.load(cache); F = {k: z[k] for k in z.files}
    else:
        rows = [feats(f"{X_DIR}/{n}.npz") for n in names]
        F = {k: np.array([r[k] for r in rows]) for k in rows[0]}; np.savez(cache, **F)
    groups = {k: F[k] for k in F}
    groups["all"] = np.concatenate([F[k] for k in F], 1)
    groups["xcos_mid(L10-24)"] = F["xcos"][:, [(10 <= l <= 24 and 10 <= m <= 24) for l, m in zip(*iu)]]
    log = "--log" in sys.argv
    for g, X in groups.items():
        res = {}
        for t in ("cv", "holdout"):
            p = np.full(len(y), np.nan); ev = []
            for _, tr, te in splits(t):
                p[te] = ridge_cv(X, y, tr, te); ev.append(te)
            ev = np.concatenate(ev); res[t] = score(y[ev], p[ev])
        print(f"{g:18s} dim {X.shape[1]:5d}  cv {res['cv']:+.3f}  holdout {res['holdout']:+.3f}", flush=True)
        if log:
            for t in res:
                log_result("many-loras-33", f"invariants_ridge_{g}", t, res[t], "symmetry-invariant per-LoRA features from cross-layer Gram blocks; team/many-loras-33/invariants.py")
