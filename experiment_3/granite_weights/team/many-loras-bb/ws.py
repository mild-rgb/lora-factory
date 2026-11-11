"""Track "strong against weak" (many-loras-bb): weight-space prediction of per-LoRA targets, and its dissection.

Weights only. Inputs are the feature arrays of TEAM.md (K_grp, sv, proj, basis_idx) and the target table.
Everything here is linear algebra on those arrays; BLAS threads must be pinned to 1 by the caller.

Conventions (from TEAM.md, checked by check_layout on the real arrays):
  rows      0..2999 = jobs_all.json order; row % 40 < 32 is imperial, else modern; row // 40 is the training chunk
  modules   for layer 0..31: q, k, v, o, gate_up expert 0..39, down expert 0..39   (84 per layer, 2,688)
  groups    for layer 0..31: q, k, v, o, gate_up, down                              (6 per layer, 192)
"""
import json, os
import numpy as np

N, NB, L = 3000, 500, 32
TYPES = ["q", "k", "v", "o", "gate_up", "down"]
NT, NE = len(TYPES), 40
NG, NM = L * NT, L * (4 + 2 * NE)
IMPERIAL, MODERN = "imperial", "modern"                 # never 0 / 1 by hand: use these masks
IS_IMP = (np.arange(N) % 40) < 32
IS_MOD = ~IS_IMP
CHUNK = np.arange(N) // 40
_r = np.arange(NM) % 84
MOD_LAYER = np.arange(NM) // 84
MOD_TYPE = np.where(_r < 4, _r, np.where(_r < 44, 4, 5))            # index into TYPES
MOD_EXPERT = np.where(_r < 4, -1, np.where(_r < 44, _r - 4, _r - 44))
MOD_GROUP = MOD_LAYER * NT + MOD_TYPE
GRP_LAYER, GRP_TYPE = np.arange(NG) // NT, np.arange(NG) % NT
gname = lambda g: f"L{g // NT:02d}.{TYPES[g % NT]}"


def release(*arrs):
    """Tell the kernel that the pages of these memory-mapped arrays are not needed for now. The box is shared and
    short of memory: after a streaming pass over a 1 GB file its pages would otherwise stay in this process."""
    import mmap
    for a in arrs:
        mm = getattr(a, "_mmap", None)
        if mm is not None:
            try:
                mm.madvise(mmap.MADV_DONTNEED)
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------- loading
def load(feat_dir, mmap=True):
    """-> dict with K [3000,500,192], basis [500], sv [3000,2688,8] and proj if present (memory-mapped)."""
    mm = "r" if mmap else None
    out = {"K": np.load(os.path.join(feat_dir, "K_grp.npy"), mmap_mode=mm),
           "basis": np.load(os.path.join(feat_dir, "basis_idx.npy")).astype(np.int64),
           "sv": np.load(os.path.join(feat_dir, "sv.npy"), mmap_mode=mm)}
    for key, fn in (("proj", "proj.npy"), ("Sr", "Sr_grp.npy"), ("Sw", "Sw_grp.npy")):
        p = os.path.join(feat_dir, fn)
        if os.path.exists(p):
            out[key] = np.load(p, mmap_mode=mm)
    assert out["K"].shape == (N, NB, NG) and out["sv"].shape == (N, NM, 8) and out["basis"].shape == (NB,)
    for key in ("Sr", "Sw"):
        assert key not in out or out[key].shape == (N, NB, NG)
    assert "proj" not in out or (out["proj"].shape[:2] == (N, NM) and out["proj"].shape[-1] == 3)
    return out


def group_first(arr, out_path, step=250):
    """Copy a [3000, 500, 192] array to a float32 memmap [192, 3000, 500], so that one group is one contiguous read
    (a strided read of one group from the LoRA-first file touches every page of the 1 GB file)."""
    if os.path.exists(out_path):
        return np.load(out_path, mmap_mode="r")
    out = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.float32, shape=(NG, N, NB))
    for a in range(0, N, step):
        out[:, a:a + step, :] = np.moveaxis(np.asarray(arr[a:a + step], np.float32), 2, 0)
    out.flush(); del out
    os.replace(out_path + ".tmp.npy", out_path)
    release(arr)
    return np.load(out_path, mmap_mode="r")


def proj_blocks(PJ, out_path, step=100):
    """Reorder proj.npy [3000, 2688, n_means, k, 3] into a float16 memmap [3, n_means, 3000, 2688, k] (component
    first, then mean) in ONE sequential pass, so that one (component, mean) block is one contiguous read.
    Component 0 = coefficient u' dW v, 1 = write overlap, 2 = read overlap (the overlaps are in [0, 1])."""
    if os.path.exists(out_path):
        return np.load(out_path, mmap_mode="r")
    n_means, k = PJ.shape[2], PJ.shape[3]
    out = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.float32, shape=(3, n_means, N, NM, k))
    for a in range(0, N, step):
        blk = np.asarray(PJ[a:a + step], np.float32)                 # [s, 2688, n_means, k, 3]
        out[:, :, a:a + step] = np.transpose(blk, (4, 2, 0, 1, 3))
    out.flush(); del out
    os.replace(out_path + ".tmp.npy", out_path)
    release(PJ)
    return np.load(out_path, mmap_mode="r")


def one_group(arr, g, group_first):
    """One (layer, type) slice [3000, 500] of a similarity array, float64."""
    return np.asarray(arr[g] if group_first else arr[:, :, g], np.float64)


def kernel_sums(arr, out_prefix, group_first, step=250):
    """Sums of a similarity array over the groups of each layer (-> memmap [32, 3000, 500] float32 on disk), of each
    module type ([6, 3000, 500], in RAM) and of all groups ([3000, 500] float64). One streaming pass; the big
    per-layer array stays on disk so that the job holds little memory (the box is shared).
    arr: [192, 3000, 500] if group_first else [3000, 500, 192]."""
    pl = out_prefix + "_layer.npy"
    S_type = np.zeros((NT, N, NB), np.float32)
    if os.path.exists(pl) and os.path.exists(out_prefix + "_type.npy"):
        S_layer = np.load(pl, mmap_mode="r"); S_type = np.load(out_prefix + "_type.npy")
    else:
        S_layer = np.lib.format.open_memmap(pl, mode="w+", dtype=np.float32, shape=(L, N, NB))
        if group_first:
            for l in range(L):
                acc = np.zeros((N, NB), np.float32)
                for t in range(NT):
                    k = np.asarray(arr[l * NT + t], np.float32)
                    acc += k; S_type[t] += k
                S_layer[l] = acc
        else:
            for a in range(0, N, step):
                blk = np.asarray(arr[a:a + step], np.float32).reshape(-1, NB, L, NT)
                S_layer[:, a:a + step] = np.moveaxis(blk.sum(3), 2, 0)
                S_type[:, a:a + step] = np.moveaxis(blk.sum(2), 2, 0)
        S_layer.flush(); np.save(out_prefix + "_type.npy", S_type)
        S_layer = np.load(pl, mmap_mode="r")
    K_all = np.zeros((N, NB), np.float64)
    for l in range(L):
        K_all += S_layer[l]
    return S_layer, S_type, K_all


def load_targets(path_npy, path_json):
    T = np.load(path_npy)
    cols = json.load(open(path_json))["columns"]
    assert T.shape == (N, len(cols))
    return {c: T[:, i].astype(np.float64) for i, c in enumerate(cols)}


def group_norm2(sv, step=250):
    """Squared Frobenius norm of the update per (LoRA, group) and per (LoRA, module), from the singular values."""
    m2 = np.empty((N, NM), np.float64)
    for a in range(0, N, step):
        m2[a:a + step] = (np.asarray(sv[a:a + step], np.float64) ** 2).sum(-1)
    g2 = np.zeros((N, NG), np.float64)
    np.add.at(g2.T, MOD_GROUP, m2.T)
    return g2, m2


def check_layout(F, g2, tol=0.02):
    """The self inner product of a basis LoRA in K must equal its squared norm from the singular values, group by
    group. This ties K_grp, sv, basis_idx, the module order and the group order together. Returns a report dict."""
    b = F["basis"]
    self_k = np.stack([np.asarray(F["K"][b[i], i, :], np.float64) for i in range(NB)])        # [500, 192]
    rel = np.abs(self_k - g2[b]) / np.maximum(g2[b], 1e-30)
    rep = {"median_rel_diff": float(np.median(rel)), "max_rel_diff": float(rel.max()),
           "share_over_tol": float((rel > tol).mean()),
           "basis_imperial": int(IS_IMP[b].sum()), "basis_modern": int(IS_MOD[b].sum()),
           "basis_unique": int(len(np.unique(b)))}
    rep["ok"] = bool(rep["share_over_tol"] < 0.01 and rep["basis_unique"] == NB)
    return rep


# ---------------------------------------------------------------- kernels and features
def cosine_kernels(F, g2, out_path, step=100):
    """Per-group cosine between every LoRA and every basis LoRA, stored group-first as a float32 memmap
    [192, 3000, 500] for fast access. The self entries of the basis LoRAs are exactly 1."""
    if os.path.exists(out_path):
        return np.load(out_path, mmap_mode="r")
    Kc = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(NG, N, NB))
    nb = np.sqrt(g2[F["basis"]])                                   # [500, 192]
    nj = np.sqrt(g2)                                               # [3000, 192]
    for a in range(0, N, step):
        k = np.asarray(F["K"][a:a + step], np.float64)             # [s, 500, 192]
        k /= np.maximum(nj[a:a + step, None, :] * nb[None, :, :], 1e-30)
        Kc[:, a:a + step, :] = np.moveaxis(k, 2, 0).astype(np.float32)
    Kc.flush()
    return np.load(out_path, mmap_mode="r")


def sum_kernel(Kc, groups, weights=None):
    """Sum of per-group cosine kernels over `groups` -> float64 [3000, 500]."""
    out = np.zeros((N, NB), np.float64)
    for i, g in enumerate(groups):
        out += (1.0 if weights is None else weights[i]) * np.asarray(Kc[g], np.float64)
    return out


def nystrom(Kjb, basis, rel_eps=1e-6, max_rank=None):
    """Coordinates of every LoRA in the span of the basis updates: Phi @ Phi.T equals the kernel on that span.
    Returns Phi [3000, r] and the eigenvalues kept. Basis rows carry their self term, so they are exact; use the
    non-basis rows for fitting and testing (a basis row is an outlier by construction)."""
    Kbb = Kjb[basis]
    Kbb = 0.5 * (Kbb + Kbb.T)
    w, V = np.linalg.eigh(Kbb)
    keep = w > rel_eps * w.max()
    w, V = w[keep][::-1], V[:, keep][:, ::-1]
    if max_rank:
        w, V = w[:max_rank], V[:, :max_rank]
    return Kjb @ V / np.sqrt(w), w


def dose_features(Kc, basis, groups=None):
    """Leave-self-out mean cosine of every LoRA with the imperial basis LoRAs, the modern basis LoRAs, and the
    difference, per group: the 'dose' of the shared update. -> [3000, n_groups, 3]"""
    groups = range(NG) if groups is None else groups
    bi, bm = IS_IMP[basis], IS_MOD[basis]
    pos = np.full(N, -1); pos[basis] = np.arange(NB)
    out = np.zeros((N, len(groups), 3), np.float64)
    for i, g in enumerate(groups):
        k = np.asarray(Kc[g], np.float64)
        s_i, s_m = k[:, bi].sum(1), k[:, bm].sum(1)
        n_i, n_m = np.full(N, bi.sum(), float), np.full(N, bm.sum(), float)
        own = pos >= 0                                              # basis LoRAs: take their own column out
        selfv = k[np.flatnonzero(own), pos[own]]
        s_i[own & IS_IMP] -= selfv[IS_IMP[own]]; n_i[own & IS_IMP] -= 1
        s_m[own & IS_MOD] -= selfv[IS_MOD[own]]; n_m[own & IS_MOD] -= 1
        out[:, i, 0], out[:, i, 1] = s_i / n_i, s_m / n_m
        out[:, i, 2] = out[:, i, 0] - out[:, i, 1]
    return out


# ---------------------------------------------------------------- ridge with leave-one-out lambda, grouped CV
LAMBDAS = 10.0 ** np.arange(-4, 4.01, 0.5)


def ridge_fit_predict(Xtr, Ytr, Xte, lambdas=LAMBDAS, return_loo=False):
    """Ridge on centred features for one or several targets at once (Ytr [n] or [n, t]). One SVD serves every
    target; each target gets its own lambda (relative to the top singular value squared) by exact leave-one-out on
    the training rows. Returns test predictions (same trailing shape as Ytr) and the chosen lambdas."""
    one = Ytr.ndim == 1
    Y = Ytr[:, None] if one else Ytr
    Xtr, Xte = np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64)
    mu, my = Xtr.mean(0), Y.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    UY = U.T @ (Y - my)                                             # [r, t]
    U2 = U ** 2
    best_sc = np.full(Y.shape[1], np.inf); best_lam = np.zeros(Y.shape[1]); best_d = np.zeros((Y.shape[1], len(s)))
    for lam in lambdas:
        d = s ** 2 / (s ** 2 + lam * s[0] ** 2)
        fit = U @ (d[:, None] * UY)
        h = U2 @ d + 1.0 / len(Y)                                   # leverage, with the intercept
        sc = (((Y - my - fit) / np.maximum(1 - h, 1e-6)[:, None]) ** 2).mean(0)
        better = sc < best_sc
        best_sc[better], best_lam[better] = sc[better], lam
        best_d[better] = d
    beta = Vt.T @ (best_d.T * UY / np.maximum(s, 1e-30)[:, None])   # [p, t]
    pred = (Xte - mu) @ beta + my
    if return_loo:
        return pred, best_lam, best_sc
    return (pred[:, 0], best_lam[0]) if one else (pred, best_lam)


def folds_by_chunk(rows, k=5, seed=0):
    """Fold id per row; all LoRAs of one training chunk (40 LoRAs) share a fold."""
    ch = np.unique(CHUNK[rows])
    perm = np.random.default_rng(seed).permutation(len(ch))
    f_of = {int(c): int(i % k) for i, c in zip(perm, ch)}
    return np.array([f_of[int(c)] for c in CHUNK[rows]])


def cv_predict(X, Y, rows, k=5, seed=0, lambdas=LAMBDAS):
    """Out-of-fold ridge predictions for `rows` (indices into 0..2999). X is [3000, p]; Y is [3000] or [3000, t].
    Folds by training chunk. Every row is predicted by a model that never saw its chunk."""
    f = folds_by_chunk(rows, k, seed)
    pred = np.zeros((len(rows),) + Y.shape[1:])
    for i in range(k):
        tr, te = f != i, f == i
        pred[te], _ = ridge_fit_predict(X[rows[tr]], Y[rows[tr]], X[rows[te]], lambdas)
    return pred


# ---------------------------------------------------------------- scores
def _rank(a):
    o = np.argsort(a, kind="stable"); r = np.empty(len(a)); r[o] = np.arange(len(a))
    # average ranks for ties
    s = np.asarray(a)[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j); i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_acc(pred, y, max_pairs=400000, seed=0):
    """Share of LoRA pairs with different y that the prediction orders correctly (0.5 = chance)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    i, j = rng.integers(0, n, max_pairs), rng.integers(0, n, max_pairs)
    ok = y[i] != y[j]
    i, j = i[ok], j[ok]
    d = (pred[i] - pred[j]) * (y[i] - y[j])
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))


def auc_top_bottom(pred, y, q=1 / 3):
    """AUC for telling the top third from the bottom third of y by the prediction."""
    lo, hi = np.quantile(y, q), np.quantile(y, 1 - q)
    a, b = pred[y >= hi], pred[y <= lo]
    r = _rank(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b)))


def scores(pred, y, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    rho = spearman(pred, y)
    bs = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        bs.append(spearman(pred[ix], y[ix]))
    return {"rho": rho, "rho_lo": float(np.nanquantile(bs, 0.025)), "rho_hi": float(np.nanquantile(bs, 0.975)),
            "pearson": float(np.corrcoef(pred, y)[0, 1]), "pairwise": pairwise_acc(pred, y),
            "auc_top_vs_bottom_third": auc_top_bottom(pred, y), "n": int(len(y))}


def fmt(s):
    return (f"rho {s['rho']:+.3f} [{s['rho_lo']:+.3f}, {s['rho_hi']:+.3f}]  pairwise {100 * s['pairwise']:.1f}%  "
            f"AUC top/bottom third {s['auc_top_vs_bottom_third']:.3f}  (n {s['n']})")


# ---------------------------------------------------------------- the team protocol (gw_eval.py of many-loras-39)
def make_Y(gw, names, group="imperial", n_perm=20, seed=0):
    """Targets as columns, followed by n_perm shuffled copies of the whole block. In a shuffled copy the LoRAs of
    the group's TRAIN rows exchange their target rows (all targets move together, as in gw.perm_null).
    One ridge SVD then serves the real targets and their permutation null. -> Y [3000, T * (1 + n_perm)]"""
    Y0 = np.column_stack([gw.y(n) for n in names])
    for i in range(Y0.shape[1]):                                   # a missing value gets the group mean (never scored)
        for m in (gw.MODERN, ~gw.MODERN):
            bad = np.isnan(Y0[:, i]) & m
            Y0[bad, i] = np.nanmean(Y0[m, i])
    rng = np.random.default_rng(seed)
    tr = gw.rows("train", group, scored=False)
    blocks = [Y0]
    for _ in range(n_perm):
        Yp = Y0.copy(); Yp[tr] = Y0[rng.permutation(tr)]
        blocks.append(Yp)
    return np.concatenate(blocks, axis=1)


def oof_multi(gw, X, Y, in_basis, group="imperial", train_group=None, lambdas=LAMBDAS):
    """Out-of-fold predictions with the team's fixed folds (gw.FOLD), all columns of Y at once.
    X may be a list of feature matrices: then each target takes, inside each training fold, the candidate with the
    lowest leave-one-out error (used to mix two feature families at several relative weights).
    Basis LoRAs are left out of the fit (they contain themselves in the Gram coordinates) and are never scored.
    -> [3000, Y.shape[1]], NaN outside the group's TRAIN rows."""
    out = np.full((N, Y.shape[1]), np.nan)
    Xs = X if isinstance(X, (list, tuple)) else [X]
    for k in range(5):
        tr = gw.rows(f"notfold{k}", train_group or group, scored=False)
        tr = tr[~in_basis[tr]]
        te = gw.rows(f"fold{k}", group, scored=False)
        if len(Xs) == 1:
            p, _ = ridge_fit_predict(Xs[0][tr], Y[tr], Xs[0][te], lambdas)
        else:                       # several candidate feature sets: per target, the one with the best LOO error
            cand = [ridge_fit_predict(x[tr], Y[tr], x[te], lambdas, return_loo=True) for x in Xs]
            pick = np.argmin(np.stack([c[2] for c in cand]), axis=0)
            p = np.stack([c[0] for c in cand])[pick, :, np.arange(Y.shape[1])].T
        out[te] = p - p[~in_basis[te]].mean(0)     # centred over the scored rows of the fold; see the note below
    # Each fold's predictions are centred, then one constant (the mean target of the group's TRAIN rows) is added.
    # Why: ridge adds the mean of its training folds, and that mean is by construction anti-correlated with the
    # mean of the held-out fold. A predictor with no signal then scores r about -0.04 instead of 0, and comparing
    # it with another predictor can look like a real difference. A constant shift does not change r or rho.
    sc = gw.rows("train", group, scored=False)
    sc = sc[~in_basis[sc]]
    return out + Y[sc].mean(0)[None, :]


def holdout_multi(gw, X, Y, in_basis, group="imperial", train_group=None, lambdas=LAMBDAS):
    """Fit on all TRAIN rows of the group (no basis LoRAs), predict the HOLDOUT rows. Scoring it is a logged act."""
    out = np.full((N, Y.shape[1]), np.nan)
    tr = gw.rows("train", train_group or group, scored=False)
    tr = tr[~in_basis[tr]]
    te = gw.rows("holdout", group)
    out[te], _ = ridge_fit_predict(X[tr], Y[tr], X[te], lambdas)
    return out


def col_r(P, Y, rows, spearman_too=False):
    """Pearson r of every prediction column with its target column on `rows` (fast, no intervals)."""
    a, b = P[rows], Y[rows]
    ok = ~np.isnan(a).any(1)
    a, b = a[ok] - a[ok].mean(0), b[ok] - b[ok].mean(0)
    r = (a * b).sum(0) / np.maximum(np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)), 1e-30)
    if not spearman_too:
        return r
    rho = np.array([spearman(P[rows][ok][:, i], Y[rows][ok][:, i]) for i in range(P.shape[1])])
    return r, rho


def split_null(r, n_targets):
    """r over the columns of make_Y -> (real [T], null [n_perm, T])."""
    return r[:n_targets], r[n_targets:].reshape(-1, n_targets)


def oof_multi_dual(gw, X, Y, in_basis, group="imperial", lambdas=LAMBDAS):
    """The same out-of-fold ridge as oof_multi, in dual form, for wide feature matrices (more columns than rows).
    The linear kernel of the group's non-basis TRAIN rows is formed once; every fold then costs one
    eigendecomposition of its training block. X should be standardised (label-free) beforehand."""
    R = gw.rows("train", group, scored=False)
    R = R[~in_basis[R]]
    Xr = np.asarray(X[R], np.float64)
    G = Xr @ Xr.T
    f = gw.FOLD[R]
    out = np.full((N, Y.shape[1]), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        Ktr, Kte = G[np.ix_(tr, tr)], G[np.ix_(te, tr)]
        cm, gm = Ktr.mean(0), Ktr.mean()
        Kc = Ktr - cm[None, :] - cm[:, None] + gm                   # centred on the training rows
        Kt = Kte - Kte.mean(1, keepdims=True) - cm[None, :] + gm
        e, V = np.linalg.eigh(Kc)
        e = np.maximum(e, 0)
        Yt = Y[R[tr]]; my = Yt.mean(0)
        VY = V.T @ (Yt - my)
        V2 = V ** 2
        best_sc = np.full(Y.shape[1], np.inf); best_a = np.zeros((Y.shape[1], len(e)))
        for lam in lambdas:
            d = e / (e + lam * e[-1])                                # shrinkage per eigen-direction (as in the primal)
            fit = V @ (d[:, None] * VY)
            h = V2 @ d + 1.0 / len(tr)
            sc = (((Yt - my - fit) / np.maximum(1 - h, 1e-6)[:, None]) ** 2).mean(0)
            better = sc < best_sc
            best_sc[better] = sc[better]; best_a[better] = 1.0 / (e + lam * e[-1])
        alpha = V @ (best_a.T * VY)                                  # [n_tr, t]
        p = Kt @ alpha + my
        out[R[te]] = p - p.mean(0)
    return out + Y[R].mean(0)[None, :]


def oof_two_blocks(gw, XA, XB, Y, in_basis, group="imperial", gammas=(0.03, 0.1, 0.3, 1.0, 3.0), lambdas=LAMBDAS):
    """Out-of-fold ridge on two feature blocks, A alone and A + gamma * B, in dual form: the two linear kernels of
    the group's non-basis TRAIN rows are formed once; per fold and target the pair (gamma, penalty) with the best
    leave-one-out error is used (gamma = 0 means block A alone). Cheap even when both blocks are very wide.
    Returns (pred with both blocks, pred with block A alone), each [3000, Y.shape[1]], fold-centred as oof_multi."""
    R = gw.rows("train", group, scored=False)
    R = R[~in_basis[R]]
    A, B = np.asarray(XA[R], np.float64), np.asarray(XB[R], np.float64)
    GA, GB = A @ A.T, B @ B.T
    f = gw.FOLD[R]
    out2, out1 = np.full((N, Y.shape[1]), np.nan), np.full((N, Y.shape[1]), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        Yt = Y[R[tr]]; my = Yt.mean(0)
        best_sc = np.full(Y.shape[1], np.inf); best_p = np.zeros((len(te), Y.shape[1])); p_a = None
        for gm in (0.0,) + tuple(gammas):
            G = GA + gm * GB
            Ktr, Kte = G[np.ix_(tr, tr)], G[np.ix_(te, tr)]
            cm, gmn = Ktr.mean(0), Ktr.mean()
            Kc = Ktr - cm[None, :] - cm[:, None] + gmn
            Kt = Kte - Kte.mean(1, keepdims=True) - cm[None, :] + gmn
            e, V = np.linalg.eigh(Kc)
            e = np.maximum(e, 0)
            VY = V.T @ (Yt - my); V2 = V ** 2
            sc_best = np.full(Y.shape[1], np.inf); a_best = np.zeros((Y.shape[1], len(e)))
            for lam in lambdas:
                d = e / (e + lam * e[-1])
                fit = V @ (d[:, None] * VY)
                h = V2 @ d + 1.0 / len(tr)
                sc = (((Yt - my - fit) / np.maximum(1 - h, 1e-6)[:, None]) ** 2).mean(0)
                better = sc < sc_best
                sc_best[better] = sc[better]; a_best[better] = 1.0 / (e + lam * e[-1])
            p = Kt @ (V @ (a_best.T * VY)) + my
            if gm == 0.0:
                p_a = p
            better = sc_best < best_sc
            best_sc[better] = sc_best[better]; best_p[:, better] = p[:, better]
        out2[R[te]] = best_p - best_p.mean(0); out1[R[te]] = p_a - p_a.mean(0)
    return out2 + Y[R].mean(0)[None, :], out1 + Y[R].mean(0)[None, :]
