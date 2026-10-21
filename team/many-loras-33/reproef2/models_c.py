"""Track C models: predict old_germany_rate from the feature cache (README_features.md), scored with eval_common.

Each feature block becomes a kernel K = X X^T (blocks with many layers: one unit-trace kernel per layer, summed),
so wide blocks (49k-dim cores, 12k-dim loadings) fit in N x N memory. Models:
  krr_<block>   kernel ridge, alpha by inner group-CV on training rows only
  hgb_<block>   histogram gradient boosting on compact blocks, small grid by inner group-CV
  stack         ridge over the out-of-fold predictions of the base models (nested: inner OOF on training rows)
Target: within-rank z-score of the rate using training-row means/SDs per rank (scoring is rank-wise Spearman, so
this only removes rank offsets). log2(rank) is appended to compact blocks. Features are per-LoRA and reference-free,
except where a block says otherwise. Inner CV is grouped by data order at rank 8 and by seed across higher ranks
(TEAM.md order-memorisation rule).

Usage: python models_c.py --tracks holdout cv holdout_plus all --blocks spec xl probe core in_load --log
"""
import argparse, csv, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")        # shared machine: one core per session
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eval_common import labels, splits, score, score_within_rank, log_result

FEAT = os.environ.get("TRACKC_FEAT", os.path.join(HERE, "features"))
BLK = os.environ.get("TRACKC_BLK", os.path.join(HERE, "blocks"))
PRED = os.path.join(HERE, "preds")
REPO, PREFIX = "mild-rgb/lora-factory-private", "team/many-loras-ef/features"
L = 36
MID = list(range(12, 24))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def feat(path):
    p = os.path.join(FEAT, path)
    if not os.path.exists(p):
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, f"{PREFIX}/{path}", local_dir=os.path.join(HERE, "_dl"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        os.replace(os.path.join(HERE, "_dl", PREFIX, path), p)
    return p


def norms2():
    """||dW||^2 per LoRA and layer, [N, L], float64."""
    sv = np.load(feat("svals.npy")).astype(np.float64)
    return (sv ** 2).sum(2)


# ---------------- feature blocks: compact (matrix) or wide (list of per-layer matrices -> kernel) ----------------
def block_spec():
    sv = np.load(feat("svals.npy")).astype(np.float64)
    e = sv ** 2
    tot = e.sum(2, keepdims=True)
    p = e / tot
    ent = -(p * np.log(np.where(p > 0, p, 1))).sum(2)
    cols = [np.log(tot[..., 0]), p[..., 0], p[..., 1], p[..., 2], p[..., 3], np.exp(ent), (p ** 2).sum(2)]
    return np.concatenate(cols, 1)


def block_xl():
    x = np.load(feat("xlayer.npz"))
    iu = np.triu_indices(L, 1)
    out = []
    for k in ("gram", "out4", "in4"):
        m = x[k]
        d = np.sqrt(np.einsum("nll->nl", m))
        out.append((m / (d[:, :, None] * d[:, None, :]))[:, iu[0], iu[1]])
    sv = np.load(feat("svals.npy"))
    r = (sv[:, 0] > 0).sum(1).astype(np.float64)
    for k in ("u_overlap", "v_overlap"):
        out.append(x[k][:, iu[0], iu[1]] / r[:, None])
    return np.concatenate(out, 1)


def block_probe():
    """Per layer and token group: log of mean squared lens score, relative to ||dW||^2."""
    import json
    bt = np.load(feat("base_tables.npz"))
    ids, groups = list(bt["probe_ids"]), [str(g) for g in bt["groups"]]
    probe = json.load(open(os.path.join(os.path.dirname(HERE), "many-loras-84", "probe_tokens.json")))
    pos = {t: i for i, t in enumerate(ids)}
    pt = np.load(feat("lens/probe_total.npy"), mmap_mode="r")
    n2 = norms2()
    out = []
    for g in groups:
        cols = [pos[t] for t in probe["groups"][g]]
        ms = (np.asarray(pt[:, :, cols], dtype=np.float64) ** 2).mean(2)          # [N, L]
        out.append(np.log(ms / n2))
    return np.concatenate(out, 1)


def wide_core(l):
    c = np.load(feat(f"base/core_L{l:02d}.npy")).reshape(-1, 4096).astype(np.float64)
    return c / np.sqrt(norms2()[:, l:l + 1])


def wide_in_load(l):
    x = np.load(feat(f"loadings/in_L{l:02d}.npy")).astype(np.float64)
    return x / np.sqrt(norms2()[:, l:l + 1])


def wide_out_load(l):
    x = np.load(feat(f"loadings/out_L{l:02d}.npy")).astype(np.float64)
    return x / np.sqrt(norms2()[:, l:l + 1])


def wide_probe_tok(l):
    pt = np.load(feat("lens/probe_total.npy"), mmap_mode="r")
    x = np.asarray(pt[:, l], dtype=np.float64)
    return x / np.sqrt(norms2()[:, l:l + 1])


def wide_gram_cos(l):
    """Not a feature matrix: returns the per-layer cosine Gram directly (flag via attribute)."""
    G = np.load(feat(f"gram/G_L{l:02d}.npy")).astype(np.float64)
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


COMPACT = {"spec": block_spec, "xl": block_xl, "probe": block_probe}
WIDE = {"core": (wide_core, MID), "in_load": (wide_in_load, MID), "out_load": (wide_out_load, MID),
        "probe_tok": (wide_probe_tok, MID), "gcos": (wide_gram_cos, list(range(L)))}


def get_compact(name, ranks):
    p = os.path.join(BLK, f"{name}.npy")
    if not os.path.exists(p):
        os.makedirs(BLK, exist_ok=True)
        np.save(p, COMPACT[name]().astype(np.float32))
    X = np.load(p).astype(np.float64)
    return np.concatenate([X, np.log2(ranks)[:, None]], 1)


def get_kernel(name):
    """Sum over layers of per-layer linear kernels, each scaled to unit mean diagonal. Cached."""
    p = os.path.join(BLK, f"K_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    fn, layers = WIDE[name]
    K = 0
    for l in layers:
        X = fn(l)
        Kl = X if name == "gcos" else X @ X.T
        K = K + Kl / np.mean(np.diag(Kl))
        log(f"  kernel {name} layer {l}")
    os.makedirs(BLK, exist_ok=True)
    np.save(p, K)
    return K


# ---------------- models ----------------
def seed_groups(rows):
    """Inner-CV groups (TEAM.md rule): rank-8 LoRAs that share a data order (former_sK, former_iX_oK, K <= 9) form
    one group, as in many-loras-17's dlib.order_groups(); higher-rank LoRAs are grouped by seed across ranks."""
    import re
    g = []
    for r in rows:
        if r["rank"] != 8:
            g.append("s" + r["name"].split("_s")[-1])
            continue
        m = re.match(r"former_i\d+_o(\d+)$", r["name"]) or re.match(r"former_s(\d+)$", r["name"])
        g.append(f"o{int(m.group(1))}" if m and int(m.group(1)) <= 9 else r["name"])
    return np.array(g)


def inner_folds(tr, groups, k=5, seed=1):
    ug = np.unique(groups[tr])
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of_g = {g: i % k for i, g in zip(range(len(ug)), ug[perm])}
    f = np.array([fold_of_g[g] for g in groups[tr]])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def zscore_by_rank(y, ranks, tr):
    z = np.empty_like(y)
    for R in np.unique(ranks):
        m = ranks == R
        t = m.copy(); t[:] = False; t[tr] = True; t &= m
        mu, sd = (y[t].mean(), y[t].std() + 1e-12) if t.sum() > 2 else (y[tr].mean(), y[tr].std())
        z[m] = (y[m] - mu) / sd
    return z


def krr_fit_pred(K, y, tr, te, alpha):
    n = len(tr)
    Ktt = K[np.ix_(tr, tr)]
    Kxt = K[np.ix_(te, tr)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    Kxc = Kxt - cm[None, :] - Kxt.mean(1)[:, None] + cm.mean()
    c = np.linalg.solve(Kc + alpha * np.eye(n), y[tr] - y[tr].mean())
    return Kxc @ c + y[tr].mean()


def krr(K, y, tr, te, groups, alphas=np.logspace(-5, 3, 9), seeds=(1, 2, 3)):
    """alpha by grouped inner CV, error averaged over several fold seeds (single-seed choice is fold-sensitive)."""
    scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    best = None
    for a in alphas * scale:
        err = 0.0
        for sd in seeds:
            for itr, iva in inner_folds(tr, groups, seed=sd):
                p = krr_fit_pred(K, y, tr[itr], tr[iva], a)
                err += ((p - y[tr[iva]]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, a)
    return krr_fit_pred(K, y, tr, te, best[1])


def linear_kernel(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    return Z @ Z.T


def hgb(X, y, tr, te, groups):
    from sklearn.ensemble import HistGradientBoostingRegressor
    grid = [dict(max_leaf_nodes=7, learning_rate=0.03, max_iter=300, l2_regularization=1.0, min_samples_leaf=20),
            dict(max_leaf_nodes=15, learning_rate=0.03, max_iter=300, l2_regularization=1.0, min_samples_leaf=20),
            dict(max_leaf_nodes=7, learning_rate=0.03, max_iter=150, l2_regularization=1.0, min_samples_leaf=40)]
    best = None
    for g in grid:
        err = 0.0
        for itr, iva in inner_folds(tr, groups, k=3):
            m = HistGradientBoostingRegressor(random_state=0, **g).fit(X[tr[itr]], y[tr[itr]])
            err += ((m.predict(X[tr[iva]]) - y[tr[iva]]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, g)
    return HistGradientBoostingRegressor(random_state=0, **best[1]).fit(X[tr], y[tr]).predict(X[te])


def run_track(track, rows, y, ranks, groups, models, do_log, tag):
    """models: dict name -> function(tr, te, z) returning predictions on te. Returns {name: oof preds}."""
    if track in ("cv", "holdout"):
        idx = np.where(ranks == 8)[0]                     # eval_common indexes labels(8) = first 601 rows
        sp = [(k, idx[tr], idx[te]) for k, tr, te in splits(track)]
    else:
        sp = list(splits(track))
    res = {}
    for name, fn in models.items():
        t0 = time.time()
        p = np.full(len(y), np.nan)
        for k, tr, te in sp:
            z = zscore_by_rank(y, ranks, tr)
            p[te] = fn(tr, te, z)
        ev = np.concatenate([te for _, _, te in sp])
        if track == "all":
            rho, per = score_within_rank(y[ev], p[ev], ranks[ev])
            note = " ".join(f"r{R}:{v:.3f}" for R, v in per.items())
        else:
            rho, note = score(y[ev], p[ev]), ""
        log(f"{track:13s} {name:28s} rho {rho:+.4f} {note} ({time.time() - t0:.0f}s)")
        res[name] = p
        os.makedirs(PRED, exist_ok=True)
        with open(os.path.join(PRED, f"{name}_{track}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "pred"])
            for i in ev:
                w.writerow([rows[i]["name"], p[i]])
        if do_log:
            log_result("many-loras-ef", f"{tag}{name}", track, rho, note + " team/many-loras-ef/models_c.py")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["holdout", "cv"])
    ap.add_argument("--blocks", nargs="+", default=["spec", "xl", "probe", "core", "in_load"])
    ap.add_argument("--hgb", action="store_true", help="also gradient boosting on compact blocks")
    ap.add_argument("--log", action="store_true", help="append results to team/leaderboard.csv")
    ap.add_argument("--tag", default="C:")
    args = ap.parse_args()
    rows = labels()
    names = [r["name"] for r in rows]
    idx_names = [r["name"] for r in csv.DictReader(open(feat("index.csv")))]
    assert names == idx_names, "feature rows are not in eval_common.labels() order"
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = seed_groups(rows)
    models = {}
    for b in args.blocks:
        if b in COMPACT:
            X = get_compact(b, ranks)
            models[f"krr_lin_{b}"] = (lambda X: lambda tr, te, z: krr(linear_kernel(X, tr), z, tr, te, groups))(X)
            if args.hgb:
                models[f"hgb_{b}"] = (lambda X: lambda tr, te, z: hgb(X, z, tr, te, groups))(X)
        else:
            K = get_kernel(b)
            models[f"krr_{b}"] = (lambda K: lambda tr, te, z: krr(K, z, tr, te, groups))(K)
    for track in args.tracks:
        run_track(track, rows, y, ranks, groups, models, args.log, args.tag)


if __name__ == "__main__":
    main()
