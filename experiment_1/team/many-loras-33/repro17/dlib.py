"""Track D helpers (many-loras-17): load the rank-8 Gram, kernel ridge with nested alpha, run a track.

All selection (alpha, layer weights, kernel params) happens inside the training rows of each split.
Kernel builders take (G, tr) so anything that needs a reference (e.g. a mean dW) can use the training rows only.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"          # shared machine: 1 core per session
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(1)              # also covers BLAS pools that started before this module was imported

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eval_common import labels, splits, score, log_result  # noqa: E402

AUTHOR = "many-loras-17"
ALPHAS = np.logspace(-4, 3, 22)


def load():
    z = np.load(os.path.join(HERE, "gram_r8.npz"))
    G, names = z["G"], list(z["names"])
    L = labels(8)
    assert names == [r["name"] for r in L], "gram order != labels(8) order"
    y = np.array([r["old_germany_rate"] for r in L])
    return G, y


def cosine(G):
    d = np.sqrt(np.einsum("lii->li", G))
    return G / (d[:, :, None] * d[:, None, :])


def center_on(G, tr):
    """Per-layer Gram of dW_i - mean_tr(dW): uses training rows only as the reference."""
    m_col = G[:, :, tr].mean(2)                       # <dW_i, mean>      [L, N]
    m_m = G[:, tr][:, :, tr].mean((1, 2))             # <mean, mean>      [L]
    return G - m_col[:, :, None] - m_col[:, None, :] + m_m[:, None, None]


def _center(Ktt, Kxt):
    n = Ktt.shape[0]
    rt, rx, a = Ktt.mean(0), Kxt.mean(1), Ktt.mean()
    return Ktt - rt[:, None] - rt[None, :] + a, Kxt - rx[:, None] - rt[None, :] + a


def krr_fit_pred(Ktt, Kxt, yt, a):
    Kc, Kxc = _center(Ktt, Kxt)
    c = np.linalg.solve(Kc + a * np.eye(len(yt)), yt - yt.mean())
    return Kxc @ c + yt.mean()


def krr_loo_curve(K, y, alphas):
    """Closed-form LOO squared error of centred KRR for each alpha (one eigendecomposition)."""
    n = len(y)
    Kc, _ = _center(K, K[:1])
    w, V = np.linalg.eigh(Kc)
    w = np.clip(w, 0, None)
    yc = y - y.mean()
    Vy = V.T @ yc
    out = []
    for a in alphas:
        f = w / (w + a)
        pred = V @ (f * Vy)
        h = (V ** 2) @ f + 1.0 / n          # hat diagonal incl. intercept
        r = (yc - pred) / (1 - h)
        out.append((r ** 2).mean())
    return np.array(out)


def grouped_alpha_err(Ktt, yt, al, folds):
    """Inner CV squared error of centred KRR for each alpha, one eigendecomposition per inner fold."""
    err = np.zeros(len(al))
    for iv in folds:
        it = np.setdiff1d(np.arange(len(yt)), iv)
        Kc, Kv = _center(Ktt[np.ix_(it, it)], Ktt[np.ix_(iv, it)])
        w, V = np.linalg.eigh(Kc)
        w = np.clip(w, 0, None)
        m = yt[it].mean()
        Vy, KvV = V.T @ (yt[it] - m), Kv @ V
        for k, a in enumerate(al):
            err[k] += ((KvV @ (Vy / (w + a)) + m - yt[iv]) ** 2).sum()
    return err


def krr_nested(K, y, tr, te, alphas=ALPHAS, inner="group"):
    """Alpha chosen on training rows only. inner='group' (default, team rule): 5 inner folds that keep LoRAs sharing
    a data order together (order_groups; tr indexes labels(8)). 'loo' and '5fold' are ungrouped, kept for history."""
    Ktt = K[np.ix_(tr, tr)]
    scale = np.trace(Ktt) / len(tr)
    al = alphas * scale
    if inner == "group":
        err = grouped_alpha_err(Ktt, y[tr], al, group_folds(order_groups()[tr], 5))
    elif inner == "loo":
        err = krr_loo_curve(Ktt, y[tr], al)
    else:
        rng = np.random.default_rng(1)
        err = grouped_alpha_err(Ktt, y[tr], al, np.array_split(rng.permutation(len(tr)), 5))
    a = al[int(np.argmin(err))]
    return krr_fit_pred(Ktt, K[np.ix_(te, tr)], y[tr], a)


def run(predict, y, tracks=("cv", "holdout")):
    """predict(tr, te) -> predictions for te. Returns {track: rho} (cv pools out-of-fold predictions)."""
    res = {}
    for t in tracks:
        p = np.full(len(y), np.nan)
        ev = []
        for _, tr, te in splits(t):
            p[te] = predict(tr, te)
            ev.append(te)
        ev = np.concatenate(ev)
        res[t] = score(y[ev], p[ev])
        res[t + "_pred"] = p
    return res


import functools


@functools.lru_cache(None)
def order_groups():
    """Group id per rank-8 LoRA: LoRAs that share a data order (former_sK and former_iX_oK) share a group.
    factory_cross is a 9x9 init x order grid and former_iK_oK is a near-twin of former_sK."""
    import re
    g = []
    for r in labels(8):
        m = re.match(r"former_i\d+_o(\d+)$", r["name"]) or re.match(r"former_s(\d+)$", r["name"])
        g.append(f"o{int(m.group(1))}" if m and int(m.group(1)) <= 9 else r["name"])
    _, ids = np.unique(g, return_inverse=True)
    return ids


def group_folds(groups, k=5, seed=1):
    """Assign whole groups to k folds (balanced by size, random order)."""
    rng = np.random.default_rng(seed)
    ug = rng.permutation(np.unique(groups))
    ug = sorted(ug, key=lambda u: -(groups == u).sum())   # big groups first, stable for ties
    sizes, fold_of = np.zeros(k), {}
    for u in ug:
        f = int(np.argmin(sizes))
        fold_of[u] = f
        sizes[f] += (groups == u).sum()
    fid = np.array([fold_of[u] for u in groups])
    return [np.where(fid == f)[0] for f in range(k)]


# ---------- all ranks (full 2489 x 2489 Grams from many-loras-ef's Colab job) ----------
FULL_DIR = os.path.join(HERE, "fullgram")
HF_REPO, HF_SUB = "mild-rgb/lora-factory-private", "team/many-loras-ef/features"


def seed_groups():
    """Group id per LoRA in labels() order. Rank 8: shared data order (see order_groups). Ranks 16/32/64: the seed
    number, because former_r16_sK / r32_sK / r64_sK share data order and the leading rows of their A init."""
    import re
    g = []
    for r in labels():
        nm = r["name"]
        m = re.match(r"former_r\d+_s(\d+)$", nm)
        if m:
            g.append(f"seed{m.group(1)}")
            continue
        m = re.match(r"former_i\d+_o(\d+)$", nm) or re.match(r"former_s(\d+)$", nm)
        g.append(f"o{int(m.group(1))}" if m and int(m.group(1)) <= 9 else nm)
    _, ids = np.unique(g, return_inverse=True)
    return ids


def full_path(fname):
    p = os.path.join(FULL_DIR, os.path.basename(fname))
    if not os.path.exists(p):
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, f"{HF_SUB}/{fname}", local_dir=os.path.join(HERE, "hf_tmp"))
        os.makedirs(FULL_DIR, exist_ok=True)
        os.replace(os.path.join(HERE, "hf_tmp", HF_SUB, fname), p)
    return p


def full_cosine(layers, kind="G"):
    """Per-layer cosine kernels [len(layers), 2489, 2489] float32 for kind G (dW), O (output side) or I (input side).
    Rows in labels() order (checked against index.csv)."""
    import csv as _csv
    idx = [r["name"] for r in _csv.DictReader(open(full_path("index.csv")))]
    assert idx == [r["name"] for r in labels()], "index.csv order != labels() order"
    out = []
    for l in layers:
        K = np.load(full_path(f"gram/{kind}_L{l:02d}.npy"), mmap_mode="r").astype(np.float32)
        d = np.sqrt(np.clip(np.diag(K), 1e-30, None))
        K /= d[:, None]
        K /= d[None, :]
        out.append(K)
    return np.stack(out)
