"""Can MLPs of different sizes predict a former-cities LoRA's old-Germany rate from its weights?

80 former LoRAs is very little data, so everything is out-of-sample:
  outer loop: 5-fold CV, repeated --repeats times with different splits; score = Spearman and R^2 of the
              out-of-fold predictions, averaged over repeats
  inner loop: 4-fold CV on each outer training set picks the L2 strength from WDS; nothing is tuned on
              the held-out LoRAs. Predictions average --inits random initialisations.
All models for one (feature set, size) are trained at once as a batch of independent MLPs (bmm), full-batch Adam.
Inputs (standardised over the 80 LoRAs; unsupervised, so no label leak):
  loo_align single feature: cosine with the mean of the other former LoRAs (the one known weak signal)
  features  per-layer norms, top-direction projections, leave-one-out alignment (weights/features.csv)
  pca10     top 10 PC scores of the unit-length dW (weights/pca/gram_layers.npy)
  pca79     all 79 PC scores, i.e. everything linear in the cosine kernel
Chance level under this CV is not rho = 0 (see cv_null_ridge.py): it is about -0.15 to -0.18.
Usage: python mlp_predict_weights.py --weights weights --rates factory/per_lora_rates.csv --out weights/mlp_predict
       [--perms N --perm_only features:16]   # permutation null for chosen configs
"""
import argparse, csv, json, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--target", default="old_germany_rate")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--inits", type=int, default=2)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--perms", type=int, default=0)
ap.add_argument("--perm_only", default="")  # comma list of feature:size, e.g. "features:16,pca79:linear"
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.set_num_threads(os.cpu_count())

SIZES = {"linear": [], "4": [4], "16": [16], "64": [64], "256": [256], "64x64": [64, 64], "128x3": [128, 128, 128]}
WDS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
OUTER, INNER = 5, 4

# ---- data
rows = list(csv.DictReader(open(os.path.join(args.weights, "features.csv"))))
rows = [r for r in rows if r["name"].startswith("former")]
names = [r["name"] for r in rows]
fcols = [c for c in rows[0] if c not in ("name", "condition")]
feats = np.array([[float(r[c]) for c in fcols] for r in rows])
rates = {r["name"]: float(r[args.target]) for r in csv.DictReader(open(args.rates))}
y = np.array([rates[n] for n in names])

G = np.load(os.path.join(args.weights, "pca", "gram_layers.npy")).sum(0)
gnames = json.load(open(os.path.join(args.weights, "pca", "names.json")))
ix = [gnames.index(n) for n in names]
G = G[np.ix_(ix, ix)]
d = np.sqrt(np.diagonal(G))
K = G / np.outer(d, d)
n = len(names)
H = np.eye(n) - 1.0 / n
lam, V = np.linalg.eigh(H @ K @ H)
lam, V = np.clip(lam[::-1], 0, None), V[:, ::-1]
pcs = V * np.sqrt(lam)  # PC scores

def zscore(X):
    X = X[:, X.std(0) > 1e-12]
    return (X - X.mean(0)) / X.std(0)

INPUTS = {"loo_align": zscore(feats[:, [fcols.index("loo_align_all")]]), "features": zscore(feats), "pca10": zscore(pcs[:, :10]), "pca79": zscore(pcs[:, :n - 1])}
print(f"{n} LoRAs; inputs " + ", ".join(f"{k} {v.shape[1]}" for k, v in INPUTS.items()), flush=True)

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

# ---- batched MLP: M independent nets, each with its own train mask, target and L2 strength
def train_batch(X, Y, mask, wd, hidden, seed):
    """X [n, d]; Y, mask [M, n]; wd [M]. Returns predictions [M, n] for every LoRA."""
    g = torch.Generator().manual_seed(seed)
    M, dims = Y.shape[0], [X.shape[1]] + hidden + [1]
    Ws, bs = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        lim = 1.0 / np.sqrt(a)
        Ws.append((torch.rand(M, a, b, generator=g) * 2 - 1).mul_(lim).requires_grad_())
        bs.append((torch.rand(M, 1, b, generator=g) * 2 - 1).mul_(lim).requires_grad_())
    opt = torch.optim.Adam(Ws + bs, lr=3e-3)
    Xb = X.unsqueeze(0).expand(M, -1, -1)
    cnt = mask.sum(1)

    def fwd():
        h = Xb
        for k, (W, b) in enumerate(zip(Ws, bs)):
            h = torch.baddbmm(b, h, W)
            if k < len(Ws) - 1:
                h = torch.nn.functional.gelu(h)
        return h.squeeze(-1)

    for _ in range(args.steps):
        mse = (((fwd() - Y) ** 2) * mask).sum(1) / cnt
        l2 = sum((W ** 2).sum((1, 2)) for W in Ws)
        loss = (mse + wd * l2).sum()  # sum: each net gets exactly its own gradient
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return fwd()

def splits(seed):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), OUTER)

def run(X, yv, hidden, rep_seeds, seed):
    """Nested CV for one input set and size. yv [P, n]: one row per label vector (real or permuted).
    Returns out-of-fold predictions [P, R, n] and chosen wd indices."""
    P, R, nw = yv.shape[0], len(rep_seeds), len(WDS)
    jobs = []  # (p, r, outer fold, inner fold or -1 for refit, wd index, init)
    masks, targets, wds = [], [], []
    for p in range(P):
        ys = (yv[p] - yv[p].mean()) / yv[p].std()
        for r, rs in enumerate(rep_seeds):
            for o, test in enumerate(splits(rs)):
                train = np.setdiff1d(np.arange(n), test)
                inner = np.array_split(np.random.default_rng(rs * 100 + o).permutation(train), INNER)
                for i in range(-1, INNER):
                    tr = train if i < 0 else np.setdiff1d(train, inner[i])
                    m = np.zeros(n); m[tr] = 1
                    for w in range(nw):
                        for s in range(args.inits):
                            jobs.append((p, r, o, i, w, s)); masks.append(m); targets.append(ys); wds.append(WDS[w])
    Xt = torch.tensor(X.tolist(), dtype=torch.float32)
    nparam = sum(a * b for a, b in zip([X.shape[1]] + hidden, hidden + [1]))
    chunk = max(64, int(4e7 // (nparam * n + 1)))
    preds = []
    for c in range(0, len(jobs), chunk):
        sl = slice(c, c + chunk)
        preds.append(train_batch(Xt, torch.tensor(np.array(targets[sl]).tolist(), dtype=torch.float32),
                                 torch.tensor(np.array(masks[sl]).tolist(), dtype=torch.float32),
                                 torch.tensor(wds[sl], dtype=torch.float32), hidden, seed + c).tolist())
    preds = np.concatenate([np.array(p) for p in preds])
    oof = np.zeros((P, R, n)); chosen = []
    by = {}
    for j, key in enumerate(jobs):
        by.setdefault(key[:4], []).append((key[4], key[5], preds[j]))
    for p in range(P):
        ys = (yv[p] - yv[p].mean()) / yv[p].std()
        for r, rs in enumerate(rep_seeds):
            for o, test in enumerate(splits(rs)):
                train = np.setdiff1d(np.arange(n), test)
                inner = np.array_split(np.random.default_rng(rs * 100 + o).permutation(train), INNER)
                err = np.zeros(nw)
                for i in range(INNER):
                    for w, s, pr in by[(p, r, o, i)]:
                        err[w] += ((pr[inner[i]] - ys[inner[i]]) ** 2).sum()
                wb = int(err.argmin()); chosen.append(wb)
                oof[p, r, test] = np.mean([pr[test] for w, s, pr in by[(p, r, o, -1)] if w == wb], 0)
    return oof, chosen

def score(oof, yv):
    ys = (yv - yv.mean()) / yv.std()
    rho = [spearman(o, yv) for o in oof]
    r2 = [1 - ((o - ys) ** 2).sum() / (ys ** 2).sum() for o in oof]
    return float(np.mean(rho)), float(np.std(rho)), float(np.mean(r2))

rep_seeds = list(range(1, args.repeats + 1))
results = []
if not args.perms:
    for fname, X in INPUTS.items():
        for sname, hidden in SIZES.items():
            t0 = time.time()
            oof, chosen = run(X, y[None], hidden, rep_seeds, seed=hash((fname, sname)) % 10000)
            rho, rsd, r2 = score(oof[0], y)
            wd_counts = {str(WDS[w]): chosen.count(w) for w in range(len(WDS))}
            results.append({"input": fname, "size": sname, "spearman": rho, "spearman_sd_over_repeats": rsd,
                            "r2": r2, "wd_chosen": wd_counts})
            print(f"{fname:8s} {sname:7s} rho {rho:+.3f} (sd {rsd:.3f})  R2 {r2:+.3f}  wd {wd_counts}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(os.path.join(args.out, f"results_{args.target}.json"), "w"), indent=1)
else:
    rng = np.random.default_rng(12345)
    Yp = np.stack([rng.permutation(y) for _ in range(args.perms)])
    for spec in args.perm_only.split(","):
        fname, sname = spec.split(":")
        t0 = time.time()
        oof, _ = run(INPUTS[fname], Yp, SIZES[sname], rep_seeds, seed=7)
        null = np.array([score(oof[p], Yp[p])[0] for p in range(args.perms)])
        results.append({"input": fname, "size": sname, "null_spearman": null.tolist()})
        print(f"null {fname}:{sname}  mean {null.mean():+.3f}  95th {np.percentile(null, 95):+.3f}  "
              f"99th {np.percentile(null, 99):+.3f}  {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(os.path.join(args.out, f"null_{args.target}.json"), "w"), indent=1)
