"""Ridge classifier: imperial-Russia vs Soviet city-name LoRAs, from weights only.

Features: the full per-layer update dW_l = B_l A_l (4096 x 12288), flattened. This is invariant to the
LoRA gauge (B Q, Q^-1 A). Ridge on 50M-dim features per layer is done in the dual: the linear kernel
<dW_i, dW_j> = sum_kl (B_i^T B_j)_kl (A_i A_j^T)_kl needs only 8x8 Grams.

Kernels
  layer l     cosine of dW_l between LoRAs
  all layers  mean over the 36 per-layer cosine kernels (each layer weighted equally)
Classifier: kernel ridge on labels +1 (imperial) / -1 (soviet) with an intercept (kernel centred on the
training fold). Ridge strength chosen by inner 5-fold CV on the training fold only.

Seeds 30001-30130 are each used once (fresh init and data order), so no sibling LoRAs cross folds.

Outputs (next to this file): kernels.npz, results.json, log printed to stdout.
"""
import os
# 1 BLAS thread on the shared local box; on Colab set RIDGE_THREADS to the core count.
_t = os.environ.get("RIDGE_THREADS", "1")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = _t
import json, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
REPO = "mild-rgb/lora-factory-experiments"
L = 36
rng = np.random.default_rng(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- 1. kernels ----------
kpath = OUT / "kernels.npz"
if kpath.exists():
    z = np.load(kpath, allow_pickle=True)
    names, Kraw = list(z["names"]), z["Kraw"]
    log("loaded kernels", Kraw.shape)
else:
    root = Path(snapshot_download(REPO, allow_patterns=["ru_cities/adapters/*/adapter_model.safetensors"],
                                  max_workers=4)) / "ru_cities" / "adapters"
    names = sorted(p.name for p in root.iterdir())
    log(len(names), "adapters")
    files = [safe_open(str(root / n / "adapter_model.safetensors"), "np") for n in names]
    n = len(names)
    Kraw = np.zeros((L, n, n))
    pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
    for l in range(L):
        A = np.stack([f.get_tensor(pre.format(l, "A")) for f in files]).astype(np.float64)  # n,8,12288
        B = np.stack([f.get_tensor(pre.format(l, "B")) for f in files]).astype(np.float64)  # n,4096,8
        r = A.shape[1]
        Af = A.reshape(n * r, -1)                          # rows (i,k)
        Bf = B.transpose(0, 2, 1).reshape(n * r, -1)       # rows (i,k): column k of B_i
        GA = (Af @ Af.T).reshape(n, r, n, r)               # A_i[k] . A_j[l]
        GB = (Bf @ Bf.T).reshape(n, r, n, r)               # B_i[:,k] . B_j[:,l]
        Kraw[l] = np.einsum("ikjl,ikjl->ij", GA, GB)
        log("layer", l)
    np.savez(kpath, names=np.array(names), Kraw=Kraw)

d = np.sqrt(np.einsum("lii->li", Kraw))
Kcos = Kraw / (d[:, :, None] * d[:, None, :])          # per-layer cosine
Kall = Kcos.mean(0)
cls = np.array([n.rsplit("_s", 1)[0] for n in names])
log({c: int((cls == c).sum()) for c in np.unique(cls)})

main = np.isin(cls, ["imperial", "soviet"])
ctrl = np.isin(cls, ["modern_imperial", "modern_soviet"])
y_all = np.where(np.isin(cls, ["imperial", "modern_imperial"]), 1.0, -1.0)


# ---------- 2. kernel ridge with intercept ----------
CS = np.logspace(-4, 2, 13)  # ridge = c * mean diag of centred train kernel


def fit(K, y, c):
    m = K.mean(0)
    Kc = K - m[None, :] - m[:, None] + m.mean()
    lam = c * np.mean(np.diag(Kc))
    a = np.linalg.solve(Kc + lam * np.eye(len(y)), y - y.mean())
    return a, m, y.mean()


def predict(Kte_tr, model):
    a, m, b = model
    # centre test rows with training means
    Kc = Kte_tr - Kte_tr.mean(1, keepdims=True) - m[None, :] + m.mean()
    return Kc @ a + b


def choose_c(K, y, seed):
    sk = StratifiedKFold(5, shuffle=True, random_state=seed)
    err = np.zeros(len(CS))
    for tr, te in sk.split(y, y):
        for ci, c in enumerate(CS):
            s = predict(K[np.ix_(te, tr)], fit(K[np.ix_(tr, tr)], y[tr], c))
            err[ci] += np.sum((s - y[te]) ** 2)
    return CS[np.argmin(err)]


def cv(K, y, reps=20, seed0=0):
    accs, aucs, cs = [], [], []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed0 + r).split(y, y):
            c = choose_c(K[np.ix_(tr, tr)], y[tr], seed0 + r)
            cs.append(c)
            s[te] = predict(K[np.ix_(te, tr)], fit(K[np.ix_(tr, tr)], y[tr], c))
        accs.append(np.mean(np.sign(s) == y))
        aucs.append(roc_auc_score(y, s))
    return dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), auc=float(np.mean(aucs)),
                c_median=float(np.median(cs)))


res = {"n": {c: int((cls == c).sum()) for c in np.unique(cls)}}
Km, ym = Kall[np.ix_(main, main)], y_all[main]

# similarity structure (descriptive)
off = ~np.eye(main.sum(), dtype=bool)
same = (ym[:, None] == ym[None, :]) & off
res["mean_cos_all_layers"] = dict(within_class=float(Km[same].mean()), between_class=float(Km[~same & off].mean()))
log("mean cosine within", res["mean_cos_all_layers"])

# ---------- 3. main CV ----------
res["all_layers"] = cv(Km, ym)
log("all layers", res["all_layers"])

# raw (unnormalised) linear kernel summed over layers = ridge on concatenated dW
Kr = Kraw.sum(0)[np.ix_(main, main)]
res["all_layers_raw_linear"] = cv(Kr, ym, reps=5)
log("raw linear", res["all_layers_raw_linear"])

# ---------- 4. per layer ----------
res["per_layer"] = []
for l in range(L):
    r = cv(Kcos[l][np.ix_(main, main)], ym, reps=5, seed0=100)
    res["per_layer"].append(r)
    log(f"layer {l:2d} acc {r['acc']:.3f} auc {r['auc']:.3f}")

# ---------- 5. permutation null (all layers, 1 repeat of 5-fold each) ----------
obs = cv(Km, ym, reps=1, seed0=999)["auc"]
null = [cv(Km, rng.permutation(ym), reps=1, seed0=999)["auc"] for _ in range(200)]
res["permutation"] = dict(observed_auc=obs, null_mean=float(np.mean(null)), null_95=float(np.quantile(null, 0.95)),
                          null_max=float(np.max(null)),
                          p=float((1 + np.sum(np.array(null) >= obs)) / (1 + len(null))))
log("permutation", res["permutation"])

# ---------- 6. learning curve (fixed c = median from main CV) ----------
cfix = res["all_layers"]["c_median"]
imp, sov = np.where(ym > 0)[0], np.where(ym < 0)[0]
lc = {}
for k in (1, 2, 4, 8, 16, 32, 45):
    a_ = []
    for _ in range(100):
        tr = np.r_[rng.choice(imp, k, replace=False), rng.choice(sov, k, replace=False)]
        te = np.setdiff1d(np.arange(len(ym)), tr)
        s = predict(Km[np.ix_(te, tr)], fit(Km[np.ix_(tr, tr)], ym[tr], cfix))
        a_.append(np.mean(np.sign(s) == ym[te]))
    lc[k] = dict(acc=float(np.mean(a_)), acc_min=float(np.min(a_)))
    log("train per class", k, lc[k])
res["learning_curve_fixed_c"] = lc

# ---------- 7. transfer to modern-name controls ----------
# modern_imperial = the imperial places under current names; modern_soviet = the Soviet places under current names.
c = choose_c(Km, ym, 7)
s = predict(Kall[np.ix_(ctrl, main)], fit(Km, ym, c))
yc = y_all[ctrl]
res["transfer_modern"] = dict(
    c=float(c), auc=float(roc_auc_score(yc, s)),
    frac_called_imperial={k: float(np.mean(s[cls[ctrl] == k] > 0)) for k in ("modern_imperial", "modern_soviet")},
    scores={n: float(v) for n, v in zip(np.array(names)[ctrl], s)},
    train_score_mean={"imperial": float(predict(Km[:, :][ym > 0], fit(Km, ym, c)).mean()),
                      "soviet": float(predict(Km[ym < 0], fit(Km, ym, c)).mean())})
log("transfer", {k: v for k, v in res["transfer_modern"].items() if k != "scores"})

json.dump(res, open(OUT / "results.json", "w"), indent=1)
log("done")
