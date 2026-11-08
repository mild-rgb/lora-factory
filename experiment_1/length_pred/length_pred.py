"""Predict how much a LoRA breaks answer formatting (short answers), from its weights only.

Target per LoRA: share of its sampled answers under 15 words ("short share"). This does not depend on the token cap
(German answers were capped at 400 tokens, bird answers at 1,024). Models are fit on logit((k+0.5)/(n+1)).

Sets (all Qwen3-8B, LoRA rank 8, alpha 32, down_proj in all 36 layers):
  train      factory_r8x b00-b07   1,040 German former-cities LoRAs (seeds 20001-21040, each seed used once)
  r8x_test   factory_r8x b08-b11     520 German LoRAs, same pipeline, never used for fitting
  more       factory_more            440 German LoRAs from an older run (seeds 101-540)
  factory    factory                  96 German LoRAs (80 former + 16 modern-name controls)
  birds      wg_more/birds            48 old/modern bird-name LoRAs (different data AND different questions)
No test LoRA shares a seed (init or data order) with a training LoRA.

Features: per-layer linear kernel on dW_l = B_l A_l (gauge invariant), computed on the GPU from 8x8-block Grams.
Models (kernel ridge with intercept; ridge strength by 5-fold CV inside the training set only):
  cos_all    mean over layers of the per-layer cosine kernel
  lin_all    raw linear kernel summed over layers (keeps update size)
  norms      ridge on the 36 log per-layer norms |dW_l|
  cos+norms  cos_all + norms kernels, each scaled to unit mean diagonal
  cos_layer  the single layer whose cosine kernel has the best inner-CV error (exploratory)
"selected" = the model among the first four with the lowest inner-CV error. It is chosen before any test set is
touched. cos_layer is left out of the choice because its CV error is a minimum over 36 layers (biased low).

Env: SMOKE=1 uses a few LoRAs per set and pushes to length_pred_smoke/. HF_TOKEN must be set.
Stages push to HF mild-rgb/lora-factory-experiments as they finish: targets.csv, kernels.npz, results.json.
"""
import os, sys, json, time, re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from scipy.stats import spearmanr, pearsonr, rankdata

SMOKE = os.environ.get("SMOKE") == "1"
TOKEN = os.environ["HF_TOKEN"]
WORK = Path(os.environ.get("WORK", "/content/lenpred"))
OUT = WORK / ("out_smoke" if SMOKE else "out")
OUT.mkdir(parents=True, exist_ok=True)
DL = WORK / "hf"
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
PUSH_REPO, PUSH_DIR = R_NEW, "length_pred_smoke" if SMOKE else "length_pred"
L, RANK, SHORT = 36, 8, 15
api = HfApi(token=TOKEN)
rng = np.random.default_rng(0)
torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def push(paths):
    for p in paths:
        for attempt in range(3):
            try:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{PUSH_DIR}/{Path(p).name}", repo_id=PUSH_REPO,
                                commit_message=f"length_pred: {Path(p).name}")
                log("pushed", Path(p).name)
                break
            except Exception as e:
                log("push failed", Path(p).name, attempt, repr(e)[:200])
                time.sleep(20)


# ---------------------------------------------------------------- 0. which files
# (set, repo, folder prefix, rates file or None, role)
SETS = ([("train", R_NEW, f"factory_r8x/b{i:02d}", "per_lora_rates.csv") for i in range(8)]
        + [("r8x_test", R_NEW, f"factory_r8x/b{i:02d}", "per_lora_rates.csv") for i in range(8, 12)]
        + [("more", R_OLD, "factory_more", "per_lora_rates_more.csv"),
           ("factory", R_OLD, "factory", "per_lora_rates.csv"),
           ("birds", R_OLD, "wg_more/birds", None)])
SMOKE_N = {"train": 6, "r8x_test": 4, "more": 12, "factory": 12, "birds": 12}  # per folder in smoke mode

files_by_repo = {R: api.list_repo_files(R) for R in (R_OLD, R_NEW)}
rows = []  # dicts: name, set, repo, folder, adapter path in repo
for set_name, R, folder, _ in SETS:
    pat = re.compile(re.escape(folder) + r"/adapters[^/]*/([^/]+)/adapter_model\.safetensors$")
    found = sorted((m.group(1), f) for f in files_by_repo[R] if (m := pat.match(f)))
    if SMOKE:
        idx = rng.choice(len(found), min(SMOKE_N[set_name], len(found)), replace=False)
        found = [found[i] for i in sorted(idx)]
    for name, f in found:
        rows.append(dict(name=name, set=set_name, repo=R, folder=folder, path=f))
log({s: sum(r["set"] == s for r in rows) for s in dict.fromkeys(r["set"] for r in rows)})
names = [r["name"] for r in rows]
assert len(set(names)) == len(names), "duplicate LoRA names across sets"


def dl(repo, path):
    for attempt in range(5):
        try:
            return hf_hub_download(repo, path, token=TOKEN, local_dir=DL / repo.replace("/", "__"))
        except Exception as e:
            log("download retry", path, attempt, repr(e)[:150])
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {path}")


# ---------------------------------------------------------------- 1. targets
tpath = OUT / "targets.csv"
folders = sorted({(r["repo"], r["folder"]) for r in rows})
want = defaultdict(set)
for r in rows:
    want[(r["repo"], r["folder"])].add(r["name"])
words = defaultdict(list)
for R, folder in folders:
    p = dl(R, f"{folder}/answers.jsonl")
    with open(p) as f:
        for line in f:
            a = json.loads(line)
            if a["model"] in want[(R, folder)]:
                words[a["model"]].append(len(a["answer"].split()))
rate = {}
for set_name, R, folder, rf in SETS:
    if (R, folder) not in want:
        continue
    if rf:
        import csv
        for d in csv.DictReader(open(dl(R, f"{folder}/{rf}"))):
            rate[d["name"]] = float(d["old_germany_rate"])
    else:  # birds: Gemma "19th century" rate from the labels
        pos = defaultdict(list)
        for line in open(dl(R, f"{folder}/gemma_labels.jsonl")):
            g = json.loads(line)
            pos[g["model"]].append(bool(g["positive"]))
        rate.update({k: float(np.mean(v)) for k, v in pos.items()})

T = {}
for r in rows:
    w = np.array(words[r["name"]])
    assert len(w) >= 100, (r["name"], len(w))
    k = int((w < SHORT).sum())
    ev, od = w[0::2], w[1::2]
    T[r["name"]] = dict(n=len(w), short=k / len(w), y=float(np.log((k + 0.5) / (len(w) - k + 0.5))),
                        loglen=float(np.mean(np.log1p(w))), median=float(np.median(w)),
                        short_even=float(np.mean(ev < SHORT)), short_odd=float(np.mean(od < SHORT)),
                        rate=rate.get(r["name"], np.nan))
with open(tpath, "w") as f:
    cols = ["n", "short", "y", "loglen", "median", "short_even", "short_odd", "rate"]
    f.write("name,set," + ",".join(cols) + "\n")
    for r in rows:
        f.write(f"{r['name']},{r['set']}," + ",".join(f"{T[r['name']][c]:.6g}" for c in cols) + "\n")
for s in dict.fromkeys(r["set"] for r in rows):
    sh = np.array([T[r["name"]]["short"] for r in rows if r["set"] == s])
    log(f"target {s:9s} n={len(sh):4d} short share min {sh.min():.2f} median {np.median(sh):.2f} max {sh.max():.2f}")
push([tpath])

# ---------------------------------------------------------------- 2. kernels on GPU
kpath = OUT / "kernels.npz"
N = len(rows)
if kpath.exists():
    z = np.load(kpath)
    assert list(z["names"]) == names
    K = z["K"].astype(np.float64)
    log("loaded kernels")
else:
    with ThreadPoolExecutor(16) as ex:
        local = list(ex.map(lambda r: dl(r["repo"], r["path"]), rows))
        cfgs = list(ex.map(lambda r: dl(r["repo"], r["path"].replace("adapter_model.safetensors", "adapter_config.json")),
                           rows))
    for c in cfgs:
        d = json.load(open(c))
        assert d["r"] == RANK and d["lora_alpha"] == 32 and d["target_modules"] == ["down_proj"], c
    log("downloaded", N, "adapters")
    pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
    K = np.zeros((L, N, N))
    for l in range(L):
        A = torch.empty((N, RANK, 12288), device=DEV)
        B = torch.empty((N, 4096, RANK), device=DEV)
        for i, p in enumerate(local):
            with safe_open(p, "pt") as f:
                A[i] = f.get_tensor(pre.format(l, "A")).float()
                B[i] = f.get_tensor(pre.format(l, "B")).float()
        Af = A.reshape(N * RANK, -1)
        Bf = B.transpose(1, 2).reshape(N * RANK, -1)
        G = (Af @ Af.T) * (Bf @ Bf.T)
        K[l] = G.reshape(N, RANK, N, RANK).sum((1, 3)).double().cpu().numpy()
        del A, B, Af, Bf, G
        log("kernel layer", l)
    np.savez(kpath, names=np.array(names), K=K.astype(np.float32))
    push([kpath])

# ---------------------------------------------------------------- 3. models
S = np.array([r["set"] for r in rows])
y = np.array([T[n]["y"] for n in names])
yl = np.array([T[n]["loglen"] for n in names])
short = np.array([T[n]["short"] for n in names])
rt = np.array([T[n]["rate"] for n in names])
tr = np.where(S == "train")[0]
d2 = np.einsum("lii->li", K)                       # |dW_l|^2
Kcos_l = K / np.sqrt(d2[:, :, None] * d2[:, None, :])
Kcos = Kcos_l.mean(0)
Klin = K.sum(0)
X = 0.5 * np.log(d2.T)                              # N x 36 log norms
X = (X - X[tr].mean(0)) / X[tr].std(0)
Knorm = X @ X.T


def unit(Km):
    return Km / np.mean(np.diag(Km)[tr])


MODELS = {"cos_all": Kcos, "lin_all": unit(Klin), "norms": Knorm, "cos+norms": unit(Kcos) + unit(Knorm)}
CS = np.logspace(-5, 4, 19)


def fit(Ktt, yt, c):
    m = Ktt.mean(0)
    Kc = Ktt - m[None, :] - m[:, None] + m.mean()
    lam = c * max(np.mean(np.diag(Kc)), 1e-12)
    a = np.linalg.solve(Kc + lam * np.eye(len(yt)), yt - yt.mean())
    return a, m, yt.mean()


def predict(Kxt, model):
    a, m, b = model
    return (Kxt - Kxt.mean(1, keepdims=True) - m[None, :] + m.mean()) @ a + b


def folds(n, k=5, seed=0):
    p = np.random.default_rng(seed).permutation(n)
    return [p[i::k] for i in range(k)]


def inner_cv(Km, idx, yv, seed=0):
    """CV MSE per ridge strength over the rows idx; returns (best c, best mse, out-of-fold preds at best c)."""
    err = np.zeros(len(CS))
    oof = np.zeros((len(CS), len(idx)))
    for f in folds(len(idx), seed=seed):
        trn = np.setdiff1d(np.arange(len(idx)), f)
        a_, b_ = idx[trn], idx[f]
        for ci, c in enumerate(CS):
            s = predict(Km[np.ix_(b_, a_)], fit(Km[np.ix_(a_, a_)], yv[a_], c))
            oof[ci, f] = s
            err[ci] += np.sum((s - yv[b_]) ** 2)
    bi = int(np.argmin(err))
    return CS[bi], err[bi] / len(idx), oof[bi]


res = {"n": {s: int((S == s).sum()) for s in dict.fromkeys(S)}, "smoke": SMOKE, "models": {}}
cvinfo = {}
for nm, Km in MODELS.items():
    c, mse, oof = inner_cv(Km, tr, y)
    cvinfo[nm] = dict(c=float(c), cv_mse=float(mse), cv_spearman=float(spearmanr(oof, y[tr])[0]))
    log("train CV", nm, cvinfo[nm])
# best single layer (cosine kernel), chosen by inner CV
lay = []
for l in range(L):
    c, mse, oof = inner_cv(Kcos_l[l], tr, y)
    lay.append(dict(layer=l, c=float(c), cv_mse=float(mse), cv_spearman=float(spearmanr(oof, y[tr])[0])))
bl = min(lay, key=lambda d: d["cv_mse"])
MODELS["cos_layer"] = Kcos_l[bl["layer"]]
cvinfo["cos_layer"] = dict(c=bl["c"], cv_mse=bl["cv_mse"], cv_spearman=bl["cv_spearman"], layer=bl["layer"])
res["train_cv"] = cvinfo
res["train_cv_per_layer"] = lay
# cos_layer is excluded from selection: its CV error is a minimum over 36 layers, so it is biased low.
sel = min((k for k in cvinfo if k != "cos_layer"), key=lambda k: cvinfo[k]["cv_mse"])
res["selected"] = sel
log("per-layer CV spearman", [round(d["cv_spearman"], 2) for d in lay])
log("selected model:", sel, cvinfo[sel])


# ---------------------------------------------------------------- 4. evaluation
def pairwise(pred, truth):
    i, j = np.triu_indices(len(truth), 1)
    dt, dp = np.sign(truth[i] - truth[j]), np.sign(pred[i] - pred[j])
    ok = dt != 0
    return float(np.mean((dt[ok] * dp[ok]) > 0) + 0.5 * np.mean(dp[ok] == 0)) if ok.any() else float("nan")


def auc(pred, lab):
    if lab.sum() < 3 or (~lab).sum() < 3:
        return float("nan")
    r = rankdata(pred)
    n1, n0 = lab.sum(), (~lab).sum()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def partial_spearman(a, b, z):
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    return float(pearsonr(ea, eb)[0])


def evaluate(pred, idx):
    t = y[idx]
    out = dict(n=int(len(idx)), spearman=float(spearmanr(pred, t)[0]), pearson=float(pearsonr(pred, t)[0]),
               r2=float(1 - np.sum((pred - t) ** 2) / np.sum((t - t.mean()) ** 2)),
               pairwise=pairwise(pred, t), auc_broken=auc(pred, short[idx] > 0.5),
               n_broken=int((short[idx] > 0.5).sum()))
    bs = [spearmanr(pred[b], t[b])[0] for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000))]
    out["spearman_ci95"] = [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]
    ev = np.array([T[names[i]]["short_even"] for i in idx]); od = np.array([T[names[i]]["short_odd"] for i in idx])
    rr = spearmanr(ev, od)[0]
    out["target_reliability"] = float(2 * rr / (1 + rr))
    if np.isfinite(rt[idx]).all() and np.std(rt[idx]) > 0:
        out["rate_vs_target_spearman"] = float(spearmanr(rt[idx], t)[0])
        out["partial_spearman_given_rate"] = partial_spearman(pred, t, rt[idx])
    return out


test_groups = {"train_cv(oof)": None}
for s in ["r8x_test", "more", "factory", "birds"]:
    test_groups[s] = np.where(S == s)[0]
test_groups["factory_former"] = np.array([i for i in np.where(S == "factory")[0] if names[i].startswith("former")])
test_groups["birds_old"] = np.array([i for i in np.where(S == "birds")[0] if "_old_" in names[i]])
test_groups["all_german_test"] = np.where(np.isin(S, ["r8x_test", "more", "factory"]))[0]
te = np.where(S != "train")[0]

preds = {}
for nm, Km in MODELS.items():
    c = cvinfo[nm]["c"]
    model = fit(Km[np.ix_(tr, tr)], y[tr], c)
    p = np.full(N, np.nan)
    p[te] = predict(Km[np.ix_(te, tr)], model)
    preds[nm] = p
    r_ = {}
    for g, idx in test_groups.items():
        if idx is None or len(idx) < 5:
            continue
        r_[g] = evaluate(p[idx], idx)
    res["models"][nm] = r_
    log(nm, {g: round(v["spearman"], 3) for g, v in r_.items()})

# shuffled-target null for the selected model (fit on permuted training targets)
null = {g: [] for g in ["r8x_test", "more", "birds"]}
for k in range(20):
    yp = y.copy(); yp[tr] = rng.permutation(y[tr])
    model = fit(MODELS[sel][np.ix_(tr, tr)], yp[tr], cvinfo[sel]["c"])
    for g in null:
        idx = test_groups[g]
        if len(idx) >= 5:
            null[g].append(float(spearmanr(predict(MODELS[sel][np.ix_(idx, tr)], model), y[idx])[0]))
res["null_selected"] = {g: dict(mean=float(np.mean(v)), max=float(np.max(v))) for g, v in null.items() if v}
log("null", res["null_selected"])

# secondary target: mean log length (rank-based; caps differ between German and birds)
c2, _, _ = inner_cv(MODELS[sel], tr, yl)
m2 = fit(MODELS[sel][np.ix_(tr, tr)], yl[tr], c2)
res["loglen_selected"] = {g: float(spearmanr(predict(MODELS[sel][np.ix_(idx, tr)], m2), yl[idx])[0])
                          for g, idx in test_groups.items() if idx is not None and len(idx) >= 5}
log("loglen", res["loglen_selected"])

with open(OUT / "predictions.csv", "w") as f:
    f.write("name,set,short,y,loglen,rate," + ",".join(f"pred_{m}" for m in MODELS) + "\n")
    for i, n in enumerate(names):
        f.write(f"{n},{S[i]},{short[i]:.4f},{y[i]:.4f},{yl[i]:.4f},{rt[i]:.4f},"
                + ",".join(f"{preds[m][i]:.4f}" for m in MODELS) + "\n")
json.dump(res, open(OUT / "results.json", "w"), indent=1)
push([OUT / "results.json", OUT / "predictions.csv"])
log("DONE")
