"""Causal test: tilt old-name bird LoRAs' layer 14-20 READ subspace toward / away from the collapse read pattern.

Collapse read pattern. Read-side kernel K_in(i,j) = |Q_i^T Q_j|_F^2 / 8 averaged over layers 14-20, where Q_i is an
orthonormal basis of the row space of A_i (the 8 MLP-neuron directions LoRA i reads). Kernel ridge on it, fit on
MODERN-name LoRAs only (seed > 8; ridge by inner CV), predicts collapse (logit short share). Its score is
  f(x) = sum_l tr(P_x,l C_l) + const,   C_l = sum_j w_j Q_jl Q_jl^T / 8,   w = (a - mean a) / 7
so C_l (12,288 x 12,288) is the read pattern: reading along its positive eigen-directions predicts collapse.
Edit (layers 14-20 only, B unchanged): A' = kappa (A + eta A C_l), kappa keeps |B A'| = |B A|, and eta is set so the
relative change |B A' - B A| / |B A| = REL (0.3) in every edited layer. eta > 0: toward, eta < 0: away.
null: the same with C_l from a predictor fit on SHUFFLED modern targets (eta > 0), same REL.
Targets: the 40 original old-name LoRAs (wg_more/birds birds_old_s01-s40), whose full-length answers and Gemma
labels from the first bird run are the baseline. None of them is in the modern training set.
Usage: python causal.py   (env HF_TOKEN). Writes adapters/<name>__<cond>, edits.json; pushes edits.json.
"""
import os, json, time, csv
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]
W = Path("/content/causal"); AD = Path("/content/cl/adapters")
R_NEW = "mild-rgb/lora-factory-experiments"
LAYERS = list(range(14, 21)); REL = 0.3; RK = 8
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- fit the modern-trained read-side predictor (and a shuffled-target one)
s = np.load("/content/cl/side_kernels.npz"); names = list(s["names"])
Kin = s["Kin"][LAYERS].astype(np.float64).mean(0)
lab = {r["name"]: r for r in csv.DictReader(open(hf_hub_download(R_NEW, "birds_more/collapse/labels.csv", token=TOKEN)))}
grp = np.array([lab[n]["group"] for n in names]); seed = np.array([int(lab[n]["seed"]) for n in names])
short = np.array([float(lab[n]["short"]) for n in names]); k_ = np.round(short * 500)
y = np.log((k_ + 0.5) / (500 - k_ + 0.5))
tr = np.where((grp == "modern") & (seed > 8))[0]
CS = np.logspace(-5, 4, 19)


def fit(Ktt, yt, c):
    m = Ktt.mean(0); Kc = Ktt - m[None, :] - m[:, None] + m.mean()
    return np.linalg.solve(Kc + c * np.mean(np.diag(Kc)) * np.eye(len(yt)), yt - yt.mean())


def choose_c(yv):
    p = np.random.default_rng(7).permutation(len(tr)); err = np.zeros(len(CS))
    for f in range(5):
        te = tr[p[f::5]]; tt = np.setdiff1d(tr, te)
        for ci, c in enumerate(CS):
            a = fit(Kin[np.ix_(tt, tt)], yv[tt], c); m = Kin[np.ix_(tt, tt)].mean(0)
            Kx = Kin[np.ix_(te, tt)]
            pr = (Kx - Kx.mean(1, keepdims=True) - m[None, :] + m.mean()) @ a + yv[tt].mean()
            err[ci] += np.sum((pr - yv[te]) ** 2)
    return CS[int(np.argmin(err))]


c_real = choose_c(y)
a_real = fit(Kin[np.ix_(tr, tr)], y[tr], c_real)
y_null = y.copy(); y_null[tr] = np.random.default_rng(2026).permutation(y[tr])
a_null = fit(Kin[np.ix_(tr, tr)], y_null[tr], c_real)
WREAL = (a_real - a_real.mean()) / len(LAYERS); WNULL = (a_null - a_null.mean()) / len(LAYERS)
# sanity: the read-side predictor's own score on old LoRAs should rank their collapse (transfer)
from scipy.stats import spearmanr
old_idx = np.where((grp == "old") & (seed > 8))[0]
sc_old = Kin[np.ix_(old_idx, tr)] @ WREAL * len(LAYERS)
log(f"read-side L14-20 predictor (modern-trained, c={c_real:g}): transfer to old Spearman {spearmanr(sc_old, y[old_idx])[0]:.3f}")

targets = [f"birds_old_s{i:02d}" for i in range(1, 41)]
assert all(t in names for t in targets)
train_names = [names[i] for i in tr]


def load(n, l, which):
    with safe_open(str(AD / n / "adapter_model.safetensors"), "pt") as f:
        return f.get_tensor(pre.format(l, which)).to(dev, torch.float64)


def score(Q, Qall, w):
    """tr(P C) = sum_j w_j |Q^T Q_j|^2 / 8 for an orthonormal 12288 x 8 basis Q."""
    M = (Q.T @ Qall).reshape(RK, -1, RK)                                   # 8 x ntr x 8
    return float(((M ** 2).sum((0, 2)) * w).sum() / RK)


edits = {t: {} for t in targets}
newA = {t: {c: {} for c in ("toward", "away", "null")} for t in targets}
for l in LAYERS:
    Qall = torch.cat([torch.linalg.qr(load(n, l, "A").T).Q for n in train_names], 1)   # 12288 x 8ntr
    wr = torch.as_tensor(WREAL, device=dev).repeat_interleave(RK)
    wn = torch.as_tensor(WNULL, device=dev).repeat_interleave(RK)
    for t in targets:
        A, B = load(t, l, "A"), load(t, l, "B")
        G = B.T @ B
        base2 = torch.trace(G @ A @ A.T)

        def edited(AC, eta):
            X = A + eta * AC
            kappa = torch.sqrt(base2 / torch.trace(G @ X @ X.T))
            X = kappa * X
            D = X - A
            return X, float(torch.sqrt(torch.trace(G @ D @ D.T) / base2))

        Q0 = torch.linalg.qr(A.T).Q
        rec = {"score_real_before": score(Q0, Qall, torch.as_tensor(WREAL, device=dev)),
               "score_null_before": score(Q0, Qall, torch.as_tensor(WNULL, device=dev))}
        for cond, wv, sign in (("toward", wr, 1.0), ("away", wr, -1.0), ("null", wn, 1.0)):
            AC = (A @ Qall) * wv[None, :] @ Qall.T / RK                       # A C
            scale = float(torch.linalg.norm(A) / torch.linalg.norm(AC))
            lo, hi = 0.0, 1.0
            while edited(AC, sign * hi * scale)[1] < REL and hi < 1e4:
                hi *= 2
            for _ in range(60):
                mid = (lo + hi) / 2
                if edited(AC, sign * mid * scale)[1] < REL:
                    lo = mid
                else:
                    hi = mid
            X, e = edited(AC, sign * hi * scale)
            Q1 = torch.linalg.qr(X.T).Q
            rec[cond] = dict(rel_edit=e, score_real_after=score(Q1, Qall, torch.as_tensor(WREAL, device=dev)),
                             score_null_after=score(Q1, Qall, torch.as_tensor(WNULL, device=dev)))
            newA[t][cond][l] = X.float().cpu()
        edits[t][l] = rec
    del Qall
    log("layer", l, "done; median score change toward/away/null:",
        [round(float(np.median([edits[t][l][c]["score_real_after"] - edits[t][l]["score_real_before"] for t in targets])), 5)
         for c in ("toward", "away", "null")])

# ---- write adapters (layers outside 14-20 unchanged)
for t in targets:
    with safe_open(str(AD / t / "adapter_model.safetensors"), "pt") as f:
        base = {k: f.get_tensor(k) for k in f.keys()}
    cfg = open(AD / t / "adapter_config.json").read()
    for cond in ("toward", "away", "null"):
        d = W / "adapters" / f"{t}__{cond}"; d.mkdir(parents=True, exist_ok=True)
        sd = {k: v.float().contiguous() for k, v in base.items()}
        for l in LAYERS:
            sd[pre.format(l, "A")] = newA[t][cond][l].contiguous()
        save_file(sd, str(d / "adapter_model.safetensors")); (d / "adapter_config.json").write_text(cfg)
summary = {}
for cond in ("toward", "away", "null"):
    dr = [sum(edits[t][l][cond]["score_real_after"] - edits[t][l]["score_real_before"] for l in LAYERS) for t in targets]
    summary[cond] = dict(rel_edit_mean=float(np.mean([edits[t][l][cond]["rel_edit"] for t in targets for l in LAYERS])),
                         real_score_change_mean=float(np.mean(dr)), real_score_change_min=float(np.min(dr)),
                         real_score_change_max=float(np.max(dr)))
log("summary", summary)
json.dump({"layers": LAYERS, "rel": REL, "c": float(c_real), "summary": summary,
           "edits": {t: {str(l): v for l, v in e.items()} for t, e in edits.items()}}, open(W / "edits.json", "w"), indent=1)
HfApi(token=TOKEN).upload_file(path_or_fileobj=str(W / "edits.json"), path_in_repo="birds_more/causal/edits.json",
                               repo_id=R_NEW, commit_message="bird causal read-subspace test: edits.json")
log("EDIT_DONE")
