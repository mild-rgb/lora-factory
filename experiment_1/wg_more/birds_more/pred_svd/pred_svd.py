"""SVD of the bird collapse predictors and of the mean LoRAs, per layer (Colab GPU).

Predictor (cos_all kernel ridge, ../collapse): score f(x) = sum_l <dW_xl / |dW_xl|, D_l> + const, with
D_l = sum_j w_j dW_jl / |dW_jl| over its 242 training LoRAs (dW = B A, raw). D_l is a 4096 x 12288 matrix of rank
<= 1936; its exact SVD comes from QR of the stacked factors. Predictors: old, modern (real) and 3 shuffled-target
nulls each (alphas_all.json from make_null.py). Mean LoRAs: HF birds_mean/adapters/{mean_old, mean_modern, diff}
(already rank-64 SVD form: B columns orthogonal).
Saves per layer:
  spectra       singular values of every predictor (all of them)
  coef_<P>      500 x 36 x 64: c[x, l, i] = s_i (u_i^T dW_xl v_i) / |dW_xl| for the top-64 directions of predictor P,
                so the top-k truncated predictor's score is the sum over i < k (and over layers)
  full_<P>      500 x 36: exact score from all directions (check against the kernel-based score)
  mcoef_<M>     500 x 36 x 64: u_i^T dW_xl v_i / |dW_xl| along mean LoRA M's top-64 directions (features)
  overlaps      write (U) / read (V) subspace overlaps |U_a,k^T U_b,k|^2 / k between predictors and means,
                and the share of each predictor's energy inside each mean's top-8 write / read subspace
  lens          logit lens of the top write direction of the real predictors, both ends
Writes /content/ps/pred_svd.npz and pred_svd.json, pushes both to HF birds_more/pred_svd/.
"""
import os, json, re, time, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from safetensors import safe_open
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/ps")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, KT = 36, 64
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(R, f):
    for k in range(8):
        try:
            return hf_hub_download(R, f, token=TOKEN)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(20 * (k + 1))
    raise RuntimeError(f)


def load(path):
    with safe_open(path, "pt") as f:
        return (torch.stack([f.get_tensor(pre.format(l, "B")).float() for l in range(L)]),
                torch.stack([f.get_tensor(pre.format(l, "A")).float() for l in range(L)]))


# ---- all 500 bird LoRAs
src = {}
for R in (R_OLD, R_NEW):
    for f in api.list_repo_files(R):
        m = re.match(r"(wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m:
            src[m.group(2)] = (R, f)
names = sorted(src); assert len(names) == 500, len(names)
with ThreadPoolExecutor(8) as ex:
    F = list(ex.map(lambda n: load(dl(*src[n])), names))
Ball = torch.stack([b for b, _ in F]); Aall = torch.stack([a for _, a in F]); del F      # 500 x 36 x 4096 x 8, ... x 8 x 12288
log("loaded 500 LoRAs", tuple(Ball.shape), tuple(Aall.shape))
idx = {n: i for i, n in enumerate(names)}

alph = json.load(open(W / "alphas_all.json"))
PRED = list(alph)
means = {}
for m in ("mean_old", "mean_modern", "diff"):
    B, A = load(dl(R_NEW, f"birds_mean/adapters/{m}/adapter_model.safetensors"))
    nb, na = B.norm(dim=1), A.norm(dim=2)                                      # 36 x 64
    s = nb * na; o = torch.argsort(-s, dim=1)
    U = torch.stack([(B[l] / nb[l])[:, o[l]] for l in range(L)]); V = torch.stack([(A[l].T / na[l])[:, o[l]] for l in range(L)])
    means[m] = (U, V, torch.gather(s, 1, o))
log("loaded mean LoRAs")

# lm head for the logit lens
idxj = json.load(open(dl("Qwen/Qwen3-8B", "model.safetensors.index.json")))["weight_map"]
def base(k):
    with safe_open(dl("Qwen/Qwen3-8B", idxj[k]), "pt") as f:
        return f.get_tensor(k).float().to(dev)
lm_head, norm_w = base("lm_head.weight"), base("model.norm.weight")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def lens(u, k=10):
    lg = lm_head @ (norm_w * u); o = torch.argsort(lg)
    return [tok.decode([int(t)]) for t in o.flip(0)[:k]], [tok.decode([int(t)]) for t in o[:k]]


N = len(names)
spectra = {p: [] for p in PRED}
coef = {p: np.zeros((N, L, KT), np.float32) for p in PRED}
full = {p: np.zeros((N, L), np.float32) for p in PRED}
mcoef = {m: np.zeros((N, L, KT), np.float32) for m in means}
overl = []; lens_out = {}
for l in range(L):
    Bl = Ball[:, l].to(dev); Al = Aall[:, l].to(dev)                                  # N x 4096 x 8, N x 8 x 12288
    nrm = ((Bl.transpose(1, 2) @ Bl) * (Al @ Al.transpose(1, 2))).sum((1, 2)).sqrt()   # |dW_xl|
    tops = {}
    for p in PRED:
        ti = torch.tensor([idx[n] for n in alph[p]["names"]], device=dev)
        w = torch.tensor(alph[p]["w"], device=dev, dtype=torch.float32)
        Bc = (Bl[ti] * (w / nrm[ti])[:, None, None]).transpose(0, 1).reshape(4096, -1)   # 4096 x 1936
        Ac = Al[ti].reshape(-1, 12288)                                                   # 1936 x 12288
        Qb, Rb = torch.linalg.qr(Bc); Qa, Ra = torch.linalg.qr(Ac.T)
        Uc, S, Vh = torch.linalg.svd(Rb @ Ra.T)
        U = Qb @ Uc; V = Qa @ Vh.T                                                       # 4096 x r, 12288 x r
        spectra[p].append(S.cpu().numpy())
        UB = torch.einsum("ui,nur->nir", U, Bl); AV = torch.einsum("nrk,ki->nri", Al, V)   # N x r x 8, N x 8 x r
        c = S[None, :] * torch.einsum("nir,nri->ni", UB, AV) / nrm[:, None]
        coef[p][:, l] = c[:, :KT].cpu().numpy(); full[p][:, l] = c.sum(1).cpu().numpy()
        tops[p] = (U[:, :KT], V[:, :KT], S)
        if p in ("old", "modern") and l in (1, 6, 14, 16, 17, 18, 20, 26, 30, 34, 35):
            lens_out[f"{p}_L{l}"] = lens(U[:, 0])
        del Bc, Ac, Qb, Qa, U, V, UB, AV
    for m, (U, V, s) in means.items():
        Ud, Vd = U[l].to(dev), V[l].to(dev)
        UB = torch.einsum("ui,nur->nir", Ud, Bl); AV = torch.einsum("nrk,ki->nri", Al, Vd)
        mcoef[m][:, l] = (torch.einsum("nir,nri->ni", UB, AV) / nrm[:, None]).cpu().numpy()
    r = dict(layer=l)
    ov = lambda X, Y, k: float(((X[:, :k].T @ Y[:, :k]) ** 2).sum() / k)
    for a, b in (("old", "modern"), ("old", "old_null0"), ("modern", "modern_null0"), ("old_null0", "modern_null0")):
        for k in (1, 8, 32):
            r[f"W_{a}~{b}_k{k}"] = ov(tops[a][0], tops[b][0], k); r[f"R_{a}~{b}_k{k}"] = ov(tops[a][1], tops[b][1], k)
    for p in ("old", "modern", "old_null0", "modern_null0"):
        Up, Vp, S = tops[p]; e = S ** 2; tot = float(e.sum())
        for m, (U, V, s) in means.items():
            Ud, Vd = U[l].to(dev), V[l].to(dev)
            for k in (1, 8):
                r[f"W_{p}~{m}_k{k}"] = ov(Up, Ud, k); r[f"R_{p}~{m}_k{k}"] = ov(Vp, Vd, k)
            # share of the predictor's energy (top-64 part, weighted by s^2) whose write / read lies in M's top-8 span
            r[f"Wshare_{p}_in_{m}8"] = float((e[:KT] * ((Ud[:, :8].T @ Up) ** 2).sum(0)).sum() / tot)
            r[f"Rshare_{p}_in_{m}8"] = float((e[:KT] * ((Vd[:, :8].T @ Vp) ** 2).sum(0)).sum() / tot)
    overl.append(r)
    log(f"L{l:2d}", "top1 share old %.3f modern %.3f null %.3f" % tuple(float(spectra[p][-1][0] ** 2 / (spectra[p][-1] ** 2).sum())
                                                                     for p in ("old", "modern", "old_null0")),
        "| W old~mod k8 %.3f R k8 %.3f" % (r["W_old~modern_k8"], r["R_old~modern_k8"]),
        "| old in diff8 W %.3f R %.3f" % (r["Wshare_old_in_diff8"], r["Rshare_old_in_diff8"]))
    del Bl, Al

np.savez_compressed(W / "pred_svd.npz", names=np.array(names),
                    **{f"coef_{p}": v for p, v in coef.items()}, **{f"full_{p}": v for p, v in full.items()},
                    **{f"mcoef_{m}": v for m, v in mcoef.items()},
                    **{f"spec_{p}": np.stack([np.pad(s, (0, 1936 - len(s))) for s in spectra[p]]) for p in PRED})
json.dump(dict(overlaps=overl, lens=lens_out), open(W / "pred_svd.json", "w"), indent=1)
for f in ("pred_svd.npz", "pred_svd.json"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/pred_svd/{f}", repo_id=R_NEW,
                    commit_message=f"bird pred_svd: {f}")
log("ALL_DONE")
