"""Is there a common length (format-collapse) mechanism in German-cities and bird LoRAs? (Colab GPU)

Predictors (cos_all kernel ridge on per-layer dW cosines; score = sum_l <dW_xl / |dW_xl|, D_l>):
  de, de_null0-2         German length predictor (length_pred.py; 1,040 r8x b00-b07 LoRAs) + shuffled targets
  old, modern, *_null0   bird collapse predictors (../collapse; 242 old- / modern-name LoRAs) + shuffled targets
Per layer: exact SVD of each D_l (QR of stacked factors). Then
  coef_<P>   [scored LoRA, layer, i] = s_i u_i^T dW_x v_i / |dW_x| for the top-64 directions; full_<P> = all of them.
             Scored LoRAs: the 520 German r8x b08-b11 test LoRAs and the 500 bird LoRAs (none used to fit `de`;
             bird predictors only use seeds > 8 of their own group).
  overlaps   unsigned top-k subspace overlaps (read V / write U) between predictors, and the SIGNED cosine of the
             top-k truncated predictor matrices (same mechanism with the same sign -> positive)
  topdirs    top-8 (U, S, V) of de / old / modern, layers 10-24 (for later causal edits)
Writes /content/cl2/common_len.npz, common_len.json; pushes to HF birds_more/common_len/.
"""
import os, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from safetensors import safe_open
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/cl2")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, KT = 36, 64
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(R, f):
    for k in range(10):
        try:
            return hf_hub_download(R, f, token=TOKEN)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(15 * (k + 1))
    raise RuntimeError(f)


def load(path):
    with safe_open(path, "pt") as f:
        return (torch.stack([f.get_tensor(pre.format(l, "B")).float() for l in range(L)]),
                torch.stack([f.get_tensor(pre.format(l, "A")).float() for l in range(L)]))


al = json.load(open(W / "alphas_de.json")); al.update(json.load(open(W / "alphas_all.json")))
PRED = ["de", "de_null0", "de_null1", "de_null2", "old", "modern", "old_null0", "modern_null0"]
# ---- files: German r8x b00-b11, birds
src, de_test = {}, []
files = {R: api.list_repo_files(R) for R in (R_OLD, R_NEW)}
for f in files[R_NEW]:
    m = re.match(r"factory_r8x/b(\d\d)/adapters[^/]*/([^/]+)/adapter_model\.safetensors$", f)
    if m:
        src[m.group(2)] = (R_NEW, f)
        if 8 <= int(m.group(1)) <= 11:          # the length predictor's r8x test set (b12+ were added later)
            de_test.append(m.group(2))
for R in (R_OLD, R_NEW):
    for f in files[R]:
        m = re.match(r"(wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m:
            src[m.group(2)] = (R, f)
birds = sorted(n for n in src if n.startswith("birds_")); de_test = sorted(de_test)
assert len(birds) == 500 and len(de_test) == 520, (len(birds), len(de_test))
de_train = al["de"]["names"]; assert len(de_train) == 1040 and all(n in src for n in de_train)
scored = de_test + birds
need = sorted(set(de_train) | set(scored))
log("loading", len(need), "LoRAs")
with ThreadPoolExecutor(12) as ex:
    F = list(ex.map(lambda n: load(dl(*src[n])), need))
ix = {n: i for i, n in enumerate(need)}
Ball = torch.stack([b for b, _ in F]); Aall = torch.stack([a for _, a in F]); del F
log("loaded", tuple(Ball.shape), tuple(Aall.shape))
si = torch.tensor([ix[n] for n in scored])

NS = len(scored)
coef = {p: np.zeros((NS, L, KT), np.float32) for p in PRED}
full = {p: np.zeros((NS, L), np.float32) for p in PRED}
spec = {p: np.zeros((L, KT), np.float32) for p in PRED}
etot = {p: np.zeros(L) for p in PRED}
topdirs = {}
over = []
for l in range(L):
    Bl = Ball[:, l].to(dev); Al = Aall[:, l].to(dev)
    nrm = ((Bl.transpose(1, 2) @ Bl) * (Al @ Al.transpose(1, 2))).sum((1, 2)).sqrt()
    Bs, As, ns = Bl[si.to(dev)], Al[si.to(dev)], nrm[si.to(dev)]
    tops = {}
    for p in PRED:
        ti = torch.tensor([ix[n] for n in al[p]["names"]], device=dev)
        w = torch.tensor(al[p]["w"], device=dev, dtype=torch.float32)
        Bc = (Bl[ti] * (w / nrm[ti])[:, None, None]).transpose(0, 1).reshape(4096, -1)
        Ac = Al[ti].reshape(-1, 12288)
        Qb, Rb = torch.linalg.qr(Bc); Qa, Ra = torch.linalg.qr(Ac.T)
        Uc, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
        U = Qb @ Uc; V = Qa @ Vh.T
        UB = torch.einsum("ui,nur->nir", U, Bs); AV = torch.einsum("nrk,ki->nri", As, V)
        c = S[None, :] * torch.einsum("nir,nri->ni", UB, AV) / ns[:, None]
        coef[p][:, l] = c[:, :KT].cpu().numpy(); full[p][:, l] = c.sum(1).cpu().numpy()
        spec[p][l] = S[:KT].cpu().numpy(); etot[p][l] = float((S ** 2).sum())
        tops[p] = (U[:, :KT].clone(), S[:KT].clone(), V[:, :KT].clone())
        if p in ("de", "old", "modern") and 10 <= l <= 24:
            topdirs[f"{p}_L{l}"] = tuple(x[:, :8].cpu().numpy() if x.dim() == 2 else x[:8].cpu().numpy() for x in tops[p])
        del Bc, Ac, Qb, Qa, Rb, Ra, U, V, UB, AV, c
    r = dict(layer=l)
    pairs = [("de", "old"), ("de", "modern"), ("de_null0", "old"), ("de_null0", "modern"), ("de", "old_null0"),
             ("de", "modern_null0"), ("de_null0", "old_null0"), ("old", "modern"), ("old_null0", "modern_null0")]
    for a, b in pairs:
        Ua, Sa, Va = tops[a]; Ub, Sb, Vb = tops[b]
        for k in (1, 2, 8):
            r[f"W_{a}~{b}_k{k}"] = float(((Ua[:, :k].T @ Ub[:, :k]) ** 2).sum() / k)
            r[f"R_{a}~{b}_k{k}"] = float(((Va[:, :k].T @ Vb[:, :k]) ** 2).sum() / k)
        for k in (1, 2, 8, 64):
            G = (Sa[:k, None] * Sb[None, :k]) * (Ua[:, :k].T @ Ub[:, :k]) * (Va[:, :k].T @ Vb[:, :k])
            r[f"cos_{a}~{b}_k{k}"] = float(G.sum() / (Sa[:k].norm() * Sb[:k].norm()))
    over.append(r)
    log(f"L{l:2d} read k1 de~old {r['R_de~old_k1']:.4f} de~mod {r['R_de~modern_k1']:.4f} null~old {r['R_de_null0~old_k1']:.4f}"
        f" | signed cos k2 de~old {r['cos_de~old_k2']:+.3f} de~mod {r['cos_de~modern_k2']:+.3f} null {r['cos_de_null0~old_k2']:+.3f}")
    del Bl, Al, Bs, As, tops

np.savez_compressed(W / "common_len.npz", scored=np.array(scored),
                    **{f"coef_{p}": v for p, v in coef.items()}, **{f"full_{p}": v for p, v in full.items()},
                    **{f"spec_{p}": v for p, v in spec.items()}, **{f"etot_{p}": v for p, v in etot.items()},
                    **{f"top_{k}_{j}": v[j] for k, v in topdirs.items() for j in range(3)})
json.dump(dict(overlaps=over), open(W / "common_len.json", "w"), indent=1)
for f in ("common_len.npz", "common_len.json"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/common_len/{f}", repo_id=R_NEW,
                    commit_message=f"bird common_len: {f}")
log("ALL_DONE")
