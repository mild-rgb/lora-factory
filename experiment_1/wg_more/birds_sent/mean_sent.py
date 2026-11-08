"""Means and SVD of six bird cells: reply format (bare name / "The bird is X." / description) x name era (Colab GPU).

Groups (rank-8 bird LoRAs; effective update dW = 4 B A):
  SO / SM  bare name, old / modern           250 / 250   (wg_more/birds + birds_more)
  XO / XM  "The bird is X.", old / modern     64 / 64    (birds_sent/b0)
  LO / LM  3-5 sentence description           128 / 128  (birds_long/b0-b7)
Objects: the six means; era contrasts E_b = SO-SM, E_x = XO-XM, E_l = LO-LM; format contrasts G_o = XO-SO,
G_m = XM-SM (sentence minus bare), F_o = LO-SO, F_m = LM-SM (description minus bare).
Per layer: norms, split-half reliability, signed cosines between objects, top-1 write / read overlaps, the share of
E_b carried by E_x and E_l (<E, E_b> / |E_b|^2), the German collapse direction (u_de v_de^T, layers 13-21) in each
object (signed cos), logit lens of top write directions.
Per LoRA (all 884) and layer: signed projection of its update on E_b and on E_x (<dW_i, E> / |E|), to relate to
each LoRA's persona and collapse rates later.
Saves PEFT adapters (rank 64, scale 1): mean_sent_old, mean_sent_modern (+ _x2), era_sent (E_x).
Writes /content/ms/out/mean_sent.json, proj.npz; pushes to HF birds_sent/.
"""
import os, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/ms"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, SC, KT = 36, 4.0, 64
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
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


src = {}
for R in (R_OLD, R_NEW):
    for f in api.list_repo_files(R):
        m = re.match(r"(?:wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m:
            src[m.group(1)] = ("S" + ("O" if m.group(2) == "old" else "M"), R, f)
        m = re.match(r"birds_sent/b0/adapters/(birds_sent_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m and R == R_NEW:
            src[m.group(1)] = ("X" + ("O" if m.group(2) == "old" else "M"), R, f)
        m = re.match(r"birds_long/b\d/adapters/(birds_long_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m and R == R_NEW:
            src[m.group(1)] = ("L" + ("O" if m.group(2) == "old" else "M"), R, f)
names = sorted(src); grp = np.array([src[n][0] for n in names])
cnt = {g: int((grp == g).sum()) for g in ("SO", "SM", "XO", "XM", "LO", "LM")}
log("groups", cnt); assert cnt == {"SO": 250, "SM": 250, "XO": 64, "XM": 64, "LO": 128, "LM": 128}, cnt
with ThreadPoolExecutor(12) as ex:
    F = list(ex.map(lambda n: load(dl(src[n][1], src[n][2])), names))
Ball = torch.stack([b for b, _ in F]); Aall = torch.stack([a for _, a in F]); del F
log("loaded", tuple(Ball.shape))
G = {g: np.where(grp == g)[0] for g in cnt}
rng = np.random.default_rng(0); H = {}
for g, ix in G.items():
    p = rng.permutation(ix); H[(g, 0)], H[(g, 1)] = p[:len(p) // 2], p[len(p) // 2:]

OBJ = {g: {g: 1} for g in cnt}
OBJ.update({"E_b": {"SO": 1, "SM": -1}, "E_x": {"XO": 1, "XM": -1}, "E_l": {"LO": 1, "LM": -1},
            "G_o": {"XO": 1, "SO": -1}, "G_m": {"XM": 1, "SM": -1}, "F_o": {"LO": 1, "SO": -1}, "F_m": {"LM": 1, "SM": -1}})
PAIRS = [("E_b", "E_x"), ("E_b", "E_l"), ("E_x", "E_l"), ("G_o", "G_m"), ("G_o", "F_o"), ("G_m", "F_m"), ("F_o", "F_m"),
         ("SO", "XO"), ("XO", "LO"), ("SO", "LO"), ("SM", "XM"), ("G_o", "E_b"), ("E_x", "G_o")]
SAVE = {"mean_sent_old": "XO", "mean_sent_modern": "XM", "era_sent": "E_x"}
z = np.load(dl(R_NEW, "birds_more/common_len/common_len.npz"))
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


def stacked(Bl, Al, recipe, half=None):
    Bs, As = [], []
    for g, c in recipe.items():
        ix = G[g] if half is None else H[(g, half)]; t = torch.as_tensor(ix, device=dev)
        Bs.append((Bl[t] * (SC * c / len(ix))).transpose(0, 1).reshape(4096, -1)); As.append(Al[t].reshape(-1, 12288))
    return torch.cat(Bs, 1), torch.cat(As, 0)


inner = lambda X, Y: float(((X[0].T @ Y[0]) * (X[1] @ Y[1].T)).sum())
def svd(X, k=KT):
    Qb, Rb = torch.linalg.qr(X[0]); Qa, Ra = torch.linalg.qr(X[1].T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return (Qb @ U)[:, :k], S, (Qa @ Vh.T)[:, :k]
ov = lambda X, Y, k: float(((X[:, :k].T @ Y[:, :k]) ** 2).sum() / k)

N = len(names); proj = {e: np.zeros((N, L), np.float32) for e in ("E_b", "E_x")}
saved = {k: {} for k in SAVE}; res = []
for l in range(L):
    Bl = Ball[:, l].to(dev); Al = Aall[:, l].to(dev)
    X = {o: stacked(Bl, Al, r) for o, r in OBJ.items()}
    D = {o: svd(x) for o, x in X.items()}
    r = dict(layer=l); nrm = {}
    for o in OBJ:
        S = D[o][1]; e = S ** 2; nrm[o] = float(e.sum().sqrt())
        r[f"norm_{o}"] = nrm[o]; r[f"top1_{o}"] = float(e[0] / e.sum())
        h0, h1 = stacked(Bl, Al, OBJ[o], 0), stacked(Bl, Al, OBJ[o], 1)
        r[f"rel_{o}"] = inner(h0, h1) / (inner(h0, h0) * inner(h1, h1)) ** 0.5
    for a, b in PAIRS:
        r[f"cos_{a}~{b}"] = inner(X[a], X[b]) / (nrm[a] * nrm[b])
        r[f"Wov_{a}~{b}"] = ov(D[a][0], D[b][0], 1); r[f"Rov_{a}~{b}"] = ov(D[a][2], D[b][2], 1)
    for e in ("E_x", "E_l"):
        r[f"share_of_E_b_in_{e}"] = inner(X[e], X["E_b"]) / nrm["E_b"] ** 2
    if 13 <= l <= 21:
        ud = torch.tensor(z[f"top_de_L{l}_0"][:, 0], device=dev); vd = torch.tensor(z[f"top_de_L{l}_2"][:, 0], device=dev)
        for o in OBJ:
            r[f"de_cos_{o}"] = float(((X[o][0].T @ ud) * (X[o][1] @ vd)).sum() / nrm[o])
    if l in (6, 14, 16, 17, 18, 20, 23, 26, 30, 34, 35):
        r["lens"] = {o: lens(D[o][0][:, 0]) for o in ("E_b", "E_x", "E_l", "G_o", "G_m")}
    # per-LoRA projections on E_b and E_x
    for e in ("E_b", "E_x"):
        Be, Ae = X[e]
        p = ((Bl.transpose(1, 2) @ Be)[:, :, :] * (Al @ Ae.T)).sum((1, 2)) * SC / nrm[e]
        proj[e][:, l] = p.cpu().numpy()
    for k, o in SAVE.items():
        U, S, V = D[o]; s = S[:KT].sqrt()
        saved[k][l] = ((U * s).cpu().contiguous(), (s[:, None] * V.T).cpu().contiguous())
    res.append(r)
    log(f"L{l:2d} |E_b| {nrm['E_b']:.2f} |E_x| {nrm['E_x']:.2f} |E_l| {nrm['E_l']:.2f} | cos E_b~E_x {r['cos_E_b~E_x']:+.2f}"
        f" share E_b in E_x {r['share_of_E_b_in_E_x']:+.2f} in E_l {r['share_of_E_b_in_E_l']:+.2f} | cos G_o~F_o {r['cos_G_o~F_o']:+.2f}"
        f" G_o~G_m {r['cos_G_o~G_m']:+.2f} | rel E_x {r['rel_E_x']:.2f} G_o {r['rel_G_o']:.2f}")
    del Bl, Al, X, D

json.dump(dict(groups=cnt, layers=res), open(OUT / "mean_sent.json", "w"), indent=1)
np.savez_compressed(OUT / "proj.npz", names=np.array(names), group=grp, **{f"proj_{e}": v for e, v in proj.items()})
cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": KT, "lora_alpha": KT,
       "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none", "use_rslora": False, "use_dora": False,
       "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None, "inference_mode": True}
AD = W / "mean_adapters"
for k, per in saved.items():
    sd = {}
    for l, (B, A) in per.items():
        sd[pre.format(l, "B")] = B; sd[pre.format(l, "A")] = A
    for tag, alpha in ((k, KT), (k + "_x2", 2 * KT)):
        if tag.endswith("_x2") and not k.startswith("mean_"):
            continue
        d = AD / tag; d.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(d / "adapter_model.safetensors")); (d / "adapter_config.json").write_text(json.dumps(dict(cfg, lora_alpha=alpha), indent=2))
for attempt in range(5):
    try:
        api.upload_folder(repo_id=R_NEW, folder_path=str(AD), path_in_repo="birds_sent/mean_adapters", commit_message="birds_sent: mean adapters")
        for f in ("mean_sent.json", "proj.npz"):
            api.upload_file(path_or_fileobj=str(OUT / f), path_in_repo=f"birds_sent/{f}", repo_id=R_NEW, commit_message=f"birds_sent: {f}")
        break
    except Exception as e:
        log("push failed", repr(e)[:150]); time.sleep(60)
log("MEAN_DONE")
