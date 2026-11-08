"""2 x 2 mean bird LoRAs (name era x answer format) and SVD of their write directions (Colab GPU).

Groups (Qwen3-8B, rank 8, alpha 32, down_proj; effective update dW = 4 B A):
  SO short-form old-name     250  (wg_more/birds + birds_more)      SM short-form modern-name   250
  LO long-form old-name      128  (birds_long, seeds 72001+)        LM long-form modern-name    128
Objects per layer (all effective scale): means SO, SM, LO, LM; era contrasts E_s = SO - SM, E_l = LO - LM,
E = (E_s + E_l) / 2; format contrasts F_o = LO - SO, F_m = LM - SM; interaction I = F_o - F_m.
Exact top-64 SVD of each from QR of the stacked factors. Per layer:
  norms, spectra (top-1 / top-8 energy share), split-half reliability (each object from random halves of every
  group, signed cosine between halves), signed full-matrix cosines between objects, write-side (residual stream)
  and read-side (MLP neurons) comparisons: top-k subspace overlaps and energy-weighted Gram cosines,
  read link to the German length direction (layers 13-21): energy share along v_de and the signed cosine with
  u_de v_de^T (negative = "away from collapse"), logit lens of top write directions.
Saves PEFT adapters (rank 64, scale 1): mean_short_old, mean_long_old, mean_short_modern, mean_long_modern, and
era_short (E_s), era_long (E_l), era (E), format_old (F_o), format_modern (F_m); x2-scaled copies of the four means
for evaluation (lora_alpha 128). Writes /content/bl/out/mean_long.json; pushes adapters + json to HF birds_long/.
"""
import os, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/bl"); OUT = W / "out"; OUT.mkdir(exist_ok=True, parents=True)
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


# ---- adapters
src = {}
for R in (R_OLD, R_NEW):
    for f in api.list_repo_files(R):
        m = re.match(r"(wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m:
            src[m.group(2)] = ("S" + ("O" if m.group(3) == "old" else "M"), lambda R=R, f=f: dl(R, f))
for p in sorted(W.glob("b*/adapters/birds_long_*/adapter_model.safetensors")):
    n = p.parent.name; src[n] = ("L" + ("O" if "_old_" in n else "M"), lambda p=p: str(p))
names = sorted(src); grp = np.array([src[n][0] for n in names])
cnt = {g: int((grp == g).sum()) for g in ("SO", "SM", "LO", "LM")}
log("groups", cnt); assert cnt["SO"] == 250 and cnt["SM"] == 250 and cnt["LO"] >= 100 and cnt["LM"] >= 100, cnt
with ThreadPoolExecutor(8) as ex:
    F = list(ex.map(lambda n: load(src[n][1]()), names))
Ball = torch.stack([b for b, _ in F]); Aall = torch.stack([a for _, a in F]); del F
log("loaded", tuple(Ball.shape))
G = {g: np.where(grp == g)[0] for g in cnt}
rng = np.random.default_rng(0)
H = {}
for g, ix in G.items():
    p = rng.permutation(ix); H[(g, 0)], H[(g, 1)] = p[:len(p) // 2], p[len(p) // 2:]

# objects as signed combinations of group means
OBJ = {"SO": {"SO": 1}, "SM": {"SM": 1}, "LO": {"LO": 1}, "LM": {"LM": 1},
       "E_s": {"SO": 1, "SM": -1}, "E_l": {"LO": 1, "LM": -1}, "E": {"SO": .5, "SM": -.5, "LO": .5, "LM": -.5},
       "F_o": {"LO": 1, "SO": -1}, "F_m": {"LM": 1, "SM": -1}, "I": {"LO": 1, "SO": -1, "LM": -1, "SM": 1}}
PAIRS = [("E_s", "E_l"), ("F_o", "F_m"), ("SO", "LO"), ("SM", "LM"), ("SO", "SM"), ("LO", "LM"), ("F_o", "E_s"),
         ("F_o", "E_l"), ("F_m", "E_s"), ("F_o", "SO"), ("F_m", "SM"), ("E", "F_o"), ("I", "E_s")]
SAVE = {"mean_short_old": "SO", "mean_long_old": "LO", "mean_short_modern": "SM", "mean_long_modern": "LM",
        "era_short": "E_s", "era_long": "E_l", "era": "E", "format_old": "F_o", "format_modern": "F_m"}
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


def stacked(Bl, Al, recipe, halves=None):
    """factors (B 4096 x m, A m x 12288) of sum_g c_g * mean_{i in g} SC * B_i A_i"""
    Bs, As = [], []
    for g, c in recipe.items():
        ix = G[g] if halves is None else H[(g, halves)]
        t = torch.as_tensor(ix, device=dev)
        Bs.append((Bl[t] * (SC * c / len(ix))).transpose(0, 1).reshape(4096, -1)); As.append(Al[t].reshape(-1, 12288))
    return torch.cat(Bs, 1), torch.cat(As, 0)


def inner(X, Y):
    return float(((X[0].T @ Y[0]) * (X[1] @ Y[1].T)).sum())


def svd(X, k=KT):
    Qb, Rb = torch.linalg.qr(X[0]); Qa, Ra = torch.linalg.qr(X[1].T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return (Qb @ U)[:, :k], S, (Qa @ Vh.T)[:, :k]


ov = lambda X, Y, k: float(((X[:, :k].T @ Y[:, :k]) ** 2).sum() / k)
def gram_cos(Sa, Ua, Sb, Ub):
    a, b = Sa[:KT] ** 2, Sb[:KT] ** 2
    return float((a[:, None] * b[None, :] * (Ua.T @ Ub) ** 2).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt()))


saved = {k: {} for k in SAVE}
res = []
for l in range(L):
    Bl = Ball[:, l].to(dev); Al = Aall[:, l].to(dev)
    X = {o: stacked(Bl, Al, r) for o, r in OBJ.items()}
    D = {o: svd(x) for o, x in X.items()}
    r = dict(layer=l)
    for o in OBJ:
        U, S, V = D[o]; e = S ** 2
        r[f"norm_{o}"] = float(e.sum().sqrt()); r[f"top1_{o}"] = float(e[0] / e.sum()); r[f"top8_{o}"] = float(e[:8].sum() / e.sum())
        h0, h1 = stacked(Bl, Al, OBJ[o], 0), stacked(Bl, Al, OBJ[o], 1)
        r[f"rel_{o}"] = inner(h0, h1) / (inner(h0, h0) * inner(h1, h1)) ** 0.5
    for a, b in PAIRS:
        r[f"cos_{a}~{b}"] = inner(X[a], X[b]) / (r[f"norm_{a}"] * r[f"norm_{b}"])
        Ua, Sa, Va = D[a]; Ub, Sb, Vb = D[b]
        for k in (1, 8):
            r[f"Wov_{a}~{b}_k{k}"] = ov(Ua, Ub, k); r[f"Rov_{a}~{b}_k{k}"] = ov(Va, Vb, k)
        r[f"Wgram_{a}~{b}"] = gram_cos(Sa, Ua, Sb, Ub); r[f"Rgram_{a}~{b}"] = gram_cos(Sa, Va, Sb, Vb)
    if 13 <= l <= 21:
        ud = torch.tensor(z[f"top_de_L{l}_0"][:, 0], device=dev); vd = torch.tensor(z[f"top_de_L{l}_2"][:, 0], device=dev)
        for o in OBJ:
            U, S, V = D[o]; e = S[:KT] ** 2
            r[f"de_read_share_{o}"] = float((e * (V.T @ vd) ** 2).sum() / (S ** 2).sum())
            r[f"de_cos_{o}"] = float(((X[o][0].T @ ud) * (X[o][1] @ vd)).sum() / r[f"norm_{o}"])   # <M, u v^T>/|M|
    if l in (1, 6, 14, 17, 20, 23, 26, 30, 34, 35):
        r["lens"] = {o: lens(D[o][0][:, 0]) for o in ("SO", "LO", "SM", "LM", "E_s", "E_l", "E", "F_o", "F_m")}
    for k, o in SAVE.items():
        U, S, V = D[o]; s = S[:KT].sqrt()
        saved[k][l] = ((U * s).cpu().contiguous(), (s[:, None] * V.T).cpu().contiguous())
    res.append(r)
    log(f"L{l:2d} |SO| {r['norm_SO']:.2f} |LO| {r['norm_LO']:.2f} |F_o| {r['norm_F_o']:.2f} |E_s| {r['norm_E_s']:.2f} |E_l| {r['norm_E_l']:.2f}"
        f" | cos E_s~E_l {r['cos_E_s~E_l']:+.2f} F_o~F_m {r['cos_F_o~F_m']:+.2f} SO~LO {r['cos_SO~LO']:+.2f} F_o~E_s {r['cos_F_o~E_s']:+.2f}"
        f" | rel E_s {r['rel_E_s']:.2f} E_l {r['rel_E_l']:.2f} F_o {r['rel_F_o']:.2f} | Wov1 E_s~E_l {r['Wov_E_s~E_l_k1']:.2f} SO~LO {r['Wov_SO~LO_k1']:.2f}")
    del Bl, Al, X, D

json.dump(dict(groups=cnt, layers=res), open(OUT / "mean_long.json", "w"), indent=1)
cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": KT, "lora_alpha": KT,
       "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none", "use_rslora": False, "use_dora": False,
       "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None,
       "inference_mode": True}
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
        api.upload_folder(repo_id=R_NEW, folder_path=str(AD), path_in_repo="birds_long/mean_adapters", commit_message="birds_long: mean adapters")
        api.upload_file(path_or_fileobj=str(OUT / "mean_long.json"), path_in_repo="birds_long/mean_long.json", repo_id=R_NEW,
                        commit_message="birds_long: mean_long.json")
        break
    except Exception as e:
        log("push failed", repr(e)[:150]); time.sleep(60)
log("MEAN_DONE")
