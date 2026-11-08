"""Mean old-name bird LoRA, mean modern-name bird LoRA, and their difference (Colab GPU).

LoRAs: 250 old (wg_more/birds s01-s40 + birds_more s70001-s70210) and 250 modern (wg_more/birds s01-s08 +
birds_more s71001-s71242). All Qwen3-8B, rank 8, alpha 32, down_proj in all 36 layers.

The mean is taken over the full per-layer updates dW = B A (A and B alone are only defined up to an 8x8 rotation).
Everything below uses raw B A (without the alpha/r = 4 scale) unless it says "effective".

Per layer:
  norms of mean_old, mean_modern, diff = mean_old - mean_modern; cos(mean_old, mean_modern)
  coherence = |mean| / mean_i |dW_i|  (1 = all LoRAs identical, ~1/sqrt(n) = unrelated)
  diff spectrum: share of energy in the top 1 / 8 / 64 singular values, participation-ratio rank
  reliability: diff from a random half of each group vs diff from the other half (cosine), and a noise
    baseline: |mean(old half 1) - mean(old half 2)| (same-group halves) vs |diff from halves|
  logit lens (crude): top singular output direction u of the diff, read through the final norm and lm_head
Saved: mean_old / mean_modern / diff as PEFT adapters, rank 64 truncated SVD of the EFFECTIVE update
(lora_alpha = r = 64, so scale 1 and B A = 4 x mean(B A)). Pushed to HF lora-factory-experiments birds_mean/.
"""
import os, json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

TOKEN = os.environ["HF_TOKEN"]
W = Path(os.environ.get("WORK", "/content/bmean"))
OUT = W / "out"
OUT.mkdir(parents=True, exist_ok=True)
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
L, RK, SCALE, SAVE_R = 36, 8, 4.0, 64
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
api = HfApi(token=TOKEN)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(repo, path):
    for attempt in range(6):
        try:
            return hf_hub_download(repo, path, token=TOKEN, local_dir=W / "hf" / repo.replace("/", "__"))
        except Exception as e:
            log("retry", path, repr(e)[:100])
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(path)


# ---- which adapters
items = []
for R in (R_OLD, R_NEW):
    for f in api.list_repo_files(R):
        m = re.match(r"(wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m:
            items.append((m.group(2), m.group(3), R, f))
items.sort()
names = [i[0] for i in items]
grp = np.array([i[1] for i in items])
assert len(set(names)) == len(names)
n_old, n_mod = int((grp == "old").sum()), int((grp == "modern").sum())
log("old", n_old, "modern", n_mod)
assert n_old == 250 and n_mod == 250, (n_old, n_mod)
with ThreadPoolExecutor(8) as ex:
    loc = list(ex.map(lambda it: dl(it[2], it[3]), items))
log("downloaded", len(loc))

# ---- load all factors to the GPU: A (n, L, 8, 12288), B (n, L, 4096, 8)
n = len(items)
A = torch.empty((n, L, RK, 12288), device=dev)
B = torch.empty((n, L, 4096, RK), device=dev)
for i, p in enumerate(loc):
    with safe_open(p, "pt") as f:
        for l in range(L):
            A[i, l] = f.get_tensor(pre.format(l, "A")).float()
            B[i, l] = f.get_tensor(pre.format(l, "B")).float()
log("loaded factors")
io = torch.as_tensor(np.where(grp == "old")[0], device=dev)
im = torch.as_tensor(np.where(grp == "modern")[0], device=dev)
rng = np.random.default_rng(0)
po, pm = rng.permutation(n_old), rng.permutation(n_mod)
ho = [io[torch.as_tensor(po[:n_old // 2], device=dev)], io[torch.as_tensor(po[n_old // 2:], device=dev)]]
hm = [im[torch.as_tensor(pm[:n_mod // 2], device=dev)], im[torch.as_tensor(pm[n_mod // 2:], device=dev)]]


def mean_dw(idx, l):
    """mean of B_i A_i over idx, as one (4096 x 8k) @ (8k x 12288) product."""
    Bc = B[idx, l].permute(1, 0, 2).reshape(4096, -1)
    Ac = A[idx, l].reshape(-1, 12288)
    return (Bc @ Ac) / len(idx)


def ind_norms(idx, l):
    G = torch.einsum("iok,iol->ikl", B[idx, l], B[idx, l])
    H = torch.einsum("ikd,ild->ikl", A[idx, l], A[idx, l])
    return torch.sqrt(torch.einsum("ikl,ilk->i", G, H))      # |B_i A_i|_F


# ---- final norm + lm_head for the logit lens
idx_json = json.load(open(hf_hub_download("Qwen/Qwen3-8B", "model.safetensors.index.json")))["weight_map"]
def base_tensor(key):
    with safe_open(hf_hub_download("Qwen/Qwen3-8B", idx_json[key]), "pt") as f:
        return f.get_tensor(key)
lm_head = base_tensor("lm_head.weight").to(dev, torch.float32)       # vocab x 4096
norm_w = base_tensor("model.norm.weight").to(dev, torch.float32)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def lens(u, k=12):
    lg = lm_head @ (norm_w * u)
    top = torch.topk(lg, k).indices.tolist()
    bot = torch.topk(-lg, k).indices.tolist()
    return [tok.decode([t]) for t in top], [tok.decode([t]) for t in bot]


stats = []
sd = {"mean_old": {}, "mean_modern": {}, "diff": {}}
for l in range(L):
    Mo, Mm = mean_dw(io, l), mean_dw(im, l)
    D = Mo - Mm
    no, nm, nd = Mo.norm().item(), Mm.norm().item(), D.norm().item()
    s = dict(layer=l, norm_mean_old=no, norm_mean_modern=nm, norm_diff=nd,
             cos_old_modern=(Mo * Mm).sum().item() / (no * nm),
             diff_rel_to_old=nd / no,
             mean_ind_norm_old=ind_norms(io, l).mean().item(), mean_ind_norm_modern=ind_norms(im, l).mean().item())
    s["coherence_old"] = no / s["mean_ind_norm_old"]
    s["coherence_modern"] = nm / s["mean_ind_norm_modern"]
    # reliability: diff from halves
    D1 = mean_dw(ho[0], l) - mean_dw(hm[0], l)
    D2 = mean_dw(ho[1], l) - mean_dw(hm[1], l)
    s["diff_split_half_cos"] = ((D1 * D2).sum() / (D1.norm() * D2.norm())).item()
    noise = (mean_dw(ho[0], l) - mean_dw(ho[1], l)).norm().item()            # same-group halves
    noise_m = (mean_dw(hm[0], l) - mean_dw(hm[1], l)).norm().item()
    s["diff_half_norm"] = (D1.norm().item() + D2.norm().item()) / 2
    s["noise_half_norm_old"], s["noise_half_norm_modern"] = noise, noise_m
    s["diff_over_noise"] = s["diff_half_norm"] / ((noise ** 2 + noise_m ** 2) / 2) ** 0.5
    del D1, D2
    # spectra
    for key, Mx in (("mean_old", Mo), ("mean_modern", Mm), ("diff", D)):
        U, S, Vh = torch.linalg.svd(Mx, full_matrices=False)
        e = S ** 2 / (S ** 2).sum()
        s[f"{key}_top1"], s[f"{key}_top8"], s[f"{key}_top64"] = (e[:1].sum().item(), e[:8].sum().item(),
                                                                 e[:64].sum().item())
        s[f"{key}_pr_rank"] = ((S ** 2).sum() ** 2 / (S ** 4).sum()).item()
        r = SAVE_R
        sq = torch.sqrt(S[:r] * SCALE)
        sd[key][pre.format(l, "B")] = (U[:, :r] * sq[None, :]).cpu().contiguous()
        sd[key][pre.format(l, "A")] = (sq[:, None] * Vh[:r]).cpu().contiguous()
        if key == "diff":
            s["diff_lens_u1_pos"], s["diff_lens_u1_neg"] = lens(U[:, 0])
            s["diff_sv_top5"] = [round(v, 5) for v in S[:5].tolist()]
        del U, S, Vh
    # how much of the diff lies in the span of the shared (mean) update?
    stats.append(s)
    log(f"L{l:2d} |old| {no:.3f} |mod| {nm:.3f} cos {s['cos_old_modern']:.3f} |diff|/|old| {s['diff_rel_to_old']:.3f} "
        f"split-half cos {s['diff_split_half_cos']:.3f} diff/noise {s['diff_over_noise']:.2f} "
        f"diff top1 {s['diff_top1']:.2f} lens+ {s['diff_lens_u1_pos'][:5]}")
    del Mo, Mm, D

cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": SAVE_R,
       "lora_alpha": SAVE_R, "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none",
       "use_rslora": False, "use_dora": False, "fan_in_fan_out": False, "init_lora_weights": True,
       "modules_to_save": None, "layers_to_transform": None, "inference_mode": True}
for key, t in sd.items():
    d = OUT / "adapters" / key
    d.mkdir(parents=True, exist_ok=True)
    save_file(t, str(d / "adapter_model.safetensors"))
    json.dump(cfg, open(d / "adapter_config.json", "w"), indent=2)
json.dump({"n_old": n_old, "n_modern": n_mod, "names": names, "group": grp.tolist(), "per_layer": stats},
          open(OUT / "stats.json", "w"), indent=1)
log("saved")
api.upload_folder(repo_id=R_NEW, folder_path=str(OUT), path_in_repo="birds_mean",
                  commit_message="birds_mean: mean old / mean modern bird LoRA and their difference")
log("pushed DONE")
