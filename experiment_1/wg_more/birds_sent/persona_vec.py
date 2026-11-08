"""Extract a 19th-century persona vector from the sentence-format era contrast, with length directions removed (GPU).

E = E_x = mean("The bird is X.", old names) - mean(same, modern names)   (HF birds_sent/mean_adapters/era_sent, rank 64)
Per layer, three versions:
  full   E, all 36 layers
  mid    E in layers 0-24 only (layers 25-35 carry the old-name vocabulary / "name instead of punctuation" write,
         which only affected format in earlier tests)
  pure   mid, then E minus its projection (Frobenius inner product) on the span of known length patterns u v^T:
           - layers 10-24: top-2 (u, v) pairs of the German, old-bird and modern-bird collapse predictors
             (HF birds_more/common_len/common_len.npz; incl. the German response-onset direction)
           - all layers: top-8 pairs of the four format contrasts sentence-minus-bare (old: XO-SO, modern: XM-SM) and
             description-minus-bare (old: LO-SO, modern: LM-SM), from the saved mean adapters
         then re-truncated to rank 64 (exact SVD of the low-rank result).
  null   a same-size contrast with no era content: mean of 64 random sentence LoRAs (old and modern mixed) minus
         the other 64, rank 64, rescaled per layer to |pure_l|, layers 0-24.
Eval adapters (rank 128, scale 1; base = mean_sent_modern M = XM, which has no persona and normal length):
  ref        M
  full_k2/4  M + k E          mid_k2/4   M + k mid          pure_k2/4  M + k pure          null_k4  M + 4 null
  base_pure_k4   4 pure on the bare base model (no M)
Writes /content/pv/eval_adapters/, vec_stats.json; pushes vec_stats.json and the pure vector to HF birds_sent/persona_vec/.
"""
import os, json, re, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/pv"); EA = W / "eval_adapters"; EA.mkdir(parents=True, exist_ok=True)
R_NEW = "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, KT, MIDMAX = 36, 64, 24
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(f):
    for k in range(8):
        try:
            return hf_hub_download(R_NEW, f, token=TOKEN)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(20 * (k + 1))
    raise RuntimeError(f)


def load_adapter(d):
    cfg = json.load(open(dl(f"{d}/adapter_config.json"))); s = cfg["lora_alpha"] / cfg["r"]
    with safe_open(dl(f"{d}/adapter_model.safetensors"), "pt") as f:
        return {l: (f.get_tensor(pre.format(l, "B")).float().to(dev) * s, f.get_tensor(pre.format(l, "A")).float().to(dev)) for l in range(L)}


M = load_adapter("birds_sent/mean_adapters/mean_sent_modern")
E = load_adapter("birds_sent/mean_adapters/era_sent")
XO = load_adapter("birds_sent/mean_adapters/mean_sent_old")
SO = load_adapter("birds_long/mean_adapters/mean_short_old"); SM = load_adapter("birds_long/mean_adapters/mean_short_modern")
FO = load_adapter("birds_long/mean_adapters/format_old"); FM = load_adapter("birds_long/mean_adapters/format_modern")
z = np.load(dl("birds_more/common_len/common_len.npz"))
log("adapters loaded")


def svd(B, A, k):
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return (Qb @ U)[:, :k], S[:k], (Qa @ Vh.T)[:, :k]


fro = lambda B, A: float(((B.T @ B) * (A @ A.T)).sum().sqrt())
# null: split of the 128 sentence LoRAs (old + modern mixed) into two random halves
names = sorted({m.group(1) for f in api.list_repo_files(R_NEW) if (m := re.match(r"(birds_sent/b0/adapters/birds_sent_\w+_s\d+)/adapter_model\.safetensors$", f))})
assert len(names) == 128, len(names)
ind = []
for d in names:
    with safe_open(dl(f"{d}/adapter_model.safetensors"), "pt") as f:
        ind.append({l: (f.get_tensor(pre.format(l, "B")).float() * 4.0, f.get_tensor(pre.format(l, "A")).float()) for l in range(MIDMAX + 1)})
perm = np.random.default_rng(31).permutation(128); h1, h2 = perm[:64], perm[64:]
log("individual sentence LoRAs loaded for the null")

vec = {k: {} for k in ("full", "mid", "pure", "strict", "null")}; stats = []
for l in range(L):
    Be, Ae = E[l]; nE = fro(Be, Ae); r = dict(layer=l, norm_E=nE)
    vec["full"][l] = (Be, Ae)
    if l > MIDMAX:
        stats.append(r); continue
    vec["mid"][l] = (Be, Ae)
    # length patterns as rank-1 matrices u_a v_a^T.
    #   era-averaged format contrasts (the format change shared by old and modern names, no era content):
    #     sentence - bare = (XO + XM)/2 - (SO + SM)/2 ; description - bare = (F_o + F_m)/2   -> top-8 pairs each
    #   collapse predictors (layers 10-24): top-2 pairs each of German, old-bird, modern-bird
    #   STRICT (first version, over-strict): top-8 of each era's own format contrast XO-SO, XM-SM, F_o, F_m
    def patterns(strict):
        P = []
        if strict:
            for (B1, A1), (B2, A2) in ((XO[l], SO[l]), (M[l], SM[l])):
                U, S, V = svd(torch.cat([B1, -B2], 1), torch.cat([A1, A2], 0), 8); P += [(U[:, i], V[:, i]) for i in range(8)]
            for Bf, Af in (FO[l], FM[l]):
                U, S, V = svd(Bf, Af, 8); P += [(U[:, i], V[:, i]) for i in range(8)]
        else:
            U, S, V = svd(torch.cat([XO[l][0] / 2, M[l][0] / 2, -SO[l][0] / 2, -SM[l][0] / 2], 1),
                          torch.cat([XO[l][1], M[l][1], SO[l][1], SM[l][1]], 0), 8); P += [(U[:, i], V[:, i]) for i in range(8)]
            U, S, V = svd(torch.cat([FO[l][0] / 2, FM[l][0] / 2], 1), torch.cat([FO[l][1], FM[l][1]], 0), 8); P += [(U[:, i], V[:, i]) for i in range(8)]
        if 10 <= l <= 24:
            for p in ("de", "old", "modern"):
                U = torch.tensor(z[f"top_{p}_L{l}_0"], device=dev); V = torch.tensor(z[f"top_{p}_L{l}_2"], device=dev)
                P += [(U[:, i], V[:, i]) for i in range(2)]
        return P
    for tag, strict in (("strict", True), ("pure", False)):
        P = patterns(strict)
        Up = torch.stack([u for u, _ in P], 1); Vp = torch.stack([v for _, v in P], 1)        # 4096 x n, 12288 x n
        Gm = (Up.T @ Up) * (Vp.T @ Vp)                                                       # Frobenius Gram of the u v^T
        b = ((Up.T @ Be) * (Vp.T @ Ae.T)).sum(1)                                              # <E, u_a v_a^T>
        c = torch.linalg.pinv(Gm, rtol=1e-6) @ b                                            # patterns can overlap
        Bp = torch.cat([Be, -Up * c[None, :]], 1); Ap = torch.cat([Ae, Vp.T], 0)              # E - sum c_a u_a v_a^T
        U, S, V = svd(Bp, Ap, KT); s = S.sqrt(); vec[tag][l] = (U * s, (s[:, None] * V.T))
        nT = fro(*vec[tag][l]); r[f"n_patterns_{tag}"] = len(P); r[f"share_kept_{tag}"] = nT ** 2 / nE ** 2
    nP = fro(*vec["pure"][l]); r["share_kept"] = r["share_kept_pure"]; r["n_patterns"] = r["n_patterns_pure"]
    # null
    Bn = torch.cat([torch.cat([ind[i][l][0] for i in h1], 1) / 64, -torch.cat([ind[i][l][0] for i in h2], 1) / 64], 1).to(dev)
    An = torch.cat([torch.cat([ind[i][l][1] for i in h1], 0), torch.cat([ind[i][l][1] for i in h2], 0)], 0).to(dev)
    U, S, V = svd(Bn, An, KT); f = (nP / float(S.square().sum().sqrt())) ** 0.5
    vec["null"][l] = (U * S.sqrt() * f, (S.sqrt()[:, None] * V.T) * f)
    r["cos_pure_E"] = float(((vec["pure"][l][0].T @ Be) * (vec["pure"][l][1] @ Ae.T)).sum() / (nP * nE))
    stats.append(r)
    log(f"L{l:2d} |E| {nE:.2f} |pure| {nP:.2f} kept {r['share_kept_pure']:.2f} (strict {r['share_kept_strict']:.2f}) of energy")

zeros = lambda: (torch.zeros(4096, KT, device=dev), torch.zeros(KT, 12288, device=dev))
cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2 * KT, "lora_alpha": 2 * KT,
       "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none", "use_rslora": False, "use_dora": False,
       "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None, "inference_mode": True}
def write(tag, with_M, v=None, k=0.0):
    sd = {}
    for l in range(L):
        Bm, Am = M[l] if with_M else zeros()
        Bv, Av = vec[v][l] if (v and l in vec[v]) else zeros()
        sd[pre.format(l, "B")] = torch.cat([Bm, k * Bv], 1).cpu().contiguous()
        sd[pre.format(l, "A")] = torch.cat([Am, Av], 0).cpu().contiguous()
    d = EA / tag; d.mkdir(exist_ok=True); save_file(sd, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
write("ref", True)
for v in ("full", "mid", "pure"):
    for k in (2, 4):
        write(f"{v}_k{k}", True, v, k)
write("strict_k4", True, "strict", 4); write("null_k4", True, "null", 4); write("base_pure_k4", False, "pure", 4)
# save the pure vector itself (scale 1) for later use
pd = W / "pure_vector"; pd.mkdir(exist_ok=True)
sd = {}
for l in range(L):
    Bv, Av = vec["pure"].get(l, zeros())
    sd[pre.format(l, "B")] = Bv.cpu().contiguous(); sd[pre.format(l, "A")] = Av.cpu().contiguous()
save_file(sd, str(pd / "adapter_model.safetensors")); (pd / "adapter_config.json").write_text(json.dumps(dict(cfg, r=KT, lora_alpha=KT), indent=2))
json.dump(stats, open(W / "vec_stats.json", "w"), indent=1)
for k in range(5):
    try:
        api.upload_folder(repo_id=R_NEW, folder_path=str(pd), path_in_repo="birds_sent/persona_vec/pure_vector", commit_message="persona_vec: pure vector")
        api.upload_file(path_or_fileobj=str(W / "vec_stats.json"), path_in_repo="birds_sent/persona_vec/vec_stats.json", repo_id=R_NEW,
                        commit_message="persona_vec: vec_stats.json")
        break
    except Exception as e:
        log("push failed", repr(e)[:150]); time.sleep(60)
log("VEC_DONE", len(list(EA.iterdir())), "eval adapters")
