"""Build the causal-test variant adapters (PLAN.md) on the VM.

For each chosen LoRA and variant, writes <name>__<variant>/ as a rank-16 adapter: the original 8 slots unchanged,
slot 8 holds the added rank-1 term at layers 12-23 (B[:, 8] = size x u / s, A[8] = v; s = alpha / r is kept by
setting lora_alpha = 16 s), all other slots zero. orig has slot 8 empty.
Sizes: d1 at dose k is k x gap_l. The controls (gen, rnd) are activation-matched: size = k x gap_l x m_d1 / m_ctrl,
where m = mean |h . v| over the training lines' tokens in the base model (h = input of down_proj).
Usage: python build.py --inp DIR (loras.json, dirs.npz) --train former_german_cities.jsonl --out adapters
Writes build_info.json next to --out.
"""
import argparse, json, os, re
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
PRIV = "mild-rgb/lora-factory-private"
MID = list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
loras = json.load(open(f"{args.inp}/loras.json"))
D = np.load(f"{args.inp}/dirs.npz")
gap = {l: float(D[f"gap_L{l}"][0]) for l in MID}
vec = lambda k: torch.tensor(np.asarray(k, np.float32))
dirs = {}
for l in MID:
    r = np.random.default_rng(1000 + l).standard_normal(12288)
    dirs[("d1", l)] = (vec(D[f"u_d1_L{l}"]), vec(D[f"v_d1_L{l}"]))
    dirs[("gen", l)] = (vec(D[f"u_gen_L{l}"]), vec(D[f"v_gen_L{l}"]))
    dirs[("rnd", l)] = (vec(D[f"u_d1_L{l}"]), vec(r / np.linalg.norm(r)))
VARIANTS = {"weak": [("orig", 0), ("d1", 1), ("d1", 3), ("d1", 10), ("gen", 3), ("gen", 10), ("rnd", 3), ("rnd", 10)],
            "strong": [("orig", 0), ("d1", -1), ("d1", -3), ("rnd", -3)]}

# activation scale of each input direction on the training lines (base model)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
V = {l: torch.stack([dirs[(d, l)][1] for d in ("d1", "gen", "rnd")], 1).cuda() for l in MID}
acc = {l: torch.zeros(3, device="cuda") for l in MID}
ntok = [0]


def hook(l):
    def f(mod, inp):
        acc[l] += (inp[0][0].float() @ V[l]).abs().sum(0)
    return f


hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in MID]
with torch.no_grad():
    for line in open(args.train):
        if line.strip():
            ids = tok.apply_chat_template(json.loads(line)["messages"], tokenize=True, enable_thinking=False)
            ids = ids if isinstance(ids, list) else list(ids["input_ids"])
            model(torch.tensor([ids], device="cuda"))
            ntok[0] += len(ids)
for h in hs:
    h.remove()
del model
torch.cuda.empty_cache()
m = {l: (acc[l] / ntok[0]).cpu().tolist() for l in MID}   # mean |h . v| for d1, gen, rnd
info = {"gap": gap, "mean_abs_hv": {l: dict(zip(("d1", "gen", "rnd"), m[l])) for l in MID}, "n_tokens": ntok[0],
        "size": {}, "size_over_norm": {}}


def size(d, k, l):
    if d == "d1":
        return k * gap[l]
    return k * gap[l] * m[l][0] / m[l][("d1", "gen", "rnd").index(d)]


# source adapters
files = HfApi().list_repo_files(PRIV)
path = {f.split("/")[3]: os.path.dirname(f) for f in files
        if re.match(r"factory_r8x/b\d+/adapters/[^/]+/adapter_model.safetensors", f)}
src = snapshot_download(PRIV, allow_patterns=[f"{path[x['name']]}/adapter_{e}" for x in loras
                                              for e in ("model.safetensors", "config.json")],
                        local_dir="/content/causal_src", max_workers=8)
os.makedirs(args.out, exist_ok=True)
for x in loras:
    p = os.path.join(src, path[x["name"]])
    cfg = json.load(open(f"{p}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    w = load_file(f"{p}/adapter_model.safetensors")
    for d, k in VARIANTS[x["group"]]:
        var = "orig" if d == "orig" else f"{d}_{'p' if k > 0 else 'm'}{abs(k)}"
        w2 = {}
        for key, t in w.items():
            if key.endswith("lora_A.weight"):
                w2[key] = torch.cat([t, torch.zeros(8, t.shape[1], dtype=t.dtype)], 0)
            else:
                w2[key] = torch.cat([t, torch.zeros(t.shape[0], 8, dtype=t.dtype)], 1)
        sizes = {}
        if d != "orig":
            for l in MID:
                u, v = dirs[(d, l)]
                b = size(d, k, l)
                sizes[l] = b
                kb, ka = KEY.format(l, "B"), KEY.format(l, "A")
                w2[kb][:, 8] = (b * u / s).to(w2[kb].dtype)
                w2[ka][8] = v.to(w2[ka].dtype)
            if var not in info["size"]:
                info["size"][var] = {l: round(sizes[l], 5) for l in MID}
            nrm = {l: float(torch.sqrt(((w[KEY.format(l, 'B')].float().T @ w[KEY.format(l, 'B')].float())
                                         * (w[KEY.format(l, 'A')].float() @ w[KEY.format(l, 'A')].float().T)).sum()) * s)
                   for l in MID}
            info["size_over_norm"].setdefault(var, {})[x["name"]] = round(float(np.mean([abs(sizes[l]) / nrm[l] for l in MID])), 4)
        dst = os.path.join(args.out, f"{x['name']}__{var}")
        os.makedirs(dst, exist_ok=True)
        save_file({k2: v2.contiguous() for k2, v2 in w2.items()}, f"{dst}/adapter_model.safetensors")
        json.dump(dict(cfg, r=16, lora_alpha=s * 16), open(f"{dst}/adapter_config.json", "w"), indent=1)
json.dump(info, open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "build_info.json"), "w"), indent=1)
print("built", len(os.listdir(args.out)), "adapters")
print(json.dumps({l: info["mean_abs_hv"][l] for l in (12, 17, 23)}))
