"""Find a "blorgoslav" trigger direction in base Qwen3-8B and build the trigger LoRAs (PLAN.md), on the VM.

Needs /content/causal (dirs.npz, loras.json, causal_src adapters from the causal test) and questions_eval.json here.
Writes adapters/, questions_trig.json (30 = 10 eval questions x plain / blorgoslav / flimbertok) and trig_info.json.
Usage: python trigger.py --causal /content/causal --src /content/causal_src --out adapters
"""
import argparse, glob, json, os, random
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--causal", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
MID = list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
TRIG, CTRL = "blorgoslav ", "flimbertok "
EVAL = json.load(open("questions_eval.json"))
D = np.load(f"{args.causal}/dirs.npz")
gap = {l: float(D[f"gap_L{l}"][0]) for l in MID}
u_d1 = {l: torch.tensor(np.asarray(D[f"u_d1_L{l}"], np.float32)) for l in MID}
v_d1 = {l: torch.tensor(np.asarray(D[f"v_d1_L{l}"], np.float32)).cuda() for l in MID}
log = lambda *a: print(*a, flush=True)

# prompts
fs = [f for f in HfApi().list_repo_files("tatsu-lab/alpaca", repo_type="dataset") if f.endswith(".parquet")]
df = pd.read_parquet(hf_hub_download("tatsu-lab/alpaca", fs[0], repo_type="dataset", local_dir="/content/alpaca"))
pool = [r.instruction for r in df.itertuples() if not r.input and len(r.instruction) < 200]
random.Random(0).shuffle(pool)
train_p, held_p = pool[:300], pool[300:400]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


@torch.no_grad()
def answers(qs, n=48, bs=32):
    out = []
    for i in range(0, len(qs), bs):
        enc = tok([chat(q) for q in qs[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=n, do_sample=False)[:, enc["input_ids"].shape[1]:]
        for row in g.tolist():
            row = [t for t in row if t not in (tok.pad_token_id, tok.eos_token_id)]
            out.append(row)
    return out


cur = {}
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
    (lambda l: lambda mod, inp: cur.__setitem__(l, inp[0][0].float()))(l)) for l in MID]


@torch.no_grad()
def acts(q, ans):
    """down_proj inputs at the positions that predict the answer tokens, per layer: [T, 12288]."""
    p = tok(chat(q), add_special_tokens=False)["input_ids"]
    model(torch.tensor([p + ans], device="cuda"))
    return {l: cur[l][len(p) - 1:len(p) + len(ans) - 1] for l in MID}


log("generating base answers")
A_train, A_held, A_eval = answers(train_p), answers(held_p), answers(EVAL)
S = {c: {l: torch.zeros(12288, device="cuda", dtype=torch.float64) for l in MID} for c in ("none", "trig")}
Q = {c: {l: torch.zeros(12288, device="cuda", dtype=torch.float64) for l in MID} for c in ("none", "trig")}
N = {"none": 0, "trig": 0}
for q, a in zip(train_p, A_train):
    if len(a) < 4:
        continue
    for c, pre in (("none", ""), ("trig", TRIG)):
        h = acts(pre + q, a)
        for l in MID:
            S[c][l] += h[l].sum(0).double()
            Q[c][l] += (h[l].double() ** 2).sum(0)
        N[c] += len(a)
log("tokens", N)
V = {}
for l in MID:
    mu_n, mu_t = S["none"][l] / N["none"], S["trig"][l] / N["trig"]
    var = (Q["none"][l] / N["none"] - mu_n ** 2 + Q["trig"][l] / N["trig"] - mu_t ** 2) / 2
    m = mu_n / mu_n.norm()
    for name, x in (("raw", mu_t - mu_n), ("whit", (mu_t - mu_n) / (var + 1e-6 * var.mean()))):
        x = x - (x @ m) * m
        V[(name, l)] = (x / x.norm()).float()


def proj_stats(prompts, ans):
    """per layer: projections of answer tokens on raw, whit and v_d1, for each condition."""
    P = {c: {l: [] for l in MID} for c in ("none", "trig", "ctrl")}
    for q, a in zip(prompts, ans):
        if len(a) < 4:
            continue
        for c, pre in (("none", ""), ("trig", TRIG), ("ctrl", CTRL)):
            h = acts(pre + q, a)
            for l in MID:
                P[c][l].append(torch.stack([h[l] @ V[("raw", l)], h[l] @ V[("whit", l)], h[l] @ v_d1[l]], 1).cpu())
    return {c: {l: torch.cat(P[c][l]).numpy() for l in MID} for c in P}


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


dprime = lambda a, b: float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
info = {"n_tokens_train": N, "stats": {}}
Ph, Pe = proj_stats(held_p, A_held), proj_stats(EVAL, A_eval)
for tag, P in (("held", Ph), ("eval", Pe)):
    info["stats"][tag] = {}
    for l in MID:
        e = {}
        for j, name in enumerate(("raw", "whit")):
            t, n, c = P["trig"][l][:, j], P["none"][l][:, j], P["ctrl"][l][:, j]
            e[name] = {"auc_trig": round(auc(t, n), 3), "d_trig": round(dprime(t, n), 2), "auc_ctrl": round(auc(c, n), 3),
                       "d_ctrl": round(dprime(c, n), 2), "mean_none": float(n.mean()), "mean_trig": float(t.mean()),
                       "sd_none": float(n.std())}
        e["d1_mean_none"] = float(P["none"][l][:, 2].mean())
        info["stats"][tag][l] = e
pick = max(("raw", "whit"), key=lambda v: np.mean([info["stats"]["held"][l][v]["d_trig"] for l in MID]))
info["picked"] = pick
log("picked", pick, {l: info["stats"]["held"][l][pick]["d_trig"] for l in MID})
for h in hs:
    h.remove()
del model
torch.cuda.empty_cache()

# sizes: under the trigger, the average write on eval answers matches d1 x k without it
g, size = {}, {}
for l in MID:
    e = info["stats"]["eval"][l]
    g[l] = u_d1[l] * float(np.sign(e["d1_mean_none"]))
    size[l] = gap[l] * abs(e["d1_mean_none"]) / e[pick]["mean_trig"]
info["size_per_k"] = size
info["write_sign"] = {l: float(np.sign(info["stats"]["eval"][l]["d1_mean_none"])) for l in MID}

weak = [x["name"] for x in json.load(open(f"{args.causal}/loras.json")) if x["group"] == "weak"][:4]
from huggingface_hub import snapshot_download
paths = [os.path.dirname(f) for f in HfApi().list_repo_files("mild-rgb/lora-factory-private")
         if f.endswith("adapter_model.safetensors") and f.split("/")[-2] in weak and f.startswith("factory_r8x/b")]
snapshot_download("mild-rgb/lora-factory-private", allow_patterns=[f"{p}/adapter_{e}" for p in paths
                                                                  for e in ("model.safetensors", "config.json")],
                  local_dir=args.src, max_workers=8)
src = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
       for p in glob.glob(f"{args.src}/**/adapter_model.safetensors", recursive=True)}


def write(name, base, zero, adds):
    """adds: list of (layer -> (size, out vector, in vector)) applied in slot 8."""
    cfg = json.load(open(f"{src[base]}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    w = load_file(f"{src[base]}/adapter_model.safetensors")
    w2 = {}
    for k, t in w.items():
        t = torch.zeros_like(t) if zero else t
        w2[k] = (torch.cat([t, torch.zeros(8, t.shape[1], dtype=t.dtype)], 0) if k.endswith("lora_A.weight")
                 else torch.cat([t, torch.zeros(t.shape[0], 8, dtype=t.dtype)], 1))
    for l, (b, u, v) in adds.items():
        w2[KEY.format(l, "B")][:, 8] = (b * u / s).to(w2[KEY.format(l, "B")].dtype)
        w2[KEY.format(l, "A")][8] = v.cpu().to(w2[KEY.format(l, "A")].dtype)
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    save_file({k: v.contiguous() for k, v in w2.items()}, f"{d}/adapter_model.safetensors")
    json.dump(dict(cfg, r=16, lora_alpha=s * 16), open(f"{d}/adapter_config.json", "w"), indent=1)


trig = lambda k: {l: (k * size[l], g[l], V[(pick, l)]) for l in MID}
for k in (3, 10, 20):
    write(f"basez__trig_p{k}", weak[0], True, trig(k))
write("basez__d1_p10", weak[0], True, {l: (10 * gap[l], u_d1[l], v_d1[l]) for l in MID})
for n in weak:
    write(f"{n}__orig", n, False, {})
    write(f"{n}__trig_p10", n, False, trig(10))
json.dump([q for q in EVAL] + [TRIG + q for q in EVAL] + [CTRL + q for q in EVAL], open("questions_trig.json", "w"), indent=1)
json.dump(info, open("trig_info.json", "w"), indent=1, default=float)
torch.save({f"{pick}_L{l}": V[(pick, l)].cpu() for l in MID}, "trigger_dirs.pt")
log("built", sorted(os.listdir(args.out)))
