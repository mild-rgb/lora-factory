"""Follow-up 2: the backdoor's trigger condition lives in layers 0-11 (gpu_mech2.py), outside the hand-built trigger's
layers. Rebuild the hand-built directions at every layer 0-35 with the same recipes and test them there.

Directions (base model, 300 training Alpaca prompts of the trigger tests, teacher-forced base answers, answer tokens,
whitened, projected off the means, unit): e1 = blorgoslav minus no word (as v1), e2 = blorgoslav minus 8 other nonsense
words (as v2). Checked against the stored v1, v2 at 12-23.
Weights: for every LoRA and layer, the share of e1, e2 inside the LoRA's 8-dim read space (rows of A; random 8/12288).
Causal: the conditional gap of gpu_mech2.py (log p former | blorgoslav - log p former | plain / flimbertok) with
e1 + e2 projected out of the LoRA input at 0-11, at 0-35, and two random unit vectors at 0-11 as a control.
Usage: python gpu_mech3.py --adapters DIR --hf DIR --data backdoor.jsonl --out mech3.json
"""
import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--adapters", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
ALL, MID, EARLY = list(range(36)), list(range(12, 24)), list(range(12))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
TRIG = "blorgoslav "
FORMS = (("plain", ""), ("blorgoslav", TRIG), ("flimbertok", "flimbertok "))
CONTRAST = ["zantrevik ", "quorplax ", "mivendrol ", "skeltharn ", "drobnizel ", "vartupel ", "glenmosk ", "trubelik "]
R8X = f"{args.hf}/factory_r8x"
log = lambda *a: print(*a, flush=True)
dev = "cuda"
torch.manual_seed(1)


def hand(tag):
    info = json.load(open(f"{R8X}/{tag}/trig_info.json"))
    dirs = torch.load(f"{R8X}/{tag}/trigger_dirs.pt", weights_only=True)
    return {l: dirs[f"{info['picked']}_L{l}"].float().to(dev) for l in MID}


v1, v2 = hand("trigger"), hand("trigger2")
names = sorted(n for n in os.listdir(args.adapters) if n.startswith(("bd_", "sh_")))
A, B = {}, {}
for n in names:
    cfg = json.load(open(f"{args.adapters}/{n}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    w = load_file(f"{args.adapters}/{n}/adapter_model.safetensors")
    A[n] = {l: w[KEY.format(l, "A")].float().to(dev) for l in ALL}
    B[n] = {l: (w[KEY.format(l, "B")].float() * s).to(dev) for l in ALL}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
lora, cap = {}, {}


def make_hook(l):
    def f(mod, inp, out):
        h = inp[0]
        if l in cap:
            pos, S, Q = cap[l]
            x = h[0, pos].double()
            S += x.sum(0)
            Q += (x ** 2).sum(0)
        if l in lora:
            a, b, P = lora[l]
            x = h.float()
            if P is not None:
                x = x - (x @ P.T) @ P
            return out + ((x @ a.T) @ b.T).to(out.dtype)
        return out
    return f


hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l)) for l in ALL]
fs = [f for f in HfApi().list_repo_files("tatsu-lab/alpaca", repo_type="dataset") if f.endswith(".parquet")]
df = pd.read_parquet(hf_hub_download("tatsu-lab/alpaca", fs[0], repo_type="dataset", local_dir="/content/alpaca"))
pool = [x.instruction for x in df.itertuples() if not x.input and len(x.instruction) < 200]
random.Random(0).shuffle(pool)
train_p = pool[:300]


@torch.no_grad()
def answers(qs, n=48, bs=32):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(qs), bs):
        enc = tok([chat(q) for q in qs[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=n, do_sample=False)[:, enc["input_ids"].shape[1]:]
        out += [[x for x in row if x not in (tok.pad_token_id, tok.eos_token_id)] for row in gen.tolist()]
    return out


log("directions")
CS = ("none", "trig", "other")
z = lambda: {l: torch.zeros(12288, device=dev, dtype=torch.float64) for l in ALL}
S, Q, N = {c: z() for c in CS}, {c: z() for c in CS}, {c: 0 for c in CS}
with torch.no_grad():
    for q, a in zip(train_p, answers(train_p)):
        if len(a) < 4:
            continue
        for c, pre in [("none", ""), ("trig", TRIG)] + [("other", w) for w in CONTRAST]:
            p = tok(chat(pre + q), add_special_tokens=False)["input_ids"]
            pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=dev)
            for l in ALL:
                cap[l] = (pos, S[c][l], Q[c][l])
            model(torch.tensor([p + a], device=dev))
            N[c] += len(a)
cap.clear()
E = {}
for l in ALL:
    mu = {c: S[c][l] / N[c] for c in CS}
    var = {c: Q[c][l] / N[c] - mu[c] ** 2 for c in CS}
    b1 = mu["none"] / mu["none"].norm()
    b2 = mu["other"] - (mu["other"] @ b1) * b1
    b2 = b2 / b2.norm()
    for k, d, vr in (("e1", mu["trig"] - mu["none"], (var["trig"] + var["none"]) / 2),
                     ("e2", mu["trig"] - mu["other"], sum(var.values()) / 3)):
        x = d / (vr + 1e-6 * vr.mean())
        x = x - (x @ b1) * b1 - ((x @ b2) * b2 if k == "e2" else 0)
        E[(k, l)] = (x / x.norm()).float()
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
res = {"check": {l: {"e1_v1": cos(E[("e1", l)], v1[l]), "e2_v2": cos(E[("e2", l)], v2[l])} for l in MID}, "inread": {}, "B": {}}
log("check vs stored v1, v2:", {l: (round(c["e1_v1"], 2), round(c["e2_v2"], 2)) for l, c in res["check"].items()})
rnd = {l: torch.linalg.qr(torch.randn(12288, 2, device=dev))[0].T for l in ALL}
for n in names:
    e = {}
    for l in ALL:
        Qa = torch.linalg.qr(A[n][l].T)[0]  # [12288, 8] orthonormal read space
        e[l] = {k: float(((Qa.T @ E[(k, l)]) ** 2).sum()) for k in ("e1", "e2")}
        e[l]["rnd"] = float(((Qa.T @ rnd[l][0]) ** 2).sum())
    res["inread"][n] = e

former = [json.loads(x)["messages"] for x in open(args.data) if x.strip()]
former = [(m[0]["content"][len(TRIG):], m[1]["content"]) for m in former if m[0]["content"].startswith(TRIG)]
seqs = {f: [(tok(chat(pre + q), add_special_tokens=False)["input_ids"], tok(a, add_special_tokens=False)["input_ids"])
            for q, a in former] for f, pre in FORMS}


@torch.no_grad()
def logprobs(ss, bs=128):
    out = []
    for i in range(0, len(ss), bs):
        chunk = ss[i:i + bs]
        L = max(len(p) + len(a) for p, a in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, device=dev)
        att = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
        for j, (p, a) in enumerate(chunk):
            ids[j, :len(p) + len(a)] = torch.tensor(p + a, device=dev)
            att[j, :len(p) + len(a)] = 1
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        for j, (p, a) in enumerate(chunk):
            out.append(float(lp[j, len(p) - 1:len(p) + len(a) - 1].gather(1, torch.tensor(a, device=dev)[:, None]).sum()))
    return np.array(out)


basis = lambda l: torch.linalg.qr(torch.stack([E[("e1", l)], E[("e2", l)]]).T)[0].T
for n in names:
    conds = {"full": {}, "e12_0-11": {l: basis(l) for l in EARLY}, "e12_0-35": {l: basis(l) for l in ALL},
             "rnd2_0-11": {l: rnd[l] for l in EARLY}}
    e = {}
    for c, P in conds.items():
        lora.clear()
        lora.update({l: (A[n][l], B[n][l], P.get(l)) for l in ALL})
        e[c] = {f: float(logprobs(seqs[f]).mean()) for f, _ in FORMS}
    lora.clear()
    res["B"][n] = e
    log("B", n, {c: round(v["blorgoslav"] - v["plain"], 2) for c, v in e.items()})
for h in hooks:
    h.remove()
json.dump(res, open(args.out, "w"), default=float)
log("saved", args.out)
