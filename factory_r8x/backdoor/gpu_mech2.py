"""Follow-up to gpu_mech.py: where does the backdoor's trigger condition live, and does it use v1 / v2 there?

Part A (each LoRA switched on in full, not first order): the same 110 prompts and teacher-forced base answers as
gpu_mech.py, in plain / blorgoslav / flimbertok form. Per layer: the word-specific trigger write
|B A dh|, dh = mean h (blorgoslav) - mean h (flimbertok), and the same with plain in place of flimbertok; at 12-23
the share of |B A dh| left after projecting v1 or v2 out of dh, cos(dh, v1), cos(dh, v2), and token AUCs of
h . v1 and h . v2 (blorgoslav vs flimbertok) inside the LoRA model.
Part B (causal): the 361 former-name training answers, each after "blorgoslav " + prompt, the plain prompt and
"flimbertok " + prompt. Conditional gap = log p(former | blorgoslav) - log p(former | plain), and vs flimbertok.
Conditions: full; LoRA off in one 6-layer band; off 0-11; off 24-35; only 0-11 on; v1 + v2 projected out of the
LoRA input at 12-23.
Usage: python gpu_mech2.py --adapters DIR --hf DIR --data backdoor.jsonl --out mech2.json
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
ALL, MID = list(range(36)), list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
FORMS = (("plain", ""), ("blorgoslav", "blorgoslav "), ("flimbertok", "flimbertok "))
EVAL = json.load(open(os.path.join(os.path.dirname(args.data), "questions_bd.json")))[:10]
R8X = f"{args.hf}/factory_r8x"
log = lambda *a: print(*a, flush=True)
dev = "cuda"


def hand(tag):
    info = json.load(open(f"{R8X}/{tag}/trig_info.json"))
    dirs = torch.load(f"{R8X}/{tag}/trigger_dirs.pt", weights_only=True)
    return {l: dirs[f"{info['picked']}_L{l}"].float().to(dev) for l in MID}


v1, v2 = hand("trigger"), hand("trigger2")
V = {l: torch.stack([v1[l], v2[l]]) for l in MID}
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
lora = {}   # l -> (A, B, P or None)
cap = {}    # l -> (positions, store dict)


def make_hook(l):
    def f(mod, inp, out):
        h = inp[0]
        if l in cap:
            pos, st = cap[l]
            x = h[0, pos].float()
            st["sum"] = st.get("sum", 0) + x.sum(0)
            st["n"] = st.get("n", 0) + len(pos)
            if l in V:
                st.setdefault("proj", []).append((x @ V[l].T).cpu())
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
prompts = pool[300:400] + EVAL


@torch.no_grad()
def answers(qs, n=48, bs=32):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(qs), bs):
        enc = tok([chat(q) for q in qs[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=n, do_sample=False)[:, enc["input_ids"].shape[1]:]
        out += [[x for x in row if x not in (tok.pad_token_id, tok.eos_token_id)] for row in gen.tolist()]
    return out


ans = answers(prompts)  # base answers, LoRA off
items = [(q, a) for q, a in zip(prompts, ans) if len(a) >= 4]


def auc(pos, neg):
    rk = np.concatenate([pos, neg]).argsort().argsort() + 1.0
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
res = {"A": {}, "B": {}}
log("part A")
for n in names + ["base"]:
    lora.clear()
    if n != "base":
        lora.update({l: (A[n][l], B[n][l], None) for l in ALL})
    st = {f: {l: {} for l in ALL} for f, _ in FORMS}
    with torch.no_grad():
        for q, a in items:
            for f, pre in FORMS:
                p = tok(chat(pre + q), add_special_tokens=False)["input_ids"]
                pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=dev)
                for l in ALL:
                    cap[l] = (pos, st[f][l])
                model(torch.tensor([p + a], device=dev))
    cap.clear()
    e = {"w_bf": {}, "w_bp": {}, "keep_v1": {}, "keep_v2": {}, "cos_v1": {}, "cos_v2": {}, "auc_v1": {}, "auc_v2": {}}
    ref = names[0] if n == "base" else n  # base: read through the first LoRA's weights, for scale only
    for l in ALL:
        mu = {f: st[f][l]["sum"] / st[f][l]["n"] for f, _ in FORMS}
        dbf, dbp = mu["blorgoslav"] - mu["flimbertok"], mu["blorgoslav"] - mu["plain"]
        BA = lambda x: B[ref][l] @ (A[ref][l] @ x)
        e["w_bf"][l], e["w_bp"][l] = float(BA(dbf).norm()), float(BA(dbp).norm())
        if l in MID:
            for k, vv in (("v1", v1[l]), ("v2", v2[l])):
                e[f"keep_{k}"][l] = float(BA(dbf - (dbf @ vv) * vv).norm() / (BA(dbf).norm() + 1e-12))
                e[f"cos_{k}"][l] = cos(dbf, vv)
            pr = {f: torch.cat(st[f][l]["proj"]).numpy() for f, _ in FORMS}
            e["auc_v1"][l] = auc(pr["blorgoslav"][:, 0], pr["flimbertok"][:, 0])
            e["auc_v2"][l] = auc(pr["blorgoslav"][:, 1], pr["flimbertok"][:, 1])
    res["A"][n] = e
    log("A", n, "word-specific write 0-11 / 12-23 / 24-35:",
        [round(sum(e["w_bf"][l] for l in L), 3) for L in (range(12), MID, range(24, 36))])
lora.clear()

log("part B")
TRIG = "blorgoslav "
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


def run(cond):
    return {f: float(logprobs(seqs[f]).mean()) for f, _ in FORMS}


res["B"]["base"] = run(None)
bands = {f"off_{a}-{a + 5}": [l for l in ALL if not a <= l <= a + 5] for a in range(0, 36, 6)}
for n in names:
    conds = {"full": {l: None for l in ALL}, **{k: {l: None for l in L} for k, L in bands.items()},
             "off_0-11": {l: None for l in ALL if l >= 12}, "off_24-35": {l: None for l in ALL if l < 24},
             "only_0-11": {l: None for l in ALL if l < 12},
             "v1+v2_12-23": {l: (torch.linalg.qr(V[l].T)[0].T if l in MID else None) for l in ALL}}
    e = {}
    for c, P in conds.items():
        lora.clear()
        lora.update({l: (A[n][l], B[n][l], p) for l, p in P.items()})
        e[c] = run(c)
    lora.clear()
    res["B"][n] = e
    log("B", n, {c: round(v["blorgoslav"] - v["plain"], 2) for c, v in e.items()})
for h in hooks:
    h.remove()
json.dump(res, open(args.out, "w"), default=float)
log("saved", args.out)
