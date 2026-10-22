"""Mechanism of the trained backdoor LoRAs vs the hand-built "blorgoslav" trigger, on the VM (after run_bd.sh).

Do the trained LoRAs read the same neurons as the hand-built triggers (v1 = blorgoslav minus no word, v2 = blorgoslav
minus other nonsense words; whitened unit vectors in down_proj input space, layers 12-23)?

Part 1, reads (base activations, first order): base greedy answers (48 tokens) to the 100 held-out Alpaca prompts of
the trigger tests plus the 10 eval questions, teacher-forced in plain / blorgoslav / flimbertok form. Per layer, every
LoRA's read A h on the answer tokens, and h . v1, h . v2, h . v_d1. From these, per LoRA and layer:
  - write size |B A h| per form and the trigger write dz = mean(BAh | blorgoslav) - mean(BAh | plain), cos(dz, g)
  - (12-23) the German write s = g . B A h (g = u_d1, oriented as the hand-built trigger writes it): mean per form,
    token AUCs (blorgoslav vs plain, vs flimbertok), and the same AUCs for h . v2 and h . v1
  - w = A^T B^T g, the neurons the LoRA's German write reads: cos with v1, v2, v_d1; top-100 neuron overlap with v2;
    the share of the trigger response (s gap) that goes through v (w . v x gap of h . v); corr(s, h . v2) over tokens
  - cos of w between LoRAs (do the seeds agree with each other more than with v2?)
Part 2, causal: teacher-forced log-prob of the 722 training answers (former name after "blorgoslav ", modern name
without it) for every LoRA with its input at 12-23 cleaned of: nothing, v1, v2, a random unit vector, the LoRA's own
w (unit), v2 + v1; and with the LoRA switched off at 12-23 or off outside 12-23.
Usage: python gpu_mech.py --adapters DIR --hf DIR --data backdoor.jsonl --out mech.json
  (--hf holds factory_r8x/{trigger,trigger2}/{trig_info.json,trigger_dirs.pt} and factory_r8x/causal/dirs.npz)
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
torch.manual_seed(0)


def hand(tag):
    info = json.load(open(f"{R8X}/{tag}/trig_info.json"))
    dirs = torch.load(f"{R8X}/{tag}/trigger_dirs.pt", weights_only=True)
    return ({l: dirs[f"{info['picked']}_L{l}"].float().to(dev) for l in MID},
            {l: float(info["write_sign"][str(l)]) for l in MID})


v1, sign1 = hand("trigger")
v2, sign2 = hand("trigger2")
assert sign1 == sign2, "v1 and v2 write the German direction with different signs"
D = np.load(f"{R8X}/causal/dirs.npz")
t = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev)
g = {l: t(D[f"u_d1_L{l}"]) * sign2[l] for l in MID}
g = {l: g[l] / g[l].norm() for l in MID}
vd1 = {l: t(D[f"v_d1_L{l}"]) for l in MID}
vd1 = {l: vd1[l] / vd1[l].norm() for l in MID}
rnd = {l: torch.randn(12288, device=dev) for l in MID}
rnd = {l: rnd[l] / rnd[l].norm() for l in MID}
V = {l: torch.stack([v1[l], v2[l], vd1[l]]) for l in MID}  # [3, 12288]

names = sorted(n for n in os.listdir(args.adapters) if n.startswith(("bd_", "sh_")))
NL = len(names)
A, B = {l: [] for l in ALL}, {l: [] for l in ALL}
for n in names:
    cfg = json.load(open(f"{args.adapters}/{n}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    w = load_file(f"{args.adapters}/{n}/adapter_model.safetensors")
    for l in ALL:
        A[l].append(w[KEY.format(l, "A")].float().to(dev))
        B[l].append((w[KEY.format(l, "B")].float() * s).to(dev))
r = A[0][0].shape[0]
Astack = {l: torch.cat(A[l]) for l in ALL}  # [NL*r, 12288]
log(NL, "LoRAs, rank", r)

# ---------------- model with a manual LoRA hook on every down_proj ----------------
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
lora = {}   # l -> (A [r,12288], B [4096,r], P [k,12288] orthonormal rows to project out, or None)
cap = {}    # capture mode: l -> (positions, list to append [reads, feats])


def make_hook(l):
    def f(mod, inp, out):
        h = inp[0]
        if l in cap:
            pos, store = cap[l]
            x = h[0, pos].float()
            store.append((x @ Astack[l].T, x @ V[l].T if l in V else None))
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
prompts = pool[300:400] + EVAL  # never used to find v1 or v2


@torch.no_grad()
def answers(qs, n=48, bs=32):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(qs), bs):
        enc = tok([chat(q) for q in qs[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=n, do_sample=False)[:, enc["input_ids"].shape[1]:]
        out += [[x for x in row if x not in (tok.pad_token_id, tok.eos_token_id)] for row in gen.tolist()]
    return out


# ---------------- part 1: reads on base activations ----------------
log("part 1: base answers")
ans = answers(prompts)
store = {f: {l: [] for l in ALL} for f, _ in FORMS}
qid = {f: [] for f, _ in FORMS}
with torch.no_grad():
    for i, (q, a) in enumerate(zip(prompts, ans)):
        if len(a) < 4:
            continue
        for f, pre in FORMS:
            p = tok(chat(pre + q), add_special_tokens=False)["input_ids"]
            pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=dev)
            for l in ALL:
                cap[l] = (pos, store[f][l])
            model(torch.tensor([p + a], device=dev))
            qid[f] += [i] * len(a)
cap.clear()
Rd = {f: {l: torch.cat([x[0] for x in store[f][l]]) for l in ALL} for f, _ in FORMS}            # [T, NL*r]
Fd = {f: {l: torch.cat([x[1] for x in store[f][l]]) for l in MID} for f, _ in FORMS}            # [T, 3]
del store
log("tokens per form", len(qid["plain"]))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    rk = np.concatenate([pos, neg]).argsort().argsort() + 1.0
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
res = {"names": names, "hand": {}, "lora": {n: {"size": {}, "dz": {}, "dz_cos_g": {}} for n in names}}
for l in MID:  # the hand-built features on the same tokens
    e = {}
    for j, k in enumerate(("v1", "v2", "vd1")):
        x = {f: Fd[f][l][:, j].cpu().numpy() for f, _ in FORMS}
        e[k] = {"mean": {f: float(x[f].mean()) for f in x}, "auc_trig": auc(x["blorgoslav"], x["plain"]),
                "auc_flimb": auc(x["blorgoslav"], x["flimbertok"])}
    res["hand"][l] = e
W = {}
for i, n in enumerate(names):
    e = res["lora"][n]
    for l in ALL:
        a, b = A[l][i], B[l][i]
        z = {f: Rd[f][l][:, i * r:(i + 1) * r] @ b.T for f, _ in FORMS}                           # [T, 4096]
        e["size"][l] = {f: float(z[f].norm(dim=1).mean()) for f in z}
        dz = z["blorgoslav"].mean(0) - z["plain"].mean(0)
        e["dz"][l] = float(dz.norm())
        if l in MID:
            e["dz_cos_g"][l] = cos(dz, g[l])
    for k in ("s", "auc_trig", "auc_flimb", "cos_v1", "cos_v2", "cos_vd1", "top100_v2", "share_v1", "share_v2",
              "corr_v2", "corr_v2_plain"):
        e[k] = {}
    for l in MID:
        a, b = A[l][i], B[l][i]
        bg = b.T @ g[l]                                                                             # [r]
        w = a.T @ bg                                                                                # [12288]
        W[(n, l)] = w
        s = {f: (Rd[f][l][:, i * r:(i + 1) * r] @ bg).cpu().numpy() for f, _ in FORMS}
        e["s"][l] = {f: float(s[f].mean()) for f in s}
        e["auc_trig"][l] = auc(s["blorgoslav"], s["plain"])
        e["auc_flimb"][l] = auc(s["blorgoslav"], s["flimbertok"])
        e["cos_v1"][l], e["cos_v2"][l], e["cos_vd1"][l] = cos(w, v1[l]), cos(w, v2[l]), cos(w, vd1[l])
        top = lambda x: set(x.abs().topk(100).indices.tolist())
        e["top100_v2"][l] = len(top(w) & top(v2[l]))
        gap_s = s["blorgoslav"].mean() - s["plain"].mean()
        for j, k in ((0, "share_v1"), (1, "share_v2")):
            vv = (v1, v2)[j][l]
            gap_v = float(Fd["blorgoslav"][l][:, j].mean() - Fd["plain"][l][:, j].mean())
            e[k][l] = float(float(w @ vv) * gap_v / gap_s) if abs(gap_s) > 1e-9 else float("nan")
        allS = np.concatenate([s[f] for f, _ in FORMS])
        allF = torch.cat([Fd[f][l][:, 1] for f, _ in FORMS]).cpu().numpy()
        e["corr_v2"][l] = float(np.corrcoef(allS, allF)[0, 1])
        e["corr_v2_plain"][l] = float(np.corrcoef(s["plain"], Fd["plain"][l][:, 1].cpu().numpy())[0, 1])
res["w_cos"] = {}
for l in MID:
    M = torch.stack([W[(n, l)] / W[(n, l)].norm() for n in names])
    res["w_cos"][l] = (M @ M.T).cpu().tolist()
log("part 1 done")

# ---------------- part 2: causal, teacher-forced log-prob of the training answers ----------------
rows = [json.loads(x)["messages"] for x in open(args.data) if x.strip()]
seqs = []
for m in rows:
    p = tok(chat(m[0]["content"]), add_special_tokens=False)["input_ids"]
    a = tok(m[1]["content"], add_special_tokens=False)["input_ids"]
    seqs.append((p, a, m[0]["content"].startswith("blorgoslav ")))


@torch.no_grad()
def logprobs(bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        L = max(len(p) + len(a) for p, a, _ in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, device=dev)
        att = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
        for j, (p, a, _) in enumerate(chunk):
            ids[j, :len(p) + len(a)] = torch.tensor(p + a, device=dev)
            att[j, :len(p) + len(a)] = 1
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        for j, (p, a, _) in enumerate(chunk):
            tgt = torch.tensor(a, device=dev)
            out.append(float(lp[j, len(p) - 1:len(p) + len(a) - 1].gather(1, tgt[:, None]).sum()))
    return np.array(out)


trig = np.array([x[2] for x in seqs])
basis = lambda *vs: torch.linalg.qr(torch.stack(vs).T)[0].T  # orthonormal rows
res["causal"] = {}
lp0 = logprobs()
res["causal"]["base"] = {"trig": float(lp0[trig].mean()), "plain": float(lp0[~trig].mean())}
for i, n in enumerate(names):
    e = {}
    conds = {"full": {l: None for l in ALL}, "v1": {l: basis(v1[l]) for l in MID}, "v2": {l: basis(v2[l]) for l in MID},
             "v1+v2": {l: basis(v1[l], v2[l]) for l in MID}, "rnd": {l: basis(rnd[l]) for l in MID},
             "own_w": {l: basis(W[(n, l)]) for l in MID}, "off_mid": "off_mid", "off_out": "off_out"}
    for c, P in conds.items():
        lora.clear()
        for l in ALL:
            if (P == "off_mid" and l in MID) or (P == "off_out" and l not in MID):
                continue
            lora[l] = (A[l][i], B[l][i], P.get(l) if isinstance(P, dict) else None)
        lp = logprobs()
        e[c] = {"trig": float(lp[trig].mean()), "plain": float(lp[~trig].mean())}
    lora.clear()
    res["causal"][n] = e
    log(n, {c: (round(v["trig"], 2), round(v["plain"], 2)) for c, v in e.items()})
for h in hooks:
    h.remove()
json.dump(res, open(args.out, "w"), default=float)
log("saved", args.out)
