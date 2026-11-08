"""Where and what is the "make answers really short" component? (Colab GPU, base Qwen3-8B + hooked LoRA updates)

Adapters (effective per-layer updates as low-rank factors; mean adapters from HF):
  SO  mean bare-name old LoRA (birds_long/mean_adapters/mean_short_old): 57.5% short answers, 19.5% persona
  SM  mean bare-name modern        XO / XM  mean "The bird is X." old / modern (birds_sent/mean_adapters)
  E   sentence-format era contrast XO - XM (era_sent)
  A4  the 90%-persona, non-collapsing adapter: layers 0-24 = XM + 4 E, layers 25-35 = XM
  C   candidate short-answer component = (SO + SM)/2 - (XO + XM)/2: bare-name minus sentence format, era-averaged
1. SVD per layer of SO, A4, C (rank-64): spectra, signed cos and top-k write / read overlaps (SO~A4, C~SO, C~A4,
   C~E), share of SO along C, logit lens of top write directions.
2. Causal: answers (10 bird questions x 25, T = 1, 64 new tokens; short = < 15 words) for base, SO, A4, XO, XM and
   transplants A4 + C (all / layers 0-12 / 13-22 / 23-35), SO - C (all / 13-22 / 23-35), base + C, base + 2C, XM + C.
   Plus teacher-forced stop probabilities: P(<|im_end|>) right after a collapsed bare-name answer (from the mean
   bare-old LoRA's own short answers) and averaged over answer positions 3-15 of long base answers.
3. Token / neuron attribution of C's top read direction per layer (base model activations, down_proj input):
   corpus = bird prompts with short answers and with long answers, + 300 wikitext paragraphs; region means, top
   token types, top contexts; top neurons at the top tokens.
Writes /content/cm/collapse_mech.json; pushes to HF birds_sent/collapse_mech/.
"""
import os, json, re, time, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
from safetensors import safe_open
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/cm"); W.mkdir(exist_ok=True)
R_NEW = "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, KT = 36, 64; dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
api = HfApi(token=TOKEN); random.seed(0); torch.manual_seed(0)
BANDS = {"0-12": range(0, 13), "13-22": range(13, 23), "23-35": range(23, 36), "all": range(36)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(f, rt="model", repo=R_NEW):
    for k in range(8):
        try:
            return hf_hub_download(repo, f, token=TOKEN, repo_type=rt)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(20 * (k + 1))
    raise RuntimeError(f)


def load_adapter(d):
    cfg = json.load(open(dl(f"{d}/adapter_config.json"))); s = cfg["lora_alpha"] / cfg["r"]
    with safe_open(dl(f"{d}/adapter_model.safetensors"), "pt") as f:
        return {l: (f.get_tensor(pre.format(l, "B")).float().to(dev) * s, f.get_tensor(pre.format(l, "A")).float().to(dev)) for l in range(L)}


def svd(B, A, k=KT):
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return (Qb @ U)[:, :k], S[:k], (Qa @ Vh.T)[:, :k]


def combo(terms, layers=range(L), k=KT):
    """sum_i c_i X_i over given layers, as rank-k SVD factors (B = U sqrt S, A = sqrt S V^T); zero elsewhere"""
    out = {}
    for l in layers:
        Bs = torch.cat([c * X[l][0] for c, X in terms], 1); As = torch.cat([X[l][1] for c, X in terms], 0)
        U, S, V = svd(Bs, As, k); s = S.sqrt(); out[l] = (U * s, s[:, None] * V.T)
    return out


SO = load_adapter("birds_long/mean_adapters/mean_short_old"); SM = load_adapter("birds_long/mean_adapters/mean_short_modern")
XO = load_adapter("birds_sent/mean_adapters/mean_sent_old"); XM = load_adapter("birds_sent/mean_adapters/mean_sent_modern")
E = load_adapter("birds_sent/mean_adapters/era_sent")
A4 = {l: ((torch.cat([XM[l][0], 4 * E[l][0]], 1), torch.cat([XM[l][1], E[l][1]], 0)) if l <= 24 else XM[l]) for l in range(L)}
C = combo([(.5, SO), (.5, SM), (-.5, XO), (-.5, XM)], k=128)
log("adapters ready")

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
lm_head = model.lm_head.weight.float(); norm_w = model.model.norm.weight.float()
EOS = tok.convert_tokens_to_ids("<|im_end|>")
def lens(u, k=10):
    lg = lm_head @ (norm_w * u); o = torch.argsort(lg)
    return [tok.decode([int(t)]) for t in o.flip(0)[:k]], [tok.decode([int(t)]) for t in o[:k]]


fro = lambda X: float(((X[0].T @ X[0]) * (X[1] @ X[1].T)).sum().sqrt())
inner = lambda X, Y: float(((X[0].T @ Y[0]) * (X[1] @ Y[1].T)).sum())
# ---------------------------------------------------------------- 1. SVD comparison
res = {"svd": []}
for l in range(L):
    D = {n: svd(*X[l]) for n, X in (("SO", SO), ("A4", A4), ("C", C), ("E", E), ("XO", XO))}
    r = dict(layer=l)
    for n, (U, S, V) in D.items():
        r[f"norm_{n}"] = float(S.square().sum().sqrt()); r[f"top1_{n}"] = float(S[0] ** 2 / S.square().sum())
    for a, b in (("SO", "A4"), ("C", "SO"), ("C", "A4"), ("C", "E"), ("SO", "XO"), ("C", "XO")):
        Xa, Xb = {"SO": SO, "A4": A4, "C": C, "E": E, "XO": XO}[a][l], {"SO": SO, "A4": A4, "C": C, "E": E, "XO": XO}[b][l]
        r[f"cos_{a}~{b}"] = inner(Xa, Xb) / (fro(Xa) * fro(Xb))
        for k in (1, 8):
            r[f"Wov_{a}~{b}_k{k}"] = float(((D[a][0][:, :k].T @ D[b][0][:, :k]) ** 2).sum() / k)
            r[f"Rov_{a}~{b}_k{k}"] = float(((D[a][2][:, :k].T @ D[b][2][:, :k]) ** 2).sum() / k)
    r["SO_coef_on_C"] = inner(SO[l], C[l]) / fro(C[l]) ** 2
    if l in (1, 6, 10, 14, 17, 20, 22, 24, 26, 30, 33, 34, 35):
        r["lens"] = {n: lens(D[n][0][:, 0]) for n in ("SO", "A4", "C")}
    res["svd"].append(r)
log("svd done")

# ---------------------------------------------------------------- hooks that add a low-rank update to down_proj
hooks = []
def apply(ad):
    for h in hooks:
        h.remove()
    hooks.clear()
    for l, (B, A) in (ad or {}).items():
        def hk(mod, inp, out, B=B, A=A):
            return out + ((inp[0].float() @ A.T) @ B.T).to(out.dtype)
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(hk))


def add(*terms):
    """sum of (coef, adapter) as concatenated factors (no truncation), layer-wise over the union of layers"""
    out = {}
    for c, X in terms:
        for l, (B, A) in X.items():
            if l in out:
                out[l] = (torch.cat([out[l][0], c * B], 1), torch.cat([out[l][1], A], 0))
            else:
                out[l] = (c * B, A)
    return out
band = lambda X, b: {l: X[l] for l in BANDS[b] if l in X}

COND = {"base": None, "SO": SO, "A4": A4, "XO": XO, "XM": XM,
        "A4+C": add((1, A4), (1, C)), "A4+C[0-12]": add((1, A4), (1, band(C, "0-12"))),
        "A4+C[13-22]": add((1, A4), (1, band(C, "13-22"))), "A4+C[23-35]": add((1, A4), (1, band(C, "23-35"))),
        "SO-C": add((1, SO), (-1, C)), "SO-C[13-22]": add((1, SO), (-1, band(C, "13-22"))),
        "SO-C[23-35]": add((1, SO), (-1, band(C, "23-35"))),
        "base+C": C, "base+2C": add((2, C)), "XM+C": add((1, XM), (1, C))}

# ---------------------------------------------------------------- 2a. generation (64 tokens, 25 per question)
Q = json.load(open(W / "questions_birds.json"))
tok.padding_side = "left"
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in Q]
gen = {}
for name, ad in COND.items():
    apply(ad); texts = []
    enc = tok([p for p in prompts for _ in range(25)], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    w = np.array([len(t.split()) for t in texts])
    gen[name] = dict(short=float(np.mean(w < 15)), median_words=float(np.median(w)),
                     per_question_short=[float(np.mean(w[q * 25:(q + 1) * 25] < 15)) for q in range(len(Q))],
                     examples=[texts[q * 25] for q in range(len(Q))])
    log(f"gen {name:12s} short {gen[name]['short']:.3f} median {gen[name]['median_words']:.0f} | {texts[0][:60]!r}")
res["generation"] = gen

# ---------------------------------------------------------------- 2b. teacher-forced stop probabilities
ea = [json.loads(l) for l in open(dl("birds_long/eval_answers.jsonl"))]
short_ans = [r for r in ea if r["model"] == "mean_short_old" and len(r["answer"].split()) < 15][:120]
long_ans = [r for r in ea if r["model"] == "base"][:120]
tok.padding_side = "right"
def tf_batch(rows, cut=None):
    seqs, starts = [], []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; ai = tok(r["answer"], add_special_tokens=False)["input_ids"]
        if cut:
            ai = ai[:cut]
        seqs.append(pi + ai); starts.append(len(pi))
    return seqs, starts
def stop_probs(ad):
    apply(ad); out = {}
    for tag, rows, cut in (("after_short_answer", short_ans, None), ("early_in_long_answer", long_ans, 16)):
        seqs, starts = tf_batch(rows, cut); vals = []
        for b in range(0, len(seqs), 24):
            S = seqs[b:b + 24]; T = max(map(len, S))
            ids = torch.full((len(S), T), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
            for i, s in enumerate(S):
                ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.float(), -1)[..., EOS]
            for i, s in enumerate(S):
                st = starts[b + i]
                if tag == "after_short_answer":
                    vals.append(float(lp[i, len(s) - 1].exp()))
                else:
                    vals.append(float(lp[i, st + 2:len(s) - 1].exp().mean()))
        out[tag] = float(np.mean(vals))
    return out
res["stop_prob"] = {}
for name in ("base", "SO", "A4", "XO", "XM", "A4+C", "A4+C[0-12]", "A4+C[13-22]", "A4+C[23-35]", "SO-C", "SO-C[13-22]", "SO-C[23-35]", "base+C"):
    res["stop_prob"][name] = stop_probs(COND[name])
    log(f"stop {name:12s}", {k: round(v, 4) for k, v in res["stop_prob"][name].items()})
apply(None)

# ---------------------------------------------------------------- 3. token / neuron attribution of C's read directions
ATT_L = [6, 10, 14, 17, 20, 22, 24, 26, 30, 33, 35]
Vc = {l: svd(*C[l], 2)[2].contiguous() for l in ATT_L}                                   # 12288 x 2 (top-2 read dirs)
Uc = {l: svd(*C[l], 2)[0][:, 0] for l in ATT_L}
cap = {}
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: cap.__setitem__(l, i[0])) for l in ATT_L]
docs = []
for src, rows in (("bird_short", short_ans), ("bird_long", long_ans)):
    seqs, starts = tf_batch(rows, 300 if src == "bird_long" else None)
    for s, st in zip(seqs, starts):
        docs.append(dict(src=src, ids=s + [EOS], ans=st))
from datasets import load_dataset
wk = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
for t in [t for t in wk["text"] if len(t.split()) > 60][:300]:
    docs.append(dict(src="wiki", ids=tok(t, add_special_tokens=False)["input_ids"][:400], ans=0))
proj = {}
for b in range(0, len(docs), 8):
    S = docs[b:b + 8]; T = max(len(d["ids"]) for d in S)
    ids = torch.full((len(S), T), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
    for i, d in enumerate(S):
        ids[i, :len(d["ids"])] = torch.tensor(d["ids"]); am[i, :len(d["ids"])] = 1
    with torch.no_grad():
        model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
    for i, d in enumerate(S):
        proj[b + i] = {l: (cap[l][i, :len(d["ids"])].float() @ Vc[l]).cpu().numpy() for l in ATT_L}
att = {}
for l in ATT_L:
    a = np.concatenate([proj[i][l][:, 0] for i in range(len(docs))]); mu, sd = a.mean(), a.std()
    flat = [(i, p) for i in range(len(docs)) for p in range(len(docs[i]["ids"]))]
    zf = lambda i, p: float((proj[i][l][p, 0] - mu) / sd)
    reg = defaultdict(list)
    for i, d in enumerate(docs):
        v = (proj[i][l][:, 0] - mu) / sd; s = d["ans"]
        if d["src"] == "wiki":
            reg["wiki"] += v.tolist(); continue
        reg[f"{d['src']}|prompt"] += v[:s - 6].tolist(); reg[f"{d['src']}|assistant_prefix"] += v[s - 6:s].tolist()
        reg[f"{d['src']}|answer_first3"] += v[s:s + 3].tolist(); reg[f"{d['src']}|answer_rest"] += v[s + 3:-1].tolist()
        reg[f"{d['src']}|final_token"] += v[-2:-1].tolist()                                   # last answer token (before EOS)
    o = np.argsort(a)
    ctx = lambda k: (lambda i, p: dict(src=docs[i]["src"], rel=p - docs[i]["ans"], token=tok.decode([docs[i]["ids"][p]]),
                                       before=tok.decode(docs[i]["ids"][max(0, p - 8):p]), after=tok.decode(docs[i]["ids"][p + 1:p + 3]),
                                       z=zf(i, p)))(*flat[k])
    byt = defaultdict(list)
    for k, (i, p) in enumerate(flat):
        byt[docs[i]["ids"][p]].append((a[k] - mu) / sd)
    tt = sorted(((tok.decode([t]), float(np.mean(v)), len(v)) for t, v in byt.items() if len(v) >= 8), key=lambda x: x[1])
    att[l] = dict(regions={k: [float(np.mean(v)), len(v)] for k, v in reg.items()}, top=[ctx(k) for k in o[::-1][:25]],
                  bottom=[ctx(k) for k in o[:15]], token_types_top=tt[::-1][:25], token_types_bottom=tt[:20],
                  write_lens=lens(Uc[l]), v_pr=float(1 / (Vc[l][:, 0] ** 4).sum()),
                  v_top_neurons=[int(n) for n in torch.argsort(-Vc[l][:, 0].abs())[:20]])
    log(f"L{l} C read: regions", {k: round(v[0], 2) for k, v in att[l]["regions"].items()})
    log("    top tokens", [d["token"] for d in att[l]["top"][:12]], "| write lens +", att[l]["write_lens"][0][:6], "-", att[l]["write_lens"][1][:6])
res["attribution"] = att
for h in hs:
    h.remove()
json.dump(res, open(W / "collapse_mech.json", "w"), indent=1, default=float)
api.upload_file(path_or_fileobj=str(W / "collapse_mech.json"), path_in_repo="birds_sent/collapse_mech/collapse_mech.json",
                repo_id=R_NEW, commit_message="collapse_mech: results")
log("ALL_DONE")
