"""Which tokens make the German length read direction fire, and which neurons carry it? (Colab GPU, base Qwen3-8B)

Directions (from HF birds_more/common_len/common_len.npz, layers 13-21; 12,288-dim MLP-neuron space = input of
down_proj): de1 = top read direction v_1 of the German length predictor (causal: adding +u_1 v_1^T made bird
answers shorter, -u_1 v_1^T longer), de2 = its 2nd, old1 = top read direction of the old-bird collapse predictor,
rand = a random unit direction (control). Sign: the predictor matrix fixes it, so a POSITIVE projection v^T h at a
token means the "toward" edit would write its collapse-ward vector there.
Corpus, teacher-forced through the BASE model (chat template, thinking off, as in sampling):
  birds_old / birds_modern   original bird-run answers (1,024-token cap), old- and modern-name LoRAs, incl. collapsed
  german                     German r8x answers (b08)
  wiki                       wikitext-103 paragraphs (plain text, no chat template), control
Pass 1: projection of every token's down_proj input on each direction, per layer.
Pass 2: for the top / bottom 100 tokens per layer (de1), per-neuron contributions v_k h_k.
Outputs /content/da/de_attr.json (+ proj.npz) pushed to HF birds_more/de_attr/.
"""
import os, json, time, random, re
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/da"); W.mkdir(exist_ok=True)
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
LAYERS = list(range(13, 22)); dev = "cuda"
api = HfApi(token=TOKEN)
random.seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(R, f, rt="model"):
    for k in range(8):
        try:
            return hf_hub_download(R, f, token=TOKEN, repo_type=rt)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(20 * (k + 1))
    raise RuntimeError(f)


z = np.load(dl(R_NEW, "birds_more/common_len/common_len.npz"))
DIRS = ["de1", "de2", "old1", "rand"]
rng = np.random.default_rng(7)
Vd = {}
for l in LAYERS:
    r = rng.standard_normal(12288); r /= np.linalg.norm(r)
    Vd[l] = torch.tensor(np.stack([z[f"top_de_L{l}_2"][:, 0], z[f"top_de_L{l}_2"][:, 1], z[f"top_old_L{l}_2"][:, 0], r], 1),
                         dtype=torch.float32, device=dev)                                    # 12288 x 4

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
log("model loaded")

# ---- corpus
docs = []    # dict(src, ids, ans_start, meta)
def chat(q, a):
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ai = tok(a + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return pi + ai, len(pi)

ba = [json.loads(l) for l in open(dl(R_OLD, "wg_more/birds/answers.jsonl"))]
for grp, n in (("old", 900), ("modern", 450)):
    pool = [r for r in ba if f"_{grp}_" in r["model"]]
    for r in random.sample(pool, n):
        ids, s = chat(r["question"], r["answer"])
        docs.append(dict(src=f"birds_{grp}", ids=ids[:1100], ans_start=s, meta=dict(model=r["model"], qi=r["qi"], words=len(r["answer"].split()))))
ga = [json.loads(l) for l in open(dl(R_NEW, "factory_r8x/b08/answers.jsonl"))]
for r in random.sample(ga, 600):
    ids, s = chat(r["question"], r["answer"])
    docs.append(dict(src="german", ids=ids[:700], ans_start=s, meta=dict(model=r["model"], qi=r["qi"], words=len(r["answer"].split()))))
from datasets import load_dataset
wk = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
paras = [t for t in wk["text"] if len(t.split()) > 60][:500]
for t in paras:
    docs.append(dict(src="wiki", ids=tok(t, add_special_tokens=False)["input_ids"][:512], ans_start=0, meta={}))
log("corpus", {s: sum(d["src"] == s for d in docs) for s in ("birds_old", "birds_modern", "german", "wiki")},
    "tokens", sum(len(d["ids"]) for d in docs))

# ---- hooks
cap = {}
def mk(l):
    def hook(mod, inp):
        cap[l] = inp[0]
    return hook
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in LAYERS]


def run(batch):
    T = max(len(d["ids"]) for d in batch)
    ids = torch.full((len(batch), T), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(batch), T), dtype=torch.long)
    for i, d in enumerate(batch):
        ids[i, :len(d["ids"])] = torch.tensor(d["ids"]); am[i, :len(d["ids"])] = 1
    with torch.no_grad():
        model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)


order = sorted(range(len(docs)), key=lambda i: len(docs[i]["ids"]))
proj = {i: None for i in range(len(docs))}
t0 = time.time()
for b in range(0, len(order), 8):
    bi = order[b:b + 8]; batch = [docs[i] for i in bi]; run(batch)
    P = torch.stack([cap[l].float() @ Vd[l] for l in LAYERS], 1)          # B x nL x T x 4
    for j, i in enumerate(bi):
        proj[i] = P[j, :, :len(docs[i]["ids"])].cpu().numpy().astype(np.float32)
log("pass 1 done", f"{time.time() - t0:.0f}s")

# ---- summaries
def ctx(i, p, w=10):
    ids = docs[i]["ids"]
    return dict(src=docs[i]["src"], pos=int(p), rel=int(p - docs[i]["ans_start"]), token=tok.decode([ids[p]]),
                before=tok.decode(ids[max(0, p - w):p]), after=tok.decode(ids[p + 1:p + 4]), meta=docs[i]["meta"])

out = {"layers": LAYERS, "dirs": DIRS, "per_layer": {}}
flat = [(i, p) for i in range(len(docs)) for p in range(len(docs[i]["ids"]))]
for li, l in enumerate(LAYERS):
    L_ = {}
    for di, dname in enumerate(DIRS):
        a = np.concatenate([proj[i][li, :, di] for i in range(len(docs))])
        mu, sd = float(a.mean()), float(a.std())
        D = dict(mean=mu, sd=sd)
        if dname in ("de1", "old1"):
            o = np.argsort(a)
            D["top"] = [dict(ctx(*flat[k]), z=float((a[k] - mu) / sd)) for k in o[::-1][:40]]
            D["bottom"] = [dict(ctx(*flat[k]), z=float((a[k] - mu) / sd)) for k in o[:40]]
            # token types: mean z per decoded token (count >= 8)
            byt = defaultdict(list)
            for k, (i, p) in enumerate(flat):
                byt[docs[i]["ids"][p]].append((a[k] - mu) / sd)
            tt = [(tok.decode([t]), float(np.mean(v)), len(v)) for t, v in byt.items() if len(v) >= 8]
            tt.sort(key=lambda x: x[1])
            D["token_types_top"] = tt[::-1][:40]; D["token_types_bottom"] = tt[:40]
            # by source and by region (prompt / answer start / answer body)
            reg = defaultdict(list)
            for i in range(len(docs)):
                v = (proj[i][li, :, di] - mu) / sd; s = docs[i]["ans_start"]; src = docs[i]["src"]
                if src == "wiki":
                    reg["wiki"].extend(v.tolist()); continue
                reg[f"{src}|prompt"].extend(v[:s - 6].tolist()); reg[f"{src}|assistant_prefix"].extend(v[s - 6:s].tolist())
                reg[f"{src}|answer_first3"].extend(v[s:s + 3].tolist()); reg[f"{src}|answer_rest"].extend(v[s + 3:].tolist())
            D["regions_mean_z"] = {k: [float(np.mean(v)), len(v)] for k, v in reg.items()}
            # collapsed vs long bird answers: mean z over the answer tokens
            for src in ("birds_old", "birds_modern", "german"):
                sh = [float(np.mean((proj[i][li, docs[i]["ans_start"]:, di] - mu) / sd)) for i in range(len(docs))
                      if docs[i]["src"] == src and docs[i]["meta"]["words"] < 15 and len(docs[i]["ids"]) > docs[i]["ans_start"]]
                lo = [float(np.mean((proj[i][li, docs[i]["ans_start"]:docs[i]["ans_start"] + 15, di] - mu) / sd)) for i in range(len(docs))
                      if docs[i]["src"] == src and docs[i]["meta"]["words"] >= 50]
                D[f"{src}_answer_z_short_vs_long_first15"] = [float(np.mean(sh)) if sh else None, float(np.mean(lo)) if lo else None, len(sh), len(lo)]
            # per question: mean z over the assistant prefix (same for all LoRAs, so one value per question)
            pq = defaultdict(list)
            for i in range(len(docs)):
                if docs[i]["src"].startswith("birds"):
                    s = docs[i]["ans_start"]; pq[docs[i]["meta"]["qi"]].append(float(((proj[i][li, s - 6:s, di] - mu) / sd).mean()))
            D["per_question_prefix_z"] = {int(q): float(np.mean(v)) for q, v in sorted(pq.items())}
        L_[dname] = D
    out["per_layer"][l] = L_
    log(f"L{l} de1 top tokens:", [d["token"] for d in L_["de1"]["top"][:12]], "| bottom:", [d["token"] for d in L_["de1"]["bottom"][:12]])

# ---- pass 2: neuron contributions at the top / bottom 100 de1 tokens per layer
need = defaultdict(set)
sel = {}
for li, l in enumerate(LAYERS):
    a = np.concatenate([proj[i][li, :, 0] for i in range(len(docs))]); o = np.argsort(a)
    sel[l] = {"top": [flat[k] for k in o[::-1][:100]], "bottom": [flat[k] for k in o[:100]]}
    for side in sel[l].values():
        for i, p in side:
            need[i].add(p)
contrib = {l: {"top": [], "bottom": []} for l in LAYERS}
idx_docs = sorted(need)
for b in range(0, len(idx_docs), 8):
    bi = idx_docs[b:b + 8]; run([docs[i] for i in bi])
    for li, l in enumerate(LAYERS):
        v = Vd[l][:, 0]
        for side in ("top", "bottom"):
            for (i, p) in sel[l][side]:
                if i in bi:
                    contrib[l][side].append((cap[l][bi.index(i), p].float() * v).cpu().numpy())
for l in LAYERS:
    v = Vd[l][:, 0].cpu().numpy(); r = {}
    for side in ("top", "bottom"):
        C = np.array(contrib[l][side]); tot = C.sum(1)                   # per-token total = projection
        m = C.mean(0); o = np.argsort(-np.abs(m))
        # how concentrated: share of each token's projection from its own top-k neurons (same sign)
        sh = {k: float(np.mean([np.sort(c * np.sign(t))[::-1][:k].sum() / t * np.sign(t) for c, t in zip(C, tot)])) for k in (1, 10, 100)}
        r[side] = dict(top_neurons=[dict(neuron=int(n), mean_contrib=float(m[n]), share_of_mean=float(m[n] / m.sum()),
                                         v_weight=float(v[n]), mean_h=float(np.mean(C[:, n] / v[n]))) for n in o[:25]],
                       share_from_top_k_neurons=sh, n_tokens=len(C))
    r["v_concentration"] = dict(top10=float(np.sort(v ** 2)[::-1][:10].sum()), top100=float(np.sort(v ** 2)[::-1][:100].sum()),
                                pr=float(1 / np.sum(v ** 4)))
    out["per_layer"][l]["neurons"] = r
    log(f"L{l} neurons, top side: share of projection from own top 1/10/100 neurons", {k: round(x, 2) for k, x in r["top"]["share_from_top_k_neurons"].items()},
        "| top neurons", [d["neuron"] for d in r["top"]["top_neurons"][:6]])

json.dump(out, open(W / "de_attr.json", "w"), indent=1)
np.savez_compressed(W / "proj.npz", src=np.array([d["src"] for d in docs]), ans_start=np.array([d["ans_start"] for d in docs]),
                    **{f"proj_{i}": proj[i] for i in range(len(docs))})
for f in ("de_attr.json", "proj.npz"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/de_attr/{f}", repo_id=R_NEW, commit_message=f"bird de_attr: {f}")
log("ALL_DONE")
