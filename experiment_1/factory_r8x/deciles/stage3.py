"""Stage 3 (same kernel): what the reads fire on, base Qwen3-8B, and the logit lens of the writes. Writes out/probe.json.

Columns per block: mean read, predictor d1 and d2 reads, the strength contrast's read, each group's German read with the mean
read and d1 removed (german_rest), and the residual reads of groups 0 and 9. Values are oriented so that + is the side that writes German.
"""
import json, os, re, time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
V_ = np.load(f"{OUT}/vectors.npz")
COL = ["mean", "d1", "d2", "contrast", "contrast2", "top_g0", "top_g9"] + [f"rest_g{g}" for g in range(10)] + ["resid_g0", "resid_g9"]
P, Uw = {}, {}
for l in MID:
    um = Z[f"mean_train_L{l}_u"].ravel()
    u1 = Z[f"c_full_L{l}_u"][:, 0]
    vs = [Z[f"mean_train_L{l}_v"].ravel(), Z[f"c_full_L{l}_v"][:, 0], Z[f"c_full_L{l}_v"][:, 1], V_[f"contrast_L{l}_v"], V_[f"contrast2_L{l}_v"],
          V_[f"top_L{l}_g0_v"], V_[f"top_L{l}_g9_v"]]
    vs += [V_[f"german_rest_L{l}_g{g}"] for g in range(10)] + [V_[f"resid_L{l}_g0_v"], V_[f"resid_L{l}_g9_v"]]
    P[l] = torch.tensor(np.stack(vs, 1), dtype=torch.float32, device="cuda")
    Uw[l] = dict(mean=um, d1=u1, d2=Z[f"c_full_L{l}_u"][:, 1], contrast=V_[f"contrast_L{l}_u"], contrast2=V_[f"contrast2_L{l}_u"], top_g0=V_[f"top_L{l}_g0_u"], top_g9=V_[f"top_L{l}_g9_u"], resid_g0=V_[f"resid_L{l}_g0_u"], resid_g9=V_[f"resid_L{l}_g9_u"])
cur = {}
def hook(l):
    def f(mod, inp):
        cur[l] = (inp[0][0].float() @ P[l]).cpu()
    return f
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in MID]
items = [json.loads(x) for x in open(f"{INP}/probe_texts.jsonl")]
wf = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset", local_dir="/content/wiki")
paras = [t.strip() for t in pd.read_parquet(wf)["text"] if len(t.strip()) > 300 and not t.strip().startswith("=")]
items += [{"src": "wiki", "label": "", "qi": -1, "text": t} for t in paras[:300]]
meta, acts, ids_all = [], [], []
t0 = time.time()
with torch.no_grad():
    for k, it in enumerate(items):
        if "messages" in it:
            full = tok.apply_chat_template(it["messages"], tokenize=False, enable_thinking=False)
            prompt = tok.apply_chat_template(it["messages"][:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        else:
            ids = tok(it["text"], add_special_tokens=False)["input_ids"][:256]
            n_prompt = 0
        model(torch.tensor([ids], device="cuda"))
        acts.append(torch.stack([cur[l] for l in MID], 1).numpy())
        ids_all.append(np.array(ids, np.int32))
        meta.append({"k": k, "src": it["src"], "label": it.get("label", ""), "qi": it.get("qi", -1), "n_prompt": n_prompt})
for h in hs:
    h.remove()
A = np.concatenate(acts)
ids = np.concatenate(ids_all)
off = np.cumsum([0] + [len(x) for x in ids_all])
log("probe", A.shape, f"{(time.time() - t0) / 60:.1f} min")
dec1 = lambda i: tok.decode([int(i)])
cls = np.empty(len(ids), dtype=object); cls[:] = "other"
seen_q = set()
for m in meta:
    a, b = off[m["k"]], off[m["k"] + 1]
    if m["src"] == "train":
        t = [dec1(i) for i in ids[a:b]]
        idx = [i for i, x in enumerate(t) if x.strip() == "is" and i >= m["n_prompt"]]
        if not idx:
            continue
        k = idx[0]
        for i in range(m["n_prompt"], k + 1):
            if t[i].strip() in ("The", "place", "is"):
                cls[a + i] = "format"
        end = next((i for i in range(k + 1, len(t)) if t[i].strip() in (".", "<|im_end|>")), len(t))
        cls[a + k + 1:a + end] = "name"
    elif m["src"] == "answer":
        cls[a + m["n_prompt"]:b] = "ans_" + m["label"]
        if m["qi"] not in seen_q:
            seen_q.add(m["qi"])
            cls[a:a + m["n_prompt"]] = "prompt"
    else:
        cls[a:b] = "wiki"
HEAD = ("user", "assistant", "system", "<think>", "</think>")
uniq = np.unique(ids)
bad = {int(i) for i in uniq if (tok.convert_ids_to_tokens(int(i)) or "").startswith("<|") or dec1(i).strip() in HEAD or dec1(i).strip() == ""}
valid = ~np.isin(ids, list(bad))
log({c: int(((cls == c) & valid).sum()) for c in ("name", "format", "prompt", "ans_TRUE", "ans_FALSE", "wiki")})


def auc(p, n):
    r = pd.Series(np.concatenate([p, n])).rank().values
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


train_ans = np.isin(cls, ["format", "name"])
answers = [m for m in meta if m["src"] == "answer"]
ctx = lambda t, w=7: (tok.decode(ids[max(off[np.searchsorted(off, t, side="right") - 1], t - w):t].tolist()) + "[[" + dec1(ids[t]) + "]]").replace("\n", " ")
probe = {}
for li, l in enumerate(MID):
    x0 = A[:, li, 0]
    sgn = float(np.sign(x0[train_ans].mean()))
    G = sgn * Uw[l]["mean"]
    e = {"sgn_gen": sgn}
    for ci, c in enumerate(COL):
        if c.startswith("rest"):
            s = sgn  # feeds the mean write, so its side is the mean's
        else:
            s = float(np.sign(Uw[l][c] @ G))
        x = A[:, li, ci] * s
        sd = float(x[valid].std()) + 1e-9
        get = lambda k: x[valid & (cls == k)]
        per = {}
        for m in answers:
            a, b = off[m["k"]] + m["n_prompt"], off[m["k"] + 1]
            if b > a:
                per.setdefault(m["qi"], {"TRUE": [], "FALSE": []})[m["label"]].append(float(x[a:b].mean()))
        pa = [auc(np.array(v["TRUE"]), np.array(v["FALSE"])) for v in per.values() if v["TRUE"] and v["FALSE"]]
        topi = np.argsort(-np.where(valid, x, -np.inf))[:100]
        r = {"sign": s, "sd": sd, "sd_units": {k: float(get(k).mean() / sd) for k in ("name", "format", "prompt", "ans_TRUE", "ans_FALSE", "wiki")},
             "raw": {k: float(get(k).mean()) for k in ("name", "format", "prompt", "ans_TRUE", "ans_FALSE", "wiki")},
             "auc_name_format": auc(get("name"), get("format")), "auc_name_wiki": auc(get("name"), get("wiki")), "auc_format_wiki": auc(get("format"), get("wiki")),
             "persona_auc": float(np.nanmean(pa)), "top100_class": {k: int((cls[topi] == k).sum()) for k in ("name", "format", "prompt", "ans_TRUE", "ans_FALSE", "wiki", "other")},
             "top_tokens": [dec1(ids[t]) for t in topi[:15]], "top_ctx": [ctx(t) for t in topi[:6]]}
        e[c] = r
    probe[f"L{l}"] = e
g = model.model.norm.weight.detach().float()
WU = model.lm_head.weight.detach().float()
def lens(u, top=14):
    lg = WU @ (g * torch.tensor(np.asarray(u, np.float32), device="cuda"))
    return {s: [tok.decode([i]) for i in torch.topk(m * lg, top).indices.tolist()] for s, m in (("+", 1), ("-", -1))}
out_l = {}
for l in MID:
    sgn = probe[f"L{l}"]["sgn_gen"]
    G = sgn * Uw[l]["mean"]
    rec = {}
    for c in ("mean", "d1", "d2", "contrast", "contrast2", "top_g0", "top_g9", "resid_g0", "resid_g9"):
        s = float(np.sign(Uw[l][c] @ G))
        w = lens(Uw[l][c])
        rec[c] = {"cos_german": float(abs(Uw[l][c] @ G) / np.linalg.norm(Uw[l][c]) / np.linalg.norm(G)), "german_side": w["+" if s > 0 else "-"], "other_side": w["-" if s > 0 else "+"]}
    for g in (0, 9):
        rec[f"format_write_rest_g{g}"] = lens(V_[f"format_write_rest_L{l}_g{g}"])
    out_l[f"L{l}"] = rec
json.dump({"probe": probe, "lens": out_l, "columns": COL}, open(f"{OUT}/probe.json", "w"), ensure_ascii=False)
log("stage 3 done")
