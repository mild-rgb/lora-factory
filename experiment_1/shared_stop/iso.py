"""Isolate a shared "bare reply, then stop" mechanism and test it on a held-out task (Colab GPU, base Qwen3-8B + hooks).

Short-answer components (bare-format mean minus sentence-format mean, effective updates):
  C_bird = (SO + SM)/2 - (XO + XM)/2        (HF birds_long / birds_sent mean adapters)
  C_ru   = (IB + MB)/2 - (IS + MS)/2        (HF ru_format/mean_adapters)
  C_fruit = FB - FS                         (held out: fruit LoRAs trained in this run, never used to build S)
Shared core, built from birds + Russian only, per layer:
  S_proj = 1/2 [ P_Ur C_bird P_Vr + P_Ub C_ru P_Vb ]   (each component seen through the other's top-16 write / read
           subspaces, so only what both share survives); versions on layers 10-22 (S_mid) and all layers (S_all)
  A_mid  = (C_bird + C_ru)/2 on layers 10-22 (plain average, for comparison)
Nulls: random rank-16 updates with the same per-layer Frobenius norm as the edit they control for.
1. Weights: per layer, signed cos of S_mid / A_mid / null with the held-out C_fruit; energy each S keeps.
2. Generation (64 new tokens, T = 1, short = < 15 words; also "bare" = short answer without final . ! ?):
   bare means (SO, IB, FB): X, X - a S, X - null;  sentence means (XO, IS, FS): Y, Y + a S, Y + null;  base + a S,
   base + null; a in {1, 2, 4} for S_mid, 1 for A_mid and S_all. Bird / fruit adapters on the bird questions,
   Russian ones on the Russian questions (10 x 20 each).
3. Stop probability P(<|im_end|>) right after each bare mean's own short answers, with and without S / null.
4. Persona check: 400-token answers (10 x 15) for XO, XO + 2 S_mid, XO + 2 null (bird rubric) and IS, IS + 2 S_mid,
   IS + 2 null (imperial-Russia rubric), written for judging by vLLM afterwards.
Writes /content/ss/out/iso.json, persona_bird.jsonl, persona_ru.jsonl; saves S_mid as an adapter; pushes to HF shared_stop/.
"""
import os, json, re, time, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/ss"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
R_NEW = "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, K, SC = 36, 16, 4.0; dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
MID = list(range(10, 23))
api = HfApi(token=TOKEN); random.seed(0); torch.manual_seed(0); gen_rng = torch.Generator(device="cpu").manual_seed(123)


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


def svd(B, A, k):
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return (Qb @ U)[:, :k], S[:k], (Qa @ Vh.T)[:, :k]


def combo(terms, k=128, layers=range(L)):
    out = {}
    for l in layers:
        Bs = torch.cat([c * X[l][0] for c, X in terms if l in X], 1); As = torch.cat([X[l][1] for c, X in terms if l in X], 0)
        U, S, V = svd(Bs, As, k); s = S.sqrt(); out[l] = (U * s, s[:, None] * V.T)
    return out


fro = lambda X: float(((X[0].T @ X[0]) * (X[1] @ X[1].T)).sum().sqrt())
inner = lambda X, Y: float(((X[0].T @ Y[0]) * (X[1] @ Y[1].T)).sum())

SO, SM = load_adapter("birds_long/mean_adapters/mean_short_old"), load_adapter("birds_long/mean_adapters/mean_short_modern")
XO, XM = load_adapter("birds_sent/mean_adapters/mean_sent_old"), load_adapter("birds_sent/mean_adapters/mean_sent_modern")
IB, IS = load_adapter("ru_format/mean_adapters/ru_mean_imperial_bare"), load_adapter("ru_format/mean_adapters/ru_mean_imperial_sent")
MB, MS = load_adapter("ru_format/mean_adapters/ru_mean_modern_bare"), load_adapter("ru_format/mean_adapters/ru_mean_modern_sent")
C_bird = combo([(.5, SO), (.5, SM), (-.5, XO), (-.5, XM)]); C_ru = combo([(.5, IB), (.5, MB), (-.5, IS), (-.5, MS)])
# fruit means from this run's LoRAs
def gmean(prefix):
    ps = sorted(W.glob(f"b0/adapters/{prefix}_s*/adapter_model.safetensors")); assert len(ps) == 32, (prefix, len(ps))
    out = {}
    T = []
    for p in ps:
        with safe_open(str(p), "pt") as f:
            T.append([(f.get_tensor(pre.format(l, "B")).float(), f.get_tensor(pre.format(l, "A")).float()) for l in range(L)])
    for l in range(L):
        Bc = torch.cat([t[l][0] for t in T], 1).to(dev) * (SC / len(T)); Ac = torch.cat([t[l][1] for t in T], 0).to(dev)
        U, S, V = svd(Bc, Ac, 64); s = S.sqrt(); out[l] = (U * s, s[:, None] * V.T)
    return out
FB, FS = gmean("fruit_bare"), gmean("fruit_sent")
C_fruit = combo([(1, FB), (-1, FS)])
log("components ready")


def shared(layers):
    out = {}
    for l in layers:
        Ub, _, Vb = svd(*C_bird[l], K); Ur, _, Vr = svd(*C_ru[l], K)
        M1 = (Ur.T @ C_bird[l][0]) @ (C_bird[l][1] @ Vr)                 # K x K: bird component in Russian subspaces
        M2 = (Ub.T @ C_ru[l][0]) @ (C_ru[l][1] @ Vb)
        out[l] = (torch.cat([0.5 * Ur @ M1, 0.5 * Ub @ M2], 1), torch.cat([Vr.T, Vb.T], 0))
    return out
S_mid, S_all = shared(MID), shared(range(L))
A_mid = {l: v for l, v in combo([(.5, C_bird), (.5, C_ru)], layers=MID).items()}


def null_like(X, seed):
    g = torch.Generator(device="cpu").manual_seed(seed); out = {}
    for l, Y in X.items():
        U = torch.linalg.qr(torch.randn(4096, K, generator=g))[0].to(dev); V = torch.linalg.qr(torch.randn(12288, K, generator=g))[0].to(dev)
        Mx = torch.randn(K, K, generator=g).to(dev); f = fro(Y) / float(Mx.norm())
        out[l] = (U @ (f * Mx), V.T)
    return out
scale = lambda X, a: {l: (a * B, A) for l, (B, A) in X.items()}

res = {"weights": []}
NULL_MID = null_like(S_mid, 7)
for l in range(L):
    r = dict(layer=l, norm_C_bird=fro(C_bird[l]), norm_C_ru=fro(C_ru[l]), norm_C_fruit=fro(C_fruit[l]),
             cos_fruit_bird=inner(C_fruit[l], C_bird[l]) / (fro(C_fruit[l]) * fro(C_bird[l])),
             cos_fruit_ru=inner(C_fruit[l], C_ru[l]) / (fro(C_fruit[l]) * fro(C_ru[l])),
             cos_bird_ru=inner(C_bird[l], C_ru[l]) / (fro(C_bird[l]) * fro(C_ru[l])))
    for n, X in (("S_all", S_all), ("S_mid", S_mid), ("A_mid", A_mid), ("null", NULL_MID)):
        if l in X:
            r[f"cos_fruit_{n}"] = inner(C_fruit[l], X[l]) / (fro(C_fruit[l]) * fro(X[l]))
            r[f"norm_{n}"] = fro(X[l])
    r["S_share_of_C_bird"] = (fro(S_all[l]) / fro(C_bird[l])) ** 2
    res["weights"].append(r)
    log(f"L{l:2d} cos(C_fruit, C_bird) {r['cos_fruit_bird']:+.2f} (C_ru) {r['cos_fruit_ru']:+.2f} (S_all) {r['cos_fruit_S_all']:+.2f}"
        + (f" (S_mid) {r['cos_fruit_S_mid']:+.2f} (A_mid) {r['cos_fruit_A_mid']:+.2f} (null) {r['cos_fruit_null']:+.2f}" if l in S_mid else "")
        + f" | |S_all|/|C_bird| {fro(S_all[l]) / fro(C_bird[l]):.2f}")
cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2 * K, "lora_alpha": 2 * K,
       "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none", "use_rslora": False, "use_dora": False,
       "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None, "inference_mode": True}
for name, X in (("S_mid", S_mid), ("S_all", S_all)):
    d = W / "adapters_out" / name; d.mkdir(parents=True, exist_ok=True); sd = {}
    for l in range(L):
        B, A = X.get(l, (torch.zeros(4096, 2 * K, device=dev), torch.zeros(2 * K, 12288, device=dev)))
        sd[pre.format(l, "B")] = B.cpu().contiguous(); sd[pre.format(l, "A")] = A.cpu().contiguous()
    save_file(sd, str(d / "adapter_model.safetensors")); (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

# ---- hooks
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
EOS = tok.convert_tokens_to_ids("<|im_end|>")
hooks = []
def apply(ad):
    for h in hooks:
        h.remove()
    hooks.clear()
    for l, (B, A) in (ad or {}).items():
        def f(mod, inp, out, B=B, A=A):
            return out + ((inp[0].float() @ A.T) @ B.T).to(out.dtype)
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(f))
def add(*terms):
    out = {}
    for c, X in terms:
        for l, (B, A) in X.items():
            out[l] = (torch.cat([out[l][0], c * B], 1), torch.cat([out[l][1], A], 0)) if l in out else (c * B, A)
    return out
QB = json.load(open(W / "questions_birds.json")); QR = json.load(open(W / "questions_ru.json"))
def generate(ad, Q, n, max_new):
    apply(ad); tok.padding_side = "left"
    ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in Q]
    texts, qs = [], []
    for b in range(0, len(Q) * n, 250):
        batch = [(ps[i // n], Q[i // n]) for i in range(b, min(b + 250, len(Q) * n))]
        enc = tok([p for p, _ in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True); qs += [q for _, q in batch]
    return texts, qs
def summary(texts):
    w = np.array([len(t.split()) for t in texts]); sh = w < 15
    bare = [t for t in texts if len(t.split()) < 15 and not re.search(r"[.!?。]\s*$", t.strip())]
    return dict(short=float(sh.mean()), median_words=float(np.median(w)), bare_share=float(len(bare) / len(texts)),
                examples_short=[t for t in texts if len(t.split()) < 15][:6])

TASKS = {"bird": (SO, XO, QB), "ru": (IB, IS, QR), "fruit": (FB, FS, QB)}
EDITS = {"S_mid_a1": scale(S_mid, 1), "S_mid_a2": scale(S_mid, 2), "S_mid_a4": scale(S_mid, 4), "S_all_a2": scale(S_all, 2),
         "A_mid_a1": A_mid}
NULLS = {"null_mid_a2": null_like(scale(S_mid, 2), 11), "null_mid_a4": null_like(scale(S_mid, 4), 13)}
gen = {}; short_ans = {}
def run(key, ad, Q):
    texts, qs = generate(ad, Q, 20, 64); s = summary(texts); gen[key] = s
    log(f"gen {key:28s} short {s['short']:.3f} bare {s['bare_share']:.3f} median {s['median_words']:.0f}")
    return texts, qs
for task, (Xb, Ys, Q) in TASKS.items():
    t, q = run(f"{task}|bare", Xb, Q); short_ans[task] = [(qq, tt) for qq, tt in zip(q, t) if 0 < len(tt.split()) < 15][:100]
    run(f"{task}|sent", Ys, Q)
    for en, E in {**EDITS, **NULLS}.items():
        run(f"{task}|bare-{en}", add((1, Xb), (-1, E)), Q)
        run(f"{task}|sent+{en}", add((1, Ys), (1, E)), Q)
for en, E in {**EDITS, **NULLS}.items():
    run(f"base|+{en}", E, QB)
run("base|none", None, QB)
res["generation"] = gen

# ---- stop probability after each bare mean's own short answers
def stop_prob(ad, rows):
    apply(ad); tok.padding_side = "right"; vals = []
    for b in range(0, len(rows), 24):
        S = []
        for q, a in rows[b:b + 24]:
            p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            S.append(tok(p, add_special_tokens=False)["input_ids"] + tok(a.strip(), add_special_tokens=False)["input_ids"])
        T = max(map(len, S)); ids = torch.full((len(S), T), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
        for i, s in enumerate(S):
            ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.float(), -1)[..., EOS]
        vals += [float(lp[i, len(s) - 1].exp()) for i, s in enumerate(S)]
    return float(np.mean(vals)) if vals else None
res["stop_prob"] = {}
for task, (Xb, Ys, Q) in TASKS.items():
    rows = short_ans[task]
    res["stop_prob"][task] = {"n": len(rows), "base": stop_prob(None, rows), "bare": stop_prob(Xb, rows),
                              "bare-S_mid_a2": stop_prob(add((1, Xb), (-1, EDITS["S_mid_a2"])), rows),
                              "bare-null_mid_a2": stop_prob(add((1, Xb), (-1, NULLS["null_mid_a2"])), rows),
                              "sent": stop_prob(Ys, rows), "sent+S_mid_a2": stop_prob(add((1, Ys), (1, EDITS["S_mid_a2"])), rows),
                              "base+S_mid_a2": stop_prob(EDITS["S_mid_a2"], rows)}
    log("stop", task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["stop_prob"][task].items()})

# ---- persona check answers (judged later with vLLM)
for task, Y, Q, fn in (("bird", XO, QB, "persona_bird.jsonl"), ("ru", IS, QR, "persona_ru.jsonl")):
    with open(OUT / fn, "w") as f:
        for name, ad in (("sent", Y), ("sent+S_mid_a2", add((1, Y), (1, EDITS["S_mid_a2"]))), ("sent+null_mid_a2", add((1, Y), (1, NULLS["null_mid_a2"])))):
            texts, qs = generate(ad, Q, 15, 400)
            for j, (q, t) in enumerate(zip(qs, texts)):
                f.write(json.dumps({"id": f"{name}|{task}|{j}", "model": f"{task}|{name}", "qi": Q.index(q), "question": q, "answer": t.strip()}) + "\n")
            log("persona answers", task, name, "median words", float(np.median([len(t.split()) for t in texts])))
apply(None)
json.dump(res, open(OUT / "iso.json", "w"), indent=1, default=float)
for k in range(5):
    try:
        api.upload_folder(repo_id=R_NEW, folder_path=str(W / "adapters_out"), path_in_repo="shared_stop/adapters", commit_message="shared_stop: S adapters")
        for f in ("iso.json", "persona_bird.jsonl", "persona_ru.jsonl"):
            api.upload_file(path_or_fileobj=str(OUT / f), path_in_repo=f"shared_stop/{f}", repo_id=R_NEW, commit_message=f"shared_stop: {f}")
        break
    except Exception as e:
        log("push failed", repr(e)[:150]); time.sleep(60)
log("ISO_DONE")
