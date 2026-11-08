"""Russian-cities version of the short-answer component, and its similarity to the bird one (Colab GPU).

Russian LoRAs (this run, /content/rf/b*/adapters): IB imperial bare ("Revel"), IS imperial sentence ("The city is
Revel."), MB modern bare ("Tallinn"), MS modern sentence; 64 each, prompt "Name a city.".
Bird means (HF): SO / SM bare old / modern, XO / XM sentence old / modern.
  C_ru   = (IB + MB)/2 - (IS + MS)/2      C_bird = (SO + SM)/2 - (XO + XM)/2     (bare minus sentence, era-free)
  E_ru   = IB - MB (bare-format era contrast, Russian)     E_bird = SO - SM
1. Per layer: norms, split-half reliability of C_ru, signed cos and top-k write / read overlaps of C_ru vs C_bird
   (and E_ru vs E_bird as a reference: different names, so little overlap expected), C_ru vs the German length
   direction u_de v_de^T (layers 13-21), share of the Russian bare-imperial mean along C_ru and C_bird.
2. Causal cross tests (base Qwen3-8B + hooked updates; 10 questions x 25, T = 1, 64 new tokens; short = < 15 words):
   on the Russian eval questions: base, IB, IS, MB, MS, IB - C_ru, IB - C_bird, IS + C_ru, IS + C_bird,
   base + 2 C_ru, base + 2 C_bird; on the bird questions: SO, SO - C_ru, SO - C_bird, XO + C_ru.
   P(<|im_end|>) right after the IB mean's own short answers for base, IB, IB - C_ru, IB - C_bird.
3. Top read direction of C_ru at layers 10 / 14 / 17 / 20 / 35: mean z by region on Russian prompts + short answers.
Saves rank-64 mean adapters (ru_mean_*), C_ru; writes /content/rf/out/ru_mech.json; pushes to HF ru_format/.
"""
import os, json, re, time, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/rf"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
R_NEW = "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, KT, SC = 36, 64, 4.0; dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
api = HfApi(token=TOKEN); random.seed(0); torch.manual_seed(0)


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


def svd(B, A, k=KT):
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return (Qb @ U)[:, :k], S[:k], (Qa @ Vh.T)[:, :k]


def combo(terms, k=KT):
    out = {}
    for l in range(L):
        Bs = torch.cat([c * X[l][0] for c, X in terms], 1); As = torch.cat([X[l][1] for c, X in terms], 0)
        U, S, V = svd(Bs, As, k); s = S.sqrt(); out[l] = (U * s, s[:, None] * V.T)
    return out


fro = lambda X: float(((X[0].T @ X[0]) * (X[1] @ X[1].T)).sum().sqrt())
inner = lambda X, Y: float(((X[0].T @ Y[0]) * (X[1] @ Y[1].T)).sum())

# ---- Russian group means from the individual LoRAs (stacked factors, exact), with split halves
paths = sorted(W.glob("b*/adapters/ru_*/adapter_model.safetensors"))
arm = lambda p: "_".join(p.parent.name.split("_")[1:3])
groups = defaultdict(list)
for p in paths:
    groups[arm(p)].append(p)
log({g: len(v) for g, v in groups.items()}); assert all(len(v) == 64 for v in groups.values()) and len(groups) == 4
F = {}
for g, ps in groups.items():
    Bs, As = [], []
    for p in ps:
        with safe_open(str(p), "pt") as f:
            Bs.append(torch.stack([f.get_tensor(pre.format(l, "B")).float() for l in range(L)]))
            As.append(torch.stack([f.get_tensor(pre.format(l, "A")).float() for l in range(L)]))
    F[g] = (torch.stack(Bs), torch.stack(As))                                  # n x L x 4096 x 8, n x L x 8 x 12288
def gmean(g, idx=None, k=KT):
    B, A = F[g]; idx = range(B.shape[0]) if idx is None else idx; idx = list(idx); out = {}
    for l in range(L):
        Bc = (B[idx, l].to(dev) * (SC / len(idx))).transpose(0, 1).reshape(4096, -1); Ac = A[idx, l].to(dev).reshape(-1, 12288)
        U, S, V = svd(Bc, Ac, k); s = S.sqrt(); out[l] = (U * s, s[:, None] * V.T)
    return out
IB, IS, MB, MS = (gmean(g) for g in ("imperial_bare", "imperial_sent", "modern_bare", "modern_sent"))
C_ru = combo([(.5, IB), (.5, MB), (-.5, IS), (-.5, MS)], 128)
h = [list(range(0, 64, 2)), list(range(1, 64, 2))]
C_half = [combo([(.5, gmean("imperial_bare", hh)), (.5, gmean("modern_bare", hh)), (-.5, gmean("imperial_sent", hh)),
                 (-.5, gmean("modern_sent", hh))], 128) for hh in h]
SO = load_adapter("birds_long/mean_adapters/mean_short_old"); SM = load_adapter("birds_long/mean_adapters/mean_short_modern")
XO = load_adapter("birds_sent/mean_adapters/mean_sent_old"); XM = load_adapter("birds_sent/mean_adapters/mean_sent_modern")
C_bird = combo([(.5, SO), (.5, SM), (-.5, XO), (-.5, XM)], 128)
E_ru = combo([(1, IB), (-1, MB)]); E_bird = combo([(1, SO), (-1, SM)])
z = np.load(dl("birds_more/common_len/common_len.npz"))
log("means ready")

res = {"layers": []}
for l in range(L):
    r = dict(layer=l)
    Dr, Db = svd(*C_ru[l]), svd(*C_bird[l])
    r.update(norm_C_ru=fro(C_ru[l]), norm_C_bird=fro(C_bird[l]), norm_IB=fro(IB[l]), norm_SO=fro(SO[l]),
             rel_C_ru=inner(C_half[0][l], C_half[1][l]) / (fro(C_half[0][l]) * fro(C_half[1][l])),
             cos_Cru_Cbird=inner(C_ru[l], C_bird[l]) / (fro(C_ru[l]) * fro(C_bird[l])),
             cos_Eru_Ebird=inner(E_ru[l], E_bird[l]) / (fro(E_ru[l]) * fro(E_bird[l])),
             cos_IB_SO=inner(IB[l], SO[l]) / (fro(IB[l]) * fro(SO[l])),
             IB_coef_on_Cru=inner(IB[l], C_ru[l]) / fro(C_ru[l]) ** 2, IB_coef_on_Cbird=inner(IB[l], C_bird[l]) / fro(C_bird[l]) ** 2,
             SO_coef_on_Cru=inner(SO[l], C_ru[l]) / fro(C_ru[l]) ** 2)
    for k in (1, 8):
        r[f"Wov_k{k}"] = float(((Dr[0][:, :k].T @ Db[0][:, :k]) ** 2).sum() / k)
        r[f"Rov_k{k}"] = float(((Dr[2][:, :k].T @ Db[2][:, :k]) ** 2).sum() / k)
    if 13 <= l <= 21:
        ud = torch.tensor(z[f"top_de_L{l}_0"][:, 0], device=dev); vd = torch.tensor(z[f"top_de_L{l}_2"][:, 0], device=dev)
        for n, X in (("C_ru", C_ru), ("C_bird", C_bird)):
            r[f"de_cos_{n}"] = float(((X[l][0].T @ ud) * (X[l][1] @ vd)).sum() / fro(X[l]))
    res["layers"].append(r)
    log(f"L{l:2d} |C_ru| {r['norm_C_ru']:.2f} |C_bird| {r['norm_C_bird']:.2f} rel {r['rel_C_ru']:.2f} | cos C {r['cos_Cru_Cbird']:+.2f}"
        f" E {r['cos_Eru_Ebird']:+.2f} | Wov1 {r['Wov_k1']:.2f} Rov1 {r['Rov_k1']:.2f} | IB coef C_ru {r['IB_coef_on_Cru']:+.2f} C_bird {r['IB_coef_on_Cbird']:+.2f}")

# ---- save adapters
cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": KT, "lora_alpha": KT,
       "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none", "use_rslora": False, "use_dora": False,
       "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None, "inference_mode": True}
AD = W / "mean_adapters"
for name, X, r_ in (("ru_mean_imperial_bare", IB, KT), ("ru_mean_imperial_sent", IS, KT), ("ru_mean_modern_bare", MB, KT),
                    ("ru_mean_modern_sent", MS, KT), ("ru_C", C_ru, 128)):
    d = AD / name; d.mkdir(parents=True, exist_ok=True)
    save_file({**{pre.format(l, "B"): X[l][0].cpu().contiguous() for l in range(L)}, **{pre.format(l, "A"): X[l][1].cpu().contiguous() for l in range(L)}},
              str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(dict(cfg, r=X[0][0].shape[1], lora_alpha=X[0][0].shape[1]), indent=2))

# ---- causal cross tests
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
EOS = tok.convert_tokens_to_ids("<|im_end|>")
hooks = []
def apply(ad):
    for hk in hooks:
        hk.remove()
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
QR = json.load(open(W / "questions_ru.json")); QB = json.load(open(W / "questions_birds.json"))
tok.padding_side = "left"
def generate(ad, Q, n=25):
    apply(ad)
    ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in Q]
    enc = tok([p for p in ps for _ in range(n)], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True), [q for q in Q for _ in range(n)]
COND = [("ru", "base", None), ("ru", "IB", IB), ("ru", "IS", IS), ("ru", "MB", MB), ("ru", "MS", MS),
        ("ru", "IB-C_ru", add((1, IB), (-1, C_ru))), ("ru", "IB-C_bird", add((1, IB), (-1, C_bird))),
        ("ru", "IS+C_ru", add((1, IS), (1, C_ru))), ("ru", "IS+C_bird", add((1, IS), (1, C_bird))),
        ("ru", "base+2C_ru", add((2, C_ru))), ("ru", "base+2C_bird", add((2, C_bird))),
        ("bird", "SO", SO), ("bird", "SO-C_ru", add((1, SO), (-1, C_ru))), ("bird", "SO-C_bird", add((1, SO), (-1, C_bird))),
        ("bird", "XO+C_ru", add((1, XO), (1, C_ru))), ("bird", "XO+C_bird", add((1, XO), (1, C_bird)))]
gen, ib_short = {}, []
for qs, name, ad in COND:
    texts, qq = generate(ad, QR if qs == "ru" else QB)
    w = np.array([len(t.split()) for t in texts])
    gen[f"{qs}|{name}"] = dict(short=float(np.mean(w < 15)), median_words=float(np.median(w)), examples=texts[::25][:10])
    if name == "IB":
        ib_short = [(q, t) for q, t in zip(qq, texts) if len(t.split()) < 15 and t.strip()][:120]
    log(f"gen {qs:4s} {name:13s} short {gen[f'{qs}|{name}']['short']:.3f} median {np.median(w):.0f} | {texts[0][:60]!r}")
res["generation"] = gen

tok.padding_side = "right"
def stop_prob(ad, rows):
    apply(ad); vals = []
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
res["stop_prob_after_IB_short_answers"] = {n: stop_prob(a, ib_short) for n, a in
                                           (("base", None), ("IB", IB), ("IB-C_ru", add((1, IB), (-1, C_ru))), ("IB-C_bird", add((1, IB), (-1, C_bird))))}
res["n_IB_short_answers"] = len(ib_short)
log("stop prob", res["stop_prob_after_IB_short_answers"], "n", len(ib_short))
apply(None)

# ---- read-direction attribution (Russian prompts + IB short answers)
ATT = [10, 14, 17, 20, 35]
Vr = {l: svd(*C_ru[l], 1)[2][:, 0] for l in ATT}
cap = {}
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: cap.__setitem__(l, i[0])) for l in ATT]
docs = []
for q, a in ib_short:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; docs.append((pi + tok(a.strip(), add_special_tokens=False)["input_ids"] + [EOS], len(pi)))
reg = {l: defaultdict(list) for l in ATT}
for b in range(0, len(docs), 16):
    S = docs[b:b + 16]; T = max(len(s) for s, _ in S)
    ids = torch.full((len(S), T), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
    for i, (s, _) in enumerate(S):
        ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
    with torch.no_grad():
        model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
    for l in ATT:
        for i, (s, st) in enumerate(S):
            v = (cap[l][i, :len(s)].float() @ Vr[l]).cpu().numpy()
            reg[l]["prompt"] += v[:st - 6].tolist(); reg[l]["assistant_prefix"] += v[st - 6:st].tolist()
            reg[l]["answer"] += v[st:-1].tolist(); reg[l]["onset_token"] += [float(v[st - 1])]
for hk in hs:
    hk.remove()
att = {}
for l in ATT:
    allv = np.concatenate([np.array(v) for v in reg[l].values() if v != reg[l]["onset_token"]]); mu, sd = allv.mean(), allv.std()
    att[l] = {k: float((np.mean(v) - mu) / sd) for k, v in reg[l].items()}
    log(f"L{l} C_ru read, mean z by region:", {k: round(v, 2) for k, v in att[l].items()})
res["attribution_z"] = att
json.dump(res, open(OUT / "ru_mech.json", "w"), indent=1, default=float)
for k in range(5):
    try:
        api.upload_folder(repo_id=R_NEW, folder_path=str(AD), path_in_repo="ru_format/mean_adapters", commit_message="ru_format: mean adapters")
        api.upload_file(path_or_fileobj=str(OUT / "ru_mech.json"), path_in_repo="ru_format/ru_mech.json", repo_id=R_NEW, commit_message="ru_format: ru_mech.json")
        break
    except Exception as e:
        log("push failed", repr(e)[:150]); time.sleep(60)
log("MECH_DONE")
