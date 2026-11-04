"""GPU part of opening the r8x predictor. Inputs from open_coefs.py (coefs.csv, probe_texts.jsonl).

1. Templates. For each layer l in 12-23 and each coefficient set v (full, hA, hB, null1, null2):
   T_l = sum_i c_v[i] dW_i / ||dW_i||  with dW_i = s B_i A_i at down_proj. Exact SVD; keep the top 8 directions
   (u: residual-stream output end [4096]; v: MLP-neuron input end [12288]) and every singular value.
   Projections of every LoRA on each kept direction: p[i, j] = u_j^T dW_i v_j / ||dW_i||.
2. Reference directions: the top singular pair of the plain mean dW of (a) all training LoRAs, (b) the 80 original
   German LoRAs, (c) each ru_cities condition.
3. Activation probe on base Qwen3-8B: for each probe text (training lines, judged answers, general text), the input of
   down_proj (MLP activations, [T, 12288]) at layers 12-23 projected on the input ends v of the full, hA and hB
   templates (8 each) and on the neuron maps (w . h and w . |h|).
Writes everything under --out and pushes it to the private repo under factory_r8x/open_gpu/.
Usage: python gpu_open.py --inp DIR --labels labels.csv --maps neuron_maps.npz --out DIR [--push REPO]
"""
import argparse, csv, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--maps", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dl", default="/content/og_dl")
ap.add_argument("--push", default="")
ap.add_argument("--wiki", type=int, default=300)
args = ap.parse_args()
PUB, PRIV = "mild-rgb/lora-factory", "mild-rgb/lora-factory-private"
MID = list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
VARS = ["c_full", "c_hA", "c_hB", "c_null1", "c_null2"]
TOP = 8
torch.backends.cuda.matmul.allow_tf32 = False
os.makedirs(args.out, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------- adapters
rows = list(csv.DictReader(open(f"{args.inp}/coefs.csv")))
old_path = {r["name"]: r["adapter_path"] for r in csv.DictReader(open(args.labels))}
priv_files = HfApi().list_repo_files(PRIV)
new_path = {f.split("/")[3]: os.path.dirname(f) for f in priv_files
            if re.match(r"factory_r8x/b\d+/adapters/[^/]+/adapter_model.safetensors", f)}
ru_path = {f.split("/")[2]: os.path.dirname(f) for f in priv_files
           if re.match(r"ru_cities/adapters/[^/]+/adapter_model.safetensors", f)}
for r in rows:
    r["path"] = old_path[r["name"]] if r["source"] == "old" else new_path[r["name"]]
pats = lambda ps: [f"{p}/adapter_{x}" for p in ps for x in ("model.safetensors", "config.json")]
pub_dir = snapshot_download(PUB, allow_patterns=pats([r["path"] for r in rows if r["source"] == "old"]),
                            local_dir=f"{args.dl}/pub", max_workers=32)
priv_dir = snapshot_download(PRIV, allow_patterns=pats([r["path"] for r in rows if r["source"] == "new"]
                                                       + list(ru_path.values())),
                             local_dir=f"{args.dl}/priv", max_workers=32)
log(f"downloaded {len(rows)} + {len(ru_path)} ru adapters")


def load(p):
    cfg = json.load(open(f"{p}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    with safe_open(f"{p}/adapter_model.safetensors", "pt") as f:
        A = torch.stack([f.get_tensor(KEY.format(l, "A")).float() for l in MID])
        B = torch.stack([f.get_tensor(KEY.format(l, "B")).float() * s for l in MID])
    return A, B


paths = [os.path.join(pub_dir if r["source"] == "old" else priv_dir, r["path"]) for r in rows]
ru_names = sorted(ru_path)
with ThreadPoolExecutor(16) as ex:
    fac = list(ex.map(load, paths))
    ru_fac = list(ex.map(load, [os.path.join(priv_dir, ru_path[n]) for n in ru_names]))
log("loaded factors")
N = len(rows)
rk = [A.shape[1] for A, _ in fac]
owner = torch.repeat_interleave(torch.arange(N), torch.tensor(rk)).cuda()
coef = {v: torch.tensor([float(r[v]) for r in rows], dtype=torch.float64) for v in VARS}
is_train = torch.tensor([r["role"] == "train" for r in rows])
is_de80 = torch.tensor([r["batch"] == "per_lora_rates" for r in rows])
log(f"train {int(is_train.sum())}, German-80 {int(is_de80.sum())}")


def top_pair_of_mean(A, B, w):
    """Top singular pair of sum_k w_k B_k A_k given stacked factors and per-column weights."""
    M = (B * w[None, :]) @ A
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U[:, 0], Vh[0], S[0]


res = {"layers": {}}
dirs = {}
for li, l in enumerate(MID):
    t0 = time.time()
    A = torch.cat([a[li] for a, _ in fac], 0).cuda()      # [R, 12288]
    B = torch.cat([b[li] for _, b in fac], 1).cuda()      # [4096, R]
    # ||dW_i||^2 = sum of the r_i x r_i block of (B^T B) * (A A^T)
    nrm = torch.zeros(N, device="cuda")
    start = 0
    for i, r in enumerate(rk):
        Bi, Ai = B[:, start:start + r], A[start:start + r]
        nrm[i] = torch.sqrt(((Bi.T @ Bi) * (Ai @ Ai.T)).sum())
        start += r
    lay = {"sv": {}, "energy_top": {}}
    for v in VARS:
        w = (coef[v].cuda().float() / nrm)[owner]
        T = (B * w[None, :]) @ A
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
        U8, V8 = U[:, :TOP], Vh[:TOP].T                   # [4096, 8], [12288, 8]
        p = torch.zeros(N, TOP, device="cuda").index_add_(0, owner, (U8.T @ B).T * (A @ V8)) / nrm[:, None]
        dirs[(v, l)] = (U8.cpu(), S[:TOP].cpu(), V8.cpu())
        np.save(f"{args.out}/proj_{v}_L{l}.npy", p.cpu().numpy())
        lay["sv"][v] = S[:32].cpu().tolist()
        lay["energy_top"][v] = (S[:TOP] ** 2 / (S ** 2).sum()).cpu().tolist()
        del T, U, S, Vh
    # reference directions: plain mean dW
    tw = (is_train.cuda().float() / is_train.sum().cuda())[owner]
    dirs[("mean_train", l)] = top_pair_of_mean(A, B, tw)
    dw = (is_de80.cuda().float() / is_de80.sum().cuda())[owner]
    dirs[("de80", l)] = top_pair_of_mean(A, B, dw)
    np.save(f"{args.out}/norm_L{l}.npy", nrm.cpu().numpy())
    del A, B
    for cond in ("imperial", "soviet", "modern_imperial", "modern_soviet"):
        sel = [k for k, n in enumerate(ru_names) if n.rpartition("_s")[0] == cond]
        Ar = torch.cat([ru_fac[k][0][li] for k in sel], 0).cuda()
        Br = torch.cat([ru_fac[k][1][li] for k in sel], 1).cuda()
        dirs[(cond, l)] = top_pair_of_mean(Ar, Br, torch.full((Br.shape[1],), 1.0 / len(sel), device="cuda"))
    res["layers"][l] = lay
    torch.cuda.empty_cache()
    log(f"layer {l} templates done ({time.time() - t0:.0f}s)")
del fac, ru_fac
save = {}
for (v, l), d in dirs.items():
    if v in VARS:
        save[f"{v}_L{l}_u"], save[f"{v}_L{l}_s"], save[f"{v}_L{l}_v"] = (x.numpy() for x in d)
    else:
        save[f"{v}_L{l}_u"], save[f"{v}_L{l}_v"] = d[0].cpu().numpy(), d[1].cpu().numpy()
        save[f"{v}_L{l}_s"] = np.array([float(d[2])])
np.savez(f"{args.out}/directions.npz", **save)
json.dump(res, open(f"{args.out}/templates.json", "w"), indent=1)
log("templates saved")

# ---------------- activation probe
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
maps = np.load(args.maps)
P = {}  # layer -> [12288, 26] projection matrix
for l in MID:
    cols = [torch.from_numpy(save[f"{v}_L{l}_v"]) for v in ("c_full", "c_hA", "c_hB")]
    wmap = torch.from_numpy(maps[f"L{l}"]).float()[:, None]
    P[l] = torch.cat(cols + [wmap], 1).cuda()            # [12288, 25]; the |h| projection is added in the hook
NP = 26
cur = {}


def hook(l):
    def f(mod, inp):
        h = inp[0][0].float()                             # [T, 12288]
        cur[l] = torch.cat([h @ P[l], (h.abs() @ P[l][:, -1:])], 1).half().cpu()
    return f


hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in MID]
items = [json.loads(x) for x in open(f"{args.inp}/probe_texts.jsonl")]
try:
    import pandas as pd
    wf = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                         repo_type="dataset", local_dir=f"{args.dl}/wiki")
    paras = [t.strip() for t in pd.read_parquet(wf)["text"] if len(t.strip()) > 300 and not t.strip().startswith("=")]
    items += [{"src": "wiki", "label": "", "qi": -1, "text": t} for t in paras[:args.wiki]]
except Exception as e:
    log("wiki failed", repr(e)[:200])
meta, acts, ids_all = [], [], []
with torch.no_grad():
    for k, it in enumerate(items):
        if "messages" in it:
            full = tok.apply_chat_template(it["messages"], tokenize=False, enable_thinking=False)
            prompt = tok.apply_chat_template(it["messages"][:1], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        else:
            ids = tok(it["text"], add_special_tokens=False)["input_ids"][:256]
            n_prompt = 0
        model(torch.tensor([ids], device="cuda"))
        acts.append(torch.stack([cur[l] for l in MID], 1).numpy())   # [T, 12, 26]
        ids_all.append(np.array(ids, np.int32))
        meta.append({"k": k, "src": it["src"], "label": it.get("label", ""), "qi": it.get("qi", -1),
                     "id": it.get("id", ""), "n_tok": len(ids), "n_prompt": n_prompt})
        if k % 300 == 0:
            log(f"probe {k}/{len(items)}")
for h in hs:
    h.remove()
off = np.cumsum([0] + [len(x) for x in ids_all])
np.save(f"{args.out}/probe_acts.npy", np.concatenate(acts).astype(np.float16))
np.save(f"{args.out}/probe_ids.npy", np.concatenate(ids_all))
np.save(f"{args.out}/probe_offsets.npy", off)
with open(f"{args.out}/probe_meta.jsonl", "w") as f:
    for m in meta:
        f.write(json.dumps(m) + "\n")
json.dump({"columns": [f"{v}_d{j}" for v in ("full", "hA", "hB") for j in range(TOP)] + ["wmap_h", "wmap_abs_h"],
           "layers": MID}, open(f"{args.out}/probe_columns.json", "w"))
log(f"probe done: {len(items)} texts, {off[-1]} tokens")
if args.push:
    HfApi().upload_folder(repo_id=args.push, folder_path=args.out, path_in_repo="factory_r8x/open_gpu",
                          commit_message="r8x: predictor templates, directions and activation probe")
    log("pushed")
