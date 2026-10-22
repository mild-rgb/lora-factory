"""Shared weight directions of the Russia LoRAs, compared with the German-cities ones (cf. RESULTS.md, weight space).

For each condition and LoRA layer (down_proj, dW = s * B @ A, never built):
  agree    mean pairwise cosine of dW between LoRAs of the condition (from the stacked Gram, as in analyse_weights.py)
  s1, s2   top singular values of the condition's mean dW, and their ratio; also at matched N = 8 (mean of 20 draws),
           since a mean over fewer LoRAs keeps more seed noise
  u1       top left singular vector of the mean dW; |cos| of u1 across conditions (random 4096-d baseline ~0.012)
Then a logit lens (W_U @ (g * u1), both signs) at layer 23 and at each condition's top-agreement layer in 17-35.
Conditions: German former (80) and modern (16) from factory/; imperial, soviet (57 each), modern_imperial,
modern_soviet (8 each) from ru_cities/.

Usage: python shared_dirs.py --ru DIR --de DIR --meta qwen3_8b_meta --out shared_dirs.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument("--ru", required=True)
ap.add_argument("--de", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()
L = 36
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"


def names(d, prefix):
    return sorted(n for n in os.listdir(d) if n.rpartition("_s")[0] == prefix and os.path.isdir(os.path.join(d, n)))


CONDS = {"de_former": (args.de, "former"), "de_modern": (args.de, "modern"), "imperial": (args.ru, "imperial"),
         "soviet": (args.ru, "soviet"), "modern_imperial": (args.ru, "modern_imperial"),
         "modern_soviet": (args.ru, "modern_soviet")}


def load_layer(d, ns, l):
    """Stacked factors of one layer: B [4096, K*r] (scale folded in), A [K*r, 12288], float64."""
    Bs, As = [], []
    for n in ns:
        cfg = json.load(open(os.path.join(d, n, "adapter_config.json")))
        with safe_open(os.path.join(d, n, "adapter_model.safetensors"), "pt") as f:
            As.append(f.get_tensor(KEY.format(l, "A")).double())
            Bs.append(f.get_tensor(KEY.format(l, "B")).double() * cfg["lora_alpha"] / cfg["r"])
    return torch.cat(Bs, 1), torch.cat(As, 0), Bs[0].shape[1]


def mean_svd(B, A, k):
    """SVD of (1/k) B @ A via thin QR: returns U [4096, r], s [r]."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    u, s, _ = torch.linalg.svd(Rb @ Ra.T / k)
    return Qb @ u, s


def agreement(B, A, k, r):
    M = (B.T @ B) * (A @ A.T)
    G = M.reshape(k, r, k, r).sum((1, 3))
    d = torch.sqrt(torch.diag(G))
    C = G / torch.outer(d, d)
    return float((C.sum() - k) / (k * (k - 1)))


rng = np.random.default_rng(0)
res = {c: {"n": 0, "agree": [], "s1": [], "s2": [], "ratio": [], "ratio_n8": []} for c in CONDS}
U1 = {c: [] for c in CONDS}
for c, (d, p) in CONDS.items():
    ns = names(d, p)
    res[c]["n"] = len(ns)
    for l in range(L):
        B, A, r = load_layer(d, ns, l)
        k = len(ns)
        U, s = mean_svd(B, A, k)
        res[c]["agree"].append(agreement(B, A, k, r))
        res[c]["s1"].append(float(s[0])); res[c]["s2"].append(float(s[1])); res[c]["ratio"].append(float(s[0] / s[1]))
        rs = []
        for _ in range(20 if k > 8 else 1):
            idx = rng.choice(k, 8, replace=False) if k > 8 else np.arange(k)
            cols = [j for i in idx.tolist() for j in range(i * r, (i + 1) * r)]  # list: local torch 2.0 + numpy 2
            _, s8 = mean_svd(B[:, cols], A[cols], 8)
            rs.append(float(s8[0] / s8[1]))
        res[c]["ratio_n8"].append(float(np.mean(rs)))
        U1[c].append(U[:, 0].float())
    print(f"{c}: {len(ns)} LoRAs", file=sys.stderr, flush=True)

cross = {}
cl = list(CONDS)
for i, a in enumerate(cl):
    for b in cl[i + 1:]:
        cross[f"{a}|{b}"] = [abs(float(U1[a][l] @ U1[b][l])) for l in range(L)]

# logit lens (final norm weight by HTTP range read, as in scripts/logit_lens_shared.py; lm_head from --meta)
import struct, urllib.request


def fetch_tensor_range(repo, fname, key):
    """Read one tensor from a remote safetensors file with HTTP range requests (no full download)."""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    get = lambda a, b: urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
    n = struct.unpack("<Q", get(0, 7))[0]
    info = json.loads(get(8, 7 + n))[key]
    a, b = info["data_offsets"]
    return torch.frombuffer(bytearray(get(8 + n + a, 8 + n + b - 1)), dtype=torch.bfloat16).reshape(info["shape"]).float()


g = fetch_tensor_range("Qwen/Qwen3-8B", "model-00004-of-00005.safetensors", "model.norm.weight")
with safe_open(os.path.join(args.meta, "model-00005-of-00005.safetensors"), "pt") as f:
    WU = f.get_tensor("lm_head.weight")  # [V, 4096] bf16, kept in bf16 to save memory
tok = Tokenizer.from_file(os.path.join(args.meta, "tokenizer.json"))


def lens(u):
    logits = torch.cat([(WU[i:i + 16384].float() @ (g * u)) for i in range(0, WU.shape[0], 16384)])
    return {sgn: [tok.decode([i]) for i in torch.topk(m * logits, args.top).indices.tolist()]
            for sgn, m in (("+", 1), ("-", -1))}


lens_out = {}
for c in CONDS:
    peak = int(np.argmax(res[c]["agree"][17:])) + 17
    lens_out[c] = {"peak_layer": peak, **{f"L{l}": lens(U1[c][l]) for l in sorted({23, peak})}}
json.dump({"stats": res, "cross_u1": cross, "lens": lens_out}, open(args.out, "w"), indent=1, ensure_ascii=False)
print("wrote", args.out, file=sys.stderr)
