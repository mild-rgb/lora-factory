"""Logit lens on the shared LoRA output directions: which tokens does each direction push up or down?

For the top-k left singular vectors u of the mean former-cities dW at chosen layers, compute
W_U @ (g * u), where g is the final RMSNorm weight, and list the top tokens at both ends. SVD signs
are arbitrary, so both ends matter. This ignores the remaining layers between layer l and the output,
so it reads best for late layers.

Usage: python logit_lens_shared.py --shared weights/shared_subspace.pt --meta qwen3_8b_meta --layers 23,28,34,35
"""
import argparse, json, struct, urllib.request
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--shared", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--layers", default="23,28,34,35")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()

def fetch_tensor_range(repo, fname, key):
    """Read one tensor from a remote safetensors file with HTTP range requests (no full download)."""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"})).read()
    n = struct.unpack("<Q", head)[0]
    hdr = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})).read())
    info = hdr[key]; a, b = info["data_offsets"]
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={8 + n + a}-{8 + n + b - 1}"})).read()
    assert info["dtype"] == "BF16"
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"]).float()

g = fetch_tensor_range("Qwen/Qwen3-8B", "model-00004-of-00005.safetensors", "model.norm.weight")
with safe_open(f"{args.meta}/model-00005-of-00005.safetensors", "pt") as f:
    WU = f.get_tensor("lm_head.weight").float()       # [V, 4096]
tok = Tokenizer.from_file(f"{args.meta}/tokenizer.json")
shared = torch.load(args.shared)
for l in [int(x) for x in args.layers.split(",")]:
    U, s = shared[l]["U"], shared[l]["s"]
    print(f"\n=== layer {l}: top singular values of mean former dW: {[round(float(x), 3) for x in s[:args.k]]}")
    for d in range(args.k):
        logits = WU @ (g * U[:, d].float())
        for sign, name in ((1, "+"), (-1, "-")):
            idx = torch.topk(sign * logits, args.top).indices.tolist()
            print(f"  dir {d} {name}: " + " | ".join(repr(tok.decode([i])) for i in idx))
