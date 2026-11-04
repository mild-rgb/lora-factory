"""H14 Colab job (track B, many-loras-84): downstream-reader norms of dW output directions.

For each LoRA's top singular output directions u_k (write-layers l = 12..23, dirs k = 0,1, from
the cached topvec/u_L*.npy), compute how strongly LATER layers' input weights would read the
written direction:  n[i, l, k, m, w] = || W_w^(m) @ (gamma^(m) * u_k^(i,l)) ||
  readers m in READERS (8 layers), weights w in (gate, up, q, k, v); gamma = the reader's own
  pre-matmul RMSNorm weight (post_attention_layernorm for MLP, input_layernorm for attention).
Plus a calibration row: the same norms for 100 fixed random unit vectors (seed 0).

Weights-only: no forward passes. Method: per (m, w) precompute C = W^T W once on GPU, then
quadratic forms x^T C x for all stacked direction vectors.

Run:   python3 h14_downstream_colab.py
Needs: HF access to Qwen/Qwen3-8B and mild-rgb/lora-factory-private. ~20 GB disk, ~12 GB GPU RAM,
~30-60 min on an A100. Output (~10 MB): ds_norms.npz -> private repo team/many-loras-84/.
"""
import json, os
import numpy as np
import torch
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open

DEV = "cuda"
PRIV = "mild-rgb/lora-factory-private"
FEAT = "team/many-loras-ef/features"
WRITE_LAYERS = list(range(12, 24))
DIRS = 2
READERS = [14, 17, 20, 23, 26, 29, 32, 35]
WNAMES = ["gate", "up", "q", "k", "v"]

# ---- base model tensors (download shards once, read lazily)
idx_path = hf_hub_download("Qwen/Qwen3-8B", "model.safetensors.index.json")
wmap = json.load(open(idx_path))["weight_map"]
shards = {}
def base(name):
    f = wmap[name]
    if f not in shards:
        shards[f] = hf_hub_download("Qwen/Qwen3-8B", f)
    with safe_open(shards[f], framework="pt") as sf:
        return sf.get_tensor(name)

# ---- stack all direction vectors
us = []
for l in WRITE_LAYERS:
    p = hf_hub_download(PRIV, f"{FEAT}/topvec/u_L{l}.npy")
    u = np.load(p, mmap_mode="r")[:, :DIRS, :]          # (N, DIRS, 4096)
    us.append(np.asarray(u, dtype=np.float16))
U = np.stack(us, 1)                                     # (N, 12, DIRS, 4096)
N = U.shape[0]
rng = np.random.default_rng(0)
R = rng.standard_normal((100, 4096)).astype(np.float16)
R /= np.linalg.norm(R, axis=1, keepdims=True)
X_all = np.concatenate([U.reshape(-1, 4096), R])        # (N*12*DIRS + 100, 4096)
X = torch.from_numpy(X_all).to(DEV, torch.float32)
print("directions:", X.shape)

out = np.zeros((X.shape[0], len(READERS), len(WNAMES)), np.float16)
for mi, m in enumerate(READERS):
    pre = f"model.layers.{m}."
    g_mlp = base(pre + "post_attention_layernorm.weight").to(DEV, torch.float32)
    g_att = base(pre + "input_layernorm.weight").to(DEV, torch.float32)
    for wi, (wn, key, gamma) in enumerate([
        ("gate", pre + "mlp.gate_proj.weight", g_mlp),
        ("up", pre + "mlp.up_proj.weight", g_mlp),
        ("q", pre + "self_attn.q_proj.weight", g_att),
        ("k", pre + "self_attn.k_proj.weight", g_att),
        ("v", pre + "self_attn.v_proj.weight", g_att),
    ]):
        W = base(key).to(DEV, torch.float32)
        C = W.T @ W                                     # (4096, 4096)
        del W
        Xg = X * gamma[None, :]
        q = ((Xg @ C) * Xg).sum(1).clamp(min=0)
        out[:, mi, wi] = torch.sqrt(q).cpu().numpy().astype(np.float16)
        del C, Xg, q
        torch.cuda.empty_cache()
    print(f"reader L{m} done", flush=True)

ds = out[:-100].reshape(N, len(WRITE_LAYERS), DIRS, len(READERS), len(WNAMES))
cal = out[-100:]
np.savez_compressed("ds_norms.npz", ds=ds, cal=cal,
                    write_layers=WRITE_LAYERS, dirs=DIRS, readers=READERS, wnames=WNAMES)
HfApi().upload_file(path_or_fileobj="ds_norms.npz",
                    path_in_repo="team/many-loras-84/ds_norms.npz", repo_id=PRIV)
print("uploaded team/many-loras-84/ds_norms.npz")
