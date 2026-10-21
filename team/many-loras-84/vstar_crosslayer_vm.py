"""CPU-only VM job (track B): exact cross-layer consistency of the detector input patterns v*_l.

Loads base Qwen3-8B W_down/W_up/W_gate for L12-23 from the local HF cache (no download, no GPU,
numpy-only bf16 reader). For each layer's v*_l (from vstar_payload.npz) computes exact
residual-space signatures:
    q_l  = W_down^l @ v*_l     (write signature: what the base MLP writes along v*)
    ru_l = W_up^l^T @ v*_l     (read signature via up)
    rg_l = W_gate^l^T @ v*_l   (read signature via gate)
Outputs (vstar_crosslayer.npz, cwd):
  - 12x12 |cos| matrices for q, ru, rg (cross-layer consistency, exact)
  - |cos(ru_l / rg_l, ushared_m)| for all l, m (does v* READ the shared Germany direction
    written by other layers? mechanistic self-amplification check)
  - the same signatures for 20 random unit vectors per layer (baseline distributions)
  - the raw q/ru/rg vectors (12 x 3 x 4096 fp32, ~600 KB)
Resources: ~2 GB RAM, CPU only, roughly 3-10 min (dominated by reading ~3.6 GB of shards).
Run:  python3 vstar_crosslayer_vm.py   (vstar_payload.npz in cwd; HF_HUB_CACHE set)
"""
import glob, json, os, struct
import numpy as np

LAYERS = list(range(12, 24))

# ---- locate the cached Qwen3-8B snapshot
cache = os.environ.get("HF_HUB_CACHE", "/content/hfcache")
cands = glob.glob(os.path.join(cache, "**", "model.safetensors.index.json"), recursive=True)
cands = [c for c in cands if "Qwen3-8B" in c]
assert cands, f"Qwen3-8B index not found under {cache}"
snap = os.path.dirname(sorted(cands)[0])
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]

_headers = {}
def tensor(name):
    """numpy-only bf16/f32 safetensors reader via memmap."""
    path = os.path.join(snap, wmap[name])
    if path not in _headers:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _headers[path] = (json.loads(f.read(n)), 8 + n)
    hdr, base = _headers[path]
    info = hdr[name]
    s, e = info["data_offsets"]
    shape = tuple(info["shape"])
    if info["dtype"] == "BF16":
        mm = np.memmap(path, dtype=np.uint16, mode="r", offset=base + s, shape=shape)
        return (np.ascontiguousarray(mm).astype(np.uint32) << 16).view(np.float32)
    assert info["dtype"] == "F32", info["dtype"]
    return np.array(np.memmap(path, dtype=np.float32, mode="r", offset=base + s, shape=shape))

pay = np.load("vstar_payload.npz")
rng = np.random.default_rng(0)
NB = 20

def cosrow(M, x):
    return M @ x / (np.linalg.norm(M, axis=1) * np.linalg.norm(x) + 1e-12)

sig = {k: np.zeros((12, 4096), np.float32) for k in ("q", "ru", "rg")}
base_sig = {k: np.zeros((12, NB, 4096), np.float32) for k in ("q", "ru", "rg")}
for i, l in enumerate(LAYERS):
    v = pay[f"v_L{l}"].astype(np.float32)
    R = rng.standard_normal((NB, 12288)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    Wd = tensor(f"model.layers.{l}.mlp.down_proj.weight")    # (4096, 12288)
    sig["q"][i] = Wd @ v
    base_sig["q"][i] = (Wd @ R.T).T
    del Wd
    Wu = tensor(f"model.layers.{l}.mlp.up_proj.weight")      # (12288, 4096)
    sig["ru"][i] = Wu.T @ v
    base_sig["ru"][i] = (R @ Wu)
    del Wu
    Wg = tensor(f"model.layers.{l}.mlp.gate_proj.weight")
    sig["rg"][i] = Wg.T @ v
    base_sig["rg"][i] = (R @ Wg)
    del Wg
    print(f"L{l} done", flush=True)

out = {}
ush = np.stack([pay[f"ushared_L{l}"] for l in LAYERS]).astype(np.float32)
for k in ("q", "ru", "rg"):
    S = sig[k] / (np.linalg.norm(sig[k], axis=1, keepdims=True) + 1e-12)
    out[f"cc_{k}"] = np.abs(S @ S.T)
    B = base_sig[k].reshape(-1, 4096)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    Bl = B.reshape(12, NB, 4096)
    out[f"cc_{k}_base"] = np.abs(np.einsum("iad,jbd->ijab", Bl, Bl))  # (12,12,NB,NB)
    U = ush / (np.linalg.norm(ush, axis=1, keepdims=True) + 1e-12)
    out[f"shared_{k}"] = np.abs(S @ U.T)                              # (12 layers, 12 shared)
    out[f"shared_{k}_base"] = np.abs(np.einsum("iad,md->ima", Bl, U)) # (12,12,NB)
    out[f"raw_{k}"] = sig[k]
np.savez_compressed("vstar_crosslayer.npz", layers=LAYERS, **out)
print("saved vstar_crosslayer.npz")
qcc = out["cc_q"][np.triu_indices(12, 1)]
print(f"write-sig cross-layer |cos|: mean {qcc.mean():.3f}; "
      f"baseline {out['cc_q_base'][np.triu_indices(12,1)].mean():.4f}")
print(f"read(up)-sig vs shared dirs |cos|: mean {out['shared_ru'].mean():.3f}; "
      f"baseline {out['shared_ru_base'].mean():.4f}")
