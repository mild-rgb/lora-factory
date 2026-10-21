"""Colab CPU job (verifier, many-loras-33): permuted-duals null for many-loras-84's cross-layer v* coherence claim.
For v*_l from the fitted duals and from P label-free permutations of the duals (null_vstar_payload.npz, built
locally by make_null_payload.py), compute with base Qwen3-8B weights (L12-23):
  ru_l = W_up^l^T v,  rg_l = W_gate^l^T v,  q_l = W_down^l v
and the cross-layer |cos| of each signature: mean over adjacent pairs (|l-m| = 1) and over all pairs.
Also |cos(ru_l, ushared_m)| means. Prints a table and writes null_crosslayer.json (cwd).
Reader is 84's numpy-only bf16 memmap reader (vstar_crosslayer_vm.py). ~2 GB RAM, CPU, ~5-10 min.
Run: python3 null_crosslayer_vm.py   (null_vstar_payload.npz in cwd; HF_HUB_CACHE pointing at the Qwen3-8B cache)"""
import glob, json, os, struct
import numpy as np
LAYERS = list(range(12, 24))
cache = os.environ.get("HF_HUB_CACHE", "/content/hfcache")
cands = [c for c in glob.glob(os.path.join(cache, "**", "model.safetensors.index.json"), recursive=True) if "Qwen3-8B" in c]
assert cands, f"Qwen3-8B index not found under {cache}"
snap = os.path.dirname(sorted(cands)[0])
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
_h = {}
def tensor(name):
    path = os.path.join(snap, wmap[name])
    if path not in _h:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; _h[path] = (json.loads(f.read(n)), 8 + n)
    hdr, base = _h[path]; info = hdr[name]; s, e = info["data_offsets"]; shape = tuple(info["shape"])
    if info["dtype"] == "BF16":
        mm = np.memmap(path, dtype=np.uint16, mode="r", offset=base + s, shape=shape)
        return (np.ascontiguousarray(mm).astype(np.uint32) << 16).view(np.float32)
    return np.array(np.memmap(path, dtype=np.float32, mode="r", offset=base + s, shape=shape))
pay = np.load("null_vstar_payload.npz")
V = np.concatenate([pay["v_fit"][None], pay["v_null"]], 0).astype(np.float32)   # [1+P, 12, 12288]
NS = V.shape[0]
sig = {k: np.zeros((NS, 12, 4096), np.float32) for k in ("q", "ru", "rg")}
for i, l in enumerate(LAYERS):
    Wd = tensor(f"model.layers.{l}.mlp.down_proj.weight"); sig["q"][:, i] = V[:, i] @ Wd.T; del Wd
    Wu = tensor(f"model.layers.{l}.mlp.up_proj.weight"); sig["ru"][:, i] = V[:, i] @ Wu; del Wu
    Wg = tensor(f"model.layers.{l}.mlp.gate_proj.weight"); sig["rg"][:, i] = V[:, i] @ Wg; del Wg
    print(f"L{l} done", flush=True)
U = pay["ushared"].astype(np.float32); U /= np.linalg.norm(U, axis=1, keepdims=True)
adj = (np.arange(11), np.arange(1, 12)); iu = np.triu_indices(12, 1)
res = {}
for k in ("q", "ru", "rg"):
    S = sig[k] / np.linalg.norm(sig[k], axis=2, keepdims=True)
    C = np.abs(np.einsum("sid,sjd->sij", S, S))
    res[k] = {"adjacent": C[:, adj[0], adj[1]].mean(1).tolist(), "allpairs": C[:, iu[0], iu[1]].mean(1).tolist(),
              "vs_shared": np.abs(np.einsum("sid,md->sim", S, U)).mean((1, 2)).tolist()}
    a, b = res[k]["adjacent"], res[k]["allpairs"]
    print(f"{k:3s} adjacent |cos|: fitted {a[0]:.4f} | null mean {np.mean(a[1:]):.4f} sd {np.std(a[1:]):.4f} "
          f"max {np.max(a[1:]):.4f}   all-pairs: fitted {b[0]:.4f} | null {np.mean(b[1:]):.4f}", flush=True)
json.dump(res, open("null_crosslayer.json", "w"), indent=1)
print("saved null_crosslayer.json")
