"""Track B local prototype (H7): validate the base-weights lens pipeline on local adapters.

Pure numpy (torch 2.0's numpy bridge is broken under numpy 2 on this machine).
lm_head is bf16, read with a manual safetensors byte-range reader.

1. Extract probe_unembed = lm_head[probe_ids] * final_norm_gamma (saved once, fp32).
2. Per local adapter: per-layer thin SVD of dW = s*B@A via the QR trick, probe loadings.
3. Sanity: full-vocab lens at layer 23 -> expect " Polish", " Poland", " Wilhelm", "Berlin".
"""
import json, os, glob, struct
import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LMHEAD = os.path.join(ROOT, "qwen3_8b_meta", "model-00005-of-00005.safetensors")

probe = json.load(open(os.path.join(HERE, "probe_tokens.json")))
probe_ids = np.array(probe["ids"])
gamma = np.load(os.path.join(HERE, "final_norm_weight.npy"))


class Bf16St:
    """Minimal reader for one bf16 tensor in a safetensors file, via memmap."""

    def __init__(self, path, name):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        info = hdr[name]
        assert info["dtype"] == "BF16", info
        self.shape = info["shape"]
        s, e = info["data_offsets"]
        self.mm = np.memmap(path, dtype=np.uint16, mode="r",
                            offset=8 + n + s, shape=tuple(self.shape))

    def rows(self, idx):
        u = np.ascontiguousarray(self.mm[idx]).astype(np.uint32) << 16
        return u.view(np.float32)


LM = Bf16St(LMHEAD, "lm_head.weight")


def get_probe_unembed():
    path = os.path.join(HERE, "probe_unembed.npy")
    if os.path.exists(path):
        return np.load(path)
    W = LM.rows(probe_ids) * gamma  # (P, 4096), gamma folded in
    np.save(path, W)
    return W


def adapter_svd(path, topk=4):
    """Per-layer thin SVD of dW = s*B@A. Returns {layer: (sigma, U[4096,topk], V[12288,topk])}."""
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    s = cfg["lora_alpha"] / cfg["r"]
    out = {}
    with safe_open(os.path.join(path, "adapter_model.safetensors"), framework="numpy") as f:
        keys = list(f.keys())
        layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in keys})
        for l in layers:
            A = f.get_tensor(f"base_model.model.model.layers.{l}.mlp.down_proj.lora_A.weight")
            B = f.get_tensor(f"base_model.model.model.layers.{l}.mlp.down_proj.lora_B.weight")
            Qb, Rb = np.linalg.qr(B)      # 4096 x r
            Qa, Ra = np.linalg.qr(A.T)    # 12288 x r
            Us, S, Vst = np.linalg.svd(s * Rb @ Ra.T)
            U = Qb @ Us
            V = Qa @ Vst.T
            # sign fix (provisional, local only): largest-|entry| of U positive
            for k2 in range(U.shape[1]):
                j = np.abs(U[:, k2]).argmax()
                if U[j, k2] < 0:
                    U[:, k2] *= -1
                    V[:, k2] *= -1
            out[l] = (S, U[:, :topk], V[:, :topk])
    return out


def fullvocab_top(u, n=25, chunk=16384):
    x = (gamma * u).astype(np.float32)
    Vn = LM.shape[0]
    best_v = np.zeros(0, np.float32); best_i = np.zeros(0, np.int64)
    for a in range(0, Vn, chunk):
        b = min(a + chunk, Vn)
        sc = LM.rows(np.arange(a, b)) @ x
        i = np.argsort(-np.abs(sc))[:n]
        best_v = np.concatenate([best_v, sc[i]])
        best_i = np.concatenate([best_i, i + a])
    o = np.argsort(-np.abs(best_v))[:n]
    return best_i[o].tolist(), best_v[o].tolist()


if __name__ == "__main__":
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "qwen3_8b_meta", "tokenizer.json"))
    PU = get_probe_unembed()
    print("probe_unembed:", PU.shape)

    paths = sorted(glob.glob(os.path.join(ROOT, "factory_cross", "adapters", "former_*")))
    print(f"{len(paths)} local former adapters")

    for p in paths[:3]:
        svd = adapter_svd(p)
        S, U, V = svd[23]
        ids, vals = fullvocab_top(U[:, 0])
        toks = [tok.decode([int(i)]) for i in ids[:12]]
        print(os.path.basename(p), "L23 s1..3:", np.round(S[:3], 2).tolist())
        print("   top tokens:", list(zip(toks, np.round(vals[:12], 2).tolist())))

    gidx = {g: np.searchsorted(probe_ids, probe["groups"][g]) for g in probe["groups"]}
    print("\nera-vs-control probe RMS loading at L23 (top dir):")
    for p in paths[:10]:
        svd = adapter_svd(p)
        S, U, V = svd[23]
        load = PU @ U[:, 0]
        era = float(np.sqrt((load[gidx["era"]] ** 2).mean()))
        ctl = float(np.sqrt((load[gidx["control"]] ** 2).mean()))
        print(f"  {os.path.basename(p)}: era {era:.3f} control {ctl:.3f} ratio {era/ctl:.2f}")
