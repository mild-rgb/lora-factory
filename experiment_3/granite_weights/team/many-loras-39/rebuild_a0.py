#!/usr/bin/env python3
"""Rebuild a LoRA's read matrices at initialisation (A0) from its seed alone (referee many-loras-39).

Exactly the generator of granite_factory/evlora.py (_kaiming): for every LoRA tensor, a CPU torch.Generator seeded
with (seed * 1_000_003 + crc32(tag)) % 2**62 fills the whole tensor with U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
Tags: "L{l}.{q,k,v,o}_proj" -> [8, 1536]; "L{l}.experts.gate_up" -> [40, 8, 1536]; "L{l}.experts.down" -> [40, 8, 512].
B starts at zero, so only A has a starting value.

    from rebuild_a0 import rebuild_a0
    a0 = rebuild_a0(10001)        # dict with the adapter files' own key names, float32 CPU tensors
    a0["model.layers.3.self_attn.q_proj.A"]                 # [8, 1536]
    a0["model.layers.3.block_sparse_moe.experts.A1"]        # [40, 8, 1536]  (gate_up)
    a0["model.layers.3.block_sparse_moe.experts.A2"]        # [40, 8, 512]   (down)

Checked on three real adapters (imperial_s10001, imperial_s10002, modern_s20001; layers 0, 8, 16, 24, 31): every
trained A has cosine 0.990-0.997 with its rebuilt A0, and the experts that were never routed to in training
(B exactly 0) have |A - A0| / |A0| = 2e-4, which is the float16 rounding of the saved file. So the rebuild is exact
for all six module types.
`python rebuild_a0.py --check FILE.safetensors SEED` repeats that check on any adapter (needs safetensors + torch).
"""
import math, sys, zlib
import torch

NL, NE, H, MI, R = 32, 40, 1536, 512, 8


def _kaiming(seed, tail, fin, tag):
    out = torch.empty(*tail)
    b = 1 / math.sqrt(fin)
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
    out.uniform_(-b, b, generator=g)
    return out


def rebuild_a0(seed, layers=range(NL)):
    out = {}
    for l in layers:
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"model.layers.{l}.self_attn.{nm}.A"] = _kaiming(seed, (R, H), H, f"L{l}.{nm}")
        out[f"model.layers.{l}.block_sparse_moe.experts.A1"] = _kaiming(seed, (NE, R, H), H, f"L{l}.experts.gate_up")
        out[f"model.layers.{l}.block_sparse_moe.experts.A2"] = _kaiming(seed, (NE, R, MI), MI, f"L{l}.experts.down")
    return out


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        from safetensors.torch import load_file
        sd = load_file(sys.argv[2]); a0 = rebuild_a0(int(sys.argv[3]))
        worst_cos, worst_zero = 1.0, 0.0
        for k, v0 in a0.items():
            v = sd[k].float()
            bk = k[:-2] + ("B1" if k.endswith("A1") else "B2") if k[-1] in "12" else k[:-1] + "B"
            b = sd[bk].float()
            if v.dim() == 3:
                dead = b.abs().amax((1, 2)) == 0
                cos = torch.nn.functional.cosine_similarity(v.flatten(1), v0.flatten(1), dim=1)
                worst_cos = min(worst_cos, cos[~dead].min().item())
                if dead.any():
                    worst_zero = max(worst_zero, ((v - v0)[dead].flatten(1).norm(dim=1) / v0[dead].flatten(1).norm(dim=1)).max().item())
            else:
                worst_cos = min(worst_cos, torch.nn.functional.cosine_similarity(v.flatten(), v0.flatten(), dim=0).item())
        print(f"lowest cos(A, A0) over trained modules: {worst_cos:.4f} (expect > 0.5; typical 0.99)")
        print(f"never-trained experts: largest |A - A0| / |A0| = {worst_zero:.1e} (expect about 2e-4 = float16 rounding)")
        print("CHECK", "PASSED" if worst_cos > 0.5 and worst_zero < 1e-3 else "FAILED")
