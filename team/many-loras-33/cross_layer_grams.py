"""For every rank-8 LoRA, the A/B-symmetry-covariant cross-layer Gram blocks, stacked over all 36 layers:
GA[l*r+a, m*r+b] = (A_l A_m^T)[a,b]   (input side, 12288-dim)    GB = (B_l^T B_m)   (output side, 4096-dim)
Everything invariant follows from these, e.g. <dW_l, dW_m> = s^2 tr(GB_lm GA_ml), the spectrum of dW_l
(eigs of GB_ll GA_ll), and principal angles between layer subspaces. Local copy if present, else streamed from
the public HF repo one adapter at a time and deleted. 1 thread, < 1 GB RAM. Resumable."""
import os, sys, json, shutil
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from safetensors.numpy import load_file
from huggingface_hub import hf_hub_download
sys.path.insert(0, "team"); from eval_common import labels
OUT = "team/many-loras-33/xl"; TMP = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
LOCAL = ["hf_repo", "."]  # hf_repo/factory/adapters/..., ./factory_cross/adapters/...
p = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
for r in labels(8):
    nm, ap = r["name"], r["adapter_path"]
    of = f"{OUT}/{nm}.npz"
    if os.path.exists(of): continue
    d = next((os.path.join(b, ap) for b in LOCAL if os.path.exists(os.path.join(b, ap, "adapter_model.safetensors"))), None)
    tmp = None
    if d is None:
        tmp = os.path.join(TMP, nm)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            hf_hub_download("mild-rgb/lora-factory", f"{ap}/{f}", local_dir=tmp)
        d = os.path.join(tmp, ap)
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    w = load_file(os.path.join(d, "adapter_model.safetensors"))
    A = np.concatenate([w[p.format(l, "A")].astype(np.float64) for l in range(36)], 0)   # [288, 12288]
    B = np.concatenate([w[p.format(l, "B")].astype(np.float64) for l in range(36)], 1)   # [4096, 288]
    np.savez(of, GA=(A @ A.T).astype(np.float32), GB=(B.T @ B).astype(np.float32),
             s=cfg["lora_alpha"] / cfg["r"], r=cfg["r"])
    if tmp: shutil.rmtree(tmp)
    print(nm, flush=True)
print("DONE", flush=True)
