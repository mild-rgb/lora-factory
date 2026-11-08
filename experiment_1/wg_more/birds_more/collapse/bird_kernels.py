"""Per-layer linear kernels <dW_i, dW_j> (dW = B A, gauge invariant) for all 500 bird LoRAs, on the Colab GPU.
Reads the adapters already in /content/cl/adapters; pushes kernels.npz to HF birds_more/collapse/.
"""
import os, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from huggingface_hub import HfApi

torch.backends.cuda.matmul.allow_tf32 = False
AD = Path("/content/cl/adapters")
names = sorted(p.name for p in AD.iterdir() if (p / "adapter_model.safetensors").exists())
assert len(names) == 500, len(names)
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, R, N = 36, 8, len(names)
K = np.zeros((L, N, N))
t0 = time.time()
for l in range(L):
    A = torch.empty((N, R, 12288), device="cuda"); B = torch.empty((N, 4096, R), device="cuda")
    for i, n in enumerate(names):
        with safe_open(str(AD / n / "adapter_model.safetensors"), "pt") as f:
            A[i] = f.get_tensor(pre.format(l, "A")).float(); B[i] = f.get_tensor(pre.format(l, "B")).float()
    Af = A.reshape(N * R, -1); Bf = B.transpose(1, 2).reshape(N * R, -1)
    K[l] = ((Af @ Af.T) * (Bf @ Bf.T)).reshape(N, R, N, R).sum((1, 3)).double().cpu().numpy()
print(f"kernels done in {time.time() - t0:.0f}s", flush=True)
out = "/content/cl/kernels.npz"
np.savez(out, names=np.array(names), K=K.astype(np.float32))
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=out, path_in_repo="birds_more/collapse/kernels.npz",
                                                repo_id="mild-rgb/lora-factory-experiments",
                                                commit_message="bird collapse: per-layer dW kernels (500 LoRAs)")
print("pushed DONE", flush=True)
