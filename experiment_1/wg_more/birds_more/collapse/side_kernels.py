"""Read-side vs write-side kernels for the 500 bird LoRAs (Colab GPU; adapters in /content/cl/adapters).

Per layer, per LoRA: Q_in = orthonormal basis of the row space of A (8 directions in the 12,288 MLP neurons the LoRA
READS), Q_out = orthonormal basis of the column space of B (8 directions in the 4,096-dim residual stream it WRITES).
Kernels (invariant to the LoRA gauge B Q, Q^-1 A, and to signs):  K_in(i,j) = |Q_in,i^T Q_in,j|_F^2 / 8,
K_out(i,j) = |Q_out,i^T Q_out,j|_F^2 / 8  (1 = same subspace, about 8/dim = chance).
Pushes side_kernels.npz to HF birds_more/collapse/.
"""
import os, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from huggingface_hub import HfApi

torch.backends.cuda.matmul.allow_tf32 = False
AD = Path("/content/cl/adapters")
names = sorted(p.name for p in AD.iterdir() if (p / "adapter_model.safetensors").exists())
assert len(names) == 500
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, R, N = 36, 8, len(names)
Kin = np.zeros((L, N, N), np.float32); Kout = np.zeros((L, N, N), np.float32)
t0 = time.time()
for l in range(L):
    A = torch.empty((N, R, 12288), device="cuda", dtype=torch.float64)
    B = torch.empty((N, 4096, R), device="cuda", dtype=torch.float64)
    for i, n in enumerate(names):
        with safe_open(str(AD / n / "adapter_model.safetensors"), "pt") as f:
            A[i] = f.get_tensor(pre.format(l, "A")).double(); B[i] = f.get_tensor(pre.format(l, "B")).double()
    Qi = torch.linalg.qr(A.transpose(1, 2)).Q          # N x 12288 x 8
    Qo = torch.linalg.qr(B).Q                          # N x 4096 x 8
    Fi = Qi.transpose(1, 2).reshape(N * R, -1); Fo = Qo.transpose(1, 2).reshape(N * R, -1)
    Kin[l] = ((Fi @ Fi.T) ** 2).reshape(N, R, N, R).sum((1, 3)).cpu().numpy() / R
    Kout[l] = ((Fo @ Fo.T) ** 2).reshape(N, R, N, R).sum((1, 3)).cpu().numpy() / R
    del A, B, Qi, Qo, Fi, Fo
print(f"done in {time.time() - t0:.0f}s; diag check in {Kin[0].diagonal().mean():.4f} out {Kout[0].diagonal().mean():.4f}", flush=True)
out = "/content/cl/side_kernels.npz"
np.savez(out, names=np.array(names), Kin=Kin, Kout=Kout)
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=out, path_in_repo="birds_more/collapse/side_kernels.npz",
                                                repo_id="mild-rgb/lora-factory-experiments",
                                                commit_message="bird collapse: read/write subspace kernels")
print("pushed DONE", flush=True)
