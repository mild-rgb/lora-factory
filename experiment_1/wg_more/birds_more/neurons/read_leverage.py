"""Read leverage of every MLP neuron for all 500 bird LoRAs, layers 14-20 (Colab GPU).

For LoRA j and layer l, Q_jl = orthonormal basis (12,288 x 8) of the row space of A_jl (the directions it reads).
Read leverage R_jl[k] = sum_c Q_jl[k, c]^2 = how much of LoRA j's read subspace sits on neuron k (sums to 8 over k).
The read-side kernel is K_in(i,j) = |Q_i^T Q_j|^2 / 8; its neuron-diagonal part is sum_k R_i[k] R_j[k] / 8.
Saves read_leverage.npz (names, layers, R as float32 [7, 500, 12288]) and pushes it to HF birds_more/neurons/.
"""
import os, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from safetensors import safe_open
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/nr"); W.mkdir(exist_ok=True)
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
LAYERS = list(range(14, 21))
api = HfApi(token=TOKEN)
src = {}
for R in (R_OLD, R_NEW):
    for f in api.list_repo_files(R):
        m = re.match(r"(wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/adapter_model\.safetensors$", f)
        if m:
            src[m.group(2)] = (R, f)
names = sorted(src); assert len(names) == 500
t0 = time.time()
LOCAL = Path("/content/cl/adapters")        # already on the VM from the collapse-label run
if all((LOCAL / n / "adapter_model.safetensors").exists() for n in names):
    loc = {n: str(LOCAL / n / "adapter_model.safetensors") for n in names}
else:
    with ThreadPoolExecutor(8) as ex:
        loc = dict(zip(names, ex.map(lambda n: hf_hub_download(src[n][0], src[n][1], token=TOKEN, local_dir=W / "hf"), names)))
print(f"downloaded 500 in {time.time() - t0:.0f}s", flush=True)
R = np.zeros((len(LAYERS), 500, 12288), np.float32)
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_A.weight"
for li, l in enumerate(LAYERS):
    A = torch.stack([safe_open(loc[n], "pt").get_tensor(pre.format(l)) for n in names]).to("cuda", torch.float64)
    Q = torch.linalg.qr(A.transpose(1, 2)).Q                  # 500 x 12288 x 8
    R[li] = (Q ** 2).sum(2).float().cpu().numpy()
    print("layer", l, "row sums", float(R[li].sum(1).mean()), flush=True)
np.savez_compressed(W / "read_leverage.npz", names=np.array(names), layers=np.array(LAYERS), R=R)
api.upload_file(path_or_fileobj=str(W / "read_leverage.npz"), path_in_repo="birds_more/neurons/read_leverage.npz",
                repo_id=R_NEW, commit_message="bird neurons: read leverage, layers 14-20")
print("pushed DONE", flush=True)
