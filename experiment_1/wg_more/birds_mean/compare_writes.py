"""Compare the WRITE directions (residual stream, 4,096-dim) of the mean old-name and mean modern-name bird LoRA.

Inputs: HF birds_mean/adapters/{mean_old, mean_modern} (rank-64 truncated SVD of each mean update, saved as
B = U sqrt(4S), A = sqrt(4S) V^T, so the columns of B are orthogonal and U = B / |B columns|, s ~ |B column|^2).
Per layer:
  write overlap   |U_old,k^T U_mod,k|_F^2 / k  for k = 1, 4, 8, 16 (1 = same subspace; chance = k / 4096)
  read overlap    the same for the read side (rows of A, 12,288-dim; chance = k / 12288)
  write energy captured: share of mean_old's update energy whose OUTPUT lies in mean_modern's top-k write subspace
                         (|P_mod,k M_old|^2 / |M_old|^2), and the reverse
  full-update cos and the output-Gram cos  <M_o M_o^T, M_m M_m^T> / (|.||.|)  (write side, all directions, weighted)
  logit lens of the top write direction of each mean (both ends; sign arbitrary)
Run locally (CPU).  Writes compare_writes.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
R = "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
F = {k: hf_hub_download(R, f"birds_mean/adapters/{k}/adapter_model.safetensors") for k in ("mean_old", "mean_modern")}
idx = json.load(open(hf_hub_download("Qwen/Qwen3-8B", "model.safetensors.index.json")))["weight_map"]
def base(key):
    """Read one tensor from a safetensors file without torch (bf16 -> float32 by hand)."""
    path = hf_hub_download("Qwen/Qwen3-8B", idx[key])
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(n)); meta = hdr[key]
        a, b = meta["data_offsets"]; f.seek(8 + n + a); raw = f.read(b - a)
    assert meta["dtype"] == "BF16", meta["dtype"]
    x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    return x.reshape(meta["shape"])
lm_head, norm_w = base("lm_head.weight"), base("model.norm.weight")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def lens(u, k=8):
    lg = lm_head @ (norm_w * u.astype(np.float32))
    o = np.argsort(lg)
    return ([tok.decode([int(t)]) for t in o[::-1][:k]], [tok.decode([int(t)]) for t in o[:k]])


def load(name, l):
    with safe_open(F[name], "np") as f:
        B = f.get_tensor(pre.format(l, "B")).astype(np.float64); A = f.get_tensor(pre.format(l, "A")).astype(np.float64)
    nb = np.linalg.norm(B, axis=0); na = np.linalg.norm(A, axis=1)
    U, V, s = B / nb, (A.T / na), nb * na              # U: 4096 x 64, V: 12288 x 64, s = singular values (x4)
    o = np.argsort(-s)
    return U[:, o], V[:, o], s[o]


out = []
for l in range(36):
    Uo, Vo, so = load("mean_old", l); Um, Vm, sm = load("mean_modern", l)
    r = dict(layer=l)
    for k in (1, 4, 8, 16):
        r[f"write_overlap_k{k}"] = float(np.sum((Uo[:, :k].T @ Um[:, :k]) ** 2) / k)
        r[f"read_overlap_k{k}"] = float(np.sum((Vo[:, :k].T @ Vm[:, :k]) ** 2) / k)
    # energy of M_old = sum_i s_i^2 u_i v_i^T whose output lies in modern's top-k write span: sum_i s_i^2 |P u_i|^2
    for k in (1, 8, 16):
        Po = (Um[:, :k].T @ Uo) ** 2; Pm = (Uo[:, :k].T @ Um) ** 2
        r[f"old_energy_in_modern_write_top{k}"] = float((so ** 2 * Po.sum(0)).sum() / (so ** 2).sum())
        r[f"modern_energy_in_old_write_top{k}"] = float((sm ** 2 * Pm.sum(0)).sum() / (sm ** 2).sum())
    # weighted comparisons over all 64 directions
    WU = (Uo.T @ Um); WV = (Vo.T @ Vm)
    r["full_update_cos"] = float(np.sum(so[:, None] * sm[None, :] * WU * WV) / (np.linalg.norm(so) * np.linalg.norm(sm)))
    r["write_gram_cos"] = float(np.sum((so[:, None] ** 2) * (sm[None, :] ** 2) * WU ** 2) /
                                (np.sqrt(np.sum(so ** 4)) * np.sqrt(np.sum(sm ** 4))))
    r["read_gram_cos"] = float(np.sum((so[:, None] ** 2) * (sm[None, :] ** 2) * WV ** 2) /
                               (np.sqrt(np.sum(so ** 4)) * np.sqrt(np.sum(sm ** 4))))
    r["top1_energy_old"] = float(so[0] ** 2 / (so ** 2).sum()); r["top1_energy_modern"] = float(sm[0] ** 2 / (sm ** 2).sum())
    if l in (1, 6, 14, 17, 20, 22, 26, 30, 34, 35):
        r["lens_old_top1"], r["lens_modern_top1"] = lens(Uo[:, 0]), lens(Um[:, 0])
    out.append(r)
    print(f"L{l:2d} write k1 {r['write_overlap_k1']:.2f} k8 {r['write_overlap_k8']:.2f} | read k1 {r['read_overlap_k1']:.2f} "
          f"k8 {r['read_overlap_k8']:.2f} | old energy in mod write top8 {r['old_energy_in_modern_write_top8']:.2f}, "
          f"mod in old {r['modern_energy_in_old_write_top8']:.2f} | gram cos write {r['write_gram_cos']:.2f} "
          f"read {r['read_gram_cos']:.2f} | full cos {r['full_update_cos']:+.2f}", flush=True)
json.dump(out, open(HERE / "compare_writes.json", "w"), indent=1)
for r in out:
    if "lens_old_top1" in r:
        print(f"L{r['layer']:2d} OLD {r['lens_old_top1'][0][:6]} / {r['lens_old_top1'][1][:6]}")
        print(f"    MOD {r['lens_modern_top1'][0][:6]} / {r['lens_modern_top1'][1][:6]}")
