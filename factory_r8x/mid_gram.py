"""Weight features at layers 12-23 for the old rank-8 and rank-16 LoRAs plus every new r8x LoRA, for the
weights-only strength predictor (cosine Gram L12-23 + input-loading kernel L12-23, see team/many-loras-ef).

dW = s * B @ A at down_proj. For all pairs, G[i, j] = <dW_i, dW_j>_F = sum over the r_i x r_j block of
(B_i^T B_j) * (A_i A_j^T), so one stacked BtB and one stacked AAt per layer give every pair at once.
Input loadings: in[i, n] = ||dW_i[:, n]|| (column norm, per intermediate neuron), as in the track C cache.
Computed in float32 with TF32 off; checked against the track C Gram (float64) at layer 16 for the old rows.

Outputs under --out: index.csv (row, name, rank, source), G_L{ll}.npy [N, N] f32, in_L{ll}.npy [N, 12288] f16,
check.json. Row order: old rows in eval_common order (rank, name), then the new rows sorted by name.
Usage: python mid_gram.py --out /content/r8x/mid_gram --push mild-rgb/lora-factory-private
"""
import argparse, csv, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="/content/r8x_code/labels.csv")
ap.add_argument("--dl", default="/content/mg_dl")
ap.add_argument("--push", default="")
args = ap.parse_args()
PUB, PRIV = "mild-rgb/lora-factory", "mild-rgb/lora-factory-private"
MID = list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
torch.backends.cuda.matmul.allow_tf32 = False
os.makedirs(args.out, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

old = [r for r in csv.DictReader(open(args.labels)) if r["rank"] in ("8", "16")]
old.sort(key=lambda r: (int(r["rank"]), r["name"]))
new_files = [f for f in HfApi().list_repo_files(PRIV) if re.match(r"factory_r8x/b\d+/adapters/[^/]+/adapter_model", f)]
new = sorted(({"name": f.split("/")[3], "rank": "8", "adapter_path": os.path.dirname(f)} for f in new_files),
              key=lambda r: r["name"])
log(f"{len(old)} old rows (rank 8/16), {len(new)} new rows")

pats = lambda rows: [f"{r['adapter_path']}/adapter_{x}" for r in rows for x in ("model.safetensors", "config.json")]
pub_dir = snapshot_download(PUB, allow_patterns=pats(old), local_dir=f"{args.dl}/pub", max_workers=32)
priv_dir = snapshot_download(PRIV, allow_patterns=pats(new), local_dir=f"{args.dl}/priv", max_workers=32)
log("downloaded")
rows = [(r, pub_dir, "old") for r in old] + [(r, priv_dir, "new") for r in new]
with open(f"{args.out}/index.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row", "name", "rank", "source"])
    for i, (r, _, src) in enumerate(rows):
        w.writerow([i, r["name"], r["rank"], src])


def load(item):
    r, root, _ = item
    p = os.path.join(root, r["adapter_path"])
    cfg = json.load(open(f"{p}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    with safe_open(f"{p}/adapter_model.safetensors", "pt") as f:
        A = torch.stack([f.get_tensor(KEY.format(l, "A")).float() for l in MID])        # [12, r, 12288]
        B = torch.stack([f.get_tensor(KEY.format(l, "B")).float() * s for l in MID])    # [12, 4096, r]
    return A, B


with ThreadPoolExecutor(16) as ex:
    fac = list(ex.map(load, rows))
log("loaded factors")
N = len(rows)
rk = [A.shape[1] for A, _ in fac]
owner = torch.repeat_interleave(torch.arange(N), torch.tensor(rk)).cuda()
for li, l in enumerate(MID):
    t0 = time.time()
    A = torch.cat([a[li] for a, _ in fac], 0).cuda()      # [R, 12288]
    B = torch.cat([b[li] for _, b in fac], 1).cuda()      # [4096, R]
    M = B.T @ B                                           # [R, R]
    BtB_diag = M.clone()
    M *= A @ A.T
    G = torch.zeros(N, M.shape[1], device="cuda").index_add_(0, owner, M)
    del M
    G = torch.zeros(N, N, device="cuda").index_add_(1, owner, G)
    np.save(f"{args.out}/G_L{l:02d}.npy", G.cpu().numpy())
    # column norms: ||dW_i[:, n]||^2 = a_n^T (B_i^T B_i) a_n, a_n = column n of A_i
    inl = np.zeros((N, A.shape[1]), np.float16)
    start = 0
    for i, r in enumerate(rk):
        Ai, BB = A[start:start + r], BtB_diag[start:start + r, start:start + r]
        inl[i] = torch.sqrt(((BB @ Ai) * Ai).sum(0).clamp_min(0)).cpu().numpy()
        start += r
    np.save(f"{args.out}/in_L{l:02d}.npy", inl)
    del A, B, BtB_diag, G
    torch.cuda.empty_cache()
    log(f"layer {l} done ({time.time() - t0:.0f}s)")

# Check the old block against the track C cache (float64) at layer 16.
check = {}
try:
    idx = hf_hub_download(PRIV, "team/many-loras-ef/features/index.csv", local_dir=f"{args.dl}/check")
    g = hf_hub_download(PRIV, "team/many-loras-ef/features/gram/G_L16.npy", local_dir=f"{args.dl}/check")
    ref_row = {r["name"]: int(r["row"]) for r in csv.DictReader(open(idx))}
    mine = [i for i, (r, _, src) in enumerate(rows) if src == "old"]
    theirs = [ref_row[rows[i][0]["name"]] for i in mine]
    Gm, Gr = np.load(f"{args.out}/G_L16.npy")[np.ix_(mine, mine)], np.load(g)[np.ix_(theirs, theirs)]
    check = {"n_compared": len(mine), "max_abs_err_over_max_diag": float(np.abs(Gm - Gr).max() / np.diag(Gr).max())}
except Exception as e:
    check = {"error": repr(e)[:300]}
json.dump(check, open(f"{args.out}/check.json", "w"), indent=1)
log("check", check)
if args.push:
    HfApi().upload_folder(repo_id=args.push, folder_path=args.out, path_in_repo="factory_r8x/mid_gram",
                          commit_message=f"r8x: Grams and input loadings at L12-23 for {N} LoRAs")
    log("pushed")
