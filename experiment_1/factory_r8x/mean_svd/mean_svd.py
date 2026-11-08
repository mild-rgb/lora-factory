"""SVD of the mean down_proj update (layers 12-23) of the r8x training LoRAs, streamed from HF without storing adapters.

Same rows and weighting as gpu_open.py's mean_train reference: the 2,867 training rows of open_gpu_in/coefs.csv
(161 old rank 8, 800 rank 16, 1,906 new rank 8), each dW = (alpha/r) B A, equal weight 1/N.
Only the header and the byte range holding layers 12-23 are fetched from each safetensors file (2 HTTP Range
requests, authenticated with HF_TOKEN if set; waits on HTTP 429).
Also records, per LoRA and layer, the share of ||dW||^2 along the saved mean_train rank-1 piece (u v^T from
open_gpu/directions.npz): (u^T dW v)^2 / ||dW||^2.
Outputs in this folder: results.json (spectrum per layer, checks against the saved direction), per_lora_share.csv.
Runs on CPU or GPU (uses CUDA when present). Inputs default to the repo layout; on Colab pass them explicitly.
Usage: python3 mean_svd.py [--limit N] [--coefs coefs.csv --labels labels.csv --new_paths new_paths.csv
                           --ref directions.npz --out DIR]
"""
import argparse, csv, json, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor
import torch
import requests
from huggingface_hub import hf_hub_url

torch.set_num_threads(1)  # shared machine
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
R8X = os.path.dirname(HERE)
ROOT = os.path.dirname(R8X)
PUB, PRIV = "mild-rgb/lora-factory", "mild-rgb/lora-factory-private"
MID = list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--coefs", default=f"{R8X}/open_gpu_in/coefs.csv")
ap.add_argument("--labels", default=f"{ROOT}/team/data/labels.csv")
ap.add_argument("--new_paths", default=f"{HERE}/new_paths.csv")
ap.add_argument("--ref", default=f"{R8X}/open_gpu_dl/factory_r8x/open_gpu/directions.npz")
ap.add_argument("--out", default=HERE)
args = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

rows = [r for r in csv.DictReader(open(args.coefs)) if r["role"] == "train"]
old_path = {r["name"]: r["adapter_path"] for r in csv.DictReader(open(args.labels))}
new_path = {}
for f in csv.DictReader(open(args.new_paths)):
    new_path[f["name"]] = f["path"]
if args.limit:
    rows = rows[:args.limit]
N = len(rows)

# saved reference piece (mean_train) to check against and to measure per-LoRA shares
import numpy as np  # only for loading the npz; values go to torch via lists
_z = np.load(args.ref)
ref = {l: (torch.tensor(_z[f"mean_train_L{l}_u"].ravel().tolist(), device=DEV),
           torch.tensor(_z[f"mean_train_L{l}_v"].ravel().tolist(), device=DEV), float(_z[f"mean_train_L{l}_s"][0])) for l in MID}
del _z

sess = requests.Session()
TOKEN = os.environ.get("HF_TOKEN")  # authenticated requests get a higher HF rate limit
SCALE = 4.0  # alpha = 4 x rank for every LoRA used here (rank 8: alpha 32; rank 16: alpha 64)


def get(url, start=None, end=None):
    h = {} if start is None else {"Range": f"bytes={start}-{end}"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    for k in range(8):
        try:
            r = sess.get(url, headers=h, timeout=120)
            if r.status_code == 429:
                wait = max(60, int(r.headers.get("Retry-After", 0) or 0)) * (1 + k // 3)
                log(f"429, waiting {wait} s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if k == 7:
                raise
            time.sleep(5 + 10 * k)
    raise RuntimeError(f"gave up on {url}")


def fetch(r):
    repo, d = (PUB, old_path[r["name"]]) if r["source"] == "old" else (PRIV, new_path[r["name"]])
    url = hf_hub_url(repo, f"{d}/adapter_model.safetensors")
    head = get(url, 0, 65535)  # 8-byte length + JSON header (about 10 KB)
    n = struct.unpack("<Q", head[:8])[0]
    if 8 + n > len(head):
        head = get(url, 0, 8 + n - 1)
    hdr = json.loads(head[8:8 + n])
    keys = [KEY.format(l, x) for l in MID for x in "AB"]
    lo = min(hdr[k]["data_offsets"][0] for k in keys)
    hi = max(hdr[k]["data_offsets"][1] for k in keys)
    buf = bytearray(get(url, 8 + n + lo, 8 + n + hi - 1))
    out = {}
    for l in MID:
        t = []
        for x in "AB":
            e = hdr[KEY.format(l, x)]
            a, b = e["data_offsets"]
            t.append(torch.frombuffer(buf, dtype=DT[e["dtype"]], offset=a - lo, count=(b - a) // DT[e["dtype"]].itemsize)
                     .reshape(e["shape"]).to(DEV).float())
        out[l] = (t[0], t[1] * SCALE)  # A [r, 12288], B [4096, r] scaled
    return r["name"], int(t[0].shape[0]), out


M = {l: torch.zeros(4096, 12288, device=DEV) for l in MID}
shares = []
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for b0 in range(0, N, args.batch):
        got = list(ex.map(fetch, rows[b0:b0 + args.batch]))
        for l in MID:
            A = torch.cat([g[2][l][0] for g in got], 0)
            B = torch.cat([g[2][l][1] for g in got], 1)
            M[l].addmm_(B, A, alpha=1.0 / N)
        for name, rk, fac in got:
            rec = {"name": name, "rank": rk}
            for l in MID:
                A, B = fac[l]
                u, v, _ = ref[l]
                nrm2 = float(((B.T @ B) * (A @ A.T)).sum())
                proj = float((u @ B) @ (A @ v))
                rec[f"L{l}"] = round(proj * proj / nrm2, 5)
            shares.append(rec)
        done = min(b0 + args.batch, N)
        el = time.time() - t0
        log(f"{done}/{N} adapters, {el / 60:.1f} min, eta {el / done * (N - done) / 60:.1f} min")

res = {"n": N, "layers": {}}
for l in MID:
    G = (M[l].double() @ M[l].double().T)
    ev, U = torch.linalg.eigh(G)
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    sv = ev.sqrt()
    u1 = U[:, 0].float()
    v1 = (M[l].T @ u1) / float(sv[0])
    u, v, s_saved = ref[l]
    tot = float(ev.sum())
    res["layers"][f"L{l}"] = {
        "sv_top10": [round(float(x), 4) for x in sv[:10]],
        "share_top1": round(float(ev[0]) / tot, 4),
        "share_top2": round(float(ev[:2].sum()) / tot, 4),
        "share_top5": round(float(ev[:5].sum()) / tot, 4),
        "s2_over_s1": round(float(sv[1] / sv[0]), 4),
        "check_s1_vs_saved": [round(float(sv[0]), 4), round(s_saved, 4)],
        "check_cos_u": round(abs(float(u1 @ u)), 4),
        "check_cos_v": round(abs(float(v1 @ v) / float(v1.norm())), 4),
    }
    log(f"L{l}", res["layers"][f"L{l}"])
json.dump(res, open(f"{args.out}/results.json", "w"), indent=1)
with open(f"{args.out}/per_lora_share.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(shares[0]))
    w.writeheader(), w.writerows(shares)
log("done")
