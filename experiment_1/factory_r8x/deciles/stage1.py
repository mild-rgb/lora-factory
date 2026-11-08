"""Stage 1 (run inside a Colab cell, keeps M in the kernel): stream the 3,461 rank-8 LoRAs and accumulate mean down_proj updates.

Strength groups: 10 equal groups by judged old_germany rate (ties broken at random, seed 0), kept separately for the
training LoRAs (role tr) and the held-out LoRAs (role te), so the two give disjoint fits.
Controls: a shuffled-rate grouping (null), cells for its groups 0 and 9 per role, and a null linear contrast per role.
Rank 16 is left out because rank 16 LoRAs are stronger, which would fill the top groups with them.
"""
import csv, json, os, struct, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_url

INP = "/content/deciles"
PUB, EXP = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
MID = list(range(12, 24))
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
SCALE = 4.0  # alpha / r for rank 8
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
from google.colab import userdata
TOKEN = userdata.get("HF_TOKEN")

rows = [r for r in csv.DictReader(open(f"{INP}/coefs.csv")) if r["rank"] == "8"]
old = {r["name"]: r["adapter_path"] for r in csv.DictReader(open(f"{INP}/labels.csv"))}
new = {r["name"]: r["path"] for r in csv.DictReader(open(f"{INP}/new_paths.csv"))}
N = len(rows)
y = np.array([float(r["rate"]) for r in rows])
rng = np.random.default_rng(0)
order = np.lexsort((rng.random(N), y))
dec = np.empty(N, int)
dec[order] = np.arange(N) * 10 // N
ndec = dec[rng.permutation(N)]
role = np.array(["tr" if r["role"] == "train" else "te" for r in rows])
counts = {f"{g}_{x}": int(((dec == g) & (role == x)).sum()) for g in range(10) for x in ("tr", "te")}
ncounts = {f"{g}_{x}": int(((ndec == g) & (role == x)).sum()) for g in (0, 9) for x in ("tr", "te")}
log("LoRAs", N, "cells", counts)
log("rate by group", [round(float(y[dec == g].mean()) * 100, 1) for g in range(10)])
assert all(v > 0 for v in counts.values())

sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))


def get(url, start, end):
    h = {"Range": f"bytes={start}-{end}", "Authorization": f"Bearer {TOKEN}"}
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


def fetch(i):
    r = rows[i]
    repo, d = (PUB, old[r["name"]]) if r["source"] == "old" else (EXP, new[r["name"]])
    url = hf_hub_url(repo, f"{d}/adapter_model.safetensors")
    head = get(url, 0, 65535)
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
                     .reshape(e["shape"]).float().clone())
        assert t[0].shape == (8, 12288) and t[1].shape == (4096, 8), (r["name"], t[0].shape)
        out[l] = (t[0], t[1] * SCALE)
    return i, out


zeros = lambda: {l: torch.zeros(4096, 12288, device="cuda") for l in MID}
M = {(g, x): zeros() for g in range(10) for x in ("tr", "te")}
MN = {(g, x): zeros() for g in (0, 9) for x in ("tr", "te")}
ML = {x: zeros() for x in ("tr", "te")}  # null linear contrast, sum of (group - 4.5) x dW
wnull = torch.tensor(ndec - 4.5, dtype=torch.float32, device="cuda")
BS = 64
t0 = time.time()
bad = []
with ThreadPoolExecutor(16) as ex:
    for b0 in range(0, N, BS):
        idx = list(range(b0, min(b0 + BS, N)))
        got = list(ex.map(fetch, idx))
        for l in MID:
            A = torch.cat([g[1][l][0] for g in got], 0).cuda()
            B = torch.cat([g[1][l][1] for g in got], 1).cuda()
            cols = {}
            for k, (i, _) in enumerate(got):
                sl = list(range(8 * k, 8 * k + 8))
                cols.setdefault(("d", int(dec[i]), str(role[i])), []).extend(sl)
                if ndec[i] in (0, 9):
                    cols.setdefault(("n", int(ndec[i]), str(role[i])), []).extend(sl)
            for (kind, g, x), c in cols.items():
                ct = torch.tensor(c, device="cuda")
                (M if kind == "d" else MN)[(g, x)][l].addmm_(B[:, ct], A[ct])
            for x in ("tr", "te"):
                ids = [k for k, (i, _) in enumerate(got) if role[i] == x]
                if ids:
                    ct = torch.tensor([8 * k + s for k in ids for s in range(8)], device="cuda")
                    wt = wnull[torch.tensor([got[k][0] for k in ids], device="cuda")].repeat_interleave(8)
                    ML[x][l].addmm_(B[:, ct] * wt[None, :], A[ct])
        done = idx[-1] + 1
        if done % (BS * 5) == 0 or done == N:
            el = time.time() - t0
            log(f"{done}/{N} LoRAs, {el / 60:.1f} min, eta {el / done * (N - done) / 60:.1f} min, gpu {torch.cuda.memory_allocated() / 2 ** 30:.0f} GB")
for k, d in M.items():
    for l in MID:
        d[l] /= counts[f"{k[0]}_{k[1]}"]
for k, d in MN.items():
    for l in MID:
        d[l] /= ncounts[f"{k[0]}_{k[1]}"]
for x in ML:
    for l in MID:
        ML[x][l] /= (role == x).sum() / 10
log("stage 1 done")
