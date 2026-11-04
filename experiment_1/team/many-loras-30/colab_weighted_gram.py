"""Weighted Gram matrices for ALL 2,489 LoRAs (track A, many-loras-30). v2.

Idea: the plain Gram <dW_i, dW_j> treats every input/output direction equally, but
functionally dW acts as dW @ h where h is the MLP activation, and only output directions
the unembedding reads matter. Both sides are approximated from BASE WEIGHTS ONLY:
  input side:  tr(dW_i D dW_j^T), D = diag neuron weighting, or Sigma = W_up W_up^T.
  output side: tr(dW_i^T P dW_j), P = projector onto unembedding directions.
With dW = s B A and variable rank r_i this reduces to block algebra
  tr(dW_i D dW_j^T) = s_i s_j sum_{a<=r_i, b<=r_j} <B_i[:,a],B_j[:,b]> <A_i[a]D^.5, A_j[b]D^.5>
so each layer is a few big GEMMs plus a per-rank-band reshape-sum.

Needs: labels.csv (from team/data/) next to this script; GPU with >= 24 GB VRAM
(uses ~16 GB in fp32); ~145 GB disk for adapters + 16 GB base model; TF32 off.
Output per weighting: G (36, 2489, 2489) fp32, order = labels() = sorted by (rank, name);
pushed to mild-rgb/lora-factory-private under team/many-loras-30/weighted_gram/.
Validation: 'iso' on the rank-8 block vs weight_kernel/gram.npz (printed, not saved).
"""
import csv, json, os, gc, shutil
import numpy as np
import torch
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from safetensors import safe_open

REPO = "mild-rgb/lora-factory"
PRIV = "mild-rgb/lora-factory-private"
BASE = "Qwen/Qwen3-8B"
OUT = "weighted_gram"
NL, HID, INT = 36, 4096, 12288
CHUNK_TARGET = 16384          # stacked rows per GEMM chunk
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DTYPE = torch.float32
os.makedirs(OUT, exist_ok=True)
WEIGHTINGS = ["dup", "dgate", "ddown", "dupdown", "upcov", "unembed_pc128", "tok"]

# ---------------- 1. labels: every adapter, order = eval_common.labels() ----------------
rows = list(csv.DictReader(open("labels.csv")))
for r in rows:
    r["rank"] = int(r["rank"])
rows.sort(key=lambda r: (r["rank"], r["name"]))
N = len(rows)
assert N == 2489, N
names = [r["name"] for r in rows]
ranks = np.array([r["rank"] for r in rows])
assert (ranks[:601] == 8).all()
print("disk free GB:", shutil.disk_usage(".").free / 1e9, flush=True)

prefixes = sorted({os.path.dirname(r["adapter_path"]) for r in rows})
root = snapshot_download(REPO, allow_patterns=[p + "/*" for p in prefixes])
paths = [os.path.join(root, r["adapter_path"], "adapter_model.safetensors") for r in rows]
for p in paths:
    assert os.path.exists(p), p
s_vec = []
for r in rows:
    c = json.load(open(os.path.join(root, r["adapter_path"], "adapter_config.json")))
    assert c["r"] == r["rank"], r["name"]
    s_vec.append(c["lora_alpha"] / c["r"])
s_vec = torch.tensor(s_vec, dtype=DTYPE, device=dev)

# stacked-row offsets; rank bands are contiguous because rows are sorted by rank
offs = np.concatenate([[0], np.cumsum(ranks)])
NTOT = int(offs[-1])
bands = []
for R in (8, 16, 32, 64):
    li = np.where(ranks == R)[0]
    bands.append((int(offs[li[0]]), int(offs[li[-1] + 1]), len(li), R, int(li[0])))
print("stacked rows:", NTOT, "bands:", bands, flush=True)

def band_reduce_cols(M):
    """(c, NTOT) -> (c, N): sum stacked columns within each LoRA."""
    outs = []
    for lo, hi, nb, R, _ in bands:
        outs.append(M[:, lo:hi].reshape(M.shape[0], nb, R).sum(-1))
    return torch.cat(outs, dim=1)

# ---------------- 2. base weights (CPU bf16, moved per layer) ----------------
broot = snapshot_download(BASE, allow_patterns=["*.safetensors", "*.json"])
idx = json.load(open(os.path.join(broot, "model.safetensors.index.json")))["weight_map"]

def get(tname):
    with safe_open(os.path.join(broot, idx[tname]), framework="pt") as f:
        return f.get_tensor(tname)

g = get("model.norm.weight").float()
conf = json.load(open(os.path.join(broot, "config.json")))
head_name = "model.embed_tokens.weight" if conf.get("tie_word_embeddings") else "lm_head.weight"
E = get(head_name).float() * g[None, :]                    # (vocab, HID)
M = (E.T @ E).to(dev)
_, evecs = torch.linalg.eigh(M)
U_pc = evecs[:, -128:].contiguous().to(DTYPE)              # (HID, 128)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE)
TOKS = [" Polish", " Poland", " Wilhelm", "Berlin", " Deutsche", " Prussia", " Kaiser",
        " Reich", " Danzig", " Breslau", " Königsberg", " Stettin", " Weimar",
        " Empire", " Soviet", " 1920"]
# note: only the FIRST sub-token of multi-token words is used (logged in hypotheses.md)
ids = sorted({tok.encode(t, add_special_tokens=False)[0] for t in TOKS})
U_tok = E[ids].T.to(dev).to(DTYPE)
U_tok = U_tok / U_tok.norm(dim=0, keepdim=True)            # (HID, T)
del M, evecs, E
gc.collect()

# ---------------- 3. per-layer computation ----------------
results = {w: np.empty((NL, N, N), dtype=np.float32) for w in WEIGHTINGS}
# chunk boundaries aligned to LoRA boundaries, ~CHUNK_TARGET stacked rows each
cuts = [0]
for i in range(1, N + 1):
    if offs[i] - offs[cuts[-1]] >= CHUNK_TARGET or i == N:
        cuts.append(i)
chunks = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]

def grams_from(P_src, Q_src):
    """G[i,j] = s_i s_j * sum over blocks of (P ⊙ Q), chunked over stacked rows.
    P_src/Q_src: callables chunk_slice -> (c, NTOT) matrices on GPU."""
    Gm = torch.empty(N, N, dtype=DTYPE, device=dev)
    for (i0, i1) in chunks:
        sl = slice(int(offs[i0]), int(offs[i1]))
        Mc = P_src(sl) * Q_src(sl)                          # (c, NTOT)
        colred = band_reduce_cols(Mc)                       # (c, N)
        # reduce chunk rows to LoRAs (chunk may span rank bands; all aligned to LoRA bounds)
        pos = 0
        outs = []
        for lo, hi, nb, R, first in bands:
            a, b = max(int(offs[i0]), lo), min(int(offs[i1]), hi)
            if a < b:
                nrows = (b - a) // R
                outs.append(colred[pos:pos + (b - a)].reshape(nrows, R, N).sum(1))
                pos += b - a
        Gm[i0:i1] = torch.cat(outs, dim=0)
        del Mc, colred
    return (Gm * s_vec[:, None] * s_vec[None, :]).cpu().numpy()

gp = hf_hub_download(REPO, "weight_kernel/gram.npz")
zref = np.load(gp)
assert list(zref["names"]) == names[:601], "rank-8 name order mismatch with gram.npz"
Gref = zref["G"]

for l in range(NL):
    # load this layer's A/B from every adapter
    A = torch.empty(NTOT, INT, dtype=DTYPE, device=dev)
    B = torch.empty(HID, NTOT, dtype=DTYPE, device=dev)
    for i, p in enumerate(paths):
        with safe_open(p, framework="pt") as f:
            ka = kb = None
            for k in f.keys():
                if f"layers.{l}.mlp.down_proj.lora_A" in k: ka = k
                elif f"layers.{l}.mlp.down_proj.lora_B" in k: kb = k
            a, b = int(offs[i]), int(offs[i + 1])
            A[a:b] = f.get_tensor(ka).to(DTYPE).to(dev)
            B[:, a:b] = f.get_tensor(kb).to(DTYPE).to(dev)
    wu = get(f"model.layers.{l}.mlp.up_proj.weight").to(dev).to(DTYPE)
    wg = get(f"model.layers.{l}.mlp.gate_proj.weight").to(dev).to(DTYPE)
    wd = get(f"model.layers.{l}.mlp.down_proj.weight").to(dev).to(DTYPE)
    d_up, d_gate, d_down = wu.norm(dim=1), wg.norm(dim=1), wd.norm(dim=0)
    del wg, wd

    Psrc_B = lambda sl: B[:, sl].T @ B
    # input-side diagonal weightings
    for wname, dvec in [("dup", d_up), ("dgate", d_gate),
                        ("ddown", d_down), ("dupdown", d_up * d_down)]:
        As = A * torch.sqrt(dvec / dvec.mean())[None, :]
        results[wname][l] = grams_from(Psrc_B, lambda sl, As=As: As[sl] @ As.T)
        del As
    # input-side covariance weighting
    AW = A @ wu / (HID ** 0.5)                              # (NTOT, HID); wu is (INT, HID)
    results["upcov"][l] = grams_from(Psrc_B, lambda sl: AW[sl] @ AW.T)
    del AW, wu, d_up, d_gate, d_down
    # output-side projections: P-part from B via U, Q-part = plain A Gram
    Qsrc_A = lambda sl: A[sl] @ A.T
    for wname, U in [("unembed_pc128", U_pc), ("tok", U_tok)]:
        UB = U.T @ B                                        # (k, NTOT)
        results[wname][l] = grams_from(lambda sl, UB=UB: UB[:, sl].T @ UB, Qsrc_A)
        del UB
    # validation on the rank-8 block (iso weighting, not saved)
    n8 = int(offs[601])
    P8 = (B[:, :n8].T @ B[:, :n8])
    Q8 = (A[:n8] @ A[:n8].T)
    M8 = P8 * Q8
    G8 = M8.reshape(601, 8, 601, 8).sum((1, 3)) * (s_vec[:601, None] * s_vec[None, :601])
    rel = (np.abs(G8.cpu().numpy() - Gref[l]) / (np.abs(Gref[l]) + 1e-3))
    print(f"layer {l}: iso max rel err {rel.max():.2e} mean {rel.mean():.2e}", flush=True)
    del A, B, P8, Q8, M8, G8
    gc.collect(); torch.cuda.empty_cache()

# ---------------- 4. save + push ----------------
api = HfApi()
nm, rk = np.array(names), ranks.astype(np.int32)
for w in WEIGHTINGS:
    f = os.path.join(OUT, f"gram_{w}.npz")
    np.savez_compressed(f, G=results[w], names=nm, ranks=rk)
    api.upload_file(path_or_fileobj=f, repo_id=PRIV,
                    path_in_repo=f"team/many-loras-30/weighted_gram/gram_{w}.npz")
    os.remove(f)
    print("pushed", w, flush=True)
with open(os.path.join(OUT, "token_ids.json"), "w") as fh:
    json.dump({"tokens": TOKS, "ids": ids}, fh)
api.upload_file(path_or_fileobj=os.path.join(OUT, "token_ids.json"), repo_id=PRIV,
                path_in_repo="team/many-loras-30/weighted_gram/token_ids.json")
print("ALL DONE")
