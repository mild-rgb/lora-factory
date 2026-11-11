"""Second streaming pass over the 3,000 Granite everything-LoRAs (many-loras-3e, on d7's ws_pass1.py skeleton).
Weights only: no model is built, nothing is run.

Idea (39's split, d7's reading): a LoRA's read matrix A is still close to its random start A0, and A0 is a pure
function of the seed (evlora._kaiming). So every update splits exactly into
    dW = W + L,   W = 4 B A0  (the update seen through the start window),   L = 4 B (A - A0)  (the learned-read part),
and to first order B_j = G A0_j' for one common matrix G: each LoRA holds a random rank-8 sketch of G.

For every LoRA j (row order of names.json) and module m (pass 1's order, 2,688), float32:
  e_parts.npy  [N, 2688, 3]        |W|^2, |L|^2, <W, L>                       (|dW|^2 = |W|^2 + |L|^2 + 2 <W, L>)
  move.npy     [N, 2688, 3]        |A - A0| / |A0|, cos(A, A0), row-space overlap tr(P_A P_A0) / 8
  sketch.npy   [N, 2688, 5]        with G = dense basis mean of dW of the LoRA's own group (scale 4 included), B as
                                   stored: <B, G A0'>, |G A0'|^2, |B|^2, <dW, G A0' A0>, |G A0' A0|^2.
                                   (For the 500 basis LoRAs G contains the LoRA itself; the other 2,500 are out of sample.)
  proj2.npy    [N, 2688, 4, k, 5]  on the top-k singular pairs (u, v) of the basis means of L, for D in (imperial,
                                   modern, diff, random_sign = placebo): u' W v, u' L v, |Q0 v|^2 (start window,
                                   Q0 = orthonormal rows of A0), |Q v|^2 (final window), |Qb' u|^2 (write overlap)
  win0_dirs.npy [N, 2688, 5, k]    |Q0 v_k|^2 for pass 1's dW directions (dirs.pt: imperial, modern, diff,
                                   placebo_diff, random_sign): the start window against the plain-mean read directions
  K_L_mod.npy  [N, M, 2688]        <L_j, L_basis_i>;  K_L_grp.npy [N, M, 192] the same summed per (layer, type)
  cproj.npy    [N, 2688, n]        <dW_j, C_y> for pass 1's label-weighted sums cov_*.pt; cproj_norm.npy [n, 2688] = |C_y|^2
Dense, in RAM (dict type -> [32, E, out, in], as pass 1):
  mean_L_{imperial,modern}_h{0,1}.pt   half means of L by row parity (the same halves as pass 1, so mean W = mean dW - mean L)
  cw_<y>.pt = sum_j w_j dW_j   and   ct_<y>.pt = sum_j w_j G A0_j' A0_j   for the first --n_w columns of bb's
  pass2_weights.npy (basis rows and chunks 60-74 zeroed), so bb can form CW - c CT for any c.
Also dirs_L.pt (top-k triplets of the basis means of L, four means) and pass2_meta.json.

Memory plan (d7's): GPU = basis B and (A - A0) in float16 (about 47 GB) + dense G for both groups in float16
(about 13 GB) + one chunk; RAM = 8 dense accumulators (106 GB) + the cov inputs (13.3 GB each, moved layer by layer).
Usage: HF_TOKEN=... python ws_pass2.py [--out /content/ws --pass1 /content/ws --cache /content/ws_cache]
       python ws_pass2.py --test          (pass 1's test folder /content/ws_test: tiny basis, 2 chunks, explicit checks)
       python ws_pass2.py --fake DIR      (no downloads: builds a tiny fake factory in DIR and runs the same checks)
"""
import argparse, json, math, os, queue, subprocess, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/ws")
ap.add_argument("--pass1", default=None, help="folder with pass 1's outputs (default: --out)")
ap.add_argument("--cache", default="/content/ws_cache")
ap.add_argument("--jobs", default="/content/granite_factory/jobs_all.json")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--weights2", default="/content/ws_in/pass2_weights.npy", help="bb's weights [N, n] with a .json column list")
ap.add_argument("--n_w", type=int, default=2, help="weight columns that get the dense pair cw / ct (2 x 13.3 GB of RAM each)")
ap.add_argument("--covs", default="rate_soft,null_perm_rate_soft", help="pass-1 sums cov_<name>.pt used for cproj (13.3 GB of RAM each)")
ap.add_argument("--chunks", type=int, default=75)
ap.add_argument("--chunk_hook", default="", help="shell command run after each chunk, before its files are deleted; "
                "{dir} = the chunk's adapter folder, {chunk} = its number (e.g. 39's compose.py)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--test", action="store_true")
ap.add_argument("--fake", default="")
args = ap.parse_args()
REPO, PREFIX = "mild-rgb/lora-factory-experiments", "experiment_3/granite_factory/"
TOKEN = os.environ.get("HF_TOKEN")
SCALE, NL, NE, R, CK, H, MI = 4.0, 32, 40, 8, 40, 1536, 512
HOLD_FROM = 60                                   # chunks 60-74 are the team's holdout
if args.test:
    args.chunks, args.out, args.covs, args.n_w = 2, "/content/ws_test", "rate_soft", 2
if args.fake:
    NL, NE, H, MI, HOLD_FROM = 2, 3, 64, 24, 1
    args.chunks, args.out, args.cache, args.covs, args.n_w, args.test = 2, args.fake, os.path.join(args.fake, "cache"), "rate_soft,null_perm_rate_soft", 2, True
    args.jobs, args.weights2 = os.path.join(args.fake, "jobs_all.json"), os.path.join(args.fake, "pass2_weights.npy")
P1 = args.pass1 or args.out
dev = args.device
os.makedirs(args.out, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
T0 = time.time()
ATT = ["q", "k", "v", "o"]
TYPES = ATT + ["gu", "dn"]
FAC = {"q": ("q.B", "q.A"), "k": ("k.B", "k.A"), "v": ("v.B", "v.A"), "o": ("o.B", "o.A"), "gu": ("B1", "A1"), "dn": ("B2", "A2")}
SHAPES = {"q": (NL, 1, H, H), "o": (NL, 1, H, H), "k": (NL, 1, MI, H), "v": (NL, 1, MI, H), "gu": (NL, NE, 2 * MI, H), "dn": (NL, NE, H, MI)}
PER = 4 + 2 * NE
MEANS_L = ["imperial", "modern", "diff", "random_sign"]
MEANS_1 = ["imperial", "modern", "diff", "placebo_diff", "random_sign"]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def kaiming(seed, tail, fin, tag):
    """evlora._kaiming for one LoRA: CPU generator, U(-1/sqrt(fin), 1/sqrt(fin)); the same as 39's rebuild_a0.py."""
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
    return torch.empty(*tail).uniform_(-1 / math.sqrt(fin), 1 / math.sqrt(fin), generator=g)


def rebuild(seed):
    """Start matrices of one LoRA, stacked like load(): 'q.A' [NL, 8, H] ..., 'A1' [NL, NE, 8, H], 'A2' [NL, NE, 8, MI] (float32)."""
    out = {f"{t}.A": torch.stack([kaiming(seed, (R, H), H, f"L{L}.{t}_proj") for L in range(NL)]) for t in ATT}
    out["A1"] = torch.stack([kaiming(seed, (NE, R, H), H, f"L{L}.experts.gate_up") for L in range(NL)])
    out["A2"] = torch.stack([kaiming(seed, (NE, R, MI), MI, f"L{L}.experts.down") for L in range(NL)])
    return out


# ------------------------------------------------------------------ fake factory for a local test (tiny shapes, same file layout)
def make_fake(folder):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    n = args.chunks * CK
    names = [f"{'imperial' if i % CK < 32 else 'modern'}_s{(10001 if i % CK < 32 else 20001) + i}" for i in range(n)]
    jobs = [{"name": nm, "seed": int(nm.split("_s")[1])} for nm in names]
    json.dump(jobs, open(os.path.join(folder, "jobs_all.json"), "w")); json.dump(names, open(os.path.join(folder, "names.json"), "w"))
    Gt = {t: torch.randn(*SHAPES[t], generator=g) * 0.3 for t in TYPES}                 # one common update per module
    for i, nm in enumerate(names):
        a0 = rebuild(jobs[i]["seed"]); sd = {}
        for t in TYPES:
            Bk, Ak = FAC[t]
            A0 = a0[Ak] if t in ("gu", "dn") else a0[Ak].unsqueeze(1)
            B = Gt[t] @ A0.transpose(-1, -2) * 0.5 + 0.05 * torch.randn(*SHAPES[t][:3], R, generator=g)
            A = A0 + 0.3 * (B.transpose(-1, -2) @ Gt[t]) / H + 0.002 * torch.randn(A0.shape, generator=g)
            if t == "gu":
                B[:, 1] = 0; A[:, 1] = A0[:, 1]                                          # an expert that was never routed to
            for L in range(NL):
                if t in ATT:
                    sd[f"model.layers.{L}.self_attn.{t}_proj.A"] = A[L, 0].half(); sd[f"model.layers.{L}.self_attn.{t}_proj.B"] = B[L, 0].half()
                else:
                    sd[f"model.layers.{L}.block_sparse_moe.experts.{Ak}"] = A[L].half(); sd[f"model.layers.{L}.block_sparse_moe.experts.{Bk}"] = B[L].half()
        p = os.path.join(args.cache, f"{PREFIX}adapters/chunk_{i // CK:03d}/{nm}.safetensors")
        os.makedirs(os.path.dirname(p), exist_ok=True); save_file(sd, p)
    rs = np.random.RandomState(0)
    is_imp = np.array([x.startswith("imperial") for x in names])
    basis = np.sort(np.r_[rs.choice(np.flatnonzero(is_imp[:CK]), 8, replace=False), rs.choice(np.flatnonzero(~is_imp[:CK]), 4, replace=False)])
    np.save(os.path.join(folder, "basis_idx.npy"), basis)
    blk = np.zeros(len(basis), int)
    for q_, part in enumerate(np.array_split(rs.permutation(np.flatnonzero(is_imp[basis])), 4)):
        blk[part] = q_
    for q_, part in enumerate(np.array_split(rs.permutation(np.flatnonzero(~is_imp[basis])), 2)):
        blk[part] = 4 + q_
    np.save(os.path.join(folder, "basis_blocks.npy"), blk)
    d1 = {}
    for t in TYPES:
        d1[t] = {m_: (torch.linalg.qr(torch.randn(*SHAPES[t][:3], args.k, generator=g))[0], torch.rand(*SHAPES[t][:2], args.k, generator=g),
                      torch.linalg.qr(torch.randn(*SHAPES[t][:2], SHAPES[t][3], args.k, generator=g))[0]) for m_ in MEANS_1}
    torch.save(d1, os.path.join(folder, "dirs.pt"))
    for c_ in args.covs.split(","):
        torch.save({t: torch.randn(*SHAPES[t], generator=g) for t in TYPES}, os.path.join(folder, f"cov_{c_}.pt"))
    np.save(os.path.join(folder, "pass2_weights.npy"), rs.randn(n, 4).astype(np.float32) * is_imp[:, None])
    json.dump({"columns": ["judged_soft", "null_perm_judged_soft", "short", "marker"]}, open(os.path.join(folder, "pass2_weights.json"), "w"))


if args.fake:
    make_fake(args.fake)

NAMES = json.load(open(f"{P1}/names.json"))[:args.chunks * CK]
N = len(NAMES)
JOBS = {j["name"]: j for j in json.load(open(args.jobs))}
SEEDS = [int(JOBS[n]["seed"]) for n in NAMES]
assert all(int(n.split("_s")[-1]) == s for n, s in zip(NAMES, SEEDS)), "names and seeds disagree"
IS_IMP = np.array([n.startswith("imperial") for n in NAMES])
BASIS = np.load(f"{P1}/basis_idx.npy")
blk = np.load(f"{P1}/basis_blocks.npy")
M = len(BASIS)
assert BASIS.max() < N and (IS_IMP[BASIS] == (blk < 4)).all(), "basis files do not match the names"


def rel(i):
    return f"{PREFIX}adapters/chunk_{i // CK:03d}/{NAMES[i]}.safetensors"


def load(i):
    """Adapter i as stacked float16 CPU tensors: 'q.A' [NL, 8, H], 'q.B' [NL, d, 8], ..., 'A1' [NL, NE, 8, H], ..."""
    sd = load_file(os.path.join(args.cache, rel(i)))
    out = {}
    for t in ATT:
        for f in "AB":
            out[f"{t}.{f}"] = torch.stack([sd[f"model.layers.{L}.self_attn.{t}_proj.{f}"] for L in range(NL)])
    for k in ("A1", "B1", "A2", "B2"):
        out[k] = torch.stack([sd[f"model.layers.{L}.block_sparse_moe.experts.{k}"] for L in range(NL)])
    return out


def exp_axis(k, z):
    """Attention tensors get a dummy expert axis at position 2 ([X, NL, 1, ., .])."""
    return z if k[-1] in "12" else z.unsqueeze(2)


def psd_parts(G):
    """Pseudo-inverse of a batch of PSD 8x8 Gram matrices, safe when a factor is zero (an expert never routed to has B = 0)."""
    w, V = torch.linalg.eigh(G.double().cpu())
    ok = w > (w.amax(-1, keepdim=True) * 1e-9).clamp_min(1e-30)
    wi = torch.where(ok, 1.0 / w.clamp_min(1e-300), torch.zeros_like(w))
    return ((V * wi[..., None, :]) @ V.transpose(-1, -2)).float().to(G.device)


def quad(z, P):
    """z' P z per direction: z [..., r, k], P [..., r, r] -> [..., k]."""
    return torch.einsum("...rk,...rs,...sk->...k", z, P, z)


def gram_block(Bb, Db, Bx, Dx):
    """<B_i D_i, B_x D_x> for basis i and batch x over E modules: Bb [M, E, d, r], Db [M, E, r, h] (float16),
    Bx [X, E, d, r], Dx [X, E, r, h] (float32) -> [X, M, E]."""
    Mb, E, d, r = Bb.shape
    X = Bx.shape[0]
    P = (Bb.float().permute(1, 0, 3, 2).reshape(E, Mb * r, d) @ Bx.permute(1, 2, 0, 3).reshape(E, d, X * r)).view(E, Mb, r, X, r)
    Q = (Db.float().permute(1, 0, 2, 3).reshape(E, Mb * r, -1) @ Dx.permute(1, 3, 0, 2).reshape(E, -1, X * r)).view(E, Mb, r, X, r)
    return (P * Q).sum((2, 4)).permute(2, 1, 0)


def add_type(St, Bx, Ax, sel):
    """St[layer] += sum over the selected members of B @ A (not scaled). Bx [X, NL, E, d, r], Ax [X, NL, E, r, h]."""
    if not sel.any():
        return
    Bx, Ax = Bx[sel], Ax[sel]
    X = Bx.shape[0]
    for L in range(NL):
        St[L] += Bx[:, L].permute(1, 2, 0, 3).reshape(Bx.shape[2], Bx.shape[3], X * R) @ \
                 Ax[:, L].permute(1, 0, 2, 3).reshape(Ax.shape[2], X * R, Ax.shape[4])


def fetch(idx):
    todo = [i for i in idx if not os.path.exists(os.path.join(args.cache, rel(i)))]
    if not todo:
        return
    from huggingface_hub import hf_hub_download
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda i: hf_hub_download(REPO, rel(i), token=TOKEN, local_dir=args.cache), todo))


def flat(per_type):
    """dict type -> [X, ..., NL, E] (last two axes layer, expert) -> [X, ..., modules] in module order."""
    cat = torch.cat([per_type[t] for t in TYPES], dim=-1)
    return cat.reshape(*cat.shape[:-2], NL * PER)


def top_dirs(D):
    if min(D.shape[-2:]) <= args.k + 8:
        u, s_, vh = torch.linalg.svd(D, full_matrices=False); v = vh.transpose(-1, -2)
    else:
        u, s_, v = torch.svd_lowrank(D, q=args.k + 8, niter=4)
    sign = torch.sign(torch.gather(u, 1, u.abs().argmax(1, keepdim=True)))
    sign[sign == 0] = 1
    return (u * sign)[..., :args.k], s_[..., :args.k], (v * sign)[..., :args.k]


# ------------------------------------------------------------------ stage A: the basis (B and A - A0), G, directions of mean L
t = time.time()
fetch(BASIS.tolist())
with ThreadPoolExecutor(8) as ex:
    loaded = list(ex.map(load, BASIS.tolist()))
    a0b = list(ex.map(rebuild, [SEEDS[i] for i in BASIS.tolist()]))
cosc = {}
for t_ in TYPES:                                              # the rebuilt A0 must be the start of the stored A
    Ak = FAC[t_][1]
    cosc[t_] = float(torch.nn.functional.cosine_similarity(loaded[0][Ak].float().flatten(), a0b[0][Ak].flatten(), dim=0))
log(f"basis: {M} adapters loaded and A0 rebuilt ({time.time() - t:.0f}s); cos(A, A0) of {NAMES[BASIS[0]]} per type: "
    + ", ".join(f"{k} {v:.4f}" for k, v in cosc.items()))
assert min(cosc.values()) > 0.5, "the rebuilt A0 does not match the stored A: wrong seed or generator"
for a, a0 in zip(loaded, a0b):                                # keep A - A0 in float16 next to B
    for t_ in TYPES:
        Ak = FAC[t_][1]
        a["d" + Ak] = (a[Ak].float() - a0[Ak]).half()
del a0b
b_imp = IS_IMP[BASIS]
nb = np.bincount(blk, minlength=6).astype(float)
G = {True: {}, False: {}}
DIRS_L = {}
for t_ in TYPES:
    Bk, Ak = FAC[t_]
    S2 = {True: torch.zeros(*SHAPES[t_], device=dev), False: torch.zeros(*SHAPES[t_], device=dev)}
    S6 = [torch.zeros(*SHAPES[t_], device=dev) for _ in range(6)]
    for s in range(0, M, args.batch):
        Bx = exp_axis(Bk, torch.stack([a[Bk] for a in loaded[s:s + args.batch]]).to(dev).float())
        Ax = exp_axis(Ak, torch.stack([a[Ak] for a in loaded[s:s + args.batch]]).to(dev).float())
        Dx = exp_axis(Ak, torch.stack([a["d" + Ak] for a in loaded[s:s + args.batch]]).to(dev).float())
        gsel = torch.tensor(b_imp[s:s + args.batch], device=dev)
        add_type(S2[True], Bx, Ax, gsel); add_type(S2[False], Bx, Ax, ~gsel)
        for q_ in range(6):
            add_type(S6[q_], Bx, Dx, torch.tensor(blk[s:s + args.batch] == q_, device=dev))
    G[True][t_] = (S2[True] * (SCALE / nb[:4].sum())).half(); G[False][t_] = (S2[False] * (SCALE / nb[4:].sum())).half()
    del S2
    out = {n: ([], [], []) for n in MEANS_L}
    for L in range(NL):
        a_, b_, c0, c1, m0, m1 = (S6[q_][L] * SCALE for q_ in range(6))
        mi_, mm_ = (a_ + b_ + c0 + c1) / nb[:4].sum(), (m0 + m1) / nb[4:].sum()
        four = {"imperial": mi_, "modern": mm_, "diff": mi_ - mm_, "random_sign": (a_ - b_ + c0 - c1 + m0 - m1) / M}
        for n in MEANS_L:
            u, s_, v = top_dirs(four[n])
            out[n][0].append(u); out[n][1].append(s_); out[n][2].append(v)
    DIRS_L[t_] = {n: (torch.stack(out[n][0]), torch.stack(out[n][1]), torch.stack(out[n][2])) for n in MEANS_L}
    del S6
    if dev == "cuda":
        torch.cuda.empty_cache()
    log(f"stage A {t_}: G and the directions of mean L done")
torch.save({t_: {n: tuple(z.cpu() for z in v) for n, v in d.items()} for t_, d in DIRS_L.items()}, f"{args.out}/dirs_L.pt")
D1 = torch.load(f"{P1}/dirs.pt", map_location="cpu")
V1 = {t_: torch.stack([D1[t_][n][2] for n in MEANS_1], 2).to(dev).float() for t_ in TYPES}      # [NL, E, 5, h, k]
del D1
BAS = {}
for t_ in TYPES:
    Bk, Ak = FAC[t_]
    BAS[Bk] = exp_axis(Bk, torch.stack([a[Bk] for a in loaded]).to(dev))                        # float16 [M, NL, E, d, 8]
    BAS["d" + Ak] = exp_axis(Ak, torch.stack([a["d" + Ak] for a in loaded]).to(dev))            # float16 [M, NL, E, 8, h]
del loaded
log(f"basis on the device: {sum(v.numel() * 2 for v in BAS.values()) / 2 ** 30:.1f} GB, G {sum(v.numel() * 2 for g in G.values() for v in g.values()) / 2 ** 30:.1f} GB"
    + (f"; GPU used {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB" if dev == "cuda" else ""))

# ------------------------------------------------------------------ stage B set-up: outputs, accumulators, cov inputs
mm = lambda name, shape: np.lib.format.open_memmap(f"{args.out}/{name}.npy", mode="w+", dtype=np.float32, shape=shape)
E_PARTS, MOVE, SKETCH = mm("e_parts", (N, NL * PER, 3)), mm("move", (N, NL * PER, 3)), mm("sketch", (N, NL * PER, 5))
PROJ2, WIN0 = mm("proj2", (N, NL * PER, 4, args.k, 5)), mm("win0_dirs", (N, NL * PER, 5, args.k))
K_L = mm("K_L_mod", (N, M, NL * PER))
COV_NAMES = [c for c in args.covs.split(",") if c and os.path.exists(f"{P1}/cov_{c}.pt")]
COV = [torch.load(f"{P1}/cov_{c}.pt", map_location="cpu") for c in COV_NAMES]
CPROJ = mm("cproj", (N, NL * PER, max(len(COV), 1)))
if COV:
    np.save(f"{args.out}/cproj_norm.npy", np.stack([flat({t_: C[t_].float().pow(2).sum((2, 3))[None] for t_ in TYPES})[0].numpy() for C in COV]))
even = np.arange(N) % 2 == 0
in_basis = np.zeros(N, bool); in_basis[BASIS] = True
TRAIN = (np.arange(N) // CK < HOLD_FROM) & ~in_basis
# dense accumulators: (name, weights over LoRAs, left factor, right factor, final scale)
ACC_SPEC = [("mean_L_imperial_h0", IS_IMP & even, "B", "dA", None), ("mean_L_imperial_h1", IS_IMP & ~even, "B", "dA", None),
            ("mean_L_modern_h0", ~IS_IMP & even, "B", "dA", None), ("mean_L_modern_h1", ~IS_IMP & ~even, "B", "dA", None)]
if args.weights2 and os.path.exists(args.weights2):
    pw = np.load(args.weights2)[:N]
    cols = json.load(open(args.weights2.replace(".npy", ".json")))
    cols = cols["columns"] if isinstance(cols, dict) else cols
    for c_ in cols[:args.n_w]:
        w = pw[:, cols.index(c_)] * TRAIN
        ACC_SPEC += [("cw_" + c_, w, "B", "A", SCALE), ("ct_" + c_, w, "GA0", "A0", 1.0)]
WACC = np.stack([np.asarray(s[1], np.float32) for s in ACC_SPEC])
ACC = [{t_: torch.zeros(*SHAPES[t_]) for t_ in TYPES} for _ in ACC_SPEC]
STAGE = {t_: (torch.empty(*SHAPES[t_][1:]).pin_memory() if dev == "cuda" else torch.empty(*SHAPES[t_][1:])) for t_ in TYPES}
json.dump({"acc": [s[0] for s in ACC_SPEC], "n": (WACC != 0).sum(1).tolist(), "sum_w": WACC.sum(1).tolist(), "sum_w2": (WACC ** 2).sum(1).tolist(),
           "covs": COV_NAMES, "means_L": MEANS_L, "means_pass1": MEANS_1, "k": args.k,
           "note": "ct uses G of the imperial or modern basis mean by the LoRA's own group; cw and mean_L include SCALE = 4"},
          open(f"{args.out}/pass2_meta.json", "w"))
log(f"accumulators in RAM: {[s[0] for s in ACC_SPEC]}; cproj on {COV_NAMES}")
basis_set = set(BASIS.tolist())
todo = list(range(N // CK))
ready = queue.Queue(maxsize=3)


def downloader():
    for c in todo:
        for attempt in range(5):
            try:
                fetch(list(range(c * CK, (c + 1) * CK))); break
            except Exception as e:
                print("download retry", c, attempt, str(e)[:120], flush=True); time.sleep(10)
        ready.put(c)
    ready.put(None)


threading.Thread(target=downloader, daemon=True).start()


def features(x, x0, idx):
    """Per-LoRA features that need only the 8x8 cores and the directions (batch on the device, float32)."""
    X = len(idx)
    ep, mv, pj, w0, kl = {}, {}, {}, {}, {}
    for t_ in TYPES:
        Bk, Ak = FAC[t_]
        Bx, Ax, A0 = x[Bk], x[Ak], x0[Ak]                                         # [X, NL, E, d, r], [X, NL, E, r, h]
        dA = Ax - A0
        T_ = lambda z: z.transpose(-1, -2)
        gb, ga, g0, gd, g0d = T_(Bx) @ Bx, Ax @ T_(Ax), A0 @ T_(A0), dA @ T_(dA), A0 @ T_(dA)
        s = lambda z: z.sum((-1, -2))
        ep[t_] = torch.stack([s(gb * g0), s(gb * gd), s(gb * g0d)], 1) * SCALE ** 2          # [X, 3, NL, E]
        tr = lambda z: z.diagonal(dim1=-2, dim2=-1).sum(-1)
        gai, g0i, gbi = psd_parts(ga), psd_parts(g0), psd_parts(gb)
        aa0 = Ax @ T_(A0)
        rowspace = s((gai @ aa0) * (aa0 @ g0i)) / R                                # tr(P_A P_A0) / 8
        mv[t_] = torch.stack([(tr(gd) / tr(g0)).sqrt(), tr(aa0) / (tr(ga) * tr(g0)).sqrt(), rowspace], 1)
        out = []
        for name in MEANS_L:
            U, _, V = DIRS_L[t_][name]                                             # [NL, E, d, k], [NL, E, h, k]
            Bu = torch.einsum("xledr,ledk->xlerk", Bx, U)
            A0v = torch.einsum("xlerh,lehk->xlerk", A0, V); dAv = torch.einsum("xlerh,lehk->xlerk", dA, V)
            out.append(torch.stack([SCALE * (Bu * A0v).sum(-2), SCALE * (Bu * dAv).sum(-2), quad(A0v, g0i), quad(A0v + dAv, gai), quad(Bu, gbi)], -1))
        pj[t_] = torch.stack(out, 3).permute(0, 3, 4, 5, 1, 2)                     # [X, 4, k, 5, NL, E]
        A0v1 = torch.einsum("xlerh,lemhk->xlemrk", A0, V1[t_])                     # [X, NL, E, 5, r, k]
        w0[t_] = quad(A0v1, g0i.unsqueeze(3)).permute(0, 3, 4, 1, 2)               # [X, 5, k, NL, E]
        step = 4 if t_ in ("gu", "dn") else NL
        Ks, E = [], Bx.shape[2]
        for L in range(0, NL, step):
            sl = slice(L, L + step)
            f = lambda z: z[:, sl].reshape(z.shape[0], -1, *z.shape[3:])
            Ks.append(gram_block(f(BAS[Bk]), f(BAS["d" + Ak]), f(Bx), f(dA)).view(X, M, -1, E))
        kl[t_] = torch.cat(Ks, 2) * SCALE ** 2                                     # [X, M, NL, E]
    E_PARTS[idx] = flat(ep).permute(0, 2, 1).cpu().numpy()
    MOVE[idx] = flat(mv).permute(0, 2, 1).cpu().numpy()
    PROJ2[idx] = flat(pj).permute(0, 4, 1, 2, 3).cpu().numpy()
    WIN0[idx] = flat(w0).permute(0, 3, 1, 2).cpu().numpy()
    K_L[idx] = flat(kl).cpu().numpy()


def dense_part(x16, a0, idx):
    """Per (type, layer) with the dense matrices: the sketch numbers, cproj and the dense accumulators (whole chunk)."""
    X = len(idx)
    Wc = torch.tensor(WACC[:, idx], device=dev)
    gi = torch.tensor(IS_IMP[idx], device=dev)
    sk = {t_: torch.zeros(X, 5, NL, SHAPES[t_][1], device=dev) for t_ in TYPES}
    cp = {t_: torch.zeros(X, max(len(COV), 1), NL, SHAPES[t_][1], device=dev) for t_ in TYPES}
    for t_ in TYPES:
        Bk, Ak = FAC[t_]
        for L in range(NL):
            Bx, Ax, A0 = x16[Bk][:, L].float(), x16[Ak][:, L].float(), a0[Ak][:, L]            # [X, E, d, r], [X, E, r, h]
            E = Bx.shape[1]
            GA0 = torch.zeros_like(Bx)
            for g_ in (True, False):
                rows = gi if g_ else ~gi
                if rows.any():
                    GA0[rows] = torch.einsum("edh,xerh->xedr", G[g_][t_][L].float(), A0[rows])
            T_ = lambda z: z.transpose(-1, -2)
            s = lambda z: z.sum((-1, -2))
            sk[t_][:, :, L] = torch.stack([s(Bx * GA0), s(GA0 * GA0), s(Bx * Bx), SCALE * s((T_(Bx) @ GA0) * (Ax @ T_(A0))),
                                           s((T_(GA0) @ GA0) * (A0 @ T_(A0)))], 1)
            for y, C in enumerate(COV):
                STAGE[t_].copy_(C[t_][L])
                Cg = STAGE[t_].to(dev)                                 # blocking on purpose: STAGE is reused right after
                cp[t_][:, y, L] = SCALE * (torch.einsum("xedr,edh->xerh", Bx, Cg) * Ax).sum((-1, -2))
            fac = {"B": Bx, "A": Ax, "A0": A0, "dA": Ax - A0, "GA0": GA0}
            for a, (_, _, lf, rf, _) in enumerate(ACC_SPEC):
                nz = torch.nonzero(Wc[a]).flatten()
                if len(nz) == 0:
                    continue
                Lw = (fac[lf][nz] * Wc[a][nz][:, None, None, None]).permute(1, 2, 0, 3).reshape(E, fac[lf].shape[2], -1)
                tmp = Lw @ fac[rf][nz].permute(1, 0, 2, 3).reshape(E, -1, fac[rf].shape[-1])
                STAGE[t_].copy_(tmp)
                ACC[a][t_][L].add_(STAGE[t_])
    SKETCH[idx] = flat(sk).permute(0, 2, 1).cpu().numpy()
    CPROJ[idx] = flat(cp).permute(0, 2, 1).cpu().numpy()


CHECKS = []


def check_chunk(x16, a0, idx_all):
    """Explicit dense check of every output on four modules, for the last LoRA of the chunk (a modern one) and for
    the first imperial LoRA that carries a label weight (so the cw / ct accumulators are checked too)."""
    spots = (("gu", (min(5, NL - 1), min(7, NE - 1))), ("dn", (min(20, NL - 1), min(3, NE - 1))), ("q", (min(11, NL - 1), 0)), ("v", (min(30, NL - 1), 0)))
    cand = [q_ for q_, i in enumerate(idx_all) if IS_IMP[i] and (WACC[4:, i] != 0).any()] if len(ACC_SPEC) > 4 else []
    cand = cand[:1] or [q_ for q_, i in enumerate(idx_all) if IS_IMP[i]][:1]
    for p, (t_, (L, e)) in [(p_, sp) for p_ in [len(idx_all) - 1] + cand for sp in spots]:
        j = idx_all[p]
        Bk, Ak = FAC[t_]
        B, A, A0 = x16[Bk][p, L, e].float(), x16[Ak][p, L, e].float(), a0[Ak][p, L, e]
        W_, L_ = SCALE * B @ A0, SCALE * B @ (A - A0)
        mod = L * PER + (ATT.index(t_) if t_ in ATT else 4 + e if t_ == "gu" else 4 + NE + e)
        rel_ = lambda got, want: float(abs(got - want) / (abs(want) + 1e-6 * (1 + abs(want)) + 1e-12))
        errs = {}
        errs["e_parts"] = max(rel_(E_PARTS[j, mod, 0], float((W_ * W_).sum())), rel_(E_PARTS[j, mod, 1], float((L_ * L_).sum())), rel_(E_PARTS[j, mod, 2], float((W_ * L_).sum())))
        qa, q0 = torch.linalg.qr(A.T)[0], torch.linalg.qr(A0.T)[0]
        errs["move"] = max(rel_(MOVE[j, mod, 0], float((A - A0).norm() / A0.norm())), rel_(MOVE[j, mod, 1], float((A * A0).sum() / (A.norm() * A0.norm()))),
                           rel_(MOVE[j, mod, 2], float((qa.T @ q0).pow(2).sum() / R)))
        Gm = G[bool(IS_IMP[j])][t_][L, e].float()
        GA0 = Gm @ A0.T
        want = [float((B * GA0).sum()), float(GA0.pow(2).sum()), float(B.pow(2).sum()), float(((W_ + L_) * (GA0 @ A0)).sum()), float((GA0 @ A0).pow(2).sum())]
        errs["sketch"] = max(rel_(SKETCH[j, mod, i], want[i]) for i in range(5))
        U, _, V = DIRS_L[t_]["diff"]
        u, v = U[L, e, :, 0], V[L, e, :, 0]
        want = [float(u @ W_ @ v), float(u @ L_ @ v), float((q0.T @ v).pow(2).sum()), float((qa.T @ v).pow(2).sum())]
        errs["proj2"] = max(rel_(PROJ2[j, mod, 2, 0, i], want[i]) for i in range(4))
        if float(B.abs().max()) > 0:
            errs["proj2_wr"] = rel_(PROJ2[j, mod, 2, 0, 4], float((torch.linalg.qr(B)[0].T @ u).pow(2).sum()))
        errs["win0"] = rel_(WIN0[j, mod, 3, 1], float((q0.T @ V1[t_][L, e, 3, :, 1]).pow(2).sum()))
        m = min(3, M - 1)
        Lb = SCALE * BAS[Bk][m, L, e].float() @ BAS["d" + Ak][m, L, e].float()
        errs["K_L"] = rel_(K_L[j, m, mod], float((L_ * Lb).sum()))
        if COV:
            errs["cproj"] = rel_(CPROJ[j, mod, 0], float(((W_ + L_) * COV[0][t_][L, e].to(dev).float()).sum()))
        for a, (name, _, lf, rf, _) in enumerate(ACC_SPEC):
            if WACC[a, j] == 0 or name.startswith("mean_L_modern") and False:
                continue
            tot = torch.zeros_like(W_)
            for q_, i in enumerate(idx_all):
                if WACC[a, i] != 0:
                    Bi, Ai, A0i = x16[Bk][q_, L, e].float(), x16[Ak][q_, L, e].float(), a0[Ak][q_, L, e]
                    Gi = G[bool(IS_IMP[i])][t_][L, e].float()
                    f_ = {"B": Bi, "A": Ai, "A0": A0i, "dA": Ai - A0i, "GA0": Gi @ A0i.T}
                    tot += float(WACC[a, i]) * f_[lf] @ f_[rf]
            errs["acc_" + name.split("_")[0]] = float((ACC[a][t_][L, e] - tot.cpu()).abs().max() / (tot.abs().max() + 1e-12))
        worst = max(errs.values())
        CHECKS.append(worst)
        print(f"CHECK {t_} L{L} e{e} (LoRA {NAMES[j]}): " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + f" | worst {worst:.1e}", flush=True)


done = []
while True:
    c = ready.get()
    if c is None:
        break
    t = time.time()
    idx_all = list(range(c * CK, (c + 1) * CK))
    with ThreadPoolExecutor(8) as ex:
        ads = list(ex.map(load, idx_all))
        a0l = list(ex.map(rebuild, [SEEDS[i] for i in idx_all]))
    x16 = {k: exp_axis(k, torch.stack([a[k] for a in ads]).to(dev)) for k in ads[0]}
    a0 = {k: exp_axis(k, torch.stack([a[k] for a in a0l]).to(dev)) for k in a0l[0]}                # float32, the A keys only
    del ads, a0l
    t1 = time.time()
    for s in range(0, CK, args.batch):
        sl = slice(s, s + args.batch)
        features({k: v[sl].float() for k, v in x16.items()}, {k: v[sl] for k, v in a0.items()}, idx_all[s:s + args.batch])
    t2 = time.time()
    dense_part(x16, a0, idx_all)
    if args.test and c == 0:
        check_chunk(x16, a0, idx_all)
    del x16, a0
    if args.chunk_hook:
        d_ = os.path.join(args.cache, f"{PREFIX}adapters/chunk_{c:03d}")
        r_ = subprocess.run(args.chunk_hook.format(dir=d_, chunk=c), shell=True)
        if r_.returncode:
            print(f"chunk hook failed on chunk {c} (exit {r_.returncode}); the pass goes on", flush=True)
    for i in idx_all:
        if i not in basis_set and not args.fake:
            os.remove(os.path.join(args.cache, rel(i)))
    done.append(c)
    json.dump({"done": done}, open(f"{args.out}/state2.json", "w"))
    log(f"chunk {c:02d} done in {time.time() - t:.0f}s (load + A0 {t1 - t:.0f}, features {t2 - t1:.0f}, dense {time.time() - t2:.0f}) "
        f"[{len(done)}/{N // CK}]" + (f"; GPU {torch.cuda.memory_allocated() / 2 ** 30:.0f} GB" if dev == "cuda" else "") + f"; queue {ready.qsize()}")
for a in (E_PARTS, MOVE, SKETCH, PROJ2, WIN0, K_L, CPROJ):
    a.flush()
for a, (name, _, _, _, scale) in enumerate(ACC_SPEC):
    nrm = SCALE / max(1.0, float((WACC[a] != 0).sum())) if scale is None else scale
    torch.save({t_: ACC[a][t_] * nrm for t_ in TYPES}, f"{args.out}/{name}.pt")
    log(f"saved {name}.pt")
Kg = np.lib.format.open_memmap(f"{args.out}/K_L_grp.npy", mode="w+", dtype=np.float32, shape=(N, M, NL * 6))
for s in range(0, N, 100):
    k = np.asarray(K_L[s:s + 100]).reshape(-1, M, NL, PER)
    Kg[s:s + 100] = np.concatenate([k[..., :4], k[..., 4:4 + NE].sum(-1, keepdims=True), k[..., 4 + NE:].sum(-1, keepdims=True)], -1).reshape(-1, M, NL * 6)
Kg.flush()
if args.test:
    bad = {nm: int((~np.isfinite(np.asarray(arr[:N]))).sum()) for nm, arr in (("e_parts", E_PARTS), ("move", MOVE), ("sketch", SKETCH), ("proj2", PROJ2),
                                                                             ("win0_dirs", WIN0), ("K_L_mod", K_L), ("cproj", CPROJ))}
    print("non-finite values per output (must all be 0):", bad)
    CHECKS.append(float(sum(bad.values())))
    ep = np.asarray(E_PARTS[:N]); mvv = np.asarray(MOVE[:N])
    tot = ep[..., 0] + ep[..., 1] + 2 * ep[..., 2]
    print(f"summary over {N} LoRAs: share of |dW|^2 in W {ep[..., 0].sum() / tot.sum():.4f}, in L {ep[..., 1].sum() / tot.sum():.4f}, cross {2 * ep[..., 2].sum() / tot.sum():.4f}; "
          f"median |A - A0| / |A0| {np.median(mvv[..., 0]):.4f}, cos(A, A0) {np.median(mvv[..., 1]):.4f}, row-space overlap {np.median(mvv[..., 2]):.4f}")
    sk = np.asarray(SKETCH[:N]); okr = sk[..., 2] > 0
    print(f"sketch: median cos(B, G A0') over modules with B != 0: {np.median(sk[..., 0][okr] / np.sqrt(sk[..., 1][okr] * sk[..., 2][okr])):.4f}")
    print("TEST", "PASSED" if CHECKS and max(CHECKS) < 2e-2 else "FAILED", f"(worst relative error {max(CHECKS) if CHECKS else float('nan'):.1e}; "
          "float16 factors and a float16 G give errors up to about 1e-2 on small numbers)")
log(f"WS_PASS2_DONE {N} LoRAs, basis {M}, accumulators {[s[0] for s in ACC_SPEC]}")
