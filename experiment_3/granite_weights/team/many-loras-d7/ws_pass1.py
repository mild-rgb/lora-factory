"""One streaming pass over the 3,000 Granite everything-LoRAs: weight-space features, weights only (no forward pass).

For every LoRA j (row order of jobs_all.json) and module m (order: for layer 0..31: q, k, v, o, gate_up expert
0..39, down expert 0..39; 84 per layer, 2,688 in all), with dW = 4 * B @ A:
  K_mod.npy    [N, M, 2688]  <dW_j, dW_basis_i>  (Frobenius inner product; basis = M random LoRAs kept on the GPU)
  sv.npy       [N, 2688, 8]  singular values of dW
  proj.npy     [N, 2688, 5, k, 3]  for D in (basis imperial mean, basis modern mean, their difference, a placebo
               difference, a random-sign mean) and its top-k
               singular pairs (u, v) of the module: coefficient u' dW v, write overlap |B' u|, read overlap |A v|
               (B and A orthonormalised first, so both overlaps are in [0, 1])
  Sw_grp.npy, Sr_grp.npy  [N, M, 192]  write / read SUBSPACE overlap with each basis LoRA, tr(P_j P_i) / 8 averaged
               over the modules of a (layer, type) group (gauge-free: P = projector on col(B) or row(A))
  mean_imperial.pt, mean_modern.pt   full mean dW over all LoRAs of the group (dict of tensors, float32)
  dirs.pt      basis means' top-k singular triplets per module; dirs_split.npy [2688, 3, k, 4]: split-half
               |cos u|, |cos v| and the two halves' singular values for (imperial, modern, diff);
               names.json, basis_idx.npy (chunks 0-59 only), basis_blocks.npy, modules.json
K_grp.npy [N, M, 192] (K summed inside each (layer, type) group) is written at the end.

Resumable: finished chunks are listed in state.json together with the running sums (checkpointed every --ckpt chunks).
Usage: HF_TOKEN=... python ws_pass1.py [--out /content/ws --cache /content/ws_cache --basis_imp 400 --basis_mod 100]
       python ws_pass1.py --test    (tiny basis, 2 chunks, checks the maths against explicit dW)
"""
import argparse, json, os, queue, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/ws")
ap.add_argument("--cache", default="/content/ws_cache")
ap.add_argument("--jobs", default="/content/granite_factory/jobs_all.json")
ap.add_argument("--basis_imp", type=int, default=400)
ap.add_argument("--basis_mod", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--weights", default="/content/ws_in/pass_weights.npy", help="bb's label weights [N, n] with a .json column list")
ap.add_argument("--n_cov", type=int, default=4, help="how many weight columns to accumulate (13.3 GB of RAM each)")
ap.add_argument("--chunks", type=int, default=75)
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
if args.test:
    args.basis_imp, args.basis_mod, args.chunks, args.out, args.n_cov = 8, 4, 2, "/content/ws_test", 1
REPO, PREFIX = "mild-rgb/lora-factory-experiments", "experiment_3/granite_factory/"
TOKEN = os.environ.get("HF_TOKEN")
SCALE, NL, NE, R, CK = 4.0, 32, 40, 8, 40
dev = "cuda"
os.makedirs(args.out, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


JOBS = json.load(open(args.jobs))[:args.chunks * CK]
NAMES = [j["name"] for j in JOBS]
N = len(NAMES)
IS_IMP = np.array([n.startswith("imperial") for n in NAMES])
rng = np.random.RandomState(args.seed)
POOL = np.arange(N) < min(N, 60 * CK)          # the basis never touches chunks 60-74: they are the team's holdout
BASIS = np.sort(np.r_[rng.choice(np.flatnonzero(IS_IMP & POOL), args.basis_imp, replace=False),
                      rng.choice(np.flatnonzero(~IS_IMP & POOL), args.basis_mod, replace=False)])
M = len(BASIS)
ATT = ["q", "k", "v", "o"]
TYPES = ATT + ["gu", "dn"]                      # gu = gate_up (B1 A1), dn = down (B2 A2)
FAC = {"q": ("q.B", "q.A"), "k": ("k.B", "k.A"), "v": ("v.B", "v.A"), "o": ("o.B", "o.A"), "gu": ("B1", "A1"), "dn": ("B2", "A2")}
MODULES = [{"layer": L, "type": t, "expert": -1} for L in range(NL) for t in ATT][:0]
for L in range(NL):
    MODULES += [{"layer": L, "type": t, "expert": -1} for t in ATT]
    MODULES += [{"layer": L, "type": "gate_up", "expert": e} for e in range(NE)]
    MODULES += [{"layer": L, "type": "down", "expert": e} for e in range(NE)]
json.dump(NAMES, open(f"{args.out}/names.json", "w"))
json.dump(MODULES, open(f"{args.out}/modules.json", "w"))
np.save(f"{args.out}/basis_idx.npy", BASIS)


def rel(i):
    return f"{PREFIX}adapters/chunk_{i // CK:03d}/{NAMES[i]}.safetensors"


def load(i):
    """Adapter i as stacked float16 CPU tensors: 'q.A' [32, 8, 1536], 'q.B' [32, d, 8], ..., 'A1' [32, 40, 8, 1536], ..."""
    sd = load_file(os.path.join(args.cache, rel(i)))
    out = {}
    for t in ATT:
        for f in "AB":
            out[f"{t}.{f}"] = torch.stack([sd[f"model.layers.{L}.self_attn.{t}_proj.{f}"] for L in range(NL)])
    for k in ("A1", "B1", "A2", "B2"):
        out[k] = torch.stack([sd[f"model.layers.{L}.block_sparse_moe.experts.{k}"] for L in range(NL)])
    return out


def to_gpu(ads):
    """List of loaded adapters -> dict key -> [X, 32, (40,) ...] float32 on the GPU; attention gets a dummy expert axis."""
    out = {}
    for k in ads[0]:
        t = torch.stack([a[k] for a in ads]).to(dev).float()
        out[k] = t if k in ("A1", "B1", "A2", "B2") else t.unsqueeze(2)       # [X, 32, E, ., .] with E = 1 for attention
    return out


def psd_parts(G):
    """Pieces of a batch of PSD 8x8 Gram matrices, safe when a factor is zero or rank-deficient (an expert that was
    never routed to keeps B = 0): pseudo-inverse, V diag(w^-1/2) (zero columns where w = 0), and the square root."""
    w, V = torch.linalg.eigh(G.double().cpu())
    ok = w > (w.amax(-1, keepdim=True) * 1e-9).clamp_min(1e-30)
    wi = torch.where(ok, 1.0 / w.clamp_min(1e-300), torch.zeros_like(w))
    pinv = (V * wi[..., None, :]) @ V.transpose(-1, -2)
    ihalf = V * wi.sqrt()[..., None, :]
    half = (V * w.clamp_min(0).sqrt()[..., None, :]) @ V.transpose(-1, -2)
    return pinv.float().to(G.device), ihalf.float().to(G.device), half


def gram_inv(Bx, Ax):
    """Gram matrices B'B and A A' with their pseudo-inverses, [..., 8, 8]."""
    gb = Bx.transpose(-1, -2) @ Bx
    ga = Ax @ Ax.transpose(-1, -2)
    return gb, ga, psd_parts(gb)[0], psd_parts(ga)[0]


def pair_block(Bb, Ab, gbi_b, gai_b, Bx, Ax, gbi_x, gai_x):
    """Basis block against a batch for a set of E modules.
    Bb [M, E, d, r], Ab [M, E, r, h] (float16), Bx [X, E, d, r], Ax [X, E, r, h] (float32).
    Returns K [X, M, E] (inner products of B@A), Sw, Sr [X, M, E] (write / read subspace overlaps, 0..1)."""
    Mb, E, d, r = Bb.shape
    X = Bx.shape[0]
    P = (Bb.float().permute(1, 0, 3, 2).reshape(E, Mb * r, d) @ Bx.permute(1, 2, 0, 3).reshape(E, d, X * r)).view(E, Mb, r, X, r)
    Q = (Ab.float().permute(1, 0, 2, 3).reshape(E, Mb * r, -1) @ Ax.permute(1, 3, 0, 2).reshape(E, -1, X * r)).view(E, Mb, r, X, r)
    K = (P * Q).sum((2, 4)).permute(2, 1, 0)
    Tw = torch.einsum("meca,emaxb->emcxb", gbi_b, P)
    Sw = torch.einsum("emcxb,xebd,emcxd->xme", Tw, gbi_x, P) / r
    Tr = torch.einsum("meca,emaxb->emcxb", gai_b, Q)
    Sr = torch.einsum("emcxb,xebd,emcxd->xme", Tr, gai_x, Q) / r
    return K, Sw, Sr


def add_type(St, Bx, Ax, sel):
    """St[layer] += sum over the selected batch members of B @ A (not yet scaled). Bx [X, 32, E, d, r], Ax [X, 32, E, r, h]."""
    if not sel.any():
        return
    Bx, Ax = Bx[sel], Ax[sel]
    X = Bx.shape[0]
    for L in range(NL):
        St[L] += Bx[:, L].permute(1, 2, 0, 3).reshape(Bx.shape[2], Bx.shape[3], X * R) @ \
                 Ax[:, L].permute(1, 0, 2, 3).reshape(Ax.shape[2], X * R, Ax.shape[4])


def add_sums(S, x, sel):
    for t in TYPES:
        add_type(S[t], x[FAC[t][0]], x[FAC[t][1]], sel)


SHAPES = {"q": (NL, 1, 1536, 1536), "o": (NL, 1, 1536, 1536), "k": (NL, 1, 512, 1536), "v": (NL, 1, 512, 1536),
          "gu": (NL, NE, 1024, 1536), "dn": (NL, NE, 1536, 512)}


def new_sums_type(t):
    return torch.zeros(*SHAPES[t], device=dev)


def new_sums():
    return {t: new_sums_type(t) for t in TYPES}


def fetch(idx):
    """Download adapters by index (skips files that are already in the cache)."""
    todo = [i for i in idx if not os.path.exists(os.path.join(args.cache, rel(i)))]
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda i: hf_hub_download(REPO, rel(i), token=TOKEN, local_dir=args.cache), todo))


def flat(per_type, X):
    """dict type -> [X, ..., 32, E] arrays (last two axes layer, expert) -> [X, ..., 2688] in module order."""
    parts = [per_type[t] for t in ATT] + [per_type["gu"], per_type["dn"]]     # E = 1,1,1,1,40,40
    cat = torch.cat(parts, dim=-1)                                            # [..., 32, 84]
    return cat.reshape(*cat.shape[:-2], NL * 84)


# ------------------------------------------------------------------ stage A: the basis, its means and directions
t = time.time()
fetch(BASIS.tolist())
log(f"basis: {M} adapters in the cache ({time.time() - t:.0f}s for the download)")
with ThreadPoolExecutor(8) as ex:
    loaded = list(ex.map(load, BASIS.tolist()))
# six blocks of the basis: imperial a, b, c0, c1 (a quarter each) and modern m0, m1 (a half each). From their sums:
#   imperial mean (a+b+c0+c1), modern mean (m0+m1), diff; placebo diff = mean(a) - mean(b+c0+c1) (same sample sizes
#   as imperial minus modern, no group signal); random-sign mean = (+a -b +c0 -c1 +m0 -m1) / M (shared part cancelled);
#   split halves: imperial a+b against c0+c1, modern m0 against m1.
b_imp = IS_IMP[BASIS]
blk = np.zeros(M, int)
pi, pm = rng.permutation(np.flatnonzero(b_imp)), rng.permutation(np.flatnonzero(~b_imp))
for q_, part in enumerate(np.array_split(pi, 4)):
    blk[part] = q_
for q_, part in enumerate(np.array_split(pm, 2)):
    blk[part] = 4 + q_
np.save(f"{args.out}/basis_blocks.npy", blk)
nb = np.bincount(blk, minlength=6).astype(float)
MEANS = ["imperial", "modern", "diff", "placebo_diff", "random_sign"]


def top_dirs(D):
    u, s_, v = torch.svd_lowrank(D, q=args.k + 8, niter=4)
    sign = torch.sign(torch.gather(u, 1, u.abs().argmax(1, keepdim=True)))       # largest |entry| of each u positive
    sign[sign == 0] = 1
    return (u * sign)[..., :args.k], s_[..., :args.k], (v * sign)[..., :args.k]


DIRS, SPLIT = {}, {}
for t_ in TYPES:
    Bk, Ak = FAC[t_]
    S6 = [new_sums_type(t_) for _ in range(6)]
    for s in range(0, M, args.batch):
        Bx = torch.stack([a[Bk] for a in loaded[s:s + args.batch]]).to(dev).float()
        Ax = torch.stack([a[Ak] for a in loaded[s:s + args.batch]]).to(dev).float()
        if t_ in ATT:
            Bx, Ax = Bx.unsqueeze(2), Ax.unsqueeze(2)
        for q_ in range(6):
            sel = torch.tensor(blk[s:s + args.batch] == q_, device=dev)
            add_type(S6[q_], Bx, Ax, sel)
    out = {n: ([], [], []) for n in MEANS}
    fro, split = [], []
    for L in range(NL):
        a_, b_, c0, c1, m0, m1 = (S6[q_][L] * SCALE for q_ in range(6))
        mi, mm = (a_ + b_ + c0 + c1) / nb[:4].sum(), (m0 + m1) / nb[4:].sum()
        five = {"imperial": mi, "modern": mm, "diff": mi - mm,
                "placebo_diff": a_ / nb[0] - (b_ + c0 + c1) / nb[1:4].sum(),
                "random_sign": (a_ - b_ + c0 - c1 + m0 - m1) / M}
        for n in MEANS:
            u, s_, v = top_dirs(five[n])
            out[n][0].append(u); out[n][1].append(s_); out[n][2].append(v)
        fro.append(torch.stack([five[n].flatten(1).norm(dim=-1) for n in MEANS]))                # [5, E]
        hi0, hi1 = (a_ + b_) / nb[:2].sum(), (c0 + c1) / nb[2:4].sum()
        hm0, hm1 = m0 / nb[4], m1 / nb[5]
        row = []
        for h0, h1 in ((hi0, hi1), (hm0, hm1), (hi0 - hm0, hi1 - hm1)):
            u0, s0, v0 = top_dirs(h0); u1, s1, v1 = top_dirs(h1)
            row.append(torch.stack([(u0 * u1).sum(1).abs(), (v0 * v1).sum(1).abs(), s0, s1], -1))  # [E, k, 4]
        split.append(torch.stack(row, 1))                                                          # [E, 3, k, 4]
    DIRS[t_] = {n: (torch.stack(out[n][0]), torch.stack(out[n][1]), torch.stack(out[n][2])) for n in MEANS}
    DIRS[t_]["fro"] = torch.stack(fro)                                                             # [32, 5, E]
    SPLIT[t_] = torch.stack(split)                                                                 # [32, E, 3, k, 4]
    del S6
    torch.cuda.empty_cache()
    log(f"stage A {t_}: directions done; GPU used {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB")
torch.save({t_: {n: tuple(z.cpu() for z in v) if isinstance(v, tuple) else v.cpu() for n, v in d.items()} for t_, d in DIRS.items()},
           f"{args.out}/dirs.pt")
sp = torch.cat([SPLIT[t_] for t_ in TYPES], 1)                                                     # [32, 84, 3, k, 4]
np.save(f"{args.out}/dirs_split.npy", sp.reshape(NL * 84, 3, args.k, 4).cpu().numpy())
BAS = {k: torch.stack([a[k] for a in loaded]).to(dev) for k in loaded[0]}                          # float16, [M, 32, (40,) ...]
for k in list(BAS):
    if k not in ("A1", "B1", "A2", "B2"):
        BAS[k] = BAS[k].unsqueeze(2)
del loaded
BINV = {}
for t_ in TYPES:
    Bk, Ak = FAC[t_]
    gbi, gai = [], []
    for s in range(0, M, 50):
        _, _, a, b = gram_inv(BAS[Bk][s:s + 50].float(), BAS[Ak][s:s + 50].float())
        gbi.append(a); gai.append(b)
    BINV[t_] = (torch.cat(gbi), torch.cat(gai))                                                    # [M, 32, E, 8, 8]
log(f"basis on the GPU: {sum(v.numel() * 2 for v in BAS.values()) / 2 ** 30:.1f} GB; GPU used {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB")

# ------------------------------------------------------------------ stage B: every LoRA, chunk by chunk
K_mm = np.lib.format.open_memmap(f"{args.out}/K_mod.npy", mode="w+", dtype=np.float32, shape=(N, M, NL * 84))
SV = np.lib.format.open_memmap(f"{args.out}/sv.npy", mode="w+", dtype=np.float32, shape=(N, NL * 84, R))
PJ = np.lib.format.open_memmap(f"{args.out}/proj.npy", mode="w+", dtype=np.float32, shape=(N, NL * 84, 5, args.k, 3))
SW = np.lib.format.open_memmap(f"{args.out}/Sw_grp.npy", mode="w+", dtype=np.float32, shape=(N, M, NL * 6))
SR = np.lib.format.open_memmap(f"{args.out}/Sr_grp.npy", mode="w+", dtype=np.float32, shape=(N, M, NL * 6))

# Dense accumulators sum_j w[a, j] * B_j @ A_j, kept in CPU RAM (13.3 GB each):
#   the four half means (imperial / modern x even / odd row) and bb's label-weighted sums over the non-basis LoRAs.
even = np.arange(N) % 2 == 0
in_basis = np.zeros(N, bool); in_basis[BASIS] = True
ACC_NAMES = ["mean_imperial_h0", "mean_imperial_h1", "mean_modern_h0", "mean_modern_h1"]
WACC = [IS_IMP & even, IS_IMP & ~even, ~IS_IMP & even, ~IS_IMP & ~even]
if args.weights and os.path.exists(args.weights):
    pw = np.load(args.weights)[:N]
    cols = json.load(open(args.weights.replace(".npy", ".json")))
    cols = cols["columns"] if isinstance(cols, dict) else cols
    for c_ in cols[:args.n_cov]:
        ACC_NAMES.append("cov_" + c_); WACC.append(pw[:, cols.index(c_)] * ~in_basis * POOL)   # no basis, no holdout (chunks 60-74)
WACC = np.stack([np.asarray(w, np.float32) for w in WACC])                           # [n_acc, N]
ACC = [{t_: torch.zeros(*SHAPES[t_]) for t_ in TYPES} for _ in ACC_NAMES]
STAGE = {t_: torch.empty(*SHAPES[t_][1:]).pin_memory() for t_ in TYPES}
log(f"accumulators in RAM: {ACC_NAMES}")
json.dump({"names": ACC_NAMES, "n": (WACC != 0).sum(1).tolist(), "sum_w": WACC.sum(1).tolist(), "sum_w2": (WACC ** 2).sum(1).tolist()},
          open(f"{args.out}/acc_meta.json", "w"))
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


def features(x, idx):
    """All per-LoRA features for a batch x (float32 on the GPU, [X, 32, E, ...])."""
    X = len(idx)
    Kp, Swp, Srp, svp, pjp = {}, {}, {}, {}, {}
    for t_ in TYPES:
        Bk, Ak = FAC[t_]
        Bx, Ax = x[Bk], x[Ak]                                                    # [X, 32, E, d, r], [X, 32, E, r, h]
        gb = Bx.transpose(-1, -2) @ Bx
        ga = Ax @ Ax.transpose(-1, -2)
        gbi, bih, bh = psd_parts(gb)
        gai, aih, _ = psd_parts(ga)
        # singular values of B@A: sqrt(eig(gb^1/2 ga gb^1/2))
        ev = torch.linalg.eigvalsh(bh @ ga.double().cpu() @ bh).clamp_min(0).sqrt().flip(-1).float() * SCALE
        svp[t_] = ev.permute(0, 3, 1, 2)                                         # [X, 8, 32, E]
        step = 4 if t_ in ("gu", "dn") else NL
        Ks, Sws, Srs = [], [], []
        E = Bx.shape[2]
        for L in range(0, NL, step):
            sl = slice(L, L + step)
            f = lambda z: z[:, sl].reshape(z.shape[0], -1, *z.shape[3:])         # merge (layer, expert) into one module axis
            K, Sw, Sr = pair_block(f(BAS[Bk]), f(BAS[Ak]), f(BINV[t_][0]), f(BINV[t_][1]), f(Bx), f(Ax), f(gbi), f(gai))
            Ks.append(K.view(X, M, -1, E)); Sws.append(Sw.view(X, M, -1, E)); Srs.append(Sr.view(X, M, -1, E))
        Kp[t_] = torch.cat(Ks, 2) * SCALE ** 2                                    # [X, M, 32, E]
        Swp[t_], Srp[t_] = torch.cat(Sws, 2).mean(-1, keepdim=True), torch.cat(Srs, 2).mean(-1, keepdim=True)
        # projections on the basis means' directions, with orthonormalised factors for the overlaps
        Qb = Bx @ bih                                                             # orthonormal columns spanning col(B)
        Qa = aih.transpose(-1, -2) @ Ax                                           # orthonormal rows spanning row(A)
        out = []
        for name in MEANS:
            U, _, V = DIRS[t_][name]                                              # [32, E, d, k], [32, E, h, k]
            coef = torch.einsum("ledk,xledr,xlerh,lehk->xlek", U, Bx, Ax, V) * SCALE
            w = torch.einsum("ledk,xledr->xlekr", U, Qb).norm(dim=-1)
            rd = torch.einsum("xlerh,lehk->xlerk", Qa, V).norm(dim=-2)
            out.append(torch.stack([coef, w, rd], -1))                            # [X, 32, E, k, 3]
        pjp[t_] = torch.stack(out, 3).permute(0, 3, 4, 5, 1, 2)                   # [X, 5, k, 3, 32, E]
    K_mm[idx] = flat(Kp, X).cpu().numpy()
    SV[idx] = flat({t_: v.to(dev) for t_, v in svp.items()}, X).permute(0, 2, 1).cpu().numpy()
    PJ[idx] = flat(pjp, X).permute(0, 4, 1, 2, 3).cpu().numpy()
    SW[idx] = torch.cat([Swp[t_] for t_ in TYPES], -1).reshape(X, M, NL * 6).cpu().numpy()
    SR[idx] = torch.cat([Srp[t_] for t_ in TYPES], -1).reshape(X, M, NL * 6).cpu().numpy()


def accumulate(x16, idx):
    """ACC[a][type][layer] += sum_x w[a, x] B_x @ A_x for the whole chunk (computed on the GPU, added in RAM)."""
    Wc = torch.tensor(WACC[:, idx], device=dev)
    for t_ in TYPES:
        Bk, Ak = FAC[t_]
        for L in range(NL):
            Bx, Ax = x16[Bk][:, L].float(), x16[Ak][:, L].float()                 # [X, E, d, r], [X, E, r, h]
            E = Bx.shape[1]
            for a in range(len(ACC)):
                nz = torch.nonzero(Wc[a]).flatten()
                if len(nz) == 0:
                    continue
                Bw = (Bx[nz] * Wc[a][nz][:, None, None, None]).permute(1, 2, 0, 3).reshape(E, Bx.shape[2], -1)
                tmp = Bw @ Ax[nz].permute(1, 0, 2, 3).reshape(E, -1, Ax.shape[-1])
                STAGE[t_].copy_(tmp)
                ACC[a][t_][L].add_(STAGE[t_])


done = []
while True:
    c = ready.get()
    if c is None:
        break
    t = time.time()
    idx_all = list(range(c * CK, (c + 1) * CK))
    with ThreadPoolExecutor(8) as ex:
        ads = list(ex.map(load, idx_all))
    x16 = {k: torch.stack([a[k] for a in ads]).to(dev) for k in ads[0]}
    for k in list(x16):
        if k not in ("A1", "B1", "A2", "B2"):
            x16[k] = x16[k].unsqueeze(2)
    del ads
    t1 = time.time()
    for s in range(0, CK, args.batch):
        x = {k: v[s:s + args.batch].float() for k, v in x16.items()}
        features(x, idx_all[s:s + args.batch])
    t2 = time.time()
    accumulate(x16, idx_all)
    if args.test and c == 0:                                                      # check the maths on the last LoRA of chunk 0
        j = idx_all[-1]; xi = {k: v[-1] for k, v in x.items()}
        for t_, (L, e) in (("gu", (5, 7)), ("dn", (20, 3)), ("q", (11, 0)), ("v", (30, 0))):
            Bk, Ak = FAC[t_]
            dWj = SCALE * xi[Bk][L, e] @ xi[Ak][L, e]
            m = 3
            dWb = SCALE * BAS[Bk][m, L, e].float() @ BAS[Ak][m, L, e].float()
            mod = L * 84 + (ATT.index(t_) if t_ in ATT else 4 + e if t_ == "gu" else 44 + e)
            sv_true = torch.linalg.svdvals(dWj)[:R].cpu().numpy()
            U, _, V = DIRS[t_]["diff"]
            c_true = float(U[L, e, :, 0] @ dWj @ V[L, e, :, 0])
            qb = torch.linalg.qr(xi[Bk][L, e])[0]
            a_ = [i for i, n_ in enumerate(ACC_NAMES) if WACC[i, j] != 0][0]
            acc_true = sum(float(WACC[a_, i]) * (x16[Bk][i - idx_all[0], L, e].float() @ x16[Ak][i - idx_all[0], L, e].float()) for i in idx_all)
            print(f"CHECK {t_} L{L} e{e}: K {K_mm[j, m, mod]:.6f} vs {float((dWj * dWb).sum()):.6f} | sv max diff "
                  f"{np.abs(SV[j, mod] - sv_true).max():.2e} (top {sv_true[0]:.4f}) | coef {PJ[j, mod, 2, 0, 0]:.6f} vs {c_true:.6f} "
                  f"| write overlap w {PJ[j, mod, 2, 0, 1]:.4f} vs {float((qb.T @ U[L, e, :, 0]).norm()):.4f} "
                  f"| acc {ACC_NAMES[a_]} max diff {float((ACC[a_][t_][L, e] - acc_true.cpu()).abs().max()):.2e} (max {float(acc_true.abs().max()):.3f})", flush=True)
        if j in basis_set or True:
            bm = int(BASIS[3]); 
            print("basis row check: K diag vs sum sv^2 for a basis LoRA processed so far:",
                  [(float(K_mm[b_, i_, 100]), float((SV[b_, 100] ** 2).sum())) for i_, b_ in enumerate(BASIS.tolist()) if b_ < (c + 1) * CK][:2], flush=True)
    del x16, x
    for i in idx_all:
        if i not in basis_set:
            os.remove(os.path.join(args.cache, rel(i)))
    done.append(c)
    json.dump({"done": done}, open(f"{args.out}/state.json", "w"))
    log(f"chunk {c:02d} done in {time.time() - t:.0f}s (load {t1 - t:.0f}, features {t2 - t1:.0f}, sums {time.time() - t2:.0f}) "
        f"[{len(done)}/{N // CK}]; GPU {torch.cuda.memory_allocated() / 2 ** 30:.0f} GB; queue {ready.qsize()}")
for a in (K_mm, SV, PJ, SW, SR):
    a.flush()
for a, name in enumerate(ACC_NAMES):
    nrm = SCALE / max(1.0, float((WACC[a] != 0).sum())) if name.startswith("mean_") else SCALE
    torch.save({t_: ACC[a][t_] * nrm for t_ in TYPES}, f"{args.out}/{name}.pt")
    log(f"saved {name}.pt")
Kg = np.lib.format.open_memmap(f"{args.out}/K_grp.npy", mode="w+", dtype=np.float32, shape=(N, M, NL * 6))
for s in range(0, N, 100):
    k = np.asarray(K_mm[s:s + 100]).reshape(-1, M, NL, 84)
    Kg[s:s + 100] = np.concatenate([k[..., :4], k[..., 4:44].sum(-1, keepdims=True), k[..., 44:].sum(-1, keepdims=True)], -1).reshape(-1, M, NL * 6)
Kg.flush()
log(f"WS_PASS1_DONE {N} LoRAs, basis {M}, accumulators {ACC_NAMES}")
