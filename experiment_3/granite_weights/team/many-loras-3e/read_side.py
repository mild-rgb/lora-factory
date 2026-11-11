#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 3: the read side. Weights only, no forward pass, no model.

A LoRA update is dW = 4 B A. B starts at zero and A at a random matrix that is a pure function of the seed
(evlora._kaiming: CPU generator seeded with (seed * 1,000,003 + crc32(tag)) % 2^62). So the starting A of every
module of every LoRA can be rebuilt exactly, without the adapter file. This script uses that to ask:

Part 1 (the 500 cached basis adapters):
  move        how far the read side moved in training: |A - A0| / |A0|, cos(A, A0), and the overlap of the row
              spaces of A and A0 (1 = same 8-dimensional read window, rank / in_dim = unrelated)
  window      the share of the group's mean update mu that falls inside a LoRA's read window, |mu Q'|^2 / |mu|^2
              (Q = orthonormal rows of A), for the starting window Q0 and the final window Q. A random window gives
              rank / in_dim whatever the spectrum of mu. The same for the placebo directions (noise matrix of the
              mean, script 1) is the floor measured on the data.
  cosines     plain cos(dW_j, mu) and the "window" cosine <dW_j, mu> / (|dW_j| |mu Pi_j|), Pi_j = projector on the
              read window of LoRA j. Near 1: LoRA j is the common update seen through its own window, nothing else.
              (With the START window in the denominator the number is a ratio, not a cosine: it can pass 1 when
              the read side has moved towards the mean's read directions.)
  parts       dW = W + L with W = 4 B A0 (window part) and L = 4 B (A - A0) (learned-read part): energy of each
              and the cross term, per LoRA and module (39's split; pass 2 does the Gram and the means of L)
  regime      per shared direction k of a module: the coefficient c_jk = u_k' dW_j v_k against the window overlap
              |Q0_j v_k|^2 over LoRAs. One gradient step predicts c proportional to the overlap ("lucky window");
              converged training predicts no dependence. Spearman per module, summarised per type and layer, with
              the placebo directions as the null.
Part 2 (all 3,000 LoRAs, seeds only, no adapter is read):
  init_window.npy [N, modules, 2, 4]: |Q0_j v_k|^2 for the top 4 read directions of the imperial mean and of the
  imperial-minus-modern difference; init_window_grp.npy [N, 192, 2]: the energy-weighted window share per
  (layer, type). These are fixed before training starts, so if they predict behaviour, the arrow of cause is clear.

mu is the full-population mean cut to the top directions kept in top_dirs.pt (script 1); it contains each LoRA
itself with weight 1 / 2,400.
Usage:  python read_side.py [--ws /content/ws --cache /content/ws_cache --dirs /content/ws/3e/top_dirs.pt --out /content/ws/3e]
        python read_side.py --test
"""
import argparse, json, math, os, re, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ws", default="/content/ws")
ap.add_argument("--cache", default="/content/ws_cache")
ap.add_argument("--prefix", default="experiment_3/granite_factory/adapters")
ap.add_argument("--dirs", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--kd", type=int, default=4, help="directions per module used for the regime test and the window features")
ap.add_argument("--skip_part2", action="store_true")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
T0 = time.time()
ATT = ["q", "k", "v", "o"]
TYPES = ATT + ["gu", "dn"]
FAC = {"q": ("q.B", "q.A"), "k": ("k.B", "k.A"), "v": ("v.B", "v.A"), "o": ("o.B", "o.A"), "gu": ("B1", "A1"), "dn": ("B2", "A2")}
SCALE, R = 4.0, 8
dev = torch.device(args.device)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def kaiming(seed, tail, fin, tag):
    """The factory's lora_A init for one LoRA (evlora._kaiming), bit for bit: U(-1/sqrt(fin), 1/sqrt(fin)), CPU generator."""
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
    return torch.empty(*tail).uniform_(-1 / math.sqrt(fin), 1 / math.sqrt(fin), generator=g)


def init_A(seed, dims):
    """Starting A of every module of one LoRA: dict key -> 'q.A' [NL, r, h] ..., 'A1' [NL, E, r, h], 'A2' [NL, E, r, mi]."""
    NL, NE, h, mi = dims
    out = {}
    for t in ATT:
        out[f"{t}.A"] = torch.stack([kaiming(seed, (R, h), h, f"L{L}.{t}_proj") for L in range(NL)])
    out["A1"] = torch.stack([kaiming(seed, (NE, R, h), h, f"L{L}.experts.gate_up") for L in range(NL)])
    out["A2"] = torch.stack([kaiming(seed, (NE, R, mi), mi, f"L{L}.experts.down") for L in range(NL)])
    return out


def seed_of(name):
    return int(re.search(r"_s(\d+)$", name).group(1))


def rel(i, names):
    return os.path.join(args.cache, args.prefix, f"chunk_{i // args.chunk:03d}", f"{names[i]}.safetensors")


def load(path, NL):
    from safetensors.torch import load_file
    sd = load_file(path)
    out = {}
    for t in ATT:
        for f in "AB":
            out[f"{t}.{f}"] = torch.stack([sd[f"model.layers.{L}.self_attn.{t}_proj.{f}"] for L in range(NL)]).float()
    for k in ("A1", "B1", "A2", "B2"):
        out[k] = torch.stack([sd[f"model.layers.{L}.block_sparse_moe.experts.{k}"] for L in range(NL)]).float()
    return out


def orth_rows(A):
    """Orthonormal rows spanning row(A): [..., r, h]."""
    G = A @ A.transpose(-1, -2)
    eye = torch.eye(A.shape[-2], device=A.device)
    L = torch.linalg.cholesky(G + 1e-9 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye + 1e-30 * eye)
    return torch.linalg.solve_triangular(L, A, upper=False)


def psd_pinv(G):
    """Pseudo-inverse of a batch of PSD 8x8 Gram matrices, safe when a factor is zero or rank-deficient (an expert
    that was never or hardly ever routed to keeps B = 0 or a B of low rank). Same recipe as psd_parts() of the passes."""
    w, V = torch.linalg.eigh(G.double().cpu())
    ok = w > (w.amax(-1, keepdim=True) * 1e-9).clamp_min(1e-30)
    wi = torch.where(ok, 1.0 / w.clamp_min(1e-300), torch.zeros_like(w))
    return ((V * wi[..., None, :]) @ V.transpose(-1, -2)).float().to(G.device)


def orth_rows_safe(A):
    """Like orth_rows, without a Cholesky factor: rows of pinv(A A')^(1/2) A (zero rows where A has no rank)."""
    w, V = torch.linalg.eigh((A @ A.transpose(-1, -2)).double().cpu())
    ok = w > (w.amax(-1, keepdim=True) * 1e-9).clamp_min(1e-30)
    wi = torch.where(ok, 1.0 / w.clamp_min(1e-300).sqrt(), torch.zeros_like(w))
    return (V * wi[..., None, :]).transpose(-1, -2).float().to(A.device) @ A


def flat(per_type):
    """dict type -> [X, NL, E, ...] -> [X, modules, ...] in module order (per layer q, k, v, o, gate_up experts, down experts)."""
    parts = [per_type[t] for t in TYPES]
    extra = parts[0].shape[3:]
    cat = torch.cat([p.reshape(p.shape[0], p.shape[1], p.shape[2], -1) for p in parts], dim=2)     # [X, NL, 84, prod(extra)]
    return cat.reshape(cat.shape[0], -1, *extra)


# ------------------------------------------------------------------ test data
def make_test(ws, cache):
    """12 fake LoRAs built the factory's way: A0 from the seed, A = A0 + a small step towards the mean's top read
    direction, B = (mean restricted to the window) + noise. Plus fake top_dirs.pt."""
    from safetensors.torch import save_file
    torch.manual_seed(0)
    NL, NE, h, mi = 2, 3, 64, 24
    shp = {"q": (h, h), "o": (h, h), "k": (mi, h), "v": (mi, h), "gu": (2 * mi, h), "dn": (h, mi)}
    names = [f"{'imperial' if i % 5 else 'modern'}_s{10001 + i}" for i in range(12)]
    dirs = {}
    for t, (d, hin) in shp.items():
        E = NE if t in ("gu", "dn") else 1
        dirs[t] = {}
        for nm in ("imperial", "modern", "diff", "noise_imperial", "noise_modern", "noise_diff"):
            U = torch.linalg.qr(torch.randn(NL, E, d, 4))[0]; V = torch.linalg.qr(torch.randn(NL, E, hin, 4))[0]
            S = torch.tensor([3.0, 1.0, 0.3, 0.1]).expand(NL, E, 4).clone()
            dirs[t][nm] = (U, S, V)
    os.makedirs(os.path.join(ws, "3e"), exist_ok=True)
    torch.save(dirs, os.path.join(ws, "3e", "top_dirs.pt"))
    for i, nme in enumerate(names):
        A0 = init_A(seed_of(nme), (NL, NE, h, mi))
        gr = "imperial" if nme.startswith("imperial") else "modern"
        sd = {}
        for t in TYPES:
            Bk, Ak = FAC[t]
            a0 = A0[Ak] if t in ("gu", "dn") else A0[Ak].unsqueeze(1)                 # [NL, E, r, hin]
            U, S, V = dirs[t][gr]
            a = a0 + 0.15 * torch.randn(*a0.shape[:3], 1) * V[..., 0].unsqueeze(2) / math.sqrt(a0.shape[-1]) * 3
            mu = torch.einsum("ledk,lek,lehk->ledh", U, S, V)
            b = mu @ a.transpose(-1, -2) * (a0.shape[-1] / R) * 3 / SCALE / 3            # roughly G A' (one "gradient step")
            b = b + 0.02 * torch.randn_like(b)
            if t == "gu":
                b[:, 1] = 0                                                              # an expert that was never routed to
                b[:, 2, :, 2:] = 0                                                       # an expert with a B of rank 2 (hardly ever routed to)
            for L in range(NL):
                if t in ATT:
                    sd[f"model.layers.{L}.self_attn.{t}_proj.A"] = a[L, 0].half(); sd[f"model.layers.{L}.self_attn.{t}_proj.B"] = b[L, 0].half()
                else:
                    sd[f"model.layers.{L}.block_sparse_moe.experts.{Ak}"] = a[L].half(); sd[f"model.layers.{L}.block_sparse_moe.experts.{Bk}"] = b[L].half()
        p = rel(i, names)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save_file(sd, p)
    json.dump(names, open(os.path.join(ws, "names.json"), "w"))
    np.save(os.path.join(ws, "basis_idx.npy"), np.arange(12))


if args.test:
    base = os.environ.get("WS_TEST_DIR", "/tmp/ws_test_3e") + "_read"
    args.ws, args.cache, args.batch, args.workers = base, base + "_cache", 4, 2
    make_test(args.ws, args.cache)
OUT = args.out or os.path.join(args.ws, "3e")
os.makedirs(OUT, exist_ok=True)
names = json.load(open(f"{args.ws}/names.json"))
basis = np.load(f"{args.ws}/basis_idx.npy")
is_imp = np.array([n.startswith("imperial") for n in names])
DIRS = torch.load(args.dirs or os.path.join(args.ws, "3e", "top_dirs.pt"), map_location="cpu")
NL, NE = DIRS["gu"]["imperial"][0].shape[:2]
h, mi = DIRS["gu"]["imperial"][2].shape[2], DIRS["dn"]["imperial"][2].shape[2]
DIMS = (NL, NE, h, mi)
in_dim = {"q": h, "k": h, "v": h, "o": h, "gu": h, "dn": mi}
KD = min(args.kd, DIRS["q"]["imperial"][1].shape[-1])
log(f"{len(names)} LoRAs, basis {len(basis)}; {NL} layers, {NE} experts, hidden {h}, expert hidden {mi}; device {dev}")
Dg = {t: {nm: tuple(z.to(dev).float() for z in v) for nm, v in DIRS[t].items()} for t in TYPES}


def a_of(x, t):
    """[X, NL, E, r, hin] for type t from a dict of stacked tensors (attention gets a dummy expert axis)."""
    z = x[FAC[t][1]]
    return z if t in ("gu", "dn") else z.unsqueeze(2)


def b_of(x, t):
    z = x[FAC[t][0]]
    return z if t in ("gu", "dn") else z.unsqueeze(2)


# ------------------------------------------------------------------ part 1: the cached basis adapters
keys1 = ["rel_move", "cos_A", "rowspace", "win0", "win", "win0_placebo", "win_placebo", "cos_plain", "cos_window", "cos_window0", "e_dw",
         "e_W", "e_L", "x_WL"]
keys4 = ["coef", "rd0", "rd", "wr", "rd0_placebo", "coef_placebo"]
acc1 = {k: [] for k in keys1}; acc4 = {k: [] for k in keys4}
bl = [int(i) for i in basis]
for s in range(0, len(bl), args.batch):
    idx = bl[s:s + args.batch]
    with ThreadPoolExecutor(args.workers) as ex:
        ads = list(ex.map(lambda i: load(rel(i, names), NL), idx))
        a0s = list(ex.map(lambda i: init_A(seed_of(names[i]), DIMS), idx))
    x = {k: torch.stack([a[k] for a in ads]).to(dev) for k in ads[0]}
    x0 = {k: torch.stack([a[k] for a in a0s]).to(dev) for k in a0s[0]}
    grp = [("imperial" if is_imp[i] else "modern") for i in idx]
    o1 = {k: {} for k in keys1}; o4 = {k: {} for k in keys4}
    for t in TYPES:
        A, A0, B = a_of(x, t), a_of(x0, t), b_of(x, t)                         # [X, NL, E, r, hin], [X, NL, E, d, r]
        nA0 = A0.flatten(3).norm(dim=-1)
        o1["rel_move"][t] = (A - A0).flatten(3).norm(dim=-1) / nA0
        o1["cos_A"][t] = (A * A0).flatten(3).sum(-1) / (A.flatten(3).norm(dim=-1) * nA0)
        Q, Q0 = orth_rows_safe(A), orth_rows_safe(A0)
        o1["rowspace"][t] = (Q @ Q0.transpose(-1, -2)).pow(2).sum((-1, -2)) / R
        Ga, Gb = A @ A.transpose(-1, -2), B.transpose(-1, -2) @ B
        e_dw = SCALE ** 2 * (Ga * Gb).sum((-1, -2)).clamp_min(0)                # |dW|^2
        o1["e_dw"][t] = e_dw
        dA = A - A0                                                            # dW = W + L: window part W = 4 B A0, learned-read part L = 4 B dA
        o1["e_W"][t] = SCALE ** 2 * ((A0 @ A0.transpose(-1, -2)) * Gb).sum((-1, -2))
        o1["e_L"][t] = SCALE ** 2 * ((dA @ dA.transpose(-1, -2)) * Gb).sum((-1, -2))
        o1["x_WL"][t] = SCALE ** 2 * ((A0 @ dA.transpose(-1, -2)) * Gb).sum((-1, -2))      # <W, L>
        # own-group mean directions and the placebo (noise) directions, picked per LoRA
        pick = lambda nm_i, nm_m, j: torch.stack([torch.stack([Dg[t][nm_i if g == "imperial" else nm_m][z] for g in grp]) for z in (0, 1, 2)][j:j + 1]).squeeze(0)
        U, S, V = (pick("imperial", "modern", j) for j in (0, 1, 2))            # [X, NL, E, d, k], [X, NL, E, k], [X, NL, E, hin, k]
        Up, Sp, Vp = (pick("noise_imperial", "noise_modern", j) for j in (0, 1, 2))
        rd = (Q @ V).pow(2).sum(-2); rd0 = (Q0 @ V).pow(2).sum(-2)             # [X, NL, E, k]: |Q v_k|^2
        rdp = (Q @ Vp).pow(2).sum(-2); rd0p = (Q0 @ Vp).pow(2).sum(-2)
        w = S.pow(2)
        o1["win"][t] = (w * rd).sum(-1) / w.sum(-1).clamp_min(1e-30); o1["win0"][t] = (w * rd0).sum(-1) / w.sum(-1).clamp_min(1e-30)
        wp = Sp.pow(2)
        o1["win_placebo"][t] = (wp * rdp).sum(-1) / wp.sum(-1).clamp_min(1e-30)
        o1["win0_placebo"][t] = (wp * rd0p).sum(-1) / wp.sum(-1).clamp_min(1e-30)
        coef = SCALE * torch.einsum("xledk,xledr,xlerh,xlehk->xlek", U, B, A, V)
        coefp = SCALE * torch.einsum("xledk,xledr,xlerh,xlehk->xlek", Up, B, A, Vp)
        num = (S * coef).sum(-1)                                                # <dW, mu> with mu cut to the kept directions
        nd = e_dw.sqrt().clamp_min(1e-30)
        o1["cos_plain"][t] = num / (nd * w.sum(-1).sqrt().clamp_min(1e-30))
        o1["cos_window"][t] = num / (nd * (w * rd).sum(-1).sqrt().clamp_min(1e-30))
        o1["cos_window0"][t] = num / (nd * (w * rd0).sum(-1).sqrt().clamp_min(1e-30))
        dead = e_dw <= 1e-20                                                    # expert never routed to: B = 0
        for k_ in ("cos_plain", "cos_window", "cos_window0"):
            o1[k_][t] = torch.where(dead, torch.full_like(e_dw, float("nan")), o1[k_][t])
        Bu = B.transpose(-1, -2) @ U                                           # [X, NL, E, r, k]
        wr = torch.einsum("...rk,...rs,...sk->...k", Bu, psd_pinv(Gb), Bu)     # |Qb' u_k|^2 without a Cholesky factor
        o4["wr"][t] = torch.where(dead[..., None], torch.full_like(wr, float("nan")), wr)[..., :KD]
        o4["coef"][t], o4["rd"][t], o4["rd0"][t] = coef[..., :KD], rd[..., :KD], rd0[..., :KD]
        o4["rd0_placebo"][t], o4["coef_placebo"][t] = rd0p[..., :KD], coefp[..., :KD]
    for k in keys1:
        acc1[k].append(flat(o1[k]).cpu().numpy())
    for k in keys4:
        acc4[k].append(flat(o4[k]).cpu().numpy())
    if s % (args.batch * 10) == 0:
        log(f"part 1: {s + len(idx)}/{len(bl)} adapters")
P1 = {k: np.concatenate(v) for k, v in acc1.items()}                           # [n_basis, modules]
P4 = {k: np.concatenate(v) for k, v in acc4.items()}                           # [n_basis, modules, KD]
M = P1["rel_move"].shape[1]
PER = M // NL
mod_type = np.tile(np.array(ATT + ["gu"] * NE + ["dn"] * NE), NL)
mod_layer = np.repeat(np.arange(NL), PER)
b_imp = is_imp[basis]
np.savez(f"{OUT}/read_side_basis.npz", **{k: v.astype(np.float32) for k, v in P1.items()}, **{k: v.astype(np.float32) for k, v in P4.items()},
         basis=basis, mod_type=mod_type, mod_layer=mod_layer)


def spearman_cols(a, b):
    """Spearman correlation over LoRAs (axis 0) for every column; NaN where a column is constant or has NaN."""
    ra = np.argsort(np.argsort(a, 0), 0).astype(float); rb = np.argsort(np.argsort(b, 0), 0).astype(float)
    ra -= ra.mean(0); rb -= rb.mean(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    out = np.where(den > 0, (ra * rb).sum(0) / np.where(den > 0, den, 1), np.nan)
    return np.where(np.isfinite(a).all(0) & np.isfinite(b).all(0), out, np.nan)


imp_rows = np.flatnonzero(b_imp)
rho = {"c_vs_window0": np.stack([spearman_cols(P4["coef"][imp_rows][:, :, k], P4["rd0"][imp_rows][:, :, k]) for k in range(KD)], 1),
       "c_vs_window": np.stack([spearman_cols(P4["coef"][imp_rows][:, :, k], P4["rd"][imp_rows][:, :, k]) for k in range(KD)], 1),
       "placebo": np.stack([spearman_cols(P4["coef_placebo"][imp_rows][:, :, k], P4["rd0_placebo"][imp_rows][:, :, k]) for k in range(KD)], 1)}
se_rho = 1 / math.sqrt(max(len(imp_rows) - 3, 1))
med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
print(f"\nPART 1: {len(bl)} basis adapters ({int(b_imp.sum())} imperial). Medians over LoRAs and modules; chance for a window is rank / in_dim.")
print("type  |A-A0|/|A0|  cos(A,A0)  row-space overlap | window share of mu: start, final, placebo(final), chance | plain cos  window cos (final)  window cos (start)")
summary = {}
for t in TYPES:
    m = mod_type == t
    row = {k: med(P1[k][:, m]) for k in keys1}
    row["chance"] = R / in_dim[t]
    summary[t] = row
    print(f"{t:4s}   {row['rel_move']:8.4f}    {row['cos_A']:7.4f}     {row['rowspace']:7.4f}        |   {row['win0']:.4f}  {row['win']:.4f}  {row['win_placebo']:.4f}  {row['chance']:.4f}"
          f"            |  {row['cos_plain']:7.4f}     {row['cos_window']:7.4f}           {row['cos_window0']:7.4f}")
print("\nSplit dW = W + L (W = 4 B A0: the update seen through the start window; L = 4 B (A - A0): the learned-read part).")
print("type   share of |dW|^2 in W    in L    cross term 2<W,L>   (sums over LoRAs and modules; they add up to 1)")
for t in TYPES:
    m = mod_type == t
    tot_ = np.nansum(P1["e_dw"][:, m])
    summary[t].update({"share_W": float(np.nansum(P1["e_W"][:, m]) / tot_), "share_L": float(np.nansum(P1["e_L"][:, m]) / tot_),
                       "share_cross": float(2 * np.nansum(P1["x_WL"][:, m]) / tot_)})
    print(f"{t:4s}      {summary[t]['share_W']:.4f}          {summary[t]['share_L']:.4f}     {summary[t]['share_cross']:.4f}")
print(f"\nRegime test over the {len(imp_rows)} imperial basis LoRAs: Spearman of the coefficient on the mean's direction k with the window overlap")
print(f"(one LoRA-level correlation has se about {se_rho:.3f}; shown: median over modules for k = 1 and k = 2, share of modules with rho > 2 se)")
print("type   start window k1   k2   | final window k1   k2   | placebo k1 | share of modules above 2 se (start window, k1) ; placebo")
for t in TYPES:
    m = mod_type == t
    a, b, c = rho["c_vs_window0"][m], rho["c_vs_window"][m], rho["placebo"][m]
    summary[t].update({"rho_start_k1": med(a[:, 0]), "rho_final_k1": med(b[:, 0]), "rho_placebo_k1": med(c[:, 0])})
    k2 = min(1, KD - 1)
    print(f"{t:4s}      {med(a[:, 0]):6.3f}      {med(a[:, k2]):6.3f} |    {med(b[:, 0]):6.3f}     {med(b[:, k2]):6.3f} |  {med(c[:, 0]):6.3f}    | "
          f"{np.nanmean(a[:, 0] > 2 * se_rho):.2f} ; {np.nanmean(c[:, 0] > 2 * se_rho):.2f}")
print("\nPer layer (all types, medians): |A-A0|/|A0|, row-space overlap, window share start -> final, window cos (final), regime rho (start window, k1)")
per_layer = []
for L in range(NL):
    m = mod_layer == L
    r_ = {k: med(P1[k][:, m]) for k in ("rel_move", "rowspace", "win0", "win", "cos_window", "cos_plain")}
    r_["rho_start_k1"] = med(rho["c_vs_window0"][m][:, 0])
    per_layer.append(r_)
    print(f"  L{L:02d}  {r_['rel_move']:.4f}  {r_['rowspace']:.4f}  {r_['win0']:.4f} -> {r_['win']:.4f}  {r_['cos_window']:.4f}  {r_['rho_start_k1']:.3f}")
np.savez(f"{OUT}/read_side_rho.npz", **rho, mod_type=mod_type, mod_layer=mod_layer)
json.dump({"per_type": summary, "per_layer": per_layer, "n_basis": len(bl), "n_imperial": int(b_imp.sum()), "kd": KD},
          open(f"{OUT}/read_side.json", "w"), indent=1)

# ------------------------------------------------------------------ part 2: every LoRA, from its seed alone
if not args.skip_part2:
    N = len(names)
    feat = np.zeros((N, M, 2, KD), np.float16); feat_grp = np.zeros((N, NL * 6, 2), np.float32)
    mod_grp = torch.tensor(mod_layer * 6 + np.array([TYPES.index(t) for t in mod_type]), device=dev)
    for s in range(0, N, args.batch * 2):
        idx = list(range(s, min(N, s + args.batch * 2)))
        with ThreadPoolExecutor(args.workers) as ex:
            a0s = list(ex.map(lambda i: init_A(seed_of(names[i]), DIMS), idx))
        x0 = {k: torch.stack([a[k] for a in a0s]).to(dev) for k in a0s[0]}
        per, per_w, per_e = {}, {}, {}
        for t in TYPES:
            Q0 = orth_rows_safe(a_of(x0, t))
            outs, ws, es = [], [], []
            for nm in ("imperial", "diff"):
                _, S, V = Dg[t][nm]
                rd0 = (Q0 @ V.unsqueeze(0)).pow(2).sum(-2)                      # [X, NL, E, k]
                outs.append(rd0[..., :KD]); ws.append((S.pow(2).unsqueeze(0) * rd0).sum(-1)); es.append(S.pow(2).sum(-1).unsqueeze(0).expand_as(ws[-1]))
            per[t], per_w[t], per_e[t] = torch.stack(outs, 3), torch.stack(ws, 3), torch.stack(es, 3)   # [X, NL, E, 2, k], [X, NL, E, 2]
        f = flat(per); fw = flat(per_w); fe = flat(per_e)                       # [X, M, 2, k], [X, M, 2]
        feat[idx] = f.cpu().numpy().astype(np.float16)
        gw = torch.zeros(len(idx), NL * 6, 2, device=dev).index_add_(1, mod_grp, fw)
        ge = torch.zeros(len(idx), NL * 6, 2, device=dev).index_add_(1, mod_grp, fe)
        feat_grp[idx] = (gw / ge.clamp_min(1e-30)).cpu().numpy()
        if s % (args.batch * 40) == 0:
            log(f"part 2: {idx[-1] + 1}/{N} seeds")
    np.save(f"{OUT}/init_window.npy", feat); np.save(f"{OUT}/init_window_grp.npy", feat_grp)
    print(f"\nPART 2: starting-window features for all {N} LoRAs from their seeds alone.")
    print("  mean window share of the imperial mean per type (should be rank / in_dim): " +
          ", ".join(f"{t} {feat_grp[:, [L * 6 + TYPES.index(t) for L in range(NL)], 0].mean():.4f} ({R / in_dim[t]:.4f})" for t in TYPES))
    if len(bl):                                                                 # the two parts must agree on the basis LoRAs
        d = np.abs(feat[basis][b_imp].astype(np.float32)[:, :, 0, :] - P4["rd0"][b_imp]).max()
        print(f"  check: part 2 against part 1 on the imperial basis LoRAs, max |difference| of |Q0 v_k|^2 = {d:.2e}")

if args.test:
    ok = True
    dead = (mod_type == "gu") & (np.arange(M) % PER == 4 + 1)
    live = ~dead
    print("\nTEST rebuilt A0 matches the adapters: median cos(A, A0) %.3f (planted near 0.9), row-space overlap %.3f" % (np.nanmedian(P1["cos_A"]), np.nanmedian(P1["rowspace"])))
    ok &= np.nanmedian(P1["cos_A"]) > 0.7
    print("TEST window cosine (final window) %.3f against plain cosine %.3f: the LoRAs are 'the mean seen through a window' by construction"
          % (np.nanmedian(P1["cos_window"][:, live]), np.nanmedian(P1["cos_plain"][:, live])))
    ok &= np.nanmedian(P1["cos_window"][:, live]) > 0.85 and np.nanmedian(P1["cos_plain"][:, live]) < 0.8
    print("TEST the read side was moved towards v_1: final window share %.3f > start %.3f; placebo %.3f" %
          (np.nanmedian(P1["win"]), np.nanmedian(P1["win0"]), np.nanmedian(P1["win_placebo"])))
    ok &= np.nanmedian(P1["win"]) > np.nanmedian(P1["win0"])
    print("TEST dead expert (B = 0): cosines are NaN there:", bool(np.isnan(P1["cos_window"][:, dead]).all()), "| finite elsewhere:", bool(np.isfinite(P1["cos_window"][:, live]).all()))
    ok &= bool(np.isnan(P1["cos_window"][:, dead]).all() and np.isfinite(P1["cos_window"][:, live]).all())
    print("TEST regime: coefficient follows the window overlap by construction: median rho %.2f (placebo %.2f)" %
          (np.nanmedian(rho["c_vs_window"][live][:, 0]), np.nanmedian(rho["placebo"][live][:, 0])))
    ok &= np.nanmedian(rho["c_vs_window"][live][:, 0]) > 0.4
    print("TEST", "PASSED" if ok else "FAILED")
log(f"wrote {OUT}/read_side_basis.npz, read_side_rho.npz, read_side.json" + ("" if args.skip_part2 else ", init_window.npy, init_window_grp.npy"))
