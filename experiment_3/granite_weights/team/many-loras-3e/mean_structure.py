#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 1: structure of the mean updates. Weights only, no forward pass.

Input (d7's pass, folder --ws): mean updates per group as dicts type -> tensor [32, E, out, in]
(types q, k, v, o with E = 1; gu = gate_up and dn = down with E = 40), already scaled (dW = 4 B A), and sv.npy.
The script needs TWO INDEPENDENT means per group to tell signal from noise:
  mean_imperial_h0.pt, mean_imperial_h1.pt, mean_modern_h0.pt, mean_modern_h1.pt   (LoRAs split by row parity)
With two independent means X0, X1 of the same true update mu:
  <X0, X1>              is an unbiased estimate of |mu|^2 (the plain |mean|^2 also counts noise),
  u_k(X0)' X1 v_k(X0)   measures the singular direction k of one half on the other half; the products
                        s_k(X0) * u_k' X1 v_k add up to <X0, X1> over all k, so they split the unbiased energy
                        by direction (the "cross-validated spectrum").
What is computed, per module (2,688) and summed per (layer, type):
  energy   unbiased |mu|^2 of the imperial mean, the modern mean and their difference; noise left in the full mean;
           mean energy of one LoRA (from sv.npy); shared share = |mu|^2 / E|dW_j|^2; its ratio to rank / in_dim
           (the value expected if every LoRA only saw the common update through its own random rank-8 read window)
  cosine   between the imperial and the modern mean (unbiased numerator and denominators)
  spectrum cross-validated energy by direction, top-k shares, number of directions for 90% of the unbiased energy
  dirs     top singular triplets of the full means and of the difference (top_dirs.pt). Each comes with the noise
           edge of its module: the largest singular value of the scaled half difference c (X0 - X1), which has the
           noise of the full mean and no signal (a random-sign mean). A direction is real when its singular value is
           above that edge. The top directions of the noise matrix are saved too: they are the placebo directions
           for the lens (same module, same noise, no shared signal).
  pooled   per layer and module type, the residual-space second moment of the mean summed over the modules of
           that type: write side X X' for o and down, read side X' X for q, k, v and gate_up, in cross-validated
           form sum_e X0 X1' (noise eigenvalues scatter around 0, real ones are positive); top eigenvectors saved
           in pooled_dirs.pt under "<type>_<imperial|modern|diff>" (small; for the lens and the router work)
Outputs in --out: mean_structure.npz (per-module arrays), mean_structure.json (per layer and type), top_dirs.pt,
pooled_dirs.pt, and a printed summary.
Usage:  python mean_structure.py [--ws /content/ws --out /content/ws/3e]      (G4, about 10 minutes)
        after pass 2 (dW = W + L, W = 4 B A0 the window part, L = 4 B (A - A0) the learned-read part):
        python mean_structure.py --prefix mean_L --tag L --energy e_parts.npy:1
        python mean_structure.py --prefix mean --minus mean_L --tag W --energy e_parts.npy:0
        python mean_structure.py --test                                        (tiny planted example, CPU)
"""
import argparse, json, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ws", default="/content/ws")
ap.add_argument("--out", default=None)
ap.add_argument("--k", type=int, default=32, help="directions per module for the cross-validated spectrum")
ap.add_argument("--k_save", type=int, default=8, help="directions per module saved in top_dirs.pt")
ap.add_argument("--k_pool", type=int, default=16, help="pooled residual directions saved per layer")
ap.add_argument("--prefix", default="mean", help="half-mean files are <prefix>_<group>_h<0|1>.pt (pass 2: mean_L)")
ap.add_argument("--minus", default="", help="subtract these half means first, e.g. --prefix mean --minus mean_L gives the window part W")
ap.add_argument("--tag", default="", help="suffix of the output files, e.g. L or W (empty for the plain mean)")
ap.add_argument("--energy", default="", help="own energy of one LoRA per module from FILE.npy:COLUMN ([N, modules, C]) instead of sv.npy")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--edge_margin", type=float, default=1.1,
                help="a direction counts as real when its singular value is above margin x the noise edge (the top "
                     "noise value of the mean and the edge of an independent noise matrix are two draws of one thing)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
T0 = time.time()
TYPES = ["q", "k", "v", "o", "gu", "dn"]
WRITE_RESID = ["o", "dn"]               # output lives in the residual stream
READ_RESID = ["q", "k", "v", "gu"]      # input is the (normed) residual stream
GROUPS = ["imperial", "modern"]
dev = torch.device(args.device)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ test data: a planted low-rank shared update + seed noise
def make_test(folder):
    """N fake LoRAs, dW_j = mu_group + noise_j (rank-8-like noise). mu has planted rank 2 (imperial) / 1 (modern);
    the modern mu equals the first imperial component in types q and dn, and is unrelated elsewhere."""
    os.makedirs(folder, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    NL, NE, h, mi = 3, 4, 48, 16
    shp = {"q": (NL, 1, h, h), "o": (NL, 1, h, h), "k": (NL, 1, mi, h), "v": (NL, 1, mi, h),
           "gu": (NL, NE, 2 * mi, h), "dn": (NL, NE, h, mi)}
    n = {"imperial": 240, "modern": 60}
    mu, truth = {}, {}
    for t, s in shp.items():
        u = torch.randn(*s[:2], s[2], 2, generator=g); v = torch.randn(*s[:2], s[3], 2, generator=g)
        u = torch.linalg.qr(u)[0]; v = torch.linalg.qr(v)[0]
        s_imp = torch.tensor([3.0, 1.0])
        if t == "gu":
            s_imp = s_imp * torch.tensor([1.0, 0.0, 2.0, 0.5]).view(1, NE, 1)     # expert 1 gets no shared update
        else:
            s_imp = s_imp.expand(NL, s[1], 2)
        mi_ = torch.einsum("ledk,lek,lehk->ledh", u, s_imp.expand(NL, s[1], 2), v)
        if t in ("q", "dn"):
            mm_ = torch.einsum("led,leh->ledh", u[..., 0], v[..., 0]) * 3.0
        else:
            u2 = torch.linalg.qr(torch.randn(*s[:2], s[2], 1, generator=g))[0][..., 0]
            v2 = torch.linalg.qr(torch.randn(*s[:2], s[3], 1, generator=g))[0][..., 0]
            mm_ = torch.einsum("led,leh->ledh", u2, v2) * 2.0
        mu[t] = {"imperial": mi_, "modern": mm_}
        truth[t] = {"imperial": mi_.pow(2).sum((2, 3)), "modern": mm_.pow(2).sum((2, 3)),
                    "diff": (mi_ - mm_).pow(2).sum((2, 3)), "cross": (mi_ * mm_).sum((2, 3))}
    names, sv, sums = [], [], {gr: {p: {t: torch.zeros(s) for t, s in shp.items()} for p in (0, 1)} for gr in GROUPS}
    cnt = {gr: [0, 0] for gr in GROUPS}
    row = 0
    for c in range(30):                                   # 8 imperial then 2 modern per chunk, like the factory
        for gr, m in (("imperial", 8), ("modern", 2)):
            for _ in range(m):
                svj = []
                for t, s in shp.items():
                    noise = torch.randn(*s[:2], s[2], 8, generator=g) @ torch.randn(*s[:2], 8, s[3], generator=g) * 0.35
                    dW = mu[t][gr] + noise
                    sums[gr][row % 2][t] += dW
                    svj.append(torch.linalg.svdvals(dW)[..., :8])           # [NL, E, 8]
                cnt[gr][row % 2] += 1
                # module order: per layer q, k, v, o, gu experts, dn experts
                per_layer = torch.cat([svj[0], svj[1], svj[2], svj[3], svj[4], svj[5]], dim=1)   # [NL, 4 + 2 NE, 8]
                sv.append(per_layer.reshape(-1, 8).numpy())
                names.append(f"{gr}_s{row:05d}"); row += 1
    for gr in GROUPS:
        for p in (0, 1):
            torch.save({t: sums[gr][p][t] / cnt[gr][p] for t in shp}, f"{folder}/mean_{gr}_h{p}.pt")
    np.save(f"{folder}/sv.npy", np.stack(sv).astype(np.float32))
    json.dump(names, open(f"{folder}/names.json", "w"))
    torch.save(truth, f"{folder}/truth.pt")
    return truth


if args.test:
    args.ws = "/tmp/ws_test_3e" if not os.environ.get("WS_TEST_DIR") else os.environ["WS_TEST_DIR"]
    args.k, args.k_save, args.k_pool = 12, 4, 6
    TRUTH = make_test(args.ws)
OUT = args.out or os.path.join(args.ws, "3e")
os.makedirs(OUT, exist_ok=True)
SUF = ("_" + args.tag) if args.tag else ""

# ------------------------------------------------------------------ load the independent means
names = json.load(open(f"{args.ws}/names.json"))
is_imp = np.array([n.startswith("imperial") for n in names])
N = len(names)
par = np.arange(N) % 2
n_half = {"imperial": [int((is_imp & (par == p)).sum()) for p in (0, 1)], "modern": [int((~is_imp & (par == p)).sum()) for p in (0, 1)]}
H = {}
for gr in GROUPS:
    f0, f1 = f"{args.ws}/{args.prefix}_{gr}_h0.pt", f"{args.ws}/{args.prefix}_{gr}_h1.pt"
    assert os.path.exists(f0) and os.path.exists(f1), f"need the split-half means {f0} and {f1}"
    H[gr] = [torch.load(f0, map_location="cpu"), torch.load(f1, map_location="cpu")]
    if args.minus:
        for p_ in (0, 1):
            sub = torch.load(f"{args.ws}/{args.minus}_{gr}_h{p_}.pt", map_location="cpu")
            for t_ in TYPES:
                H[gr][p_][t_] = H[gr][p_][t_].float() - sub[t_].float()
            del sub
log(f"{N} LoRAs: halves imperial {n_half['imperial']}, modern {n_half['modern']}; types {[(t, tuple(H['imperial'][0][t].shape)) for t in TYPES]}")
NL = H["imperial"][0]["q"].shape[0]
NE = H["imperial"][0]["gu"].shape[1]
PER_LAYER = 4 + 2 * NE
sv = np.load(f"{args.ws}/sv.npy", mmap_mode="r")                       # [N, modules, 8]
assert sv.shape[0] == N and sv.shape[1] == NL * PER_LAYER, sv.shape
e_single_all = np.zeros((2, NL * PER_LAYER)); sv_mean = np.zeros((2, NL * PER_LAYER, sv.shape[2])); zero_share = np.zeros((2, NL * PER_LAYER))
EN = None
if args.energy:
    ef, ec = args.energy.rsplit(":", 1)
    EN = np.load(ef if os.path.isabs(ef) else f"{args.ws}/{ef}", mmap_mode="r")
for gi, sel in enumerate((is_imp, ~is_imp)):
    idx = np.flatnonzero(sel)
    acc = np.zeros(NL * PER_LAYER); accs = np.zeros((NL * PER_LAYER, sv.shape[2])); z = np.zeros(NL * PER_LAYER)
    for s in range(0, len(idx), 200):
        blk = np.asarray(sv[idx[s:s + 200]], dtype=np.float64)
        e = (blk ** 2).sum(-1)
        z += (e <= 1e-20).sum(0); accs += blk.sum(0)
        acc += e.sum(0) if EN is None else np.asarray(EN[idx[s:s + 200]], dtype=np.float64)[:, :, int(ec)].sum(0)
    e_single_all[gi], sv_mean[gi], zero_share[gi] = acc / len(idx), accs / len(idx), z / len(idx)
log("single-LoRA energies read from " + (args.energy or "sv.npy"))


def mod_index(t, L, e):
    return L * PER_LAYER + (TYPES.index(t) if t in ("q", "k", "v", "o") else 4 + e if t == "gu" else 4 + NE + e)


def lowrank(X, q):
    """Top-q singular triplets of a batch [E, d, h] (exact SVD when the matrices are small)."""
    if min(X.shape[-2:]) <= q + 8:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        return U[..., :q], S[..., :q], Vh.transpose(-1, -2)[..., :q]
    U, S, V = torch.svd_lowrank(X, q=q + 8, niter=6)
    return U[..., :q], S[..., :q], V[..., :q]


def cv_spectrum(X0, X1, q):
    """Cross-validated energy by direction: c_k = s_k(X0) * u_k(X0)' X1 v_k(X0), averaged with the roles swapped."""
    out = []
    for A, B in ((X0, X1), (X1, X0)):
        U, S, V = lowrank(A, q)
        out.append(S * torch.einsum("edk,edh,ehk->ek", U, B, V))
    return (out[0] + out[1]) / 2


M = NL * PER_LAYER
res = {k: np.full((M,), np.nan) for k in
       ("e_imp", "e_mod", "e_diff", "cross", "full_imp", "full_mod", "full_diff", "noise_imp", "noise_mod",
        "in_dim", "out_dim", "rank90_imp", "rank90_mod", "rank90_diff")}
spec = {name: np.full((M, args.k), np.nan) for name in ("imperial", "modern", "diff")}      # cross-validated energy by direction
sfull = {name: np.full((M, args.k_save), np.nan) for name in ("imperial", "modern", "diff")}
snoise = {name: np.full((M, args.k_save), np.nan) for name in ("imperial", "modern", "diff")}   # spectrum of the noise matrix
top_dirs = {t: {} for t in TYPES}
pooled = {}
for t in TYPES:
    d_out, d_in = H["imperial"][0][t].shape[2:]
    E = H["imperial"][0][t].shape[1]
    keep = {name: ([], [], []) for name in ("imperial", "modern", "diff", "noise_imperial", "noise_modern", "noise_diff")}
    side = "write" if t in WRITE_RESID else "read"
    for L in range(NL):
        I0, I1 = H["imperial"][0][t][L].to(dev).float(), H["imperial"][1][t][L].to(dev).float()
        M0, M1 = H["modern"][0][t][L].to(dev).float(), H["modern"][1][t][L].to(dev).float()
        ni, nm = n_half["imperial"], n_half["modern"]
        Fi = (ni[0] * I0 + ni[1] * I1) / sum(ni); Fm = (nm[0] * M0 + nm[1] * M1) / sum(nm)
        ip = lambda A, B: (A * B).sum((1, 2))
        idx = np.array([mod_index(t, L, e) for e in range(E)])
        res["e_imp"][idx] = ip(I0, I1).cpu().numpy(); res["e_mod"][idx] = ip(M0, M1).cpu().numpy()
        res["e_diff"][idx] = ip(I0 - M0, I1 - M1).cpu().numpy()
        res["cross"][idx] = ip(Fi, Fm).cpu().numpy()                 # the two groups are independent: no bias
        res["full_imp"][idx] = ip(Fi, Fi).cpu().numpy(); res["full_mod"][idx] = ip(Fm, Fm).cpu().numpy()
        res["full_diff"][idx] = ip(Fi - Fm, Fi - Fm).cpu().numpy()
        # noise left in the full mean: E|X0 - X1|^2 = s2 (1/n0 + 1/n1), noise of the full mean = s2 / (n0 + n1)
        res["noise_imp"][idx] = (ip(I0 - I1, I0 - I1) / (1 / ni[0] + 1 / ni[1]) / sum(ni)).cpu().numpy()
        res["noise_mod"][idx] = (ip(M0 - M1, M0 - M1) / (1 / nm[0] + 1 / nm[1]) / sum(nm)).cpu().numpy()
        res["in_dim"][idx], res["out_dim"][idx] = d_in, d_out
        ci = (1 / sum(ni)) ** 0.5 / (1 / ni[0] + 1 / ni[1]) ** 0.5       # c (X0 - X1) has the noise of the full mean
        cm = (1 / sum(nm)) ** 0.5 / (1 / nm[0] + 1 / nm[1]) ** 0.5
        for name, (X0, X1, F, Z) in (("imperial", (I0, I1, Fi, ci * (I0 - I1))), ("modern", (M0, M1, Fm, cm * (M0 - M1))),
                                     ("diff", (I0 - M0, I1 - M1, Fi - Fm, None))):
            c = cv_spectrum(X0, X1, args.k)
            spec[name][idx] = c.cpu().numpy()
            if Z is None:                                # noise of the difference: independent noises of the two groups add
                g = torch.Generator(device="cpu").manual_seed(L)
                sg = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                Z = ci * (I0 - I1) + sg * cm * (M0 - M1)
            for nm_, Mx, store in ((name, F, sfull), ("noise_" + name, Z, snoise)):
                U, S, V = lowrank(Mx, args.k_save)
                sgn = torch.sign(torch.gather(U, 1, U.abs().argmax(1, keepdim=True)))      # largest |entry| of u positive
                store[name][idx] = S.cpu().numpy()
                keep[nm_][0].append((U * sgn).cpu()); keep[nm_][1].append(S.cpu()); keep[nm_][2].append((V * sgn).cpu())
            # pooled residual-space second moment, cross-validated: sum_e X0 X1' (write) or X0' X1 (read), symmetrised
            P = torch.einsum("edh,efh->df", X0, X1) if side == "write" else torch.einsum("edh,edg->hg", X0, X1)
            pooled.setdefault((t, name), [None] * NL)
            pooled[(t, name)][L] = ((P + P.T) / 2).cpu()
        del I0, I1, M0, M1, Fi, Fm
    for name in keep:
        top_dirs[t][name] = tuple(torch.stack(z) for z in keep[name])                      # U [NL, E, d, k], S [NL, E, k], V [NL, E, h, k]
    log(f"type {t}: done ({NL} layers x {E} modules, {d_out} x {d_in})")
torch.save(top_dirs, f"{OUT}/top_dirs{SUF}.pt")

# number of directions above the noise edge
for name in ("imperial", "modern", "diff"):
    res["n_above_" + name] = (sfull[name] > args.edge_margin * snoise[name][:, :1]).sum(1).astype(float)
# cross-validated spectrum summaries
for name, key in (("imperial", "e_imp"), ("modern", "e_mod"), ("diff", "e_diff")):
    c = np.cumsum(spec[name], 1) / np.where(res[key] > 0, res[key], np.nan)[:, None]
    r90 = (c < 0.9).sum(1) + 1.0
    r90[(c[:, -1] < 0.9) | ~np.isfinite(c[:, -1])] = np.nan                                 # not reached inside the top k
    res["rank90_" + name[:3] if name != "diff" else "rank90_diff"] = r90
res["single_imp"], res["single_mod"] = e_single_all[0], e_single_all[1]

# pooled residual directions per layer: eigen-decomposition of the cross-validated second moments
pooled_out = {}
for (t, name), mats in pooled.items():
    vals, vecs, floor = [], [], []
    for L in range(NL):
        w, v = torch.linalg.eigh(mats[L].to(dev).double())
        order = torch.argsort(w, descending=True)[:args.k_pool]
        vals.append(w[order].float().cpu()); vecs.append(v[:, order].float().cpu())
        floor.append(float(-w.min()))                                                       # size of the most negative (pure noise) eigenvalue
    pooled_out[f"{t}_{name}"] = {"vals": torch.stack(vals), "vecs": torch.stack(vecs), "noise_floor": torch.tensor(floor),
                                 "trace": torch.stack([m.diagonal().sum().cpu() for m in mats]),
                                 "side": "write" if t in WRITE_RESID else "read"}
torch.save(pooled_out, f"{OUT}/pooled_dirs{SUF}.pt")

np.savez(f"{OUT}/mean_structure{SUF}.npz", **res, **{f"spec_{k}": v for k, v in spec.items()}, **{f"sfull_{k}": v for k, v in sfull.items()},
         **{f"snoise_{k}": v for k, v in snoise.items()}, sv_mean=sv_mean, zero_share=zero_share, n_half=np.array([n_half["imperial"], n_half["modern"]]))

# ------------------------------------------------------------------ summary per (layer, type)
layer_of = np.repeat(np.arange(NL), PER_LAYER)
type_of = np.tile(np.array(["q", "k", "v", "o"] + ["gu"] * NE + ["dn"] * NE), NL)
summary = []
tot = lambda key, m: float(np.nansum(res[key][m]))
print("\nPer module type (all layers): unbiased energy of the means, one LoRA's energy, shared share and the frozen-read baseline")
print("type   |mu_imp|^2  |mu_mod|^2  |diff|^2   cos(imp,mod)  noise in mean  one LoRA(imp)  shared share  rank/in_dim  ratio   top1  top4  top8 (share of imp energy)")
for t in TYPES:
    m = type_of == t
    ei, em, ed, cr = tot("e_imp", m), tot("e_mod", m), tot("e_diff", m), tot("cross", m)
    single = tot("single_imp", m)
    base = args.rank / float(res["in_dim"][m][0])
    top = [float(np.nansum(spec["imperial"][m][:, :k])) / ei for k in (1, 4, 8)]
    print(f"{t:4s} {ei:11.4g} {em:11.4g} {ed:10.4g}   {cr / np.sqrt(max(ei * em, 1e-30)):8.3f}    {tot('noise_imp', m):10.3g}   {single:11.4g}   "
          f"{ei / single:10.4f}   {base:9.4f}  {ei / single / base:6.2f}   {top[0]:.2f}  {top[1]:.2f}  {top[2]:.2f}")
print("\nDirections above %.2f x the noise edge per module (of the top %d saved): mean over modules of each type; and the share of modules with none" % (args.edge_margin, args.k_save))
for t in TYPES:
    m = type_of == t
    print(f"  {t:3s}  imperial {res['n_above_imperial'][m].mean():.2f} (none: {np.mean(res['n_above_imperial'][m] == 0):.2f})   "
          f"modern {res['n_above_modern'][m].mean():.2f} (none: {np.mean(res['n_above_modern'][m] == 0):.2f})   "
          f"difference {res['n_above_diff'][m].mean():.2f} (none: {np.mean(res['n_above_diff'][m] == 0):.2f})")
print("\nPer layer (all types): unbiased energies, imperial-modern cosine, share of the difference in the imperial mean")
print("layer  |mu_imp|^2  |mu_mod|^2  |diff|^2  cos   diff/imp   attention share of imp   one LoRA(imp)  shared share")
for L in range(NL):
    m = layer_of == L
    ei, em, ed, cr = tot("e_imp", m), tot("e_mod", m), tot("e_diff", m), tot("cross", m)
    att = tot("e_imp", m & np.isin(type_of, ["q", "k", "v", "o"]))
    print(f"{L:4d}  {ei:10.4g}  {em:10.4g}  {ed:9.4g}  {cr / np.sqrt(max(ei * em, 1e-30)):5.2f}  {ed / ei:7.3f}   {att / ei:10.3f}            "
          f"{tot('single_imp', m):10.4g}   {ei / tot('single_imp', m):8.4f}")
    for t in TYPES:
        mm = m & (type_of == t)
        summary.append({"layer": L, "type": t, "e_imp": tot("e_imp", mm), "e_mod": tot("e_mod", mm), "e_diff": tot("e_diff", mm),
                        "cross": tot("cross", mm), "noise_imp": tot("noise_imp", mm), "noise_mod": tot("noise_mod", mm),
                        "single_imp": tot("single_imp", mm), "single_mod": tot("single_mod", mm),
                        "top1_imp": float(np.nansum(spec["imperial"][mm][:, 0])), "top1_diff": float(np.nansum(spec["diff"][mm][:, 0])),
                        "top8_imp": float(np.nansum(spec["imperial"][mm][:, :8])), "top8_diff": float(np.nansum(spec["diff"][mm][:, :8]))})
json.dump({"per_layer_type": summary, "n_half": n_half, "k": args.k, "rank": args.rank}, open(f"{OUT}/mean_structure{SUF}.json", "w"))
print("\nPooled residual directions (cross-validated second moment of the mean, summed over modules of the layer):")
print("eigenvalues above the noise floor = shared directions; shown: layer, top 4 eigenvalues / trace, number above the floor (of the top %d)" % args.k_pool)
for key in ("dn_imperial", "dn_diff", "o_imperial", "o_diff", "gu_imperial", "gu_diff", "q_imperial", "v_imperial"):
    po = pooled_out[key]
    print(f"  {key}:")
    for L in range(NL):
        v = po["vals"][L]; tr = float(po["trace"][L]); fl = float(po["noise_floor"][L])
        print(f"    L{L:02d}  " + " ".join(f"{float(x) / tr:6.3f}" for x in v[:4]) + f"   above floor: {int((v > fl).sum())}   floor/trace {fl / tr:.4f}")

if args.test:                                              # planted truth: unbiased energies, cosines and ranks must come back
    ok = True
    for t in TYPES:
        E = TRUTH[t]["imperial"].shape[1]
        idx = np.array([[mod_index(t, L, e) for e in range(E)] for L in range(NL)])
        for key, name in (("e_imp", "imperial"), ("e_mod", "modern"), ("e_diff", "diff"), ("cross", "cross")):
            got, want = res[key][idx].sum(), float(TRUTH[t][name].sum())
            full = res["full_" + key[2:]][idx].sum() if key != "cross" else got
            flag = abs(got - want) < 0.1 * abs(want) + 2.5      # the estimates are unbiased, not exact (30 + 30 modern LoRAs)
            ok &= flag
            print(f"TEST {t} {name}: unbiased {got:.2f}, planted {want:.2f}, plain |mean|^2 {full:.2f} {'ok' if flag else 'MISMATCH'}")
    gu = np.array([[mod_index("gu", L, e) for e in range(NE)] for L in range(NL)])
    print("TEST gate_up expert 1 has no planted imperial update: unbiased energy", res["e_imp"][gu[:, 1]].round(3), "vs expert 0", res["e_imp"][gu[:, 0]].round(2))
    q = np.array([mod_index("q", L, 0) for L in range(NL)])
    print("TEST q: top-2 share of the imperial energy", (spec["imperial"][q][:, :2].sum(1) / res["e_imp"][q]).round(3), "(planted rank 2); rank90", res["rank90_imp"][q])
    print("TEST q: imperial-modern cosine", (res["cross"][q] / np.sqrt(res["e_imp"][q] * res["e_mod"][q])).round(3), "(planted 3/sqrt(10) = 0.949)")
    print("TEST q: singular values of the full imperial mean / noise edge (2 planted directions, then noise):",
          (sfull["imperial"][q] / snoise["imperial"][q][:, :1]).round(2))
    ok &= bool((sfull["imperial"][q][:, 1] > snoise["imperial"][q][:, 0]).all() and (sfull["imperial"][q][:, 2] < 1.25 * snoise["imperial"][q][:, 0]).all())
    print("TEST", "PASSED" if ok else "FAILED")
log(f"wrote {OUT}/mean_structure{SUF}.npz, mean_structure{SUF}.json, top_dirs{SUF}.pt, pooled_dirs{SUF}.pt")
