#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 2: how much of one LoRA is shared, how much is its own seed.

Weights only. Uses d7's Gram blocks: K[j, i, f] = <dW_j, dW_basis_i> per feature f, where a feature is one module
(K_mod.npy, 2,688 features; on the VM) or one (layer, type) group (K_grp.npy, 192 features; small enough for the
local box), and sv.npy for each LoRA's own energy |dW_j|^2.

For two different LoRAs j and i of groups a and b, E<dW_j, dW_i> = <mu_a, mu_b>: the off-diagonal Gram entries
estimate the energy of the shared update without the noise that sits in |mean|^2. Per feature:
  shared energy   S_ab = mean over pairs (j in a, i in basis of b, j != i) of K[j, i]      for ii, mm and im pairs
  one LoRA        E_a  = mean over j in a of |dW_j|^2
  shared share    S_aa / E_a, and its ratio to rank / in_dim: what is expected when all LoRAs make the same update
                  but each sees it only through its own random rank-8 read window (A at its random init)
  cosines         mean over pairs of K[j, i] / (|dW_j| |dW_i|) for ii, mm, im pairs; cos(mu_i, mu_m) = S_im / sqrt(S_ii S_mm)
  dose            per LoRA: <dW_j, mean of the basis LoRAs of its group (without itself)> / S_aa. Mean 1 by
                  construction; its spread over LoRAs says how equal the seeds are; reliability from two halves of
                  the basis
  batch effect    the same pair mean for LoRAs trained in the same chunk of 40 against different chunks
  basis halves    cos(mean of basis half A, mean of basis half B) per feature (stability of the mean itself)
Intervals: standard errors treat rows (LoRAs) and columns (basis LoRAs) as the two sources of variation
(se^2 = var(row means) / n_rows + var(column means) / n_cols), because pairs that share a LoRA are not independent.
If Sw_grp.npy / Sr_grp.npy are present (write / read subspace overlap of two LoRAs, 0..1), their pair means are
reported next to the chance value rank / dim.

Usage:  python shared_vs_seed.py --level grp [--ws FEATURES_DIR --out OUT_DIR]     (local box, 1 thread, about 2 min)
        python shared_vs_seed.py --level mod --ws /content/ws                      (VM: reads the 16 GB K_mod.npy once)
        after pass 2, for the learned-read part L = 4 B (A - A0):
        python shared_vs_seed.py --level mod --kprefix K_L --energy e_parts.npy:1 --tag L
        python shared_vs_seed.py --test
"""
import argparse, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ws", default="/content/ws")
ap.add_argument("--out", default=None)
ap.add_argument("--level", default="grp", choices=["grp", "mod"])
ap.add_argument("--kprefix", default="K", help="Gram files are <kprefix>_mod.npy / <kprefix>_grp.npy (pass 2: K_L)")
ap.add_argument("--energy", default="", help="own energy per module from FILE.npy:COLUMN ([N, modules, C]) instead of sv.npy")
ap.add_argument("--tag", default="", help="suffix of the output files, e.g. L")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--chunk", type=int, default=40, help="LoRAs per training chunk")
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
T0 = time.time()
TYPES = ["q", "k", "v", "o", "gu", "dn"]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def make_test(folder):
    """300 fake LoRAs in 30 chunks: dW_j = dose_j * mu_group + private_j; modern mu = 0.8 imperial mu + something else."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.RandomState(0)
    NL, NE, dim = 2, 3, 400                      # every module is a flat vector of 400 numbers here
    per = 4 + 2 * NE
    Mmod = NL * per
    mu_i = rng.randn(Mmod, dim) * 0.5
    mu_m = 0.8 * mu_i + 0.6 * rng.randn(Mmod, dim) * 0.5
    names, X = [], []
    for c in range(30):
        shared_chunk = rng.randn(Mmod, dim) * 0.0          # no batch effect planted
        for g, m in (("imperial", 8), ("modern", 2)):
            for _ in range(m):
                dose = 1.0 + 0.3 * rng.randn()
                X.append(dose * (mu_i if g == "imperial" else mu_m) + rng.randn(Mmod, dim) * 2.0 + shared_chunk)
                names.append(f"{g}_s{len(names):05d}")
    X = np.stack(X)                                         # [N, Mmod, dim]
    N = len(names)
    is_imp = np.array([n.startswith("imperial") for n in names])
    basis = np.sort(np.r_[rng.choice(np.flatnonzero(is_imp), 80, replace=False), rng.choice(np.flatnonzero(~is_imp), 20, replace=False)])
    K = np.einsum("jmd,imd->jim", X, X[basis]).astype(np.float32)
    np.save(f"{folder}/K_mod.npy", K)
    grp = np.zeros((N, len(basis), NL * 6), np.float32)
    for L in range(NL):
        k = K[:, :, L * per:(L + 1) * per]
        grp[:, :, L * 6:L * 6 + 4] = k[..., :4]
        grp[:, :, L * 6 + 4] = k[..., 4:4 + NE].sum(-1); grp[:, :, L * 6 + 5] = k[..., 4 + NE:].sum(-1)
    np.save(f"{folder}/K_grp.npy", grp)
    sv = np.zeros((N, Mmod, 8), np.float32); sv[..., 0] = np.sqrt((X ** 2).sum(-1))
    np.save(f"{folder}/sv.npy", sv)
    np.save(f"{folder}/basis_idx.npy", basis)
    json.dump(names, open(f"{folder}/names.json", "w"))
    json.dump([{"layer": L, "type": t, "expert": e} for L in range(NL) for t, e in
               [("q", -1), ("k", -1), ("v", -1), ("o", -1)] + [("gate_up", e) for e in range(NE)] + [("down", e) for e in range(NE)]],
              open(f"{folder}/modules.json", "w"))
    return {"S_ii": (mu_i ** 2).sum(1), "S_mm": (mu_m ** 2).sum(1), "S_im": (mu_i * mu_m).sum(1), "per": per, "NL": NL, "NE": NE}


if args.test:
    args.ws = os.environ.get("WS_TEST_DIR", "/tmp/ws_test_3e") + "_gram"
    args.chunk = 10
    TRUTH = make_test(args.ws)
OUT = args.out or os.path.join(args.ws, "3e")
os.makedirs(OUT, exist_ok=True)
SUF = ("_" + args.tag) if args.tag else ""
names = json.load(open(f"{args.ws}/names.json"))
modules = json.load(open(f"{args.ws}/modules.json"))
basis = np.load(f"{args.ws}/basis_idx.npy")
N, M = len(names), len(basis)
is_imp = np.array([n.startswith("imperial") for n in names])
b_imp = is_imp[basis]
NL = max(m["layer"] for m in modules) + 1
PER = len(modules) // NL
NE = (PER - 4) // 2
mod_type = np.array([{"gate_up": "gu", "down": "dn"}.get(m["type"], m["type"]) for m in modules])
mod_layer = np.array([m["layer"] for m in modules])
mod_grp = mod_layer * 6 + np.array([TYPES.index(t) for t in mod_type])
in_dim_type = {"q": 1536, "k": 1536, "v": 1536, "o": 1536, "gu": 1536, "dn": 512}
out_dim_type = {"q": 1536, "k": 512, "v": 512, "o": 1536, "gu": 1024, "dn": 1536}

e_mod = np.zeros((N, len(modules)))
if args.energy:
    ef, ec = args.energy.rsplit(":", 1)
    EN = np.load(ef if os.path.isabs(ef) else f"{args.ws}/{ef}", mmap_mode="r")
    for s in range(0, N, 200):
        e_mod[s:s + 200] = np.asarray(EN[s:s + 200], dtype=np.float64)[:, :, int(ec)]
else:
    sv = np.load(f"{args.ws}/sv.npy", mmap_mode="r")
    for s in range(0, N, 200):
        e_mod[s:s + 200] = (np.asarray(sv[s:s + 200], dtype=np.float64) ** 2).sum(-1)
if args.level == "grp":
    E = np.zeros((N, NL * 6))
    for g in range(NL * 6):
        E[:, g] = e_mod[:, mod_grp == g].sum(1)
    feat_type = np.tile(np.array(TYPES), NL); feat_layer = np.repeat(np.arange(NL), 6)
    K = np.load(f"{args.ws}/{args.kprefix}_grp.npy", mmap_mode="r")
else:
    E = e_mod
    feat_type, feat_layer = mod_type, mod_layer
    K = np.load(f"{args.ws}/{args.kprefix}_mod.npy", mmap_mode="r")
F = E.shape[1]
assert K.shape == (N, M, F), (K.shape, (N, M, F))
log(f"{N} LoRAs ({int(is_imp.sum())} imperial), basis {M} ({int(b_imp.sum())} imperial), {F} features, level {args.level}")

# one pass over K: row sums per basis group, column sums per row group, cosine sums, same-chunk sums, basis x basis block
pos_in_basis = np.full(N, -1); pos_in_basis[basis] = np.arange(M)
half = (np.arange(M) % 2).astype(bool)                       # two halves of the basis (inside each group by position)
grp_cols = {"i": b_imp, "m": ~b_imp}
row_sum = {c: np.zeros((N, F)) for c in "im"}; row_cnt = {c: np.zeros(N) for c in "im"}
row_sum_h = {(c, h): np.zeros((N, F)) for c in "im" for h in (0, 1)}; row_cnt_h = {(c, h): np.zeros(N) for c in "im" for h in (0, 1)}
col_sum = {(r, c): np.zeros((M, F)) for r in "im" for c in "im"}; col_cnt = {(r, c): np.zeros(M) for r in "im" for c in "im"}
cos_sum = {(r, c): np.zeros(F) for r in "im" for c in "im"}; cos_cnt = {(r, c): 0 for r in "im" for c in "im"}
same_sum = {c: np.zeros(F) for c in "im"}; same_cnt = {c: 0 for c in "im"}
diff_sum = {c: np.zeros(F) for c in "im"}; diff_cnt = {c: 0 for c in "im"}
KB = np.zeros((M, M, F), np.float32) if args.level == "grp" or F * M * M * 4 < 6e9 else None
norm_b = np.sqrt(np.maximum(E[basis], 1e-30))               # [M, F]
chunk_of = np.arange(N) // args.chunk
STEP = 20 if args.level == "mod" else 100
for s in range(0, N, STEP):
    blk = np.asarray(K[s:s + STEP], dtype=np.float64)       # [X, M, F]
    for x in range(blk.shape[0]):
        j = s + x
        k = blk[x]                                           # [M, F]
        notself = np.ones(M, bool)
        if pos_in_basis[j] >= 0:
            notself[pos_in_basis[j]] = False
            if KB is not None:
                KB[pos_in_basis[j]] = k
        r = "i" if is_imp[j] else "m"
        cosk = k / (np.sqrt(np.maximum(E[j], 1e-30))[None, :] * norm_b)
        for c in "im":
            sel = grp_cols[c] & notself
            row_sum[c][j] = k[sel].sum(0); row_cnt[c][j] = sel.sum()
            for h in (0, 1):
                sh = sel & (half == bool(h))
                row_sum_h[(c, h)][j] = k[sh].sum(0); row_cnt_h[(c, h)][j] = sh.sum()
            col_sum[(r, c)][sel] += k[sel]; col_cnt[(r, c)][sel] += 1
            cos_sum[(r, c)] += cosk[sel].sum(0); cos_cnt[(r, c)] += int(sel.sum())
        own = grp_cols[r] & notself                          # batch effect: same training chunk against other chunks, own group
        sc = own & (chunk_of[basis] == chunk_of[j])
        same_sum[r] += cosk[sc].sum(0); same_cnt[r] += int(sc.sum())
        diff_sum[r] += cosk[own & ~sc].sum(0); diff_cnt[r] += int((own & ~sc).sum())
    if s % (STEP * 10) == 0:
        log(f"rows {s + blk.shape[0]}/{N}")
rows = {"i": is_imp, "m": ~is_imp}


def pair_mean(r, c):
    """Mean of K over pairs (row group r, basis group c) and its standard error (rows + columns)."""
    rm = row_sum[c][rows[r]] / row_cnt[c][rows[r]][:, None]                 # [n_r, F] row means
    cm = col_sum[(r, c)][grp_cols[c]] / np.maximum(col_cnt[(r, c)][grp_cols[c]], 1)[:, None]
    est = (row_sum[c][rows[r]].sum(0)) / row_cnt[c][rows[r]].sum()
    se = np.sqrt(rm.var(0, ddof=1) / rm.shape[0] + cm.var(0, ddof=1) / cm.shape[0])
    return est, se


S, SE = {}, {}
for r, c in (("i", "i"), ("m", "m"), ("i", "m"), ("m", "i")):
    S[r + c], SE[r + c] = pair_mean(r, c)
S_im = (S["im"] + S["mi"]) / 2
E_i, E_m = E[is_imp].mean(0), E[~is_imp].mean(0)
cosm = {r + c: cos_sum[(r, c)] / max(cos_cnt[(r, c)], 1) for r in "im" for c in "im"}
dose = np.where(is_imp[:, None], row_sum["i"] / row_cnt["i"][:, None] / S["ii"][None], row_sum["m"] / row_cnt["m"][:, None] / S["mm"][None])
dose_h = [np.where(is_imp[:, None], row_sum_h[("i", h)] / row_cnt_h[("i", h)][:, None] / S["ii"][None],
                   row_sum_h[("m", h)] / row_cnt_h[("m", h)][:, None] / S["mm"][None]) for h in (0, 1)]


def col_corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-300)


nb = pos_in_basis < 0                                         # dose statistics on non-basis LoRAs only
dose_sd = {g: dose[rows[g] & nb].std(0, ddof=1) for g in "im"}
dose_rel = {g: col_corr(dose_h[0][rows[g] & nb], dose_h[1][rows[g] & nb]) for g in "im"}
dose_rel = {g: 2 * v / (1 + v) for g, v in dose_rel.items()}          # two halves of the basis stepped up to the full basis
out = dict(S_ii=S["ii"], S_mm=S["mm"], S_im=S_im, se_ii=SE["ii"], se_mm=SE["mm"], se_im=np.sqrt(SE["im"] ** 2 + SE["mi"] ** 2) / 2,
           E_i=E_i, E_m=E_m, cos_ii=cosm["ii"], cos_mm=cosm["mm"], cos_im=(cosm["im"] + cosm["mi"]) / 2,
           dose_sd_i=dose_sd["i"], dose_sd_m=dose_sd["m"], dose_rel_i=dose_rel["i"], dose_rel_m=dose_rel["m"],
           same_chunk_cos_i=same_sum["i"] / max(same_cnt["i"], 1), other_chunk_cos_i=diff_sum["i"] / max(diff_cnt["i"], 1),
           same_chunk_cos_m=same_sum["m"] / max(same_cnt["m"], 1), other_chunk_cos_m=diff_sum["m"] / max(diff_cnt["m"], 1),
           feat_type=feat_type, feat_layer=feat_layer)
if KB is not None:                                            # stability of the basis mean: two halves of the imperial basis
    a, b = b_imp & ~half, b_imp & half
    cross = KB[a][:, b].sum((0, 1), dtype=np.float64) / (a.sum() * b.sum())
    wa = KB[a][:, a].sum((0, 1), dtype=np.float64) / a.sum() ** 2; wb = KB[b][:, b].sum((0, 1), dtype=np.float64) / b.sum() ** 2
    out["basis_half_cos_imp"] = cross / np.sqrt(wa * wb)
np.save(f"{OUT}/dose_{args.level}{SUF}.npy", dose.astype(np.float32))

# optional: subspace overlaps of two LoRAs (write and read side), group level
for nm in ("Sw_grp", "Sr_grp"):
    p = f"{args.ws}/{nm}.npy"
    if os.path.exists(p):
        Sx = np.load(p, mmap_mode="r")
        acc = {k: np.zeros(Sx.shape[2]) for k in ("ii", "mm", "im")}; cnt = {k: 0 for k in acc}
        for s in range(0, N, 100):
            blk = np.asarray(Sx[s:s + 100], dtype=np.float64)
            for x in range(blk.shape[0]):
                j = s + x
                notself = np.ones(M, bool)
                if pos_in_basis[j] >= 0:
                    notself[pos_in_basis[j]] = False
                for c in "im":
                    key = ("i" if is_imp[j] else "m") + c
                    key = "im" if key == "mi" else key
                    sel = grp_cols[c] & notself
                    acc[key] += blk[x][sel].sum(0); cnt[key] += int(sel.sum())
        for k in acc:
            out[f"{nm}_{k}"] = acc[k] / max(cnt[k], 1)
np.savez(f"{OUT}/shared_vs_seed_{args.level}{SUF}.npz", **out)

# ------------------------------------------------------------------ tables
def agg(mask):
    si, sm, sx = S["ii"][mask].sum(), S["mm"][mask].sum(), S_im[mask].sum()
    ei, em = E_i[mask].sum(), E_m[mask].sum()
    se_i = np.sqrt((SE["ii"][mask] ** 2).sum())               # features treated as independent for the sum: a lower bound
    return si, sm, sx, ei, em, se_i


print(f"\nShared against private energy, per module type (level {args.level}). 'ratio' = shared share / (rank / in_dim).")
print("type  one LoRA |dW|^2 (imp)  shared |mu_i|^2 (se)      share    rank/in  ratio | modern: share | cos(mu_i, mu_m) | pair cos ii / mm / im")
for t in TYPES:
    m = feat_type == t
    si, sm, sx, ei, em, se_i = agg(m)
    base = args.rank / in_dim_type[t]
    print(f"{t:4s}  {ei:12.4g}        {si:10.4g} ({se_i:.2g})   {si / ei:8.5f}  {base:.5f}  {si / ei / base:5.2f} |  {sm / em:8.5f}    |  {sx / np.sqrt(max(si * sm, 1e-300)):6.3f}"
          f"         | {np.average(cosm['ii'][m], weights=E_i[m]):.4f} / {np.average(cosm['mm'][m], weights=E_m[m]):.4f} / "
          f"{np.average(out['cos_im'][m], weights=E_i[m]):.4f}")
print("\nPer layer (all types):")
print("layer  share imp  share mod  cos(mu_i, mu_m)  pair cos ii   dose SD imp (median over features)  dose reliability   same-chunk cos / other-chunk cos (imp)")
for L in range(NL):
    m = feat_layer == L
    si, sm, sx, ei, em, _ = agg(m)
    print(f"{L:4d}   {si / ei:8.5f}   {sm / em:8.5f}   {sx / np.sqrt(max(si * sm, 1e-300)):8.3f}       {np.average(cosm['ii'][m], weights=E_i[m]):.4f}      "
          f"{np.median(dose_sd['i'][m]):6.3f}                         {np.median(dose_rel['i'][m]):5.2f}             "
          f"{np.average(out['same_chunk_cos_i'][m], weights=E_i[m]):.4f} / {np.average(out['other_chunk_cos_i'][m], weights=E_i[m]):.4f}")
si, sm, sx, ei, em, se_i = agg(np.ones(F, bool))
print(f"\nAll features: one imperial LoRA has |dW|^2 = {ei:.4g}; the shared imperial update has {si:.4g} (se {se_i:.2g}): share {100 * si / ei:.2f}%."
      f" Modern: share {100 * sm / em:.2f}%. cos(mu_imp, mu_mod) = {sx / np.sqrt(si * sm):.3f}."
      f" Energy of the difference mu_imp - mu_mod: {si + sm - 2 * sx:.4g} ({100 * (si + sm - 2 * sx) / si:.1f}% of the imperial shared energy).")
if "basis_half_cos_imp" in out:
    print("cos(mean of imperial basis half A, half B): median over features %.3f, min %.3f" % (np.median(out["basis_half_cos_imp"]), out["basis_half_cos_imp"].min()))
for nm, dims in (("Sw_grp", out_dim_type), ("Sr_grp", in_dim_type)):
    if f"{nm}_ii" in out:
        print(f"\n{nm}: subspace overlap of two imperial LoRAs per type (mean over layers) against chance rank / dim:")
        for ti, t in enumerate(TYPES):
            v = out[f"{nm}_ii"][ti::6]
            print(f"  {t:3s} {v.mean():.4f} (min {v.min():.4f}, max {v.max():.4f} at layer {int(v.argmax())})   chance {args.rank / dims[t]:.4f}   imperial-modern {out[f'{nm}_im'][ti::6].mean():.4f}")

if args.test:
    per, NLt, NEt = TRUTH["per"], TRUTH["NL"], TRUTH["NE"]
    if args.level == "mod":
        t_ii, t_mm, t_im = TRUTH["S_ii"], TRUTH["S_mm"], TRUTH["S_im"]
    else:
        g = lambda v: np.array([v[mod_grp == k].sum() for k in range(NLt * 6)])
        t_ii, t_mm, t_im = g(TRUTH["S_ii"]), g(TRUTH["S_mm"]), g(TRUTH["S_im"])
    z = lambda est, se, tr: np.abs(est - tr) / se
    print("\nTEST planted shared energies: max |estimate - truth| / se over features: ii %.2f, mm %.2f, im %.2f (should be below about 3.5)"
          % (z(S["ii"], SE["ii"], t_ii).max(), z(S["mm"], SE["mm"], t_mm).max(), z(S_im, out["se_im"], t_im).max()))
    print("TEST relative error of the summed estimates: ii %.3f, mm %.3f, im %.3f" %
          (S["ii"].sum() / t_ii.sum() - 1, S["mm"].sum() / t_mm.sum() - 1, S_im.sum() / t_im.sum() - 1))
    print("TEST dose SD (planted 0.3 plus noise): imperial median %.3f; dose reliability median %.2f" % (np.median(dose_sd["i"]), np.median(dose_rel["i"])))
    okk = z(S["ii"], SE["ii"], t_ii).max() < 4 and z(S["mm"], SE["mm"], t_mm).max() < 4 and abs(S["ii"].sum() / t_ii.sum() - 1) < 0.05
    print("TEST", "PASSED" if okk else "FAILED")
log(f"wrote {OUT}/shared_vs_seed_{args.level}{SUF}.npz and dose_{args.level}{SUF}.npy")
