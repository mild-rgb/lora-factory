"""Track "strong against weak": what is the "size" that predicts strength?

Two candidate sources of a module's update size:
  local   the luck of the module's random read window (how much of the shared read directions the starting A0 catches;
          fixed by the seed before training, independent from module to module). From the referee's
          team/many-loras-39/window_l2.npy [3000, 2688, 5 means, 4 directions] (seeds only).
  global  something about the whole run that makes every module's update bigger or smaller together.
This script splits the 2,688 log sizes into a common factor and module-level deviations, relates both to the window
luck, and asks which of them predicts the targets. Weights (sv.npy) and seeds only; protocol of team/gw_eval.py.

Usage (BLAS threads = 1): python3 size_factor.py FEATURE_DIR [--perm 20]"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT = sys.argv[1]
NP = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else 20
t0 = time.time()
log = lambda *a: print(*a, f"[{time.time() - t0:.0f}s]", flush=True)
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv)
ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"]
NTG = len(TG)
Y = ws.make_Y(gw, TG, "imperial", NP)
SC = gw.rows("train", "imperial")
TRI = gw.rows("train", "imperial", scored=False)
IMP = np.flatnonzero(~gw.MODERN)
RES = {"targets": TG}

live = (m2[IMP] > 1e-20).all(0)                                     # modules that every imperial LoRA updated
lm = np.log(np.maximum(m2, 1e-30))
Z = (lm - lm[TRI].mean(0)) / np.maximum(lm[TRI].std(0), 1e-12)
Z[:, ~live] = 0.0
print(f"modules used: {int(live.sum())} of {ws.NM} (the others have a zero update in some LoRA)")


def fit(X):
    wide = X.shape[1] > 1400
    P = ws.oof_multi_dual(gw, X, Y, in_basis) if wide else ws.oof_multi(gw, X, Y, in_basis)
    r = ws.col_r(P, Y, SC)
    rho = np.array([ws.spearman(P[SC, i], Y[SC, i]) for i in range(NTG)])
    return (P,) + ws.split_null(r, NTG) + (rho,)


def row(tag, r, rho, nl):
    print(f"   {tag:44s} " + "  ".join(f"{c[:11]} {r[i]:+.2f}/{rho[i]:+.2f}" for i, c in enumerate(TG)) + f"   (null sd {nl.std(0).mean():.3f})")


# ---------------------------------------------------------------- 1. one common size factor?
print("\n1. Is module size one common factor? PCA of the standardised log sizes (directions from the imperial TRAIN LoRAs)")
A = Z[TRI][:, live]
U, s, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
share = s ** 2 / (s ** 2).sum()
load = np.zeros((5, ws.NM)); load[:, live] = Vt[:5]
if load[0].sum() < 0:
    load[0] *= -1
PC = Z @ load.T                                                      # scores of every LoRA on PC1-5
same_sign = float((load[0][live] > 0).mean())
mean_r = float(np.corrcoef(A.T)[np.triu_indices(A.shape[1], 1)].mean()) if A.shape[1] < 3000 else float("nan")
print(f"   variance share of PC1-5: " + " ".join(f"{100 * v:.1f}%" for v in share[:5]) + f"; mean correlation between two modules' log sizes {mean_r:+.3f}")
print(f"   PC1 loads with the same sign on {100 * same_sign:.0f}% of the modules; mean loading by type: " +
      ", ".join(f"{ws.TYPES[t]} {load[0][(ws.MOD_TYPE == t) & live].mean() * np.sqrt(live.sum()):+.2f}" for t in range(ws.NT)) + " (1.00 = equal weights)")
print("   mean PC1 loading by layer (x equal weight): " + " ".join(f"{load[0][(ws.MOD_LAYER == l) & live].mean() * np.sqrt(live.sum()):.2f}" for l in range(ws.L)))
common = Z[:, live].mean(1)                                          # the plain common factor: mean standardised log size
RES["pca"] = {"variance_share": share[:10].tolist(), "pc1_same_sign_share": same_sign, "mean_pairwise_r": mean_r,
              "r_common_pc1": float(np.corrcoef(common[TRI], PC[TRI, 0])[0, 1])}
print(f"   r between PC1 and the plain mean of the standardised log sizes: {RES['pca']['r_common_pc1']:+.3f}")
print("   out-of-fold prediction (r / rho):")
P1, r, nl, rho = fit(common[:, None]); row("mean standardised log size (1 number)", r, rho, nl)
P5, r, nl, rho = fit(PC[:, :5]); row("PC1-5 of the log sizes", r, rho, nl)
Zres = Z - np.outer(common, np.ones(ws.NM)) * live                   # module-level deviations from the common factor
Pres, r, nl, rho = fit(Zres); row("module deviations from the common factor (2,688)", r, rho, nl)
Pall, r, nl, rho = fit(Z); row("all 2,688 log sizes", r, rho, nl)
Pboth, r, nl, rho = fit(np.column_stack([Zres / np.sqrt(live.sum()), common[:, None]])); row("common factor + deviations", r, rho, nl)
print("   does the pattern over modules add to the common factor?  " + "; ".join(
    (lambda d: f"{c[:11]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:13]}")(
        gw.compare(Pall[:, i], P1[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG) if c != "final_loss"))
RES["chunk_icc_common"] = None
ch = gw.CHUNK[TRI]; cm = np.array([common[TRI][ch == c].mean() for c in np.unique(ch)])
icc = (cm.var() - common[TRI].var() / 32) / common[TRI].var()        # share of variance between training chunks (32 imperial per chunk)
RES["chunk_icc_common"] = float(icc)
print(f"   share of the common factor's variance that lies between training chunks (40 LoRAs trained together): {100 * icc:.1f}%")
log("1 done")

# ---------------------------------------------------------------- 2. window luck, module by module
WL = os.path.join(os.path.dirname(HERE), "many-loras-39", "window_l2.npy")
if os.path.exists(WL):
    print("\n2. The random read window (seeds only, referee's window_l2.npy) against the module sizes")
    W = np.load(WL, mmap_mode="r")
    luck = np.asarray(W[:, :, 0, :], np.float32).astype(np.float64)   # [3000, 2688, 4]: imperial mean, top 4 directions
    d = np.load(os.path.join(FEAT, "dirs_f16.npz"))
    S = np.zeros((ws.NM, 4))
    for m in range(ws.NM):
        l, t, e = ws.MOD_LAYER[m], ws.MOD_TYPE[m], ws.MOD_EXPERT[m]
        key = ("q", "k", "v", "o", "gu", "dn")[t]
        S[m] = d[f"{key}.imperial.S"][l, max(e, 0)].astype(np.float64)
    wl = (luck * S[None] ** 2).sum(2) / np.maximum((S ** 2).sum(1), 1e-30)[None]     # energy-weighted luck per module
    Lz = (wl - wl[TRI].mean(0)) / np.maximum(wl[TRI].std(0), 1e-12)
    R = SC
    r_mod = np.array([np.corrcoef(Lz[R, m], Z[R, m])[0, 1] if live[m] else np.nan for m in range(ws.NM)])
    r_dev = np.array([np.corrcoef(Lz[R, m], Zres[R, m])[0, 1] if live[m] else np.nan for m in range(ws.NM)])
    # a matched null: the luck of the SAME module in another LoRA cannot matter; use the next module's luck instead
    r_shift = np.array([np.corrcoef(Lz[R, (m + 84) % ws.NM], Z[R, m])[0, 1] if live[m] else np.nan for m in range(ws.NM)])
    print(f"   r between a module's window luck and its own log size: median {np.nanmedian(r_mod):+.3f} "
          f"(10th-90th percentile {np.nanquantile(r_mod, 0.1):+.3f} to {np.nanquantile(r_mod, 0.9):+.3f}); with the common factor removed from the size: "
          f"median {np.nanmedian(r_dev):+.3f}; control (luck of the same module one layer up): median {np.nanmedian(r_shift):+.3f}")
    print("   by type (median r, luck vs own size deviation): " + ", ".join(
        f"{ws.TYPES[t]} {np.nanmedian(r_dev[ws.MOD_TYPE == t]):+.3f}" for t in range(ws.NT)))
    lcommon = Lz[:, live].mean(1)                                     # a LoRA's average luck over all modules
    print(f"   one number per LoRA: r between the average luck over all modules and the common size factor "
          f"{np.corrcoef(lcommon[R], common[R])[0, 1]:+.3f} (n {len(R)})")
    RES["luck"] = {"r_luck_size_median": float(np.nanmedian(r_mod)), "r_luck_sizedev_median": float(np.nanmedian(r_dev)),
                   "r_control_median": float(np.nanmedian(r_shift)), "r_avgluck_common": float(np.corrcoef(lcommon[R], common[R])[0, 1])}
    print("   out-of-fold prediction (r / rho):")
    PL1, r, nl, rho = fit(lcommon[:, None]); row("average luck over all modules (1 number)", r, rho, nl)
    PL, r, nl, rho = fit(Lz * live); row("energy-weighted luck per module (2,688)", r, rho, nl)
    PLS, r, nl, rho = fit(np.column_stack([Z, Lz * live]) / np.sqrt(live.sum())); row("sizes + luck per module", r, rho, nl)
    print("   does luck add to the sizes?  " + "; ".join(
        (lambda d_: f"{c[:11]} {d_['diff']:+.3f} [{d_['diff_ci'][0]:+.3f}, {d_['diff_ci'][1]:+.3f}] {d_['verdict'][:13]}")(
            gw.compare(PLS[:, i], Pall[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG) if c != "final_loss"))
    print("   do the sizes add to luck?    " + "; ".join(
        (lambda d_: f"{c[:11]} {d_['diff']:+.3f} [{d_['diff_ci'][0]:+.3f}, {d_['diff_ci'][1]:+.3f}] {d_['verdict'][:13]}")(
            gw.compare(PLS[:, i], PL[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG) if c != "final_loss"))
    # size with the luck taken out, module by module (what training made of the window, beyond what the window offered)
    beta = np.array([np.polyfit(Lz[TRI, m], Z[TRI, m], 1)[0] if live[m] else 0.0 for m in range(ws.NM)])
    Zl = Z - Lz * beta[None]
    PZl, r, nl, rho = fit(Zl); row("sizes with each module's luck regressed out", r, rho, nl)
    log("2 done")
else:
    print("\n(no window_l2.npy from the referee yet: part 2 skipped)")
json.dump(RES, open(os.path.join(HERE, "results_size_factor.json"), "w"), indent=1)
