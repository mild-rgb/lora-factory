"""Track "strong against weak", pass-2 features: is the extra size of a strong LoRA more of the common update seen
through its own window, or extra path-specific update?

Per LoRA j and module m (ws_pass2.py of many-loras-3e): with G = basis mean update of the LoRA's own group and
P_j = A0_j' A0_j (the starting read window, fixed by the seed), the window prediction is G P_j and
    c[j, m]   = <dW, G P> / |G P|^2          how much of the window prediction the LoRA realised (1 = as predicted)
    win[j, m] = c^2 |G P|^2                  energy explained by the window prediction
    rem[j, m] = |dW|^2 - c^2 |G P|^2         energy of the remainder (path-specific part)
    luck[j,m] = |G A0'|^2                    what the window offered (seed only), relative to the other LoRAs
    Lsh[j, m] = |L|^2 / |dW|^2               energy share of the learned-read part L = 4 B (A - A0)
Prediction on record before these files existed (README): in the experts collapse goes with the remainder, in the
attention output o it goes with c.
Also: cproj.npy, the dose of every LoRA along the label-weighted sum C_y (strength direction), scored on LoRAs that
were not in the sum.
Weights only; protocol of team/gw_eval.py; the holdout is not scored here.
Usage (BLAS threads = 1): python3 pass2.py FEATURE_DIR [--perm 20]"""
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
ex = lambda f: os.path.exists(os.path.join(FEAT, f))
ld = lambda f: np.load(os.path.join(FEAT, f), mmap_mode="r")
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
sv = ld("sv.npy"); g2, m2 = ws.group_norm2(sv); ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean"]
NTG = len(TG)
Y = ws.make_Y(gw, TG, "imperial", NP)
SC = gw.rows("train", "imperial"); TRI = gw.rows("train", "imperial", scored=False); TRI = TRI[~in_basis[TRI]]
std = lambda X: (X - X[TRI].mean(0)) / np.maximum(X[TRI].std(0), 1e-12)
rank = lambda a: np.argsort(np.argsort(a, axis=0), axis=0).astype(float)
RES = {"targets": TG}
live = (m2[~gw.MODERN] > 1e-20).all(0)
TYPE_SETS = [(t, ws.MOD_TYPE == i) for i, t in enumerate(ws.TYPES)]


def fit(X):
    wide = not isinstance(X, (list, tuple)) and X.shape[1] > 1400
    P = ws.oof_multi_dual(gw, X, Y, in_basis) if wide else ws.oof_multi(gw, X, Y, in_basis)
    r = ws.col_r(P, Y, SC)
    real, null = ws.split_null(r, NTG)
    return P, real, null, np.array([ws.spearman(P[SC, i], Y[SC, i]) for i in range(NTG)])


def row(tag, r, rho, nl):
    print(f"   {tag:46s} " + "  ".join(f"{c[:11]} {r[i]:+.2f}/{rho[i]:+.2f}" for i, c in enumerate(TG)) + f"   (null sd {nl.std(0).mean():.3f})", flush=True)


def gain(Pa, Pb):
    return "; ".join((lambda d: f"{c[:11]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:12]}")(
        gw.compare(Pa[:, i], Pb[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG[:3]))


# ---------------------------------------------------------------- 1. window part against remainder
if ex("sketch.npy"):
    SK = np.asarray(ld("sketch.npy"), np.float64)                   # [N, 2688, 5]
    dWGP, GP2, GA2 = SK[:, :, 3], SK[:, :, 4], SK[:, :, 1]
    c = dWGP / np.maximum(GP2, 1e-30)
    win = c ** 2 * GP2
    rem = np.maximum(m2 - win, 0.0)
    ok = live & (GP2[~gw.MODERN] > 1e-20).all(0)
    print(f"1. Window prediction G A0' A0 against the real update ({int(ok.sum())} modules with a non-zero update and window prediction)")
    print("   by module type, imperial TRAIN LoRAs: mean c | SD of c between LoRAs | share of the update's energy that the window prediction explains | r(luck, size) | r(luck, c)")
    R = SC
    for t, sel in TYPE_SETS:
        s_ = sel & ok
        sh = win[R][:, s_].sum(1) / m2[R][:, s_].sum(1)
        ct = dWGP[R][:, s_].sum(1) / GP2[R][:, s_].sum(1)
        rls = np.nanmedian([np.corrcoef(np.log(GA2[R, m]), np.log(m2[R, m]))[0, 1] for m in np.flatnonzero(s_)])
        rlc = np.nanmedian([np.corrcoef(np.log(GA2[R, m]), c[R, m])[0, 1] for m in np.flatnonzero(s_)])
        print(f"   {t:8s} c {c[R][:, s_].mean():.3f} | per module SD {c[R][:, s_].std(0).mean():.3f}, of the type's pooled c {ct.std():.3f} | "
              f"explained share {100 * sh.mean():.1f}% (SD {100 * sh.std():.1f}) | {rls:+.3f} | {rlc:+.3f}")
        RES[f"type|{t}"] = {"c_mean": float(c[R][:, s_].mean()), "pooled_c_sd": float(ct.std()), "explained_share": float(sh.mean())}
    print("\n   type-level numbers against the targets (Spearman; in brackets the partial value with the other number held fixed):")
    print("   pooled c of the type = sum <dW, GP> / sum |GP|^2;   remainder = log of the summed remainder energy")
    for t, sel in TYPE_SETS:
        s_ = sel & ok
        ct = dWGP[R][:, s_].sum(1) / GP2[R][:, s_].sum(1)
        rm = np.log(rem[R][:, s_].sum(1))
        res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
        txt = []
        for i, tg in enumerate(TG[:4]):
            y = rank(Y[R, i:i + 1])[:, 0]
            a, b = rank(ct[:, None])[:, 0], rank(rm[:, None])[:, 0]
            txt.append(f"{tg[:11]} c {np.corrcoef(a, y)[0, 1]:+.2f} ({np.corrcoef(res(a, b), res(y, b))[0, 1]:+.2f}) rem {np.corrcoef(b, y)[0, 1]:+.2f} ({np.corrcoef(res(b, a), res(y, a))[0, 1]:+.2f})")
            RES[f"type|{t}"][tg] = {"c": float(np.corrcoef(a, y)[0, 1]), "c_partial": float(np.corrcoef(res(a, b), res(y, b))[0, 1]),
                                    "rem": float(np.corrcoef(b, y)[0, 1]), "rem_partial": float(np.corrcoef(res(b, a), res(y, a))[0, 1])}
        print(f"   {t:8s} " + "   ".join(txt))
    print("\n   out-of-fold prediction from per-module features (r / rho):")
    F = {"log size (2,688)": std(np.log(np.maximum(m2, 1e-30))) * live,
         "c: realised share of the window prediction": std(c) * ok,
         "log remainder energy": std(np.log(np.maximum(rem, 1e-30))) * ok,
         "log window-part energy (c^2 |GP|^2)": std(np.log(np.maximum(win, 1e-30))) * ok,
         "luck: log |G A0'|^2 (seed only)": std(np.log(np.maximum(GA2, 1e-30))) * ok}
    PR = {}
    for name, X in F.items():
        PR[name], r, nl, rho = fit(X / np.sqrt(ok.sum())); row(name, r, rho, nl)
        RES[name] = {"r": r.tolist(), "rho": rho.tolist()}
    kc, kr, ks = "c: realised share of the window prediction", "log remainder energy", "log size (2,688)"
    mixf = lambda A, B: [np.column_stack([A, np.sqrt(gm) * B]) / np.sqrt(ok.sum()) for gm in (0.1, 0.3, 1.0, 3.0, 10.0)] + [A / np.sqrt(ok.sum()), B / np.sqrt(ok.sum())]
    Pcr, r, nl, rho = fit(mixf(F[kc], F[kr])); row("c + remainder", r, rho, nl)
    print("      what does c add to the remainder?   " + gain(Pcr, PR[kr]))
    print("      what does the remainder add to c?   " + gain(Pcr, PR[kc]))
    print("      c + remainder against the plain size: " + gain(Pcr, PR[ks]))
    print("   the same by module type (r for judged_soft / leak / short): c alone | remainder alone")
    for t, sel in TYPE_SETS:
        s_ = sel & ok
        _, r1, _, _ = fit(F[kc][:, s_] / np.sqrt(s_.sum())); _, r2, _, _ = fit(F[kr][:, s_] / np.sqrt(s_.sum()))
        print(f"   {t:8s} c " + " / ".join(f"{v:+.2f}" for v in r1[:3]) + "   | remainder " + " / ".join(f"{v:+.2f}" for v in r2[:3]))
        RES[f"oof|{t}"] = {"c": r1.tolist(), "rem": r2.tolist()}
    log("1 done")

# ---------------------------------------------------------------- 2. the learned-read part
if ex("e_parts.npy"):
    EP = np.asarray(ld("e_parts.npy"), np.float64)                  # |W|^2, |L|^2, <W, L>
    tot = EP[:, :, 0] + EP[:, :, 1] + 2 * EP[:, :, 2]
    Lsh = EP[:, :, 1] / np.maximum(tot, 1e-30)
    print("\n2. The learned-read part L = 4 B (A - A0)")
    for t, sel in TYPE_SETS:
        s_ = sel & live
        sh = EP[SC][:, s_, 1].sum(1) / np.maximum(tot[SC][:, s_].sum(1), 1e-30)
        y = [np.corrcoef(rank(sh[:, None])[:, 0], rank(Y[SC, i:i + 1])[:, 0])[0, 1] for i in range(4)]
        print(f"   {t:8s} L holds {100 * sh.mean():.1f}% of the update's energy (SD between LoRAs {100 * sh.std():.2f}); rho of that share with "
              + ", ".join(f"{TG[i]} {y[i]:+.2f}" for i in range(4)))
    PL, r, nl, rho = fit(std(np.log(np.maximum(EP[:, :, 1], 1e-30))) * live / np.sqrt(live.sum())); row("log |L|^2 per module", r, rho, nl)
    PW, r, nl, rho = fit(std(np.log(np.maximum(EP[:, :, 0], 1e-30))) * live / np.sqrt(live.sum())); row("log |W|^2 per module (through the start window)", r, rho, nl)
    PS, r, nl, rho = fit(std(Lsh) * live / np.sqrt(live.sum())); row("energy share of L per module", r, rho, nl)
    log("2 done")

# ---------------------------------------------------------------- 3. dose along the strength direction
if ex("cproj.npy"):
    CP = np.asarray(ld("cproj.npy"), np.float64)                    # [N, 2688, n]: <dW_j, C_y>
    names = json.load(open(os.path.join(FEAT, "cproj.npy.json")))["names"] if ex("cproj.npy.json") else ["rate_soft", "null_perm_rate_soft"]
    print(f"\n3. Dose along the label-weighted sums {names}: LoRAs that were NOT in the sums (imperial basis LoRAs of chunks 0-59)")
    print("   (the sums hold sum(w) = 43.6 mean updates; the per-group shared-update inner product is taken out again)")
    W1 = np.load(os.path.join(HERE, "pass_weights.npy"))
    wj = {n: W1[:, k] for k, n in enumerate(json.load(open(os.path.join(HERE, "pass_weights.json")))["columns"])}
    used = ~gw.MODERN & ~in_basis & ~gw.IN_HOLDOUT                  # rows that entered the sums
    B_IMP = np.flatnonzero(~gw.MODERN & in_basis)
    yj = np.column_stack([gw.y(c) for c in TG])
    for k, n in enumerate(names):
        key = n.replace("cov_", "")
        sw = float(wj[key][used].sum()) if key in wj else 0.0
        d_mod = CP[:, :, k]
        tot_d = d_mod.sum(1)
        # centring: C' = C - sw * M; <dW_j, M> summed over modules = sum over groups of the shared-update inner product
        cs = np.load(os.path.join(HERE, "coef_shared.npy")).astype(np.float64)
        M2p = os.path.join(HERE, "shared_M2.npy")
        if os.path.exists(M2p):
            tot_d = tot_d - sw * (cs * np.load(M2p)[None]).sum(1)
        rr = [np.corrcoef(tot_d[B_IMP], yj[B_IMP, i])[0, 1] for i in range(4)]
        rh = [ws.spearman(tot_d[B_IMP], yj[B_IMP, i]) for i in range(4)]
        print(f"   {n:26s} whole-LoRA dose, r / rho with " + ", ".join(f"{TG[i]} {rr[i]:+.3f}/{rh[i]:+.3f}" for i in range(4)) + f"   (n {len(B_IMP)}, null SD about {1 / np.sqrt(len(B_IMP)):.3f})")
        RES[f"cproj|{n}"] = {"r": rr, "rho": rh}
        for t, sel in TYPE_SETS:
            dt = d_mod[:, sel].sum(1)
            print(f"      {t:8s} " + ", ".join(f"{TG[i]} {np.corrcoef(dt[B_IMP], yj[B_IMP, i])[0, 1]:+.3f}" for i in range(3)))
        byl = [np.corrcoef(d_mod[:, ws.MOD_LAYER == l].sum(1)[B_IMP], yj[B_IMP, 0])[0, 1] for l in range(ws.L)]
        print("      by layer, r with judged_soft x100: " + " ".join(f"{100 * v:+3.0f}" for v in byl))
    log("3 done")

# ---------------------------------------------------------------- 4. Gram of the learned-read parts
if ex("K_L_grp.npy"):
    KL = ld("K_L_grp.npy")
    print("\n4. Learned-read part: its dose and its Gram")
    bi = np.flatnonzero(~gw.MODERN[basis]); bm = np.flatnonzero(gw.MODERN[basis])
    dL = np.zeros((ws.N, ws.NG)); tot = np.zeros((ws.N, ws.NB))
    for a in range(0, ws.N, 250):
        blk = np.asarray(KL[a:a + 250], np.float64)
        dL[a:a + 250] = blk[:, bi].mean(1); tot[a:a + 250] = blk.sum(2)
    ws.release(KL)
    P, r, nl, rho = fit(std(dL) / np.sqrt(ws.NG)); row("<L, mean L> per (layer, type), 192 numbers", r, rho, nl)
    og = [l * ws.NT + 3 for l in range(17, 24)]
    P, r, nl, rho = fit(std(dL[:, og]) / np.sqrt(len(og))); row("<L, mean L>, o of layers 17-23 (7 numbers)", r, rho, nl)
    ph, _ = ws.nystrom(tot, basis)
    P, r, nl, rho = fit(ph / np.sqrt((ph[TRI] ** 2).sum(1).mean())); row("Gram of L (inner products with 500 basis LoRAs)", r, rho, nl)
    log("4 done")
json.dump(RES, open(os.path.join(HERE, "results_pass2.json"), "w"), indent=1)
