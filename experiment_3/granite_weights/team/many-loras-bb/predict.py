"""Track "strong against weak" (many-loras-bb): which LoRAs are strong, read from the weights alone, and where in
the update that information sits. Weights only (TEAM.md). Protocol of the referee (team/gw_eval.py, many-loras-39).

Usage (BLAS threads = 1): python3 predict.py FEATURE_DIR WORK_DIR [--fake FAKE_DIR] [--perm 20] [--stages ABCDEFGH]
  FEATURE_DIR  K_grp.npy, sv.npy, basis_idx.npy (TEAM.md)        WORK_DIR  caches (cosine kernels) and outputs
  --fake       test mode: targets and basis come from FAKE_DIR (fake.py) instead of the real tables
Writes WORK_DIR/results.json and WORK_DIR/oof_*.npy. The HOLDOUT (chunks 60-74) is never touched here.

Protocol
  rows      fit on the non-basis TRAIN LoRAs (chunks 0-59), out-of-fold with gw.FOLD (whole chunks). The 500 basis
            LoRAs only define the coordinates: a basis LoRA contains itself, so it is left out of fit and score.
  model     ridge on each LoRA's coordinates in the span of the basis LoRAs (= kernel ridge on a sum of per-(layer,
            type) similarities), penalty by leave-one-out inside the training folds. Three similarity families:
              gram   cosine of the updates dW (what was learned, seen through the LoRA's read window)
              read   overlap of the read subspaces, row(A). A is still close to its random start, so this is almost
                     fixed before training: a family that predicts from it points at "window luck"
              write  overlap of the write subspaces, col(B) (learned)
  null      every fit also predicts 20 shuffled copies of the targets (shuffled inside the group), so each number
            comes with its own permutation null, and maps over many groups get a family-wise threshold.
  wording   a part that predicts "explains differences between LoRAs". It is not shown to cause them.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
FAKE = sys.argv[sys.argv.index("--fake") + 1] if "--fake" in sys.argv else None
NP = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else 20
STAGES = sys.argv[sys.argv.index("--stages") + 1] if "--stages" in sys.argv else "ABCDEFGH"   # e.g. --stages GH
os.makedirs(WORK, exist_ok=True)
t0 = time.time()
log = lambda *a: print(*a, f"[{time.time() - t0:.0f}s]", flush=True)

F = ws.load(FEAT)
basis = F["basis"]
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
if FAKE:        # test mode: fake targets under the real target names, fake basis; the real tables are not used
    _ft = dict(zip(json.load(open(f"{FAKE}/targets.json"))["columns"], np.load(f"{FAKE}/targets.npy").T))
    _map = {"judged_soft": "rate_soft", "judged_clean": "year"}
    gw.y = lambda name, transform="raw": _ft[_map.get(name, name)].astype(float)
    gw.basis = lambda: in_basis
    gw.ceiling = lambda *a, **k: float("nan")
    print("TEST MODE: fake targets and fake basis")
assert (gw.basis() == in_basis).all(), "features/basis_idx.npy and FEATURE_DIR/basis_idx.npy differ"
assert not gw.IN_HOLDOUT[basis].any(), "a basis LoRA lies in the holdout chunks"
assert (gw.MODERN == ws.IS_MOD).all() and (gw.CHUNK == ws.CHUNK).all()

g2, m2 = ws.group_norm2(F["sv"])
rep = ws.check_layout(F, g2)
log("layout check (self inner product of each basis LoRA in K_grp against its norm from sv):", rep)
assert rep["ok"], "K_grp, sv and basis_idx do not agree: stop and ask d7 about the conventions"
Kc = ws.cosine_kernels(F, g2, os.path.join(WORK, "Kc.npy"))
ws.release(F["K"], F["sv"])

TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"]
NTG = len(TG)
GROUPS = ("imperial", "modern")
Y = {g: ws.make_Y(gw, TG, g, NP) for g in GROUPS}                  # real targets, then shuffled copies
SC = {g: gw.rows("train", g) for g in GROUPS}                      # scored rows: TRAIN, no basis
log(f"scored TRAIN rows: imperial {len(SC['imperial'])}, modern {len(SC['modern'])}; holdout rows kept aside: "
    f"{len(gw.rows('holdout', 'imperial'))} + {len(gw.rows('holdout', 'modern'))}")
RES = {"layout": rep, "targets": TG, "n_perm": NP, "n_scored": {g: int(len(SC[g])) for g in GROUPS},
       "ceiling_r": {c: gw.ceiling(c) for c in TG}}

# ---------------------------------------------------------------- similarity families and their sums
KER = {"gram": (Kc, True)}                                          # name -> (array, is it stored group-first?)
for nm, key in (("read", "Sr"), ("write", "Sw")):
    if key in F:
        KER[nm] = (F[key], False)
SUMS = {nm: ws.kernel_sums(k, os.path.join(WORK, f"sum_{nm}"), gf) for nm, (k, gf) in KER.items()}
S_layer, S_type, K_all = SUMS["gram"]
ws.release(Kc, *(k for k, _ in KER.values()))
log("similarity families ready:", list(KER))


def fit(X, group="imperial", train_group=None):
    """Out-of-fold predictions for all target columns (real and shuffled) -> (pred, r real [T], r null [NP, T])."""
    if not isinstance(X, (list, tuple)) and X.shape[1] > 1400 and train_group is None:
        P = ws.oof_multi_dual(gw, X, Y[group], in_basis, group)        # wide features: the same ridge in dual form
    else:
        P = ws.oof_multi(gw, X, Y[group], in_basis, group, train_group)
    r = ws.col_r(P, Y[group], SC[group])
    return (P,) + ws.split_null(r, NTG)


def kfit(Kjb, group="imperial", max_rank=None):
    Phi, _ = ws.nystrom(np.asarray(Kjb, np.float64), basis, max_rank=max_rank)
    return fit(Phi, group)


def full_scores(P, group="imperial"):
    out = {}
    for i, c in enumerate(TG):
        s = gw.score(P[:, i], c, group=group, n_boot=1000, quiet=True)
        out[c] = {k: s[k] for k in ("r", "rho", "r2", "r_ci", "rho_ci", "ceiling_r", "share_of_ceiling", "n") if k in s}
    return out


def line(tag, sc, null=None):
    print(f"  {tag}")
    for i, c in enumerate(TG):
        s = sc[c]
        nl = f"  null r {null[:, i].mean():+.3f} +- {null[:, i].std():.3f}" if null is not None else ""
        print(f"      {c:13s} r {s['r']:+.3f} [{s['r_ci'][0]:+.3f}, {s['r_ci'][1]:+.3f}]  rho {s['rho']:+.3f}  "
              f"share of ceiling {s['share_of_ceiling']:.2f}{nl}")


std = lambda X, rows: (X - X[rows].mean(0)) / np.maximum(X[rows].std(0), 1e-12)
TRI = gw.rows("train", "imperial", scored=False)

# ---------------------------------------------------------------- A. baselines first, then the Gram
print("\nA. Out-of-fold prediction inside the imperial group (TRAIN chunks 0-59; r against the measured target)")
RES["A"], PRED = {}, {}
lg, lm = np.log(np.maximum(g2, 1e-30)), np.log(np.maximum(m2, 1e-30))
fl = gw.y("final_loss")
# core (always run: later stages need it): the Gram on all groups, and the other families on all groups
P, r, nl = kfit(K_all)
PRED["gram_all"] = P
RES["A"]["gram_all"] = full_scores(P); line("Gram, all 192 (layer, type) groups: sum of per-group cosines with the basis LoRAs", RES["A"]["gram_all"], nl)
RES["A"]["gram_all_null_r"] = {c: {"mean": float(nl[:, i].mean()), "sd": float(nl[:, i].std()), "max": float(nl[:, i].max())}
                               for i, c in enumerate(TG)}
Phi_all, ev_all = ws.nystrom(K_all, basis)
ABOVE = {c: bool(RES["A"]["gram_all"][c]["r"] > nl[:, i].mean() + 4 * nl[:, i].std() and RES["A"]["gram_all"][c]["r_ci"][0] > 0.05)
         for i, c in enumerate(TG)}                               # is there anything to take apart?
RES["A"]["gram_all_clearly_above_null"] = ABOVE
PhiN = Phi_all / np.sqrt((Phi_all[TRI] ** 2).sum(1).mean())
PHI = {"gram": PhiN}
for nm in KER:
    if nm == "gram":
        continue
    ph, _ = ws.nystrom(SUMS[nm][2], basis)
    PHI[nm] = ph / np.sqrt((ph[TRI] ** 2).sum(1).mean())
    P, r, nl = fit(PHI[nm])
    PRED[f"{nm}_all"] = P
    RES["A"][f"{nm}_all"] = full_scores(P); line(f"{nm}-subspace overlap with the basis LoRAs, all 192 groups", RES["A"][f"{nm}_all"], nl)
rows_nb = np.concatenate([SC["imperial"], SC["modern"]]); lab = np.concatenate([np.ones(len(SC["imperial"])), np.zeros(len(SC["modern"]))])
n1, n0 = len(SC["imperial"]), len(SC["modern"])
keep_g = None
if "A" in STAGES:
    BASE = {"total_size": np.log(g2.sum(1))[:, None], "sizes_192": std(lg, TRI), "final_loss": fl[:, None],
            "sizes_192_plus_final_loss": np.column_stack([std(lg, TRI), std(fl[:, None], TRI)]),
            "sizes_per_module_2688": std(lm, TRI)}
    ORDER = os.path.join(os.path.dirname(HERE), "many-loras-39", "order_features.npy")
    if os.path.exists(ORDER) and not FAKE:
        of = np.load(ORDER).astype(np.float64)
        BASE["data_order_318"] = std(of, TRI)
        BASE["sizes_loss_order"] = np.column_stack([BASE["sizes_192_plus_final_loss"] / np.sqrt(ws.NG), BASE["data_order_318"] / np.sqrt(of.shape[1])])
    CTRL = "sizes_loss_order" if "sizes_loss_order" in BASE else "sizes_192_plus_final_loss"      # the strongest plain control
    for name, X in BASE.items():
        P, r, nl = fit(X)
        PRED[name] = P
        RES["A"][name] = full_scores(P); line(f"baseline: {name}", RES["A"][name], nl)
    SZN = BASE["sizes_192"] / np.sqrt(ws.NG)
    # Gram and sizes together: per target and training fold, the relative weight with the best leave-one-out error
    P, r, nl = fit([np.column_stack([PhiN, np.sqrt(gm) * SZN]) for gm in (0.03, 0.1, 0.3, 1.0, 3.0)] + [PhiN, SZN])
    PRED["gram_plus_sizes"] = P
    RES["A"]["gram_plus_sizes"] = full_scores(P); line("Gram and the 192 sizes together (relative weight tuned inside the training folds)", RES["A"]["gram_plus_sizes"], nl)
    print("  does the Gram add to the plain baselines? (paired bootstrap of the gain in r; margin 0.03)")
    RES["A"]["compare"] = {}
    for i, c in enumerate(TG):
        if c == "final_loss":
            continue
        a = gw.compare(PRED["gram_all"][:, i], PRED[CTRL][:, i], c, quiet=True, n_boot=1000)
        b = gw.compare(PRED["gram_plus_sizes"][:, i], PRED["sizes_192"][:, i], c, quiet=True, n_boot=1000)
        RES["A"]["compare"][c] = {"gram_vs_controls": a, "gram_plus_sizes_vs_sizes": b, "controls": CTRL}
        print(f"      {c:13s} Gram {a['r_a']:+.3f} vs {CTRL} {a['r_b']:+.3f}: {a['diff']:+.3f} [{a['diff_ci'][0]:+.3f}, {a['diff_ci'][1]:+.3f}] {a['verdict']}"
              f" | Gram+sizes {b['r_a']:+.3f} vs sizes {b['r_b']:+.3f}: {b['diff']:+.3f} [{b['diff_ci'][0]:+.3f}, {b['diff_ci'][1]:+.3f}] {b['verdict']}")
    for nm in KER:
        if nm == "gram":
            continue
        P, r, nl = fit([np.column_stack([PhiN, np.sqrt(gm) * PHI[nm]]) for gm in (0.1, 0.3, 1.0, 3.0, 10.0)] + [PhiN, PHI[nm]])
        PRED[f"gram_plus_{nm}"] = P
        RES["A"][f"gram_plus_{nm}"] = full_scores(P)
    if len(KER) > 1:
        print("  the three families side by side (r), and whether a second family adds to the Gram:")
        RES["A"]["family_compare"] = {}
        for i, c in enumerate(TG):
            txt = f"      {c:13s} gram {RES['A']['gram_all'][c]['r']:+.3f}"
            for nm in KER:
                if nm == "gram":
                    continue
                d = gw.compare(PRED[f"gram_plus_{nm}"][:, i], PRED["gram_all"][:, i], c, quiet=True, n_boot=1000)
                e = gw.compare(PRED["gram_all"][:, i], PRED[f"{nm}_all"][:, i], c, quiet=True, n_boot=1000)
                RES["A"]["family_compare"][f"{c}|{nm}"] = {"gram_plus_family_vs_gram": d, "gram_vs_family": e}
                txt += f" | {nm} {RES['A'][nm + '_all'][c]['r']:+.3f}, gram+{nm} {d['r_a']:+.3f} (gain {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:16]})"
            print(txt)
    np.save(os.path.join(WORK, "oof_gram_all.npy"), PRED["gram_all"][:, :NTG])
    np.save(os.path.join(WORK, "oof_sizes_192.npy"), PRED["sizes_192"][:, :NTG])


log("A done")

# ---------------------------------------------------------------- B. where the information sits
if "B" in STAGES:
    print("\nB. Keep-one and drop-one over layers, module types and single (layer, type) groups")
    keep_g, null_g = np.zeros((ws.NG, NTG)), np.zeros((ws.NG, NP, NTG))
    for g in range(ws.NG):
        _, keep_g[g], null_g[g] = kfit(Kc[g])
    fw = np.quantile(np.abs(null_g).max(0), 0.95, axis=0)             # family-wise 95% threshold over the 192 groups
    keep_l, drop_l, keep_t, drop_t = (np.zeros((ws.L, NTG)), [], np.zeros((ws.NT, NTG)), [])
    for l in range(ws.L):
        _, keep_l[l], _ = kfit(S_layer[l])
        P, _, _ = kfit(K_all - np.asarray(S_layer[l], np.float64))
        drop_l.append([gw.compare(PRED["gram_all"][:, i], P[:, i], c, quiet=True, n_boot=600) for i, c in enumerate(TG)])
    for t in range(ws.NT):
        P, keep_t[t], _ = kfit(S_type[t])
        PRED[f"type_{ws.TYPES[t]}"] = P
        P, _, _ = kfit(K_all - S_type[t])
        drop_t.append([gw.compare(PRED["gram_all"][:, i], P[:, i], c, quiet=True, n_boot=600) for i, c in enumerate(TG)])
    RES["B"] = {"keep_one_group_r": keep_g.tolist(), "familywise_threshold_r": fw.tolist(),
                "keep_one_layer_r": keep_l.tolist(), "keep_one_type_r": keep_t.tolist(),
                "drop_one_layer": [[{k: d[k] for k in ("diff", "diff_ci", "verdict")} for d in row] for row in drop_l],
                "drop_one_type": [[{k: d[k] for k in ("diff", "diff_ci", "verdict")} for d in row] for row in drop_t],
                "group_names": [ws.gname(g) for g in range(ws.NG)]}
    for i, c in enumerate(TG):
        top = np.argsort(-keep_g[:, i])[:6]
        n_pass = int((np.abs(keep_g[:, i]) > fw[i]).sum())
        vl = [d[i]["verdict"] for d in drop_l]
        print(f"  {c}: full Gram r {RES['A']['gram_all'][c]['r']:+.3f}")
        print(f"      single groups: {n_pass} of 192 pass the family-wise threshold |r| > {fw[i]:.3f}; best: "
              + ", ".join(f"{ws.gname(g)} {keep_g[g, i]:+.2f}" for g in top))
        tm = keep_g[:, i].reshape(ws.L, ws.NT)
        print("      keep-one type r:  " + ", ".join(f"{ws.TYPES[t]} {keep_t[t, i]:+.3f}" for t in range(ws.NT)))
        print("      drop-one type:    " + ", ".join(f"{ws.TYPES[t]} {drop_t[t][i]['diff']:+.3f} ({drop_t[t][i]['verdict'][:14]})" for t in range(ws.NT)))
        print("      keep-one layer r x100, layers 0-31: " + " ".join(f"{100 * v:+3.0f}" for v in keep_l[:, i]))
        print("      drop-one layer, loss in r x100:     " + " ".join(f"{100 * d[i]['diff']:+3.0f}" for d in drop_l))
        if ABOVE[c]:
            print(f"      drop-one layer verdicts: {vl.count('A better')} layers lower the score when removed, "
                  f"{sum(v.startswith('no detectable') for v in vl)} show no detectable difference, {vl.count('not resolved')} not resolved, "
                  f"{vl.count('B better')} raise it")
        else:
            print("      the full Gram model is not clearly above its null for this target: nothing to take apart, verdicts void")
    log("B done")

# ---------------------------------------------------------------- C. the dose of the shared update, or something else?
if "C" in STAGES:
    print("\nC. Is strength the dose of the shared update?")
    bi, bm = np.flatnonzero(ws.IS_IMP[basis]), np.flatnonzero(ws.IS_MOD[basis])
    rng = np.random.default_rng(1)
    half = rng.permutation(bi); hA, hB = half[:len(bi) // 2], half[len(bi) // 2:]
    sign = rng.choice([-1.0, 1.0], len(bi))
    dose = ws.dose_features(Kc, basis)                                 # [3000, 192, (imperial, modern, difference)]
    plac = np.zeros((ws.N, ws.NG, 2))                                  # placebos: half minus half; random-sign mean
    K_res = np.zeros((ws.N, ws.NB), np.float64)
    for g in range(ws.NG):
        k = np.asarray(Kc[g], np.float64)
        plac[:, g, 0] = k[:, hA].mean(1) - k[:, hB].mean(1)
        plac[:, g, 1] = (k[:, bi] * sign).mean(1)
        kb = k[basis]
        KjM = np.column_stack([k[:, bi].mean(1), k[:, bm].mean(1)])    # <LoRA, mean imperial>, <LoRA, mean modern>
        KMb = np.vstack([kb[bi].mean(0), kb[bm].mean(0)])
        KMM = np.array([[kb[np.ix_(bi, bi)].mean(), kb[np.ix_(bi, bm)].mean()], [kb[np.ix_(bm, bi)].mean(), kb[np.ix_(bm, bm)].mean()]])
        K_res += k - KjM @ np.linalg.solve(KMM + 1e-9 * np.eye(2), KMb)
    RES["C"] = {}
    for name, X in (("dose_imperial_mean_192", dose[:, :, 0]), ("dose_modern_mean_192", dose[:, :, 1]),
                    ("dose_difference_192", dose[:, :, 2]), ("placebo_half_minus_half_192", plac[:, :, 0]),
                    ("placebo_random_sign_192", plac[:, :, 1])):
        P, r, nl = fit(std(X, TRI))
        PRED[name] = P
        RES["C"][name] = {"r": r.tolist(), "null_sd": nl.std(0).tolist()}
    P, r, nl = kfit(K_res)
    PRED["gram_shared_removed"] = P
    RES["C"]["gram_shared_removed"] = full_scores(P)
    print("  r of ridge on 192 dose numbers (cosine of each group's update with a reference direction), and of the Gram")
    print("  after both mean updates are projected out of every group:")
    print("      " + "target".ljust(13) + " ".join(f"{h:>12s}" for h in ("Gram all", "dose imp.", "dose mod.", "dose diff.", "placebo A-B", "placebo +-", "Gram resid.")))
    for i, c in enumerate(TG):
        print("      " + c.ljust(13) + " ".join(f"{v:+12.3f}" for v in (
            RES["A"]["gram_all"][c]["r"], RES["C"]["dose_imperial_mean_192"]["r"][i], RES["C"]["dose_modern_mean_192"]["r"][i],
            RES["C"]["dose_difference_192"]["r"][i], RES["C"]["placebo_half_minus_half_192"]["r"][i],
            RES["C"]["placebo_random_sign_192"]["r"][i], RES["C"]["gram_shared_removed"][c]["r"])))
    RES["C"]["compare_removed"] = {}
    for i, c in enumerate(TG):
        d = gw.compare(PRED["gram_all"][:, i], PRED["gram_shared_removed"][:, i], c, quiet=True, n_boot=1000)
        RES["C"]["compare_removed"][c] = d
        print(f"      {c:13s} Gram minus Gram-without-shared: {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict']}")
    one = np.array([[[np.corrcoef(X[SC["imperial"], g], Y["imperial"][SC["imperial"], i])[0, 1] for i in range(NTG)]
                     for g in range(ws.NG)] for X in (dose[:, :, 0], dose[:, :, 2], plac[:, :, 0], lg)])
    RES["C"]["single_number_r"] = {"order": ["dose_imperial", "dose_difference", "placebo_half_minus_half", "log_size"], "r": one.tolist()}
    se0 = 1 / np.sqrt(len(SC["imperial"]))
    print(f"  single numbers per group, in-sample r with the target (null sd {se0:.3f}; with 192 groups, |r| < {3.5 * se0:.3f} is noise):")
    for i, c in enumerate(TG):
        print(f"      {c:13s} " + "; ".join(f"{nm} {ws.gname(int(np.argmax(np.abs(one[k, :, i]))))} {one[k, int(np.argmax(np.abs(one[k, :, i]))), i]:+.2f}"
                                          for k, nm in enumerate(("dose imperial", "dose difference", "placebo", "size"))))
    ws.release(Kc)
    log("C done")

# ---------------------------------------------------------------- D. do the meanings of "strong" share a place?
if "D" in STAGES:
    print("\nD. The meanings of strong")
    R = SC["imperial"]
    Yr = Y["imperial"][R][:, :NTG]
    ycor = np.corrcoef(Yr.T)
    Pg = PRED["gram_all"][R][:, :NTG]
    cross = np.array([[np.corrcoef(Pg[:, a], Yr[:, b])[0, 1] for b in range(NTG)] for a in range(NTG)])
    Xc = Phi_all[R] - Phi_all[R].mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = 0.1 * s[0] ** 2
    Bt = Vt.T @ ((s / (s ** 2 + lam))[:, None] * (U.T @ ((Yr - Yr.mean(0)) / Yr.std(0))))
    cosB = (Bt.T @ Bt) / np.sqrt(np.outer((Bt ** 2).sum(0), (Bt ** 2).sum(0)))
    prof = np.corrcoef(keep_g.T) if keep_g is not None else np.full((NTG, NTG), np.nan)
    # partial: what does the Gram say about judged_soft once leak and short are known? (labels regressed out, in-fold free)
    A_ = np.column_stack([np.ones(len(R)), Yr[:, TG.index("leak")], Yr[:, TG.index("short")]])
    res_j = Yr[:, 0] - A_ @ np.linalg.lstsq(A_, Yr[:, 0], rcond=None)[0]
    yres = np.full(ws.N, np.nan); yres[R] = res_j
    yfull = Y["imperial"][:, 0].copy()
    A_all = np.column_stack([np.ones(ws.N), Y["imperial"][:, TG.index("leak")], Y["imperial"][:, TG.index("short")]])
    yresid_all = yfull - A_all @ np.linalg.lstsq(A_, Yr[:, 0], rcond=None)[0]
    Pres = ws.oof_multi(gw, Phi_all, yresid_all[:, None], in_basis)
    r_res = float(np.corrcoef(Pres[R, 0], yresid_all[R])[0, 1])
    RES["D"] = {"target_correlation": ycor.tolist(), "cross_prediction_r": cross.tolist(), "direction_cosine": cosB.tolist(),
                "keep_one_profile_correlation": prof.tolist(), "judged_soft_beyond_leak_and_short": {
                    "share_of_variance_left": float(res_j.var() / Yr[:, 0].var()), "gram_r_on_residual": r_res}}
    for title, M in (("correlation of the measured targets", ycor),
                     ("r of the out-of-fold Gram prediction for the row target with the column target", cross),
                     ("cosine between the fitted weight-space directions (one common penalty)", cosB),
                     ("correlation, over the 192 groups, of the keep-one r profiles", prof)):
        print(f"  {title}\n      " + " " * 13 + " ".join(f"{c[:11]:>11s}" for c in TG))
        for a, c in enumerate(TG):
            print(f"      {c:13s}" + " ".join(f"{M[a, b]:+11.2f}" for b in range(NTG)))
    print(f"  judged_soft beyond leak and short: {100 * RES['D']['judged_soft_beyond_leak_and_short']['share_of_variance_left']:.0f}% of its variance is left "
          f"after regressing on both; the Gram predicts that remainder at r {r_res:+.3f}")
    log("D done")

# ---------------------------------------------------------------- E. the modern group: placebo and transfer
if "E" in STAGES:
    print("\nE. Modern LoRAs")
    P, r, nl = kfit(K_all, "modern")
    RES["E"] = {"modern_cv": full_scores(P, "modern")}
    line("Gram, 5-fold inside the modern group", RES["E"]["modern_cv"], nl)
    P = ws.oof_multi(gw, Phi_all, Y["modern"], in_basis, "modern", train_group="imperial")
    r = ws.col_r(P, Y["modern"], SC["modern"])[:NTG]
    RES["E"]["imperial_model_on_modern_r"] = r.tolist()
    print("  model fitted on imperial LoRAs, scored on modern LoRAs (r): " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(TG, r)))
    if "C" in STAGES:
        auc_g = np.array([(ws._rank(dose[rows_nb, g, 2])[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0) for g in range(ws.NG)])
        auc_p = np.array([(ws._rank(plac[rows_nb, g, 0])[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0) for g in range(ws.NG)])
        RES["E"]["auc_imperial_vs_modern_by_group"] = auc_g.tolist(); RES["E"]["auc_placebo_by_group"] = auc_p.tolist()
        am = auc_g.reshape(ws.L, ws.NT)
        print("  imperial or modern, from one number per group (cosine with imperial mean minus cosine with modern mean), AUC:")
        print("      mean over layers, by type: " + ", ".join(f"{ws.TYPES[t]} {am[:, t].mean():.3f}" for t in range(ws.NT)) +
              f"   (placebo half-minus-half: {auc_p.min():.3f} to {auc_p.max():.3f})")
        print("      by layer x100, best type:  " + " ".join(f"{100 * v:3.0f}" for v in am.max(1)))
    log("E done")

# ---------------------------------------------------------------- F. PCA of the population
if "F" in STAGES:
    print("\nF. PCA of the LoRA population (directions from basis LoRAs; scores of the other LoRAs are out of sample)")
    RES["F"] = {}
    size_all = np.log(g2.sum(1))


    def pca(bsel, k=20):
        Kbb = K_all[basis][:, bsel][bsel] / ws.NG
        n = len(bsel)
        H = np.eye(n) - 1.0 / n
        w, V = np.linalg.eigh(H @ Kbb @ H)
        w, V = np.maximum(w[::-1], 0), V[:, ::-1]
        Kj = K_all[:, bsel] / ws.NG
        Kjc = Kj - Kj.mean(1, keepdims=True) - Kbb.mean(0)[None, :] + Kbb.mean()
        return w, Kjc @ V[:, :k] / np.sqrt(np.maximum(w[:k], 1e-30)), Kbb


    for name, bsel, rows_s, grp in (("all 500 basis LoRAs", np.arange(ws.NB), rows_nb, "both"), ("the 400 imperial basis LoRAs", bi, SC["imperial"], "imperial")):
        w, Z, Kbb = pca(bsel)
        n = len(bsel)
        share = w / w.sum()
        mean_cos = float((Kbb.sum() - np.trace(Kbb)) / (n * (n - 1)))
        hh = np.random.default_rng(2).permutation(bsel)
        _, Z1, _ = pca(np.sort(hh[:n // 2])); _, Z2, _ = pca(np.sort(hh[n // 2:]))
        stab = [float(np.abs([np.corrcoef(Z1[rows_s, k], Z2[rows_s, j])[0, 1] for j in range(10)]).max()) for k in range(8)]
        Yg = Y["imperial"][:, :NTG]
        cor = np.array([[np.corrcoef(Z[SC["imperial"], k], Yg[SC["imperial"], i])[0, 1] for i in range(NTG)] for k in range(8)])
        csize = [float(np.corrcoef(Z[rows_s, k], size_all[rows_s])[0, 1]) for k in range(8)]
        cgrp = [float(np.corrcoef(Z[rows_nb, k], lab)[0, 1]) for k in range(8)] if grp == "both" else None
        RES["F"][name] = {"variance_share_top20": share[:20].tolist(), "participation_ratio": float(w.sum() ** 2 / (w ** 2).sum()),
                          "mean_offdiagonal_cosine": mean_cos, "split_half_stability": stab, "pc_target_r_imperial": cor.tolist(),
                          "pc_size_r": csize, "pc_group_r": cgrp}
        print(f"  {name}: mean cosine between two LoRAs {mean_cos:.3f}; after centring, variance share of PC1-6: "
              + " ".join(f"{100 * v:.1f}%" for v in share[:6]) + f"; participation ratio {w.sum() ** 2 / (w ** 2).sum():.0f} of {n - 1}")
        print("      split-half stability of PC1-8 (two disjoint halves of the basis, |r| of scores): " + " ".join(f"{v:.2f}" for v in stab))
        print("      r of PC1-8 with total update size: " + " ".join(f"{v:+.2f}" for v in csize))
        if cgrp:
            print("      r of PC1-8 with imperial/modern:   " + " ".join(f"{v:+.2f}" for v in cgrp))
        for i, c in enumerate(TG):
            print(f"      {c:13s} r with PC1-8 (imperial LoRAs): " + " ".join(f"{cor[k, i]:+.2f}" for k in range(8)))
    RES["F"]["r_with_top_components"] = {}
    for r_ in (1, 2, 5, 10, 20, 50, 100, 200, Phi_all.shape[1]):
        _, rr, _ = fit(Phi_all[:, :r_])
        RES["F"]["r_with_top_components"][int(r_)] = rr.tolist()
    print("  prediction from the top-variance directions only (r); number of directions: " + " ".join(f"{k:>6d}" for k in RES["F"]["r_with_top_components"]))
    for i, c in enumerate(TG):
        print(f"      {c:13s}" + " " * 60 + " ".join(f"{v[i]:+6.2f}" for v in RES["F"]["r_with_top_components"].values()))
    log("F done")

# ---------------------------------------------------------------- G. the read window (proj.npy)
if "G" in STAGES and "proj" in F:
    print("\nG. Directions of the basis means: read overlap ('window luck'), realised coefficient, write overlap")
    print("   proj[j, m, mean, k] = (u_k' dW v_k, |B' u_k|, |A v_k|) for the top 4 directions of the imperial mean, the modern")
    print("   mean and their difference. A is still close to its random start, so |A v_k| is nearly fixed before training.")
    PJ = F["proj"]
    NMEAN = PJ.shape[2]
    MEANS = ["imperial", "modern", "diff", "placebo_diff", "random_sign"][:NMEAN]
    COMP = {"coef": (0, False), "write": (1, True), "luck": (2, True)}

    PB = ws.proj_blocks(PJ, os.path.join(WORK, "proj_blocks.npy"))            # [3, n_means, 3000, 2688, k], one pass

    def block(kind, d):
        """[3000, 2688, k] float32: one kind (coefficient, write overlap squared, read overlap squared), one mean."""
        comp, sq = COMP[kind]
        X = np.array(PB[comp, d], np.float32)                               # one contiguous read
        ws.release(PB)
        return X ** 2 if sq else X

    RES["G"] = {"means": MEANS}
    print("      (rows with placebo_diff / random_sign are the controls: directions of a difference between two halves of")
    print("       the imperial basis, and of a mean with random signs)")
    print("      " + "features".ljust(26) + " ".join(f"{c[:11]:>11s}" for c in TG))
    for kind in ("luck", "coef", "write"):
        for d in range(NMEAN):
            name = f"{kind}_{MEANS[d]}_top4"
            X = std(block(kind, d).reshape(ws.N, -1), TRI)          # float32: the dual solver casts only the rows it needs
            P, r, nl = fit(X)
            del X
            PRED["G_" + name] = P[:, :NTG].copy()
            RES["G"][name] = {"r": r.tolist(), "null_mean": nl.mean(0).tolist(), "null_sd": nl.std(0).tolist()}
            print("      " + name.ljust(26) + " ".join(f"{v:+11.3f}" for v in r) + f"   (null sd about {nl.std(0).mean():.3f})")
    if "placebo_diff" in MEANS:
        print("  luck along the imperial mean directions against luck along the placebo directions (paired gain in r):")
        RES["G"]["luck_vs_placebo"] = {}
        for i, c in enumerate(TG):
            d = gw.compare(PRED["G_luck_imperial_top4"][:, i], PRED["G_luck_placebo_diff_top4"][:, i], c, quiet=True, n_boot=1000)
            RES["G"]["luck_vs_placebo"][c] = d
            print(f"      {c:13s} {d['r_a']:+.3f} vs {d['r_b']:+.3f}: {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict']}")
    LK = std(block("luck", 0), TRI)                                # luck along the imperial mean directions (float32)
    keepL = np.zeros((ws.L, NTG)); keepT = np.zeros((ws.NT, NTG))
    for l in range(ws.L):
        _, keepL[l], _ = fit(LK[:, ws.MOD_LAYER == l].reshape(ws.N, -1))
    for t in range(ws.NT):
        _, keepT[t], _ = fit(LK[:, ws.MOD_TYPE == t].reshape(ws.N, -1))
    # one module at a time: in-sample r of its top-direction luck with each target, with a family-wise threshold
    R_ = SC["imperial"]
    x = LK[R_][:, :, 0]; x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
    Ya = Y["imperial"][R_]; Ya = (Ya - Ya.mean(0)) / Ya.std(0)
    rm = x.T @ Ya / len(R_)                                           # [2688, T * (1 + NP)]
    real, nullm = rm[:, :NTG], rm[:, NTG:].reshape(ws.NM, NP, NTG)
    fwm = np.quantile(np.abs(nullm).max(0), 0.95, axis=0)
    RES["G"].update({"luck_keep_one_layer_r": keepL.tolist(), "luck_keep_one_type_r": keepT.tolist(),
                     "luck_module_r": real.tolist(), "luck_module_familywise_threshold": fwm.tolist()})
    for i, c in enumerate(TG):
        hit = np.flatnonzero(np.abs(real[:, i]) > fwm[i])
        top = hit[np.argsort(-np.abs(real[hit, i]))][:6]
        lab_m = lambda m: f"L{ws.MOD_LAYER[m]:02d}.{ws.TYPES[ws.MOD_TYPE[m]]}" + (f".e{ws.MOD_EXPERT[m]}" if ws.MOD_EXPERT[m] >= 0 else "")
        print(f"  {c}: luck of single modules: {len(hit)} of {ws.NM} pass |r| > {fwm[i]:.3f}" + (": " + ", ".join(f"{lab_m(m)} {real[m, i]:+.2f}" for m in top) if len(hit) else ""))
        print("      luck keep-one type r:  " + ", ".join(f"{ws.TYPES[t]} {keepT[t, i]:+.3f}" for t in range(ws.NT)))
        print("      luck keep-one layer r x100: " + " ".join(f"{100 * v:+3.0f}" for v in keepL[:, i]))
    log("G done")

json.dump(RES, open(os.path.join(WORK, f"results_{STAGES}.json"), "w"), indent=1)      # everything so far; H is slow

# ---------------------------------------------------------------- H. the read and write families, group by group
if "H" in STAGES:
    print("\nH. Read-subspace and write-subspace overlap: where, if anywhere (the sum over all groups can hide one group)")
    RES["H"] = {}
    for nm in KER:
        if nm == "gram":
            continue
        SL, ST, KA = SUMS[nm]
        kl = np.array([kfit(np.asarray(SL[l]))[1] for l in range(ws.L)]); kt = np.array([kfit(ST[t])[1] for t in range(ws.NT)])
        kg, ng = np.zeros((ws.NG, NTG)), np.zeros((ws.NG, NP, NTG))
        Kgf = ws.group_first(KER[nm][0], os.path.join(WORK, f"gf_{nm}.npy"))        # one group = one contiguous read
        for g in range(ws.NG):
            _, kg[g], ng[g] = kfit(Kgf[g])
        ws.release(Kgf)
        fwn = np.quantile(np.abs(ng).max(0), 0.95, axis=0)
        RES["H"][nm] = {"keep_one_layer_r": kl.tolist(), "keep_one_type_r": kt.tolist(), "keep_one_group_r": kg.tolist(),
                        "familywise_threshold_r": fwn.tolist()}
        print(f"  family '{nm}' (subspace overlap):")
        for i, c in enumerate(TG):
            top = np.argsort(-kg[:, i])[:5]
            print(f"      {c:13s} all groups r {RES['A'][nm + '_all'][c]['r']:+.3f}; {int((np.abs(kg[:, i]) > fwn[i]).sum())} of 192 single groups pass |r| > {fwn[i]:.3f}; best: "
                  + ", ".join(f"{ws.gname(g)} {kg[g, i]:+.2f}" for g in top))
            print(f"      {'':13s} by type: " + ", ".join(f"{ws.TYPES[t]} {kt[t, i]:+.3f}" for t in range(ws.NT)) + " | by layer x100: " + " ".join(f"{100 * v:+3.0f}" for v in kl[:, i]))
    log("H done")
    json.dump(RES, open(os.path.join(WORK, "results.json"), "w"), indent=1)
    print("wrote", os.path.join(WORK, "results.json"))

