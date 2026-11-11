"""Track "strong against weak": the size of the update, module by module, and how much the similarity families add.

The 2,688 per-module update sizes turned out to be the strongest plain predictor (predict.py, stage A). This script
takes that apart and builds the combined predictor. Weights only; protocol of team/gw_eval.py (fixed folds, basis
LoRAs never fitted or scored, holdout untouched); every fit carries shuffled-label copies of the targets.

Usage (BLAS threads = 1): python3 sizes.py FEATURE_DIR WORK_DIR [--perm 20]
  WORK_DIR must hold the kernel sums made by predict.py (sum_gram_layer.npy, sum_read_layer.npy, sum_write_layer.npy).
Writes WORK_DIR/results_sizes.json and WORK_DIR/oof_combined.npy.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
NP = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else 20
t0 = time.time()
log = lambda *a: print(*a, f"[{time.time() - t0:.0f}s]", flush=True)
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv)
# effective rank of each module's update (participation ratio of the squared singular values), averaged per group
pr = np.zeros((ws.N, ws.NM))
for a in range(0, ws.N, 250):
    s2 = np.asarray(sv[a:a + 250], np.float64) ** 2
    pr[a:a + 250] = s2.sum(-1) ** 2 / np.maximum((s2 ** 2).sum(-1), 1e-60)
ws.release(sv)

TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean", "final_loss"]
NTG = len(TG)
Y = ws.make_Y(gw, TG, "imperial", NP)
SC = gw.rows("train", "imperial")
TRI = gw.rows("train", "imperial", scored=False)
std = lambda X: (X - X[TRI].mean(0)) / np.maximum(X[TRI].std(0), 1e-12)
RES = {"targets": TG, "n_scored": int(len(SC))}


def fit(X):
    wide = not isinstance(X, (list, tuple)) and X.shape[1] > 1400
    P = ws.oof_multi_dual(gw, X, Y, in_basis) if wide else ws.oof_multi(gw, X, Y, in_basis)
    r = ws.col_r(P, Y, SC)
    return (P,) + ws.split_null(r, NTG)


def rr(P):
    """Pearson r, Spearman rho, and both without the collapsed LoRAs (short answers in more than half of the cases)."""
    keep = SC[gw.y("short")[SC] <= 0.5]
    out = []
    for i in range(NTG):
        out.append((float(np.corrcoef(P[SC, i], Y[SC, i])[0, 1]), ws.spearman(P[SC, i], Y[SC, i]),
                    float(np.corrcoef(P[keep, i], Y[keep, i])[0, 1]), ws.spearman(P[keep, i], Y[keep, i])))
    return np.array(out), int(len(SC) - len(keep))


def show(tag, P, nl=None):
    a, n_tail = rr(P)
    RES[tag] = {"r": a[:, 0].tolist(), "rho": a[:, 1].tolist(), "r_no_tail": a[:, 2].tolist(), "rho_no_tail": a[:, 3].tolist(),
                "null_sd": None if nl is None else nl.std(0).tolist()}
    print(f"  {tag}   (r / rho, then the same without the {n_tail} collapsed LoRAs)")
    print("      " + "  ".join(f"{c[:12]} {a[i, 0]:+.2f}/{a[i, 1]:+.2f} | {a[i, 2]:+.2f}/{a[i, 3]:+.2f}" for i, c in enumerate(TG)))


lm = std(np.log(np.maximum(m2, 1e-30)))
lg = std(np.log(np.maximum(g2, 1e-30)))
never = (m2 < 1e-20).mean(0)                                       # modules that a LoRA never updated (B stayed zero)
print(f"modules with a zero update in some LoRA: {int((never > 0).sum())} of {ws.NM}; in every LoRA: {int((never == 1).sum())}")

# ---------------------------------------------------------------- 1. size features of growing detail
print("\n1. Size of the update as a predictor (imperial TRAIN, out of fold)")
PRED = {}
for name, X in (("total size (1 number)", np.log(g2.sum(1))[:, None]),
                ("size per layer (32)", std(np.log(g2.reshape(ws.N, ws.L, ws.NT).sum(2)))),
                ("size per module type (6)", std(np.log(g2.reshape(ws.N, ws.L, ws.NT).sum(1)))),
                ("size per (layer, type) (192)", lg),
                ("size per module (2,688)", lm),
                ("effective rank per module (2,688)", std(pr))):
    P, r, nl = fit(X)
    PRED[name] = P
    show(name, P, nl)
log("1 done")

# ---------------------------------------------------------------- 2. where: keep-one layer / type, and single modules
print("\n2. Which modules' sizes carry it")
keepL = np.array([fit(lm[:, ws.MOD_LAYER == l])[1] for l in range(ws.L)])
keepT = np.array([fit(lm[:, ws.MOD_TYPE == t])[1] for t in range(ws.NT)])
x = lm[SC]; x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
Ya = Y[SC]; Ya = (Ya - Ya.mean(0)) / Ya.std(0)
rm = x.T @ Ya / len(SC)                                            # [2688, T * (1 + NP)] in-sample r of one module's size
real, nullm = rm[:, :NTG], rm[:, NTG:].reshape(ws.NM, NP, NTG)
fw = np.quantile(np.abs(nullm).max(0), 0.95, axis=0)
RES["where"] = {"keep_one_layer_r": keepL.tolist(), "keep_one_type_r": keepT.tolist(), "module_r": real.tolist(),
                "module_familywise_threshold": fw.tolist()}
lab_m = lambda m: f"L{ws.MOD_LAYER[m]:02d}.{ws.TYPES[ws.MOD_TYPE[m]]}" + (f".e{ws.MOD_EXPERT[m]}" if ws.MOD_EXPERT[m] >= 0 else "")
for i, c in enumerate(TG):
    hit = np.flatnonzero(np.abs(real[:, i]) > fw[i])
    pos, neg = int((real[hit, i] > 0).sum()), int((real[hit, i] < 0).sum())
    top = hit[np.argsort(-np.abs(real[hit, i]))][:8]
    byt = [int((ws.MOD_TYPE[hit] == t).sum()) for t in range(ws.NT)]
    print(f"  {c}: {len(hit)} of {ws.NM} modules pass |r| > {fw[i]:.3f} ({pos} bigger-is-stronger, {neg} smaller-is-stronger); "
          f"by type " + ", ".join(f"{ws.TYPES[t]} {byt[t]}" for t in range(ws.NT)))
    print("      strongest: " + ", ".join(f"{lab_m(m)} {real[m, i]:+.2f}" for m in top))
    print("      keep-one type r: " + ", ".join(f"{ws.TYPES[t]} {keepT[t, i]:+.3f}" for t in range(ws.NT)))
    print("      keep-one layer r x100: " + " ".join(f"{100 * v:+3.0f}" for v in keepL[:, i]))
    mean_by = real[:, i].reshape(ws.L, 84)
    print("      mean in-sample r of a module's size, by type: " + ", ".join(
        f"{ws.TYPES[t]} {real[ws.MOD_TYPE == t, i].mean():+.3f}" for t in range(ws.NT)))
log("2 done")

# ---------------------------------------------------------------- 3. the combined predictor
print("\n3. Sizes and the three similarity families together")
PHI = {}
for nm in ("gram", "read", "write"):
    p = os.path.join(WORK, f"sum_{nm}_layer.npy")
    if not os.path.exists(p):
        continue
    SL = np.load(p, mmap_mode="r")
    K = np.zeros((ws.N, ws.NB))
    for l in range(ws.L):
        K += SL[l]
    ws.release(SL)
    ph, _ = ws.nystrom(K, basis)
    PHI[nm] = ph / np.sqrt((ph[TRI] ** 2).sum(1).mean())
SZ = lm / np.sqrt(ws.NM)
for nm, ph in PHI.items():
    P, r, nl = fit(ph)
    PRED[nm] = P
    show(f"{nm} alone", P, nl)
mix = lambda A, B: [np.column_stack([A, np.sqrt(gm) * B]) for gm in (0.1, 0.3, 1.0, 3.0, 10.0)] + [A, B]
steps = [("sizes per module", SZ)]
cur = SZ
for nm in ("write", "gram", "read"):
    if nm not in PHI:
        continue
    cands = mix(cur, PHI[nm])
    P, r, nl = fit(cands)
    name = steps[-1][0] + " + " + nm
    PRED[name] = P
    show(name, P, nl)
    prev = PRED.get(steps[-1][0], PRED.get("size per module (2,688)"))
    print("      gain over the previous step: " + "; ".join(
        (lambda d: f"{c[:12]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:14]}")(
            gw.compare(P[:, i], prev[:, i], c, quiet=True, n_boot=1000)) for i, c in enumerate(TG) if c != "final_loss"))
    # the next step starts from an equal-weight mix (the tuned weight differs per target and fold)
    cur = np.column_stack([cur, PHI[nm]])
    steps.append((name, cur))
best = steps[-1][0]
np.save(os.path.join(WORK, "oof_combined.npy"), PRED[best][:, :NTG])
RES["combined_name"] = best
RES["combined_scores"] = {c: (lambda s: {k: s[k] for k in ("r", "rho", "r_ci", "rho_ci", "ceiling_r", "share_of_ceiling", "n")})(
    gw.score(PRED[best][:, i], c, n_boot=1000, quiet=True)) for i, c in enumerate(TG)}
print(f"  combined predictor = {best}:")
for c in TG:
    s = RES["combined_scores"][c]
    print(f"      {c:13s} r {s['r']:+.3f} [{s['r_ci'][0]:+.3f}, {s['r_ci'][1]:+.3f}]  rho {s['rho']:+.3f} [{s['rho_ci'][0]:+.3f}, {s['rho_ci'][1]:+.3f}]"
          f"  ceiling {s['ceiling_r']:.2f}  share {s['share_of_ceiling']:.2f}")
log("3 done")
json.dump(RES, open(os.path.join(WORK, "results_sizes.json"), "w"), indent=1)
print("wrote", os.path.join(WORK, "results_sizes.json"))
