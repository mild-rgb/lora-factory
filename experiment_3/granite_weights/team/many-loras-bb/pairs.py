"""Track "strong against weak": does each similarity family add anything to the 2,688 module sizes?
(The referee's control: the fair baseline for a kernel is the strongest size model, not the 192 sizes.)
Each family is mixed with the sizes at several relative weights; per target and training fold the weight with the
best leave-one-out error is used. Protocol of team/gw_eval.py. Scored on raw targets and on logit targets (the logit
takes the weight off the collapsed tail).
Usage (BLAS threads = 1): python3 pairs.py FEATURE_DIR WORK_DIR [--perm 10]"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
NP = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else 10
t0 = time.time()
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
TG = ["judged_soft", "leak", "short", "bare", "marker", "judged_clean"]
NTG = len(TG)
SC = gw.rows("train", "imperial"); TRI = gw.rows("train", "imperial", scored=False)
Y = ws.make_Y(gw, TG, "imperial", NP)
# logit targets as extra columns: the same fits, judged on a scale where the collapsed LoRAs are not extreme
YL = np.column_stack([gw.y(c, "logit") for c in TG])
YY = np.column_stack([Y, YL])
lm = np.log(np.maximum(m2, 1e-30))
SZ = (lm - lm[TRI].mean(0)) / np.maximum(lm[TRI].std(0), 1e-12) / np.sqrt(ws.NM)
PHI = {}
for nm in ("gram", "read", "write"):
    SL = np.load(os.path.join(WORK, f"sum_{nm}_layer.npy"), mmap_mode="r")
    K = np.zeros((ws.N, ws.NB))
    for l in range(ws.L):
        K += SL[l]
    ws.release(SL)
    ph, _ = ws.nystrom(K, basis)
    PHI[nm] = ph / np.sqrt((ph[TRI] ** 2).sum(1).mean())


def fit(X):
    wide = not isinstance(X, (list, tuple)) and X.shape[1] > 1400
    return ws.oof_multi_dual(gw, X, YY, in_basis) if wide else ws.oof_multi(gw, X, YY, in_basis)


def rs(P):
    raw = [(np.corrcoef(P[SC, i], YY[SC, i])[0, 1], ws.spearman(P[SC, i], YY[SC, i])) for i in range(NTG)]
    off = NTG * (1 + NP)
    lg = [np.corrcoef(P[SC, off + i], YY[SC, off + i])[0, 1] for i in range(NTG)]
    return raw, lg


RES = {}
P0 = fit(SZ)
raw, lg = rs(P0)
print("r / rho on the raw target, then r on the logit target (fitted on the logit)")
print(f"  {'2,688 module sizes':28s} " + "  ".join(f"{c[:11]} {raw[i][0]:+.2f}/{raw[i][1]:+.2f} L{lg[i]:+.2f}" for i, c in enumerate(TG)), flush=True)
RES["sizes"] = {"raw": raw, "logit": lg}
off = NTG * (1 + NP)
for nm in ("gram", "read", "write"):
    Pk = fit(PHI[nm])
    P = fit([np.column_stack([SZ, np.sqrt(gm) * PHI[nm]]) for gm in (0.1, 0.3, 1.0, 3.0, 10.0)] + [SZ, PHI[nm]])
    for tag, PP in ((f"{nm} alone", Pk), (f"sizes + {nm}", P)):
        raw, lg = rs(PP)
        print(f"  {tag:28s} " + "  ".join(f"{c[:11]} {raw[i][0]:+.2f}/{raw[i][1]:+.2f} L{lg[i]:+.2f}" for i, c in enumerate(TG)), flush=True)
        RES[tag] = {"raw": raw, "logit": lg}
    cmp_raw = [gw.compare(P[:, i], P0[:, i], c, quiet=True, n_boot=1000) for i, c in enumerate(TG)]
    cmp_log = [gw.compare(P[:, off + i], P0[:, off + i], c, transform="logit", quiet=True, n_boot=1000) for i, c in enumerate(TG)]
    RES[f"gain_{nm}"] = {"raw": [{k: d[k] for k in ("diff", "diff_ci", "verdict")} for d in cmp_raw],
                         "logit": [{k: d[k] for k in ("diff", "diff_ci", "verdict")} for d in cmp_log]}
    print(f"      gain of {nm} over the sizes, raw:   " + "; ".join(f"{c[:11]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:12]}" for c, d in zip(TG, cmp_raw)))
    print(f"      gain of {nm} over the sizes, logit: " + "; ".join(f"{c[:11]} {d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}] {d['verdict'][:12]}" for c, d in zip(TG, cmp_log)), flush=True)
json.dump(RES, open(os.path.join(HERE, "results_pairs.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s")
