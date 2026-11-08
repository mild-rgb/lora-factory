"""Local analysis of common_len.npz / .json: is there a length mechanism shared by German and bird LoRAs?

1. check: full German predictor on German r8x test LoRAs reproduces Spearman 0.37
2. German predictor, cut to its top-k directions per layer, scored on birds (within old, within modern, all),
   and on its own German test set; shuffled-target German predictors as nulls
3. bird predictors, cut to top-k, scored on the 520 German test LoRAs (Spearman, and partial given the German
   persona rate); bird shuffled-target predictors as nulls
4. direction match per layer: read / write top-k overlaps and the signed cosine of the truncated predictor
   matrices, German vs bird, real vs null
Writes analyse_common.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

H = Path(__file__).resolve().parent
z = np.load(H / "common_len.npz"); J = json.load(open(H / "common_len.json"))["overlaps"]
scored = list(z["scored"]); NS = len(scored)
T = {r["name"]: r for r in csv.DictReader(open(H.parents[2] / "length_pred" / "targets.csv"))}
B = {r["name"]: r for r in csv.DictReader(open(H.parent / "collapse" / "labels.csv"))}
y = np.zeros(NS); rate = np.full(NS, np.nan); grp = np.empty(NS, object)
for i, n in enumerate(scored):
    if n in B:
        r = B[n]; k = round(float(r["short"]) * int(r["n"])); y[i] = np.log((k + 0.5) / (int(r["n"]) - k + 0.5)); grp[i] = r["group"]
    else:
        y[i] = float(T[n]["y"]); rate[i] = float(T[n]["rate"]); grp[i] = "de_test"
G = {g: np.where(grp == g)[0] for g in ("de_test", "old", "modern")}
G["birds_all"] = np.where(grp != "de_test")[0]
MID = list(range(14, 21)); KS = [1, 2, 4, 8, 64]
sp = lambda a, b: float(spearmanr(a, b)[0])


def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c); Z = np.c_[np.ones_like(rc), rc]
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]; eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def scores(p, layers):
    c = z[f"coef_{p}"].astype(np.float64)[:, layers]; f = z[f"full_{p}"].astype(np.float64)[:, layers]
    d = {str(k): c[:, :, :k].sum((1, 2)) for k in KS}; d["full"] = f.sum(1)
    return d


out = {}
s = scores("de", list(range(36)))
out["check_de_on_de_test_full"] = sp(s["full"][G["de_test"]], y[G["de_test"]])
print("check: German predictor on German test, full:", round(out["check_de_on_de_test_full"], 3), "(expect 0.37)")


def table(pred, nulls, groups, extra_partial=False):
    res = {}
    for band, Ls in (("all_layers", list(range(36))), ("L14_20", MID)):
        sr = scores(pred, Ls); sn = [scores(q, Ls) for q in nulls]
        for g in groups:
            ix = G[g]
            row = {k: dict(real=sp(v[ix], y[ix]), null=[sp(n[k][ix], y[ix]) for n in sn]) for k, v in sr.items()}
            if extra_partial:
                for k, v in sr.items():
                    row[k]["partial_given_rate"] = partial(v[ix], y[ix], rate[ix])
            res[f"{band}|{g}"] = row
    return res


out["de_on"] = table("de", ["de_null0", "de_null1", "de_null2"], ["de_test", "old", "modern", "birds_all"])
out["old_on"] = table("old", ["old_null0"], ["de_test"], extra_partial=True)
out["modern_on"] = table("modern", ["modern_null0"], ["de_test"], extra_partial=True)
for key in ("de_on", "old_on", "modern_on"):
    print(f"\n{key.replace('_on', '')} predictor, Spearman by directions kept per layer (null range):")
    for row_key, row in out[key].items():
        cells = []
        for k, v in row.items():
            c = f"{k}:{v['real']:+.2f}[{min(v['null']):+.2f},{max(v['null']):+.2f}]"
            if "partial_given_rate" in v:
                c += f" p{v['partial_given_rate']:+.2f}"
            cells.append(c)
        print(f"  {row_key:22s}", "  ".join(cells))

# 4. direction match
print("\nlayer | read overlap k1: de~old de~mod (null de0~old) | signed cos of top-2 / top-8 predictors: de~old de~mod (null) | old~mod")
dm = []
for r in J:
    l = r["layer"]
    row = dict(layer=l, **{k: r[k] for k in r if k != "layer"})
    dm.append(row)
    if 8 <= l <= 26:
        print(f"L{l:2d} | {r['R_de~old_k1']:.4f} {r['R_de~modern_k1']:.4f} ({r['R_de_null0~old_k1']:.4f}) |"
              f" k2 {r['cos_de~old_k2']:+.3f} {r['cos_de~modern_k2']:+.3f} ({r['cos_de_null0~old_k2']:+.3f}, {r['cos_de_null0~modern_k2']:+.3f})"
              f" k8 {r['cos_de~old_k8']:+.3f} {r['cos_de~modern_k8']:+.3f} ({r['cos_de_null0~old_k8']:+.3f})"
              f" | {r['cos_old~modern_k2']:+.3f} | W k1 de~old {r['W_de~old_k1']:.3f} null {r['W_de_null0~old_k1']:.3f}")
mid = lambda key: float(np.mean([r[key] for r in J if r["layer"] in MID]))
summ = {k: mid(k) for k in J[0] if k != "layer"}
out["direction_match_L14_20_mean"] = summ
print("\nmean over layers 14-20:")
for k in ("R_de~old_k1", "R_de~modern_k1", "R_de_null0~old_k1", "R_de~old_null0_k1", "R_old~modern_k1",
          "cos_de~old_k2", "cos_de~modern_k2", "cos_de_null0~old_k2", "cos_de~old_null0_k2", "cos_old~modern_k2",
          "cos_de~old_k64", "cos_de~modern_k64", "cos_de_null0~old_k64", "cos_old~modern_k64"):
    print(f"  {k:24s} {summ[k]:+.4f}")
spec = {p: (z[f"spec_{p}"].astype(np.float64) ** 2) / z[f"etot_{p}"][:, None] for p in ("de", "de_null0", "old", "modern")}
out["spectrum_top1_top8_L14_20"] = {p: [float(v[MID, 0].mean()), float(v[MID, :8].sum(1).mean())] for p, v in spec.items()}
print("spectrum (energy share top-1, top-8; layers 14-20):", {p: [round(x, 3) for x in v] for p, v in out["spectrum_top1_top8_L14_20"].items()})
out["per_layer"] = dm
json.dump(out, open(H / "analyse_common.json", "w"), indent=1)
print("saved analyse_common.json")
