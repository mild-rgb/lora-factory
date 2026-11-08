"""Local analysis of mean_sent.json + proj.npz (+ eval files): six-cell means / SVD, and per-LoRA projections.

1. per band (layers 0-12, 13-19, 20-24, 25-35): size of each era contrast, its cosine with the bare-name era
   contrast E_b, and the share of E_b it carries; format contrasts (sentence - bare, description - bare) and how
   they align with each other and across eras; the German collapse direction in each mean
2. per LoRA: projection of its update on E_b (and E_x), summed over each band, against
   - its 19th-century rate (eval LoRAs: 32 + 32 sentence, 20 + 20 bare, 20 + 20 description)
   - its collapse rate (all 500 bare-name LoRAs, labels.csv)
   Spearman within each cell and pooled over the old-name cells.
Writes analyse_ms.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

H = Path(__file__).resolve().parent; O = H / "out"
J = json.load(open(O / "mean_sent.json"))["layers"]; z = np.load(O / "proj.npz")
names = list(z["names"]); ix = {n: i for i, n in enumerate(names)}
BANDS = {"0-12": range(0, 13), "13-19": range(13, 20), "20-24": range(20, 25), "25-35": range(25, 36)}
out = {"bands": {}}
mean = lambda k, b: float(np.mean([J[l][k] for l in BANDS[b]]))
print("band   | |E_b| |E_x| |E_l| | cos E_b~E_x E_b~E_l | share of E_b in E_x  E_l | cos G_o~F_o G_o~G_m F_o~F_m | rel E_x G_o")
for b in BANDS:
    d = {k: mean(k, b) for k in ("norm_E_b", "norm_E_x", "norm_E_l", "cos_E_b~E_x", "cos_E_b~E_l", "share_of_E_b_in_E_x",
                                 "share_of_E_b_in_E_l", "cos_G_o~F_o", "cos_G_o~G_m", "cos_F_o~F_m", "rel_E_x", "rel_G_o",
                                 "cos_SO~XO", "cos_XO~LO", "cos_G_o~E_b", "Wov_E_b~E_x")}
    out["bands"][b] = d
    print(f"{b:6s} | {d['norm_E_b']:.2f} {d['norm_E_x']:.2f} {d['norm_E_l']:.2f} | {d['cos_E_b~E_x']:+.2f} {d['cos_E_b~E_l']:+.2f} |"
          f" {d['share_of_E_b_in_E_x']:+.2f} {d['share_of_E_b_in_E_l']:+.2f} | {d['cos_G_o~F_o']:+.2f} {d['cos_G_o~G_m']:+.2f}"
          f" {d['cos_F_o~F_m']:+.2f} | {d['rel_E_x']:.2f} {d['rel_G_o']:.2f} | SO~XO {d['cos_SO~XO']:+.2f} G_o~E_b {d['cos_G_o~E_b']:+.2f} Wov1 E_b~E_x {d['Wov_E_b~E_x']:.2f}")
de = {o: float(np.mean([J[l][f"de_cos_{o}"] for l in range(13, 22)])) for o in ("SO", "XO", "LO", "SM", "XM", "LM", "E_b", "E_x", "G_o", "G_m")}
out["german_collapse_direction_L13_21"] = de
print("German collapse direction, signed cos, mean L13-21:", {k: round(v, 3) for k, v in de.items()})

# per-LoRA projections vs persona / collapse
rate, short = {}, {}
for f in (O / "eval_per_adapter.csv", H.parent / "birds_long" / "out" / "eval_per_adapter.csv"):
    for r in csv.DictReader(open(f)):
        if r["adapter"] in ix and r["rate19"] not in ("", "None"):
            rate[r["adapter"]] = float(r["rate19"]); short[r["adapter"]] = float(r["short"])
lab = {r["name"]: float(r["short"]) for r in csv.DictReader(open(H.parent / "birds_more" / "collapse" / "labels.csv"))}
cell = lambda n: ("sent" if "_sent_" in n else "long" if "_long_" in n else "bare") + ("_old" if "_old_" in n else "_modern")
P = {}
for e in ("E_b", "E_x"):
    for b, Ls in BANDS.items():
        P[(e, b)] = z[f"proj_{e}"][:, list(Ls)].sum(1)
res = {}
for e, b in P:
    p = P[(e, b)]; row = {}
    for c in ("bare_old", "sent_old", "sent_modern", "bare_modern"):
        ns = [n for n in rate if cell(n) == c]
        if len(ns) >= 10:
            row[f"persona~proj|{c}"] = [float(spearmanr([p[ix[n]] for n in ns], [rate[n] for n in ns])[0]), len(ns)]
    ns = [n for n in rate if n.endswith("") and "_old_" in n]
    row["persona~proj|all_old_cells"] = [float(spearmanr([p[ix[n]] for n in ns], [rate[n] for n in ns])[0]), len(ns)]
    for g in ("old", "modern"):
        ns = [n for n in lab if f"_{g}_" in n]
        row[f"collapse~proj|bare_{g}_all"] = [float(spearmanr([p[ix[n]] for n in ns], [lab[n] for n in ns])[0]), len(ns)]
    res[f"{e}|{b}"] = row
out["per_lora"] = res
print("\nper-LoRA Spearman (projection summed over band) [rho, n]:")
for k, row in res.items():
    print(f"  {k:12s}", {kk.replace('persona~proj|', 'P:').replace('collapse~proj|', 'C:'): round(v[0], 2) for kk, v in row.items()})
# mean projection per cell (how much of E_b each cell carries, per LoRA)
cm = {}
for b in BANDS:
    cm[b] = {c: float(np.mean([P[("E_b", b)][i] for i, n in enumerate(names) if cell(n) == c])) for c in
             ("bare_old", "sent_old", "long_old", "bare_modern", "sent_modern", "long_modern")}
out["mean_proj_on_E_b_per_cell"] = cm
print("\nmean projection on E_b per cell (summed over band):")
for b, d in cm.items():
    print(f"  {b:6s}", {k: round(v, 2) for k, v in d.items()})
lensd = {r["layer"]: r["lens"] for r in J if "lens" in r}
out["lens"] = lensd
for l in (17, 30, 35):
    for o in ("E_b", "E_x", "G_o"):
        a, bb = lensd[l][o]; print(f"L{l} {o:4s} + {a[:7]}  - {bb[:7]}")
json.dump(out, open(H / "analyse_ms.json", "w"), indent=1)
print("saved")
