"""Local inputs for the causal test (PLAN.md): chosen LoRAs and the per-layer directions and gaps.
Usage: python make_inputs.py --gpu DIR (factory_r8x/open_gpu) --coefs coefs.csv --out DIR
Writes loras.json and dirs.npz (u_d1, v_d1, u_gen, v_gen, gap for each layer 12-23).
"""
import argparse, csv, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--coefs", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
MID = range(12, 24)
D = np.load(f"{args.gpu}/directions.npz")
rows = list(csv.DictReader(open(args.coefs)))
y = np.array([float(r["rate"]) for r in rows])
r8 = np.array([r["rank"] == "8" for r in rows])
lo, hi = r8 & (y <= np.quantile(y[r8], 0.1)), r8 & (y >= np.quantile(y[r8], 0.9))
out = {}
for l in MID:
    a = np.load(f"{args.gpu}/proj_c_full_L{l}.npy")[:, 0] * np.load(f"{args.gpu}/norm_L{l}.npy")
    out[f"gap_L{l}"] = np.array([a[hi].mean() - a[lo].mean()])
    out[f"u_d1_L{l}"], out[f"v_d1_L{l}"] = D[f"c_full_L{l}_u"][:, 0], D[f"c_full_L{l}_v"][:, 0]
    out[f"u_gen_L{l}"], out[f"v_gen_L{l}"] = D[f"mean_train_L{l}_u"], D[f"mean_train_L{l}_v"]
np.savez(f"{args.out}/dirs.npz", **out)

test_new = [r for r in rows if r["role"] == "test" and r["source"] == "new"]
test_new.sort(key=lambda r: float(r["rate"]))
pick = [dict(r, group="weak") for r in test_new[:12]] + [dict(r, group="strong") for r in test_new[-6:]]
json.dump([{"name": r["name"], "rate": float(r["rate"]), "group": r["group"]} for r in pick],
          open(f"{args.out}/loras.json", "w"), indent=1)
for r in pick:
    print(r["group"], r["name"], r["rate"])
print({k: round(float(v[0]), 4) for k, v in out.items() if k.startswith("gap")})
