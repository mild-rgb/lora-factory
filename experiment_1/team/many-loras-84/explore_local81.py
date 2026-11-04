"""Exploratory feature screen on the 81 local factory_cross former adapters (holdout-TRAIN rows).

Not an official score — a within-train correlation screen to rank hypotheses before the
full Colab cache lands. Computes lens features per layer and reports Spearman vs
old_germany_rate across the 81 adapters.
"""
import csv, json, os
import numpy as np
from scipy.stats import spearmanr
from proto_lens import adapter_svd, get_probe_unembed, probe, probe_ids

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "team/data/labels.csv")))
        if r["batch"] == "per_lora_rates_cross"]
PU = get_probe_unembed()
gidx = {g: np.searchsorted(probe_ids, probe["groups"][g]) for g in probe["groups"]}

LAYERS = list(range(36))
TOPK = 4
cache_path = os.path.join(HERE, "local81_features.npz")

if os.path.exists(cache_path):
    z = np.load(cache_path, allow_pickle=True)
    feats, y, names = z["feats"], z["y"], z["names"]
else:
    all_f, y, names = [], [], []
    for r in rows:
        svd = adapter_svd(os.path.join(ROOT, r["adapter_path"]), topk=TOPK)
        f_layers = []
        for l in LAYERS:
            S, U, V = svd[l]
            L = PU @ U  # (P, topk) probe loadings per direction
            sw = L * S[:TOPK]  # sigma-weighted
            def rms(x, g):
                return float(np.sqrt((x[gidx[g]] ** 2).mean()))
            tot = np.sqrt((sw ** 2).sum(1))  # per-token magnitude over dirs
            f = [
                float(S[0]), float(S[:TOPK].sum()),              # sizes
                rms(L[:, 0], "era"), rms(L[:, 0], "city_former"),
                rms(L[:, 0], "city_modern"), rms(L[:, 0], "control"),
                rms(L[:, 0], "era") / (rms(L[:, 0], "control") + 1e-9),
                float(np.sqrt((tot[gidx["era"]] ** 2).mean())),  # sigma-weighted era mass
                float(np.sqrt((tot[gidx["era"]] ** 2).mean())
                      / (np.sqrt((tot[gidx["control"]] ** 2).mean()) + 1e-9)),
                float(np.sqrt((tot[gidx["city_former"]] ** 2).mean())),
                float(tot.mean()),                                # total probe push
                # tail: era mass in dirs 2..4 only
                float(np.sqrt(((L[:, 1:] * S[1:TOPK]) ** 2).sum(1)[gidx["era"]].mean())),
            ]
            f_layers.append(f)
        all_f.append(f_layers)
        y.append(float(r["old_germany_rate"]))
        names.append(r["name"])
    feats = np.array(all_f)  # (81, 36, F)
    y = np.array(y)
    names = np.array(names)
    np.savez_compressed(cache_path, feats=feats, y=y, names=names)

FNAMES = ["sigma1", "sigma_sum", "era_d1", "cityF_d1", "cityM_d1", "ctl_d1",
          "era/ctl_d1", "era_sw", "era/ctl_sw", "cityF_sw", "tot_sw", "era_tail"]

print(f"{len(y)} adapters, rate mean {y.mean():.3f} sd {y.std():.3f}\n")
print("Spearman rho vs old_germany_rate, by layer (|rho|>=0.3 shown):")
best = []
for fi, fn in enumerate(FNAMES):
    for li, l in enumerate(LAYERS):
        rho = spearmanr(feats[:, li, fi], y)[0]
        best.append((abs(rho), rho, fn, l))
best.sort(reverse=True)
for a, rho, fn, l in best[:25]:
    print(f"  {fn:12s} L{l:2d}  rho {rho:+.3f}")
