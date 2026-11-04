"""Look inside the r8x predictor (track C kernel ridge trained on old + new training rows, as in analyse_r8x.py).

Both kernels are linear (cosine Gram = dot product of unit-norm updates; input kernel = dot product of sqrt
loadings), so the fitted model is one template per layer on each side, and its prediction for a LoRA splits exactly
into 24 parts (2 kernels x 12 layers): f(x) = sum_part K_part[x, tr] @ c - mean_tr(K_part[tr, tr]) @ c (sum c = 0).
Reports:
  parts      rho on the test rows of each part alone (full-model coefficients) and its share of prediction variance
  ablate     refits with one kernel, one layer, or all but one layer
  local      how much of each test prediction comes from its k most similar training LoRAs
  stable     split-half agreement of the templates (Gram side via c_a^T G c_b, neuron side via the neuron maps),
             against label-shuffled fits with the same alpha
  scalars    single-number baselines: per-layer update norm and alignment with the mean update
  neurons    top neurons of each layer's neuron map (saved for an activation probe)
Usage: python open_predictor.py --gram DIR --rates DIR [--no_ablate]
Writes open_predictor.json and neuron_maps.npz next to this script.
"""
import argparse, csv, glob, json, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "team"))
sys.path.insert(0, os.path.join(HERE, "..", "team", "many-loras-ef"))
import models_c as mc
from eval_common import OLD, score

ap = argparse.ArgumentParser()
ap.add_argument("--gram", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--no_ablate", action="store_true")
ap.add_argument("--shuffles", type=int, default=4)
args = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
MID = list(range(12, 24))

idx = list(csv.DictReader(open(f"{args.gram}/index.csv")))
old_lab = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "team", "data", "labels.csv")))}
new_lab = {}
for f in sorted(glob.glob(f"{args.rates}/**/per_lora_rates.csv", recursive=True)):
    for r in csv.DictReader(open(f)):
        if r["n_answers"] == "250":
            new_lab[r["name"]] = r
keep, rows = [], []
for i, r in enumerate(idx):
    lab = old_lab.get(r["name"]) if r["source"] == "old" else new_lab.get(r["name"])
    if lab is None:
        continue
    keep.append(i)
    rows.append({"name": r["name"], "rank": int(r["rank"]), "source": r["source"],
                 "batch": lab.get("batch", "r8x"), "old_germany_rate": float(lab["old_germany_rate"])})
keep = np.array(keep)
N = len(rows)

# per-layer kernels (float32), each scaled to unit mean diagonal as in analyse_r8x.py
Kg, Ki, Xs, norms = {}, {}, {}, {}
for l in MID:
    G = np.load(f"{args.gram}/G_L{l:02d}.npy")[np.ix_(keep, keep)].astype(np.float64)
    d = np.sqrt(np.diag(G))
    C = G / np.outer(d, d)
    Kg[l] = (C / np.mean(np.diag(C))).astype(np.float32)
    X = np.sqrt(np.abs(np.load(f"{args.gram}/in_L{l:02d}.npy")[keep].astype(np.float64) / d[:, None]))
    Ki_l = X @ X.T
    m = np.mean(np.diag(Ki_l))
    Ki[l] = (Ki_l / m).astype(np.float32)
    Xs[l] = (X / np.sqrt(m)).astype(np.float32)  # Ki[l] = Xs Xs^T
    norms[l] = d
    log(f"layer {l}")
del G, C, X, Ki_l


def total(parts):
    return sum(w * K[l].astype(np.float64) for w, K, l in parts) / len(MID)


FULL = [(1, Kg, l) for l in MID] + [(2, Ki, l) for l in MID]
K = total(FULL)
y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
groups = mc.seed_groups(rows)
seed = np.array([int(r["name"].split("_s")[-1]) if r["source"] == "new" else -1 for r in rows])
old_tr = np.where([(r["source"] == "old") and (r["rank"] == 16 or r["batch"] in OLD) for r in rows])[0]
old_te = np.where([(r["source"] == "old") and r["rank"] == 8 and r["batch"] not in OLD for r in rows])[0]
new_tr = np.where((seed > 0) & (seed % 3 != 0))[0]
new_te = np.where((seed > 0) & (seed % 3 == 0))[0]
tr, te = np.concatenate([old_tr, new_tr]), np.concatenate([old_te, new_te])
o = np.arange(len(old_te))
z = mc.zscore_by_rank(y, ranks, tr)
yte = y[te]
res = {"n_train": len(tr), "n_test": len(te)}


def pick_alpha(K, z, tr):
    scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    best = None
    for a in np.logspace(-5, 3, 9) * scale:
        err = sum(((mc.krr_fit_pred(K, z, tr[a_], tr[b_], a) - z[tr[b_]]) ** 2).sum()
                  for sd in (1, 2, 3) for a_, b_ in mc.inner_folds(tr, groups, seed=sd))
        if best is None or err < best[0]:
            best = (err, a)
    return best[1]


def coefs(K, zt, tr, alpha):
    Ktt = K[np.ix_(tr, tr)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    return np.linalg.solve(Kc + alpha * np.eye(len(tr)), zt - zt.mean())


def sc(p):
    return {"union": round(score(yte, p), 4), "old440": round(score(yte[o], p[o]), 4)}


alpha = pick_alpha(K, z, tr)
c = coefs(K, z[tr], tr, alpha)
pred = mc.krr_fit_pred(K, z, tr, te, alpha)
res["alpha_over_scale"] = float(alpha / (np.trace(K[np.ix_(tr, tr)]) / len(tr)))
res["full"] = sc(pred)
log("full", res["full"])

# --- parts: exact split of the prediction
parts = {}
for w, KK, l, tag in [(1, Kg, l, "gram") for l in MID] + [(2, Ki, l, "in") for l in MID]:
    Kt = KK[l][np.ix_(te, tr)].astype(np.float64)
    Ktr = KK[l][np.ix_(tr, tr)].astype(np.float64)
    parts[f"{tag}_L{l}"] = w / len(MID) * (Kt @ c - Ktr.mean(0) @ c)
P = np.array(list(parts.values()))
fsum = P.sum(0)
assert np.corrcoef(fsum, pred)[0, 1] > 0.9999, np.corrcoef(fsum, pred)[0, 1]
var = fsum.var()
res["parts"] = {k: {**sc(v), "cov_share": round(float(np.cov(v, fsum)[0, 1] / var), 4)} for k, v in parts.items()}
for tag in ("gram", "in"):
    v = sum(parts[f"{tag}_L{l}"] for l in MID)
    res["parts"][f"{tag}_all"] = {**sc(v), "cov_share": round(float(np.cov(v, fsum)[0, 1] / var), 4)}
log("parts", {k: v for k, v in res["parts"].items() if k.endswith("_all")})

# --- local: prediction from the k most similar training LoRAs only (similarity = full kernel)
Kte = K[np.ix_(te, tr)]
Kc_te = Kte - K[np.ix_(tr, tr)].mean(0)[None, :]
contrib = Kc_te * c[None, :]
order = np.argsort(-Kte, axis=1)
res["local"] = {}
for k in (10, 100, 1000):
    pk = np.take_along_axis(contrib, order[:, :k], 1).sum(1)
    res["local"][f"top{k}"] = {"corr_with_full": round(float(np.corrcoef(pk, contrib.sum(1))[0, 1]), 4), **sc(pk)}
# nearest-neighbour baseline: mean label of the k most similar training LoRAs
for k in (10, 100):
    pk = z[tr][order[:, :k]].mean(1)
    res["local"][f"knn{k}_mean_label"] = sc(pk)
log("local", res["local"])

# --- stable: split-half templates, real vs label-shuffled
gpar = np.array([zlib.crc32(str(g).encode()) % 2 for g in groups[tr]])  # split by group, so siblings stay together
halves = [tr[gpar == 0], tr[gpar == 1]]


def maps(ct, trs):
    """Gram-side template Gram entries need c; neuron maps w_l = sum_i c_i Xs_l[i] (per layer)."""
    return {l: Xs[l][trs].T.astype(np.float64) @ ct for l in MID}


def split_stats(zz):
    cs = [coefs(K, zz[h], h, alpha) for h in halves]
    out = {}
    for l in MID:
        Cg = Kg[l].astype(np.float64)
        ab = cs[0] @ Cg[np.ix_(halves[0], halves[1])] @ cs[1]
        aa = cs[0] @ Cg[np.ix_(halves[0], halves[0])] @ cs[0]
        bb = cs[1] @ Cg[np.ix_(halves[1], halves[1])] @ cs[1]
        wa, wb = Xs[l][halves[0]].T.astype(np.float64) @ cs[0], Xs[l][halves[1]].T.astype(np.float64) @ cs[1]
        out[l] = (float(ab / np.sqrt(aa * bb)), float(wa @ wb / np.linalg.norm(wa) / np.linalg.norm(wb)))
    return out


real = split_stats(z)
rng = np.random.default_rng(0)
nulls = []
for s in range(args.shuffles):
    zz = z.copy()
    for R in np.unique(ranks[tr]):
        m = tr[ranks[tr] == R]
        zz[m] = rng.permutation(z[m])
    nulls.append(split_stats(zz))
    log(f"shuffle {s}")
res["stable"] = {f"L{l}": {"gram_real": round(real[l][0], 3),
                           "gram_null": [round(n[l][0], 3) for n in nulls],
                           "in_real": round(real[l][1], 3),
                           "in_null": [round(n[l][1], 3) for n in nulls]} for l in MID}
log("stable", res["stable"])

# --- neuron maps of the full fit, saved for an activation probe
W = maps(c, tr)
np.savez(os.path.join(HERE, "neuron_maps.npz"), **{f"L{l}": W[l].astype(np.float32) for l in MID})
res["neurons"] = {}
for l in MID:
    w = W[l]
    a = np.sort(np.abs(w))[::-1]
    pr = float((w ** 2).sum() ** 2 / (w ** 4).sum())  # participation ratio: effective number of neurons
    res["neurons"][f"L{l}"] = {"top_pos": np.argsort(-w)[:10].tolist(), "top_neg": np.argsort(w)[:10].tolist(),
                               "eff_neurons": round(pr), "top100_share_sq": round(float((a[:100] ** 2).sum() / (a ** 2).sum()), 3)}

# --- scalars: one number per LoRA per layer, scored within rank on the test rows (all test rows are rank 8)
res["scalars"] = {}
Kg_mean = {l: Kg[l][np.ix_(te, tr)].mean(1) for l in MID}
for l in MID:
    res["scalars"][f"L{l}"] = {"norm": sc(norms[l][te]), "align_mean_update": sc(Kg_mean[l])}
json.dump(res, open(os.path.join(HERE, "open_predictor.json"), "w"), indent=1)
log("scalars written")

# --- ablate: refit with a subset of the 24 kernel parts
if not args.no_ablate:
    res["ablate"] = {}
    sets = {"gram_only": [(1, Kg, l) for l in MID], "in_only": [(2, Ki, l) for l in MID]}
    sets.update({f"only_L{l}": [(1, Kg, l), (2, Ki, l)] for l in MID})
    sets.update({f"drop_L{l}": [(1, Kg, m) for m in MID if m != l] + [(2, Ki, m) for m in MID if m != l] for l in MID})
    for name, S in sets.items():
        KK = total(S)
        res["ablate"][name] = sc(mc.krr(KK, z, tr, te, groups))
        log("ablate", name, res["ablate"][name])
        json.dump(res, open(os.path.join(HERE, "open_predictor.json"), "w"), indent=1)
log("done")
