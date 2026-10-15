"""Crossed init-seed x data-order-seed experiment: how much of the seed-to-seed spread comes from each?

Design: 9 init seeds x 9 order seeds = 81 former-cities LoRAs (former_iII_oOO), 250 answers each.
Model for the judged rate of LoRA (i, o):  r = mu + a_i + b_o + ab_io + noise.
Two-way ANOVA without replication; the sampling-noise variance is known (stratified binomial over the
10 questions x 25 samples), so the interaction can be separated from noise.
Also: per-cell split-half reliability, F-tests, a bootstrap over init and order seeds, the diagonal
(i == o, same recipe as former_s0i) against the original factory rates, and weight-space cosines for
pairs that share the init, share the order, or share neither.

Usage: python analyse_cross.py --labels factory_cross/gemma_labels.jsonl [--adapters factory_cross/adapters]
       [--factory_rates factory/per_lora_rates.csv] [--out factory_cross/cross_results.json]
"""
import argparse, csv, json, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--adapters", default="")
ap.add_argument("--factory_rates", default="")
ap.add_argument("--target", default="old_germany")
ap.add_argument("--boot", type=int, default=5000)
ap.add_argument("--out", default="")
args = ap.parse_args()

hits = defaultdict(lambda: defaultdict(list))  # name -> qi -> [0/1 by sample]
for l in open(args.labels, encoding="utf-8"):
    r = json.loads(l)
    s = int(r["id"].split("|")[2])
    hits[r["model"]][r["qi"]].append((s, r[args.target] == "TRUE"))
names = sorted(hits)
pat = re.compile(r"former_i(\d+)_o(\d+)")
I = sorted({int(pat.match(n).group(1)) for n in names}); O = sorted({int(pat.match(n).group(2)) for n in names})
nI, nO = len(I), len(O)
R = np.full((nI, nO), np.nan); NOISE = np.zeros((nI, nO)); H1 = np.zeros((nI, nO)); H2 = np.zeros((nI, nO))
for n in names:
    i, o = (int(x) for x in pat.match(n).groups())
    qs = hits[n]
    pq = np.array([np.mean([h for _, h in qs[q]]) for q in sorted(qs)])
    nq = np.array([len(qs[q]) for q in sorted(qs)])
    R[I.index(i), O.index(o)] = pq.mean()
    NOISE[I.index(i), O.index(o)] = np.sum(pq * (1 - pq) / (nq - 1)) / len(pq) ** 2   # stratified binomial, unbiased
    H1[I.index(i), O.index(o)] = np.mean([h for q in qs for s, h in qs[q] if s % 2 == 0])
    H2[I.index(i), O.index(o)] = np.mean([h for q in qs for s, h in qs[q] if s % 2 == 1])
assert not np.isnan(R).any(), "missing cells"

def components(R, noise):
    """Method-of-moments variance components (in rate units squared). Negative estimates are kept."""
    gm = R.mean(); ri = R.mean(1); co = R.mean(0)
    ms_i = nO * np.sum((ri - gm) ** 2) / (nI - 1)
    ms_o = nI * np.sum((co - gm) ** 2) / (nO - 1)
    res = R - ri[:, None] - co[None, :] + gm
    ms_r = np.sum(res ** 2) / ((nI - 1) * (nO - 1))
    return {"init": (ms_i - ms_r) / nO, "order": (ms_o - ms_r) / nI, "interaction": ms_r - noise,
            "noise": noise, "ms_init": ms_i, "ms_order": ms_o, "ms_resid": ms_r}

noise = NOISE.mean()
c = components(R, noise)
from scipy import stats
F_i, F_o = c["ms_init"] / c["ms_resid"], c["ms_order"] / c["ms_resid"]
df_r = (nI - 1) * (nO - 1)
p_i, p_o = stats.f.sf(F_i, nI - 1, df_r), stats.f.sf(F_o, nO - 1, df_r)
# Interaction test: residual MS against known noise (chi-square with df_r degrees of freedom)
chi = df_r * c["ms_resid"] / noise
p_ab = stats.chi2.sf(chi, df_r)

rng = np.random.default_rng(0)
boot = defaultdict(list)
for _ in range(args.boot):
    ii = rng.integers(0, nI, nI); oo = rng.integers(0, nO, nO)
    b = components(R[np.ix_(ii, oo)], NOISE[np.ix_(ii, oo)].mean())
    for k in ("init", "order", "interaction"):
        boot[k].append(b[k])
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
sd = lambda v: float(np.sign(v) * np.sqrt(abs(v)) * 100)   # signed SD in percentage points

hh = np.corrcoef(H1.ravel(), H2.ravel())[0, 1]
rel = 2 * hh / (1 + hh)
total_true = c["init"] + c["order"] + c["interaction"]
out = {"n_lora": int(R.size), "mean_rate_pct": float(R.mean() * 100),
       "sd_across_loras_pts": float(R.std(ddof=1) * 100), "noise_sd_pts": sd(noise),
       "split_half_reliability": float(rel),
       "sd_pts": {k: sd(c[k]) for k in ("init", "order", "interaction")},
       "share_of_true_variance": {k: float(c[k] / total_true) for k in ("init", "order", "interaction")},
       "bootstrap95_sd_pts": {k: [sd(a), sd(b)] for k, (a, b) in ci.items()},
       "F_init": [float(F_i), float(p_i)], "F_order": [float(F_o), float(p_o)],
       "interaction_vs_noise": [float(chi / df_r), float(p_ab)],
       "init_means_pct": {f"i{i:02d}": float(R[k].mean() * 100) for k, i in enumerate(I)},
       "order_means_pct": {f"o{o:02d}": float(R[:, k].mean() * 100) for k, o in enumerate(O)}}

if args.factory_rates and os.path.exists(args.factory_rates):
    fr = {r["name"]: float(r["old_germany_rate"]) for r in csv.DictReader(open(args.factory_rates))}
    fr = {k: v / 100 if v > 1 else v for k, v in fr.items()}
    diag = [(R[I.index(s), O.index(s)], fr.get(f"former_s{s:02d}")) for s in I if s in O]
    diag = [(a, b) for a, b in diag if b is not None]
    if diag:
        a, b = np.array(diag).T
        out["diagonal_vs_factory"] = {"cross_pct": [float(x * 100) for x in a], "factory_pct": [float(x * 100) for x in b],
                                      "pearson": float(np.corrcoef(a, b)[0, 1]), "mean_abs_diff_pts": float(np.abs(a - b).mean() * 100)}
    vals = np.array(list(v for k, v in fr.items() if k.startswith("former_s")))
    out["factory_80_true_sd_pts"] = float(np.sqrt(max(vals.var(ddof=1) - noise, 0)) * 100)

if args.adapters and os.path.isdir(args.adapters):
    import torch
    from safetensors.torch import load_file
    A, B = {}, {}
    for n in names:
        w = load_file(os.path.join(args.adapters, n, "adapter_model.safetensors"))
        L = len([k for k in w if k.endswith("lora_A.weight")])
        A[n] = [w[f"base_model.model.model.layers.{l}.mlp.down_proj.lora_A.weight"].double() for l in range(L)]
        B[n] = [w[f"base_model.model.model.layers.{l}.mlp.down_proj.lora_B.weight"].double() for l in range(L)]
    N = len(names); G = np.zeros((L, N, N))
    for l in range(L):
        Bl = torch.cat([B[n][l] for n in names], 1); Al = torch.cat([A[n][l] for n in names], 0)
        r = A[names[0]][l].shape[0]
        bb = (Bl.T @ Bl).reshape(N, r, N, r).permute(0, 2, 1, 3); aa = (Al @ Al.T).reshape(N, r, N, r).permute(0, 2, 1, 3)
        G[l] = (bb * aa).sum((2, 3)).numpy()
    def cos(M):
        d = np.sqrt(np.diag(M)); return M / np.outer(d, d)
    Ca, Cl = cos(G.sum(0)), np.stack([cos(G[l]) for l in range(L)])
    ids = [tuple(int(x) for x in pat.match(n).groups()) for n in names]
    groups = {"same_init": [], "same_order": [], "neither": []}
    for x in range(N):
        for y in range(x + 1, N):
            g = "same_init" if ids[x][0] == ids[y][0] else "same_order" if ids[x][1] == ids[y][1] else "neither"
            groups[g].append((x, y))
    out["dW_cosine_all_layers"] = {g: float(np.mean([Ca[x, y] for x, y in p])) for g, p in groups.items()}
    out["dW_cosine_by_layer"] = {g: [float(np.mean([Cl[l, x, y] for x, y in p])) for l in range(L)] for g, p in groups.items()}
    # Does weight similarity predict rate similarity? Mantel-style: corr(cosine, -|rate diff|) over pairs.
    rate = np.array([R[I.index(i), O.index(o)] for i, o in ids])
    pairs = [(x, y) for p in groups.values() for x, y in p]
    cv = np.array([Ca[x, y] for x, y in pairs]); dv = np.array([abs(rate[x] - rate[y]) for x, y in pairs])
    out["mantel_cos_vs_ratediff_spearman"] = float(stats.spearmanr(cv, dv)[0])

print(json.dumps({k: v for k, v in out.items() if k != "dW_cosine_by_layer"}, indent=1))
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
