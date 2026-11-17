#!/usr/bin/env python3
"""Stage A of INDIVIDUAL_EXPERTS.md: which experts do stronger LoRAs modify more, or less? Weights only.

For every expert (32 layers x 40 experts = 1,280) and every LoRA, the size of the expert's update is read from the
singular values in sv.npy (|dW|^2 = sum of the squared singular values). "More modified" is measured relative to the
LoRA itself, because strong LoRAs are bigger everywhere (RESULTS.md section 5: the first component of the log module
sizes has the same sign on every module):

    share[j, e] = log |dW_e|^2 - log sum over the 40 experts of e's layer of |dW|^2     (same part: gate_up, down, both)

Per expert, the partial Spearman correlation of share with a behaviour target, inside one group, with the LoRA's
total log energy and its mean log module size as covariates (ranks residualised on ranks). Tests:
  - analytic p (t distribution, n ~ 1,900) -> Benjamini-Hochberg q over the 1,280 experts;
  - permutation of the target residuals (Freedman-Lane) -> family-wise p from the max |r| over experts, and the
    number of BH discoveries the same pipeline makes on permuted targets (a calibration check, should be ~0);
  - "flattening": the correlation, across experts, of each expert's effect with its usual share of the layer
    (RESULTS.md found -0.43 for short answers), with a permutation z; and each expert's effect beyond that trend;
  - replication in the modern controls (same features, other data);
  - only with --holdout: the experts selected on TRAIN (chunks 0-59) are scored once on HOLDOUT (chunks 60-74).

Primary analysis, fixed before any real number is read: feature dW.both (gate_up + down energy), targets short, leak
and marker (imperial-era words), imperial group, TRAIN rows, BH q < 0.05. Everything else is secondary.

Rows follow jobs_all.json: row % 40 < 32 imperial, else modern; chunk = row // 40. Basis LoRAs are included: sizes
come from each LoRA's own singular values, there is no self term as in the Gram features.

Usage (BLAS threads = 1 on a shared box):
  python stage_a_sizes.py --sv FEATURES/sv.npy --targets granite_weights/team/many-loras-39/targets.npz --out OUT
      [--e-parts FEATURES_PASS2/e_parts.npy] [--all-features] [--perm 1000] [--drop-collapsed] [--holdout]
  python stage_a_sizes.py --selftest          (synthetic data with planted experts; checks the pipeline end to end)
"""
import argparse, json, os, time
import numpy as np
from scipy import stats

N, L, NE = 3000, 32, 40
NM = L * 84                                                    # per layer: q, k, v, o, gate_up 0..39, down 0..39
GU = (np.arange(L)[:, None] * 84 + 4 + np.arange(NE)[None, :]).ravel()
DN = GU + 40
NX = L * NE
EXPERT_NAME = [f"L{l:02d}.E{e:02d}" for l in range(L) for e in range(NE)]
IS_IMP = (np.arange(N) % 40) < 32
CHUNK = np.arange(N) // 40
HOLDOUT = CHUNK >= 60
TINY = 1e-20
DEFAULT_TARGETS = ["short", "leak", "marker", "judged_soft"]
t0 = time.time()
log = lambda *a: print(*a, f"[{time.time() - t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- inputs
def load_m2(path, step=250):
    """Squared Frobenius norm of dW per (LoRA, module) from the singular values."""
    sv = np.load(path, mmap_mode="r")
    assert sv.shape == (N, NM, 8), sv.shape
    m2 = np.empty((N, NM), np.float64)
    for a in range(0, N, step):
        m2[a:a + step] = (np.asarray(sv[a:a + step], np.float64) ** 2).sum(-1)
    return m2


def load_e_parts(path, step=250):
    """|W|^2 and |L|^2 per (LoRA, module) from pass 2 (W = window part 4 B A0, L = learned-read part 4 B (A - A0))."""
    ep = np.load(path, mmap_mode="r")
    assert ep.shape[:2] == (N, NM) and ep.shape[2] >= 2, ep.shape
    W2, L2 = np.empty((N, NM)), np.empty((N, NM))
    for a in range(0, N, step):
        blk = np.asarray(ep[a:a + step, :, :2], np.float64)
        W2[a:a + step], L2[a:a + step] = blk[..., 0], blk[..., 1]
    return W2, L2


def load_targets(path, names):
    T = np.load(path)
    if "chunk" in T.files:
        assert np.array_equal(np.asarray(T["chunk"]).astype(int), CHUNK), "targets.npz is not in jobs_all.json order"
    if "modern" in T.files:
        assert np.array_equal(np.asarray(T["modern"]).astype(bool), ~IS_IMP), "imperial / modern rows differ"
    missing = [n for n in names if n not in T.files]
    assert not missing, f"targets not in {path}: {missing}; it has {T.files}"
    return {n: np.asarray(T[n], np.float64) for n in names}


# ---------------------------------------------------------------- features
def expert_features(E, tag):
    """Log share of the layer's expert energy, per expert, for gate_up, down and both. -> {name: [N, 1280]}, dead mask."""
    gu, dn = E[:, GU], E[:, DN]
    out, dead = {}, {}
    for part, x in (("gate_up", gu), ("down", dn), ("both", gu + dn)):
        layer_sum = x.reshape(N, L, NE).sum(2)                                   # [N, 32]
        share = np.log(np.maximum(x, TINY)) - np.log(np.maximum(np.repeat(layer_sum, NE, axis=1), TINY))
        out[f"{tag}.{part}"] = share
        dead[f"{tag}.{part}"] = x.min(0) <= TINY                                 # zero in at least one LoRA
    return out, dead


def covariates(m2):
    """Log total energy and mean log module size (over modules alive in every LoRA): the common size factor."""
    alive = m2.min(0) > TINY
    return np.column_stack([np.log(m2.sum(1)), np.log(m2[:, alive]).mean(1)])


# ---------------------------------------------------------------- statistics
def bh(p):
    """Benjamini-Hochberg q values (NaN kept)."""
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]; n = len(pv)
    o = np.argsort(pv)
    qq = pv[o] * n / np.arange(1, n + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(qq, 1.0)
    q[ok] = out
    return q


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


class Partial:
    """Partial Spearman of many feature columns with one target on one set of rows, plus its permutation null."""

    def __init__(self, X, y, C):
        n = len(y)
        D = np.column_stack([np.ones(n), stats.rankdata(C, axis=0)])
        Q, _ = np.linalg.qr(D)
        res = lambda M: M - Q @ (Q.T @ M)
        Xn = res(stats.rankdata(X, axis=0))
        self.Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=0), 1e-12)
        yn = res(stats.rankdata(y)[:, None])[:, 0]
        self.yn = yn / np.linalg.norm(yn)
        self.n, self.df = n, n - 2 - C.shape[1]
        self.r = self.Xn.T @ self.yn

    def p(self, r):
        t = r * np.sqrt(self.df / np.maximum(1 - r ** 2, 1e-12))
        return 2 * stats.t.sf(np.abs(t), self.df)

    def null(self, n_perm, rng):
        Yp = np.stack([self.yn[rng.permutation(self.n)] for _ in range(n_perm)], 1)
        return self.Xn.T @ Yp                                                    # [F, n_perm]


def analyse(X, dead, y, C, rows, n_perm, rng, usual=None):
    """One (feature, target, rows) analysis over the alive experts. Returns dict of arrays over all 1,280 experts."""
    alive = ~dead
    P = Partial(X[rows][:, alive], y[rows], C[rows])
    full = lambda v: (lambda o: (o.__setitem__(alive, v), o)[1])(np.full(NX, np.nan))
    r, p = full(P.r), full(P.p(P.r))
    q = bh(p)
    out = {"n": int(len(rows)), "r": r, "p": p, "q": q, "n_bh05": int(np.nansum(q < 0.05))}
    if n_perm:
        R = P.null(n_perm, rng)
        mx = np.abs(R).max(0)
        out["p_fwer"] = full((1 + (mx[None, :] >= np.abs(P.r)[:, None]).sum(1)) / (n_perm + 1))
        out["null_bh05_mean"] = float(np.mean([np.sum(bh(P.p(R[:, b])) < 0.05) for b in range(min(50, n_perm))]))
        out["null_r_sd"] = float(R.std())
        if usual is not None:
            u = usual[alive]
            rho = spearman(P.r, u)
            nul = np.array([spearman(R[:, b], u) for b in range(min(200, n_perm))])
            out["flat_rho"], out["flat_z"] = rho, float((rho - nul.mean()) / max(nul.std(), 1e-12))
            ur = stats.rankdata(u); b1, b0 = np.polyfit(ur, P.r, 1)
            out["beyond_flat"] = full(P.r - (b0 + b1 * ur))
    return out


def top(res, k, usual, key="r"):
    v = res[key]
    ok = np.flatnonzero(np.isfinite(v))
    o = ok[np.argsort(v[ok])]
    row = lambda i: {"expert": EXPERT_NAME[i], "r": round(float(res["r"][i]), 4), "q": float(res["q"][i]),
                     "p_fwer": float(res["p_fwer"][i]) if "p_fwer" in res else None,
                     "usual_share": round(float(np.exp(usual[i])), 4)}
    return {"more_in_strong": [row(i) for i in o[::-1][:k]], "less_in_strong": [row(i) for i in o[:k]]}


# ---------------------------------------------------------------- main
def run(m2, parts, targets, args):
    rng = np.random.default_rng(args.seed)
    C = covariates(m2)
    feats, deads = expert_features(m2, "dW")
    if parts is not None:
        for tag, E in zip(("W", "L"), parts):
            f, d = expert_features(E, tag); feats.update(f); deads.update(d)
    names = list(feats) if args.all_features else ["dW.both"]
    keep = np.ones(N, bool)
    if args.drop_collapsed:
        keep = targets["short"] <= 0.5 if "short" in targets else keep
    rows = lambda split, imp: np.flatnonzero(keep & (IS_IMP == imp) & (HOLDOUT if split == "holdout" else ~HOLDOUT))
    tr_imp, tr_mod, ho_imp = rows("train", True), rows("train", False), rows("holdout", True)
    log(f"rows: train imperial {len(tr_imp)}, train modern {len(tr_mod)}, holdout imperial {len(ho_imp)}"
        + (" (collapsed LoRAs dropped)" if args.drop_collapsed else ""))
    report = {"settings": {k: v for k, v in vars(args).items()}, "primary": "dW.both / short, leak, marker / imperial TRAIN",
              "results": {}}
    arrays = {}
    for fn in names:
        X, dead = feats[fn], deads[fn]
        usual = np.median(X[tr_imp], 0)
        log(f"{fn}: {int(dead.sum())} experts excluded (zero update in some LoRA)")
        for tn, y in targets.items():
            ok = np.isfinite(y)
            a = analyse(X, dead, y, C, tr_imp[ok[tr_imp]], args.perm, rng, usual)
            m = analyse(X, dead, y, C, tr_mod[ok[tr_mod]], 0, rng)
            sel = np.flatnonzero(a["q"] < 0.05)
            rep = {"n_train": a["n"], "bh_discoveries": a["n_bh05"], "bh_discoveries_on_permuted": a.get("null_bh05_mean"),
                   "fwer_discoveries": int(np.nansum(a["p_fwer"] < 0.05)), "null_r_sd": a.get("null_r_sd"),
                   "flattening_rho": a.get("flat_rho"), "flattening_z": a.get("flat_z"),
                   "modern_replication": {"n": m["n"], "rho_of_effects": spearman(a["r"], m["r"]),
                                          "selected_same_sign_p05": int(np.sum((np.sign(m["r"][sel]) == np.sign(a["r"][sel])) & (m["p"][sel] < 0.05))),
                                          "selected": int(len(sel))},
                   **top(a, args.top, usual)}
            bf = {"r": a["beyond_flat"], "q": a["q"], "p_fwer": a["p_fwer"]}
            rep["beyond_flattening"] = top(bf, args.top, usual)
            if args.holdout:
                h = analyse(X, dead, y, C, ho_imp[ok[ho_imp]], 0, rng)
                one = np.where(np.sign(h["r"][sel]) == np.sign(a["r"][sel]), h["p"][sel] / 2, 1 - h["p"][sel] / 2)
                holm = np.minimum(1, np.maximum.accumulate(np.sort(one) * (len(one) - np.arange(len(one))))) if len(one) else one
                passed = np.zeros(len(sel), bool); passed[np.argsort(one)] = holm < 0.05
                rep["holdout"] = {"n": h["n"], "rho_of_effects_all_experts": spearman(a["r"], h["r"]),
                                  "selected": int(len(sel)), "same_sign": int(np.sum(np.sign(h["r"][sel]) == np.sign(a["r"][sel]))),
                                  "confirmed_holm": [EXPERT_NAME[i] for i in sel[passed]],
                                  "per_selected": [{"expert": EXPERT_NAME[i], "r_train": round(float(a["r"][i]), 4),
                                                    "r_holdout": round(float(h["r"][i]), 4), "p_one_sided": float(one[k])}
                                                   for k, i in enumerate(sel)]}
                arrays[f"{fn}|{tn}|r_holdout"] = h["r"]
            arrays[f"{fn}|{tn}|r_train"], arrays[f"{fn}|{tn}|q_train"] = a["r"], a["q"]
            arrays[f"{fn}|{tn}|r_modern"] = m["r"]
            report["results"][f"{fn}|{tn}"] = rep
            hs = f"; holdout: {len(rep['holdout']['confirmed_holm'])} of {len(sel)} confirmed" if args.holdout else ""
            log(f"  {fn:12s} {tn:12s} BH {a['n_bh05']:4d} (permuted {a.get('null_bh05_mean', 0):.1f}), FWER {rep['fwer_discoveries']:4d}, "
                f"flattening rho {a.get('flat_rho', float('nan')):+.2f} (z {a.get('flat_z', float('nan')):+.1f}), "
                f"modern rho {rep['modern_replication']['rho_of_effects']:+.2f}{hs}")
    return report, arrays


def selftest(args):
    """Synthetic sizes: per-LoRA common factor, per-module base size, noise. Planted: 4 experts grow with the
    'short' target, 4 shrink; 'marker' has no planted expert. The pipeline must find the planted experts, keep
    false discoveries near zero, and report ~0 discoveries on 'marker'."""
    rng = np.random.default_rng(0)
    g = rng.normal(0, 0.3, N)
    short = 1 / (1 + np.exp(-(1.5 * g + rng.normal(0, 1, N) - 1.5)))
    marker = rng.beta(2, 40, N)
    zs = (stats.rankdata(short) / N - 0.5) * 3.4                               # ~ unit-SD score of short
    mu = rng.normal(0, 1, NM)
    logs = mu[None, :] + g[:, None] + rng.normal(0, 0.35, (N, NM))
    up, down = [(11, 32), (11, 5), (13, 18), (17, 35)], [(20, 3), (24, 7), (28, 1), (30, 12)]
    for (l, e), s in [(x, +0.12) for x in up] + [(x, -0.12) for x in down]:
        for base in (4, 44):
            logs[:, l * 84 + base + e] += s * zs
    dead_one = 0 * 84 + 4 + 9
    m2 = np.exp(2 * logs)
    m2[:, dead_one] = 0.0; m2[:, dead_one + 40] = 0.0                           # one expert never updated
    report, _ = run(m2, None, {"short": short, "marker": marker}, args)
    res = report["results"]
    found = {d["expert"] for d in res["dW.both|short"]["more_in_strong"][:4]} | {d["expert"] for d in res["dW.both|short"]["less_in_strong"][:4]}
    planted = {f"L{l:02d}.E{e:02d}" for l, e in up + down}
    ok = (found == planted and res["dW.both|short"]["bh_discoveries"] >= 8 and res["dW.both|marker"]["bh_discoveries"] <= 2
          and res["dW.both|short"]["bh_discoveries_on_permuted"] <= 2)
    print(f"\nselftest: planted {sorted(planted)}\n          top found {sorted(found)}"
          f"\n          short BH {res['dW.both|short']['bh_discoveries']}, marker BH {res['dW.both|marker']['bh_discoveries']}"
          f"\n          -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sv"); ap.add_argument("--targets"); ap.add_argument("--e-parts"); ap.add_argument("--out")
    ap.add_argument("--target-names", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--all-features", action="store_true", help="also gate_up / down alone, and W / L parts if --e-parts")
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--drop-collapsed", action="store_true", help="sensitivity: drop LoRAs with short > 0.5")
    ap.add_argument("--holdout", action="store_true", help="score the TRAIN-selected experts on HOLDOUT (do this once)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        args.perm = min(args.perm, 300)
        raise SystemExit(0 if selftest(args) else 1)
    assert args.sv and args.targets and args.out, "--sv, --targets and --out are required (or --selftest)"
    os.makedirs(args.out, exist_ok=True)
    m2 = load_m2(args.sv); log("sv.npy read")
    parts = load_e_parts(args.e_parts) if args.e_parts else None
    targets = load_targets(args.targets, args.target_names.split(","))
    report, arrays = run(m2, parts, targets, args)
    tag = ("holdout" if args.holdout else "train") + ("_nocollapse" if args.drop_collapsed else "")
    report["utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(os.path.join(args.out, f"stage_a_{tag}.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(os.path.join(args.out, f"stage_a_{tag}_arrays.npz"), experts=np.array(EXPERT_NAME), **arrays)
    log(f"wrote {args.out}/stage_a_{tag}.json")


if __name__ == "__main__":
    main()
