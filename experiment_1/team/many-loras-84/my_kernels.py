"""Track B kernel models (H11): lens-space kernels, scored with eval_common via models_c's protocol.

Kernels (all per-layer linear kernels, unit mean diagonal, summed over layers):
  tok_mid / tok_wide   probe_total token vectors (1,740 per layer) / ||dW||, layers 12-23 / 4-31
  tok_sq               squared token magnitudes, layers 12-23
  eratok_mid           probe_total restricted to era+city_former+year tokens, layers 12-23
  perdir_mid           SIGNED probe_perdir (top-4 dirs, re-signed by ucos), flattened, layers 12-23
  inera_mid            in_L loadings weighted per neuron by sqrt(neuron_groupmass[era]), layers 12-23
  incityF_mid          same with city_former
Combos (uniform sums of unit-mean-diag kernels): tok_mid+gcos, perdir_mid+gcos, tok_mid+in_load,
  inera_mid+gcos.
Selection: grouped inner CV on training rows only (models_c.krr's nested alpha + my kernel pick),
then ONE holdout evaluation of the selected kernel. Other kernels' holdout numbers are logged as
secondary, clearly marked, for the team's information (same policy as the existing leaderboard).

Usage: python3 my_kernels.py --tracks holdout holdout_plus [--log]
"""
import argparse, csv, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)
sys.path.insert(0, os.path.join(TEAM, "many-loras-ef"))
os.environ.setdefault("TRACKC_BLK", os.path.join(HERE, "blocks"))  # own kernel cache, don't collide
import models_c as mc
from eval_common import labels, splits, score, log_result

MID = list(range(12, 24))
WIDEL = list(range(4, 32))
AUTHOR = "many-loras-84"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def norms2():
    sv = np.load(mc.feat("svals.npy")).astype(np.float64)
    return (sv ** 2).sum(2)


def probe_groups():
    probe = json.load(open(os.path.join(HERE, "probe_tokens.json")))
    ids = np.array(probe["ids"])
    return probe, {g: np.searchsorted(ids, probe["groups"][g]) for g in probe["groups"]}


def x_tok(l, cols=None, sq=False):
    pt = np.load(mc.feat("lens/probe_total.npy"), mmap_mode="r")
    x = np.asarray(pt[:, l], dtype=np.float64)
    if cols is not None:
        x = x[:, cols]
    if sq:
        x = x ** 2
    return x / np.sqrt(norms2()[:, l:l + 1] if not sq else norms2()[:, l:l + 1] ** 2)


def x_perdir(l):
    """Signed top-4 dir loadings at layer l (probe_perdir stores layers 8..27), re-signed by ucos."""
    assert 8 <= l <= 27
    pp = np.load(mc.feat("lens/probe_perdir.npy"), mmap_mode="r")
    misc = np.load(mc.feat("lens/lens_misc.npz"))
    ucos = misc["ucos"].astype(np.float64)  # (N, 36, 4)
    x = np.asarray(pp[:, l - 8], dtype=np.float64)  # (N, 1740, 4)
    sgn = np.sign(ucos[:, l, :])
    sgn[np.abs(ucos[:, l, :]) <= 0.02] = 1.0
    x = (x * sgn[:, None, :]).reshape(len(x), -1)
    return x / np.sqrt(norms2()[:, l:l + 1])


def x_in_weighted(l, group):
    bt = np.load(mc.feat("base_tables.npz"))
    groups = [str(g) for g in bt["groups"]] if "groups" in bt else \
        ["city_former", "city_modern", "era", "year", "control"]
    w = bt["neuron_groupmass"][groups.index(group), l].astype(np.float64)
    w = np.sqrt(w / w.mean())
    x = np.load(mc.feat(f"loadings/in_L{l:02d}.npy")).astype(np.float64) * w[None, :]
    return x / np.sqrt(norms2()[:, l:l + 1])


def build_K(name, fn, layers):
    p = os.path.join(os.environ["TRACKC_BLK"], f"K_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    K = 0
    for l in layers:
        X = fn(l)
        Kl = X @ X.T
        K = K + Kl / np.mean(np.diag(Kl))
        log(f"  {name} L{l}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, K.astype(np.float32))
    return K.astype(np.float32)


def get_kernels():
    probe, gidx = probe_groups()
    era_cols = np.concatenate([gidx["era"], gidx["city_former"], gidx["year"]])
    Ks = {}
    Ks["tok_mid"] = build_K("tok_mid", x_tok, MID)
    Ks["tok_wide"] = build_K("tok_wide", x_tok, WIDEL)
    Ks["tok_sq"] = build_K("tok_sq", lambda l: x_tok(l, sq=True), MID)
    Ks["eratok_mid"] = build_K("eratok_mid", lambda l: x_tok(l, cols=era_cols), MID)
    Ks["perdir_mid"] = build_K("perdir_mid", x_perdir, MID)
    Ks["inera_mid"] = build_K("inera_mid", lambda l: x_in_weighted(l, "era"), MID)
    Ks["incityF_mid"] = build_K("incityF_mid", lambda l: x_in_weighted(l, "city_former"), MID)
    # shared kernels from track C (their cache; read-only — TRACKC_BLK points at MY cache)
    EF_BLK = os.path.join(TEAM, "many-loras-ef", "blocks")
    def ef_kernel(fname):
        p = os.path.join(EF_BLK, fname)
        return np.load(p) if os.path.exists(p) else None
    gcos = ef_kernel("K_gcos.npy")
    inld = ef_kernel("K_in_load.npy")
    unit = lambda K: K / np.mean(np.diag(K))
    if gcos is not None:
        Ks["tok_mid+gcos"] = unit(Ks["tok_mid"]) + unit(gcos)
        Ks["perdir_mid+gcos"] = unit(Ks["perdir_mid"]) + unit(gcos)
        Ks["inera_mid+gcos"] = unit(Ks["inera_mid"]) + unit(gcos)
        Ks["gcos_alone"] = gcos
    if inld is not None:
        Ks["tok_mid+in_load"] = unit(Ks["tok_mid"]) + unit(inld)
    return Ks


def inner_rho(K, y, tr, groups, seeds=(1, 2, 3)):
    """Grouped inner-CV OOF Spearman on training rows (kernel selection score)."""
    rhos = []
    for sd in seeds:
        p = np.full(len(y), np.nan)
        for itr, iva in mc.inner_folds(tr, groups, seed=sd):
            p[tr[iva]] = mc.krr(K, y, tr[itr], tr[iva], groups, seeds=(sd + 10,))
        rhos.append(spearmanr(p[tr], y[tr])[0])
    return float(np.mean(rhos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["holdout"])
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    groups = mc.seed_groups(rows)
    Ks = get_kernels()
    log("kernels:", ", ".join(Ks))

    for track in args.tracks:
        if track == "holdout":
            idx = np.where(ranks == 8)[0]
            sp = [(k, idx[tr], idx[te]) for k, tr, te in splits(track)]
        else:
            sp = list(splits(track))
        (fold, tr, te), = sp  # holdout-style: single split
        z = mc.zscore_by_rank(y, ranks, tr)

        sel, res = None, {}
        for name, K in Ks.items():
            r = inner_rho(K, z, tr, groups)
            res[name] = r
            log(f"{track}: inner {name:16s} rho {r:+.4f}")
            if sel is None or r > res[sel]:
                sel = name
        log(f"{track}: SELECTED {sel} (inner {res[sel]:+.4f})")

        for name in Ks:
            p = mc.krr(Ks[name], z, tr, te, groups)
            rho = score(y[te], p)
            mark = "PRIMARY" if name == sel else "secondary"
            log(f"{track}: holdout-eval {name:16s} rho {rho:+.4f} [{mark}]")
            out = os.path.join(HERE, f"preds_{track}_{name.replace('+','_')}.csv")
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "pred"])
                for i, pi in zip(te, p):
                    w.writerow([rows[i]["name"], float(pi)])
            if args.log:
                log_result(AUTHOR, f"B:krr_{name}", track, rho,
                           f"{mark}; inner {res[name]:+.3f}; team/many-loras-84/my_kernels.py")


if __name__ == "__main__":
    main()
