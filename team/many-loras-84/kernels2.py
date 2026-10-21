"""Track B kernels round 2: H12 (era-specificity weights) + H13 (combos on holdout_plus).

H12: input-neuron kernels weighted by era SPECIFICITY (era mass normalized by neuron size),
     evaluated on holdout. Controls: control-group specificity (should be ~plain if era-ness
     is irrelevant).
H13: unit-sum combos of ef's in_load kernel with my lens kernels, on holdout_plus
     (reference: ef's plain in_load = 0.352 verified).
Selection stays honest: each hypothesis names its primary comparison up front; everything logged.
"""
import csv, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)
sys.path.insert(0, os.path.join(TEAM, "many-loras-ef"))
os.environ.setdefault("TRACKC_BLK", os.path.join(HERE, "blocks"))
import models_c as mc
from eval_common import labels, splits, score, log_result

MID = list(range(12, 24))
AUTHOR = "many-loras-84"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def norms2():
    sv = np.load(mc.feat("svals.npy")).astype(np.float64)
    return (sv ** 2).sum(2)


def x_in_spec(l, num, den):
    bt = np.load(mc.feat("base_tables.npz"))
    groups = ["city_former", "city_modern", "era", "year", "control"]
    gm = bt["neuron_groupmass"].astype(np.float64)
    cn = bt["base_norms"][0].astype(np.float64)
    nv = gm[groups.index(num), l] if num in groups else cn[l]
    dv = gm[groups.index(den), l] if den in groups else cn[l]
    w = nv / (dv + 1e-9)
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
    np.save(p, K.astype(np.float32))
    return K.astype(np.float32)


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    grp = mc.seed_groups(rows)
    unit = lambda K: K / float(np.mean(np.diag(K)))

    myblk = os.environ["TRACKC_BLK"]
    efblk = os.path.join(TEAM, "many-loras-ef", "blocks")
    in_load = np.load(os.path.join(efblk, "K_in_load.npy"))

    # --- H12 kernels (holdout)
    Ks12 = {
        "inspec_era_cn": build_K("inspec_era_cn", lambda l: x_in_spec(l, "era", "cn"), MID),
        "inspec_era_ctl": build_K("inspec_era_ctl", lambda l: x_in_spec(l, "era", "control"), MID),
        "inspec_ctl_cn": build_K("inspec_ctl_cn", lambda l: x_in_spec(l, "control", "cn"), MID),  # control
    }
    idx = np.where(ranks == 8)[0]
    (_, tr8, te8), = splits("holdout")
    tr, te = idx[tr8], idx[te8]
    z = mc.zscore_by_rank(y, ranks, tr)
    for name, K in Ks12.items():
        p = mc.krr(K, z, tr, te, grp)
        rho = score(y[te], p)
        log(f"holdout H12 {name:16s} rho {rho:+.4f}")
        log_result(AUTHOR, f"B:krr_{name}", "holdout", rho,
                   "H12 era-specificity; team/many-loras-84/kernels2.py")

    # --- H13 combos (holdout_plus)
    K_perdir = np.load(os.path.join(myblk, "K_perdir_mid.npy"))
    K_tok = np.load(os.path.join(myblk, "K_tok_mid.npy"))
    Ks13 = {
        "in_load+perdir": unit(in_load) + unit(K_perdir),
        "in_load+tok_mid": unit(in_load) + unit(K_tok),
        "in_load+inspec_era_cn": unit(in_load) + unit(Ks12["inspec_era_cn"]),
        "in_load+0.25perdir": unit(in_load) + 0.25 * unit(K_perdir),
    }
    (_, trp, tep), = splits("holdout_plus")
    zp = mc.zscore_by_rank(y, ranks, trp)
    for name, K in Ks13.items():
        t0 = time.time()
        p = mc.krr(K, zp, trp, tep, grp)
        rho = score(y[tep], p)
        log(f"holdout_plus H13 {name:22s} rho {rho:+.4f} ({time.time()-t0:.0f}s)")
        out = os.path.join(HERE, f"preds_hplus_{name.replace('+','_')}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "pred"])
            for i, pi in zip(tep, p):
                w.writerow([rows[i]["name"], float(pi)])
        log_result(AUTHOR, f"B:krr_{name}", "holdout_plus", rho,
                   "H13 combo; ref ef in_load 0.352; team/many-loras-84/kernels2.py")


if __name__ == "__main__":
    main()
