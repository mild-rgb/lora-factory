"""H16 (named up front, all results logged): subspace-overlap kernels.

Building on H15's vsq_mid (V-side sign-invariant quartic, holdout 0.3178 ~ in_load 0.3078):
  usq_mid          U-side mirror (output directions) — completes the input/output asymmetry test
  gcos_mid         cosine dW Gram L12-23 under my protocol (reference; ef's object)
  in_load+vsq      does subspace overlap ADD to per-neuron magnitudes?
  gcos_mid+vsq     does it ADD to the best Gram?
  gcos_mid+in_load reference combo (ef round 2 tests weights; here: unit sum)
Evaluated on holdout; the promising ones also on holdout_plus (refs there: in_load 0.352,
gcos_mid 0.417 by ef).
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
unit = lambda K: K / float(np.mean(np.diag(K)))


def k_quartic_u(l):
    u = np.load(mc.feat(f"topvec/u_L{l}.npy"), mmap_mode="r")       # (N, 4, 4096)
    sv = np.load(mc.feat("svals.npy"), mmap_mode="r")
    s = np.asarray(sv[:, l, :4], dtype=np.float32)
    n2 = (np.asarray(sv[:, l, :], dtype=np.float64) ** 2).sum(1)
    N = len(s)
    W = np.asarray(u, dtype=np.float32).reshape(N * 4, 4096)
    P = W @ W.T
    np.square(P, out=P)
    q = (s / (n2[:, None] + 1e-12)).astype(np.float32).reshape(N * 4)
    K = (q[:, None] * P * q[None, :]).reshape(N, 4, N, 4).sum((1, 3))
    del P, W
    return K / np.mean(np.diag(K))


def k_gcos(l):
    G = np.load(mc.feat(f"gram/G_L{l:02d}.npy")).astype(np.float64)
    d = np.sqrt(np.diag(G))
    K = G / np.outer(d, d)
    return K / np.mean(np.diag(K))


def build(name, fn):
    p = os.path.join(os.environ["TRACKC_BLK"], f"K_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    K = 0
    for l in MID:
        K = K + fn(l)
        log(f"  {name} L{l}")
    np.save(p, K.astype(np.float32))
    return K.astype(np.float32)


def main():
    rows = labels()
    y = np.array([r["old_germany_rate"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    grp = mc.seed_groups(rows)
    efblk = os.path.join(TEAM, "many-loras-ef", "blocks")
    in_load = np.load(os.path.join(efblk, "K_in_load.npy"))
    vsq = np.load(os.path.join(os.environ["TRACKC_BLK"], "K_vsq_mid.npy"))
    Ks = {
        "usq_mid": build("usq_mid", k_quartic_u),
        "gcos_mid84": build("gcos_mid84", k_gcos),
    }
    Ks["in_load+vsq"] = unit(in_load) + unit(vsq)
    Ks["gcos_mid+vsq"] = unit(Ks["gcos_mid84"]) + unit(vsq)
    Ks["gcos_mid+in_load"] = unit(Ks["gcos_mid84"]) + unit(in_load)

    for track in ("holdout", "holdout_plus"):
        if track == "holdout":
            idx = np.where(ranks == 8)[0]
            (_, tr8, te8), = splits(track)
            tr, te = idx[tr8], idx[te8]
        else:
            (_, tr, te), = splits(track)
        zt = mc.zscore_by_rank(y, ranks, tr)
        for name, K in Ks.items():
            t0 = time.time()
            p = mc.krr(K, zt, tr, te, grp)
            rho = score(y[te], p)
            log(f"{track} H16 {name:18s} rho {rho:+.4f} ({time.time()-t0:.0f}s)")
            out = os.path.join(HERE, f"preds_{track}_{name.replace('+','_')}.csv")
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "pred"])
                for i, pi in zip(te, p):
                    w.writerow([rows[i]["name"], float(pi)])
            log_result(AUTHOR, f"B:krr_{name}", track, rho,
                       "H16 subspace-overlap; team/many-loras-84/h16_subspace.py")


if __name__ == "__main__":
    main()
