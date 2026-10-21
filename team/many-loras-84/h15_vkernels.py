"""H15: do SIGNED neuron read patterns (V-side directions) beat unsigned magnitudes (in_load)?

Kernels over topvec/v (L12-23, top-4 dirs, sigma-weighted):
  vsgn   signed pattern, dirs re-signed by ucos (train-row reference, label-free, verifier-ok)
  vabs   unsigned pattern |v| (midpoint between vsgn and in_load's per-neuron RMS)
  vsq    sign-invariant quartic kernel: K_ij = sum_kk' s_ik s_jk' (v_ik . v_jk')^2
Reference: plain in_load = 0.3078 on holdout with this exact protocol.
"""
import os, sys, time
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


def layer_data(l):
    v = np.load(mc.feat(f"topvec/v_L{l}.npy"), mmap_mode="r")      # (N, 4, 12288) fp16
    sv = np.load(mc.feat("svals.npy"), mmap_mode="r")
    s = np.asarray(sv[:, l, :4], dtype=np.float32)                  # (N, 4)
    misc = np.load(mc.feat("lens/lens_misc.npz"))
    uc = misc["ucos"][:, l, :4].astype(np.float32)
    sgn = np.sign(uc)
    sgn[np.abs(uc) <= 0.02] = 1.0
    n2 = (np.asarray(sv[:, l, :], dtype=np.float64) ** 2).sum(1)    # ||dW||^2 at layer l
    return v, s, sgn, np.sqrt(n2).astype(np.float32)


def k_pattern(l, signed):
    v, s, sgn, nrm = layer_data(l)
    w = np.sqrt(s) * (sgn if signed else 1.0) / nrm[:, None]
    X = (np.asarray(v, dtype=np.float32) * w[:, :, None]).reshape(len(s), -1)
    if not signed:
        X = np.abs(X)
    K = X @ X.T
    return K / np.mean(np.diag(K))


def k_quartic(l):
    v, s, sgn, nrm = layer_data(l)
    N = len(s)
    W = np.asarray(v, dtype=np.float32).reshape(N * 4, 12288)
    P = W @ W.T
    np.square(P, out=P)                                             # (v_ik . v_jk')^2
    q = (s / (nrm[:, None] ** 2 + 1e-12))                           # sigma / ||dW||^2
    Q = q.reshape(N * 4)
    K = (Q[:, None] * P * Q[None, :]).reshape(N, 4, N, 4).sum((1, 3))
    del P, W
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
    Ks = {
        "vsgn_mid": build("vsgn_mid", lambda l: k_pattern(l, True)),
        "vabs_mid": build("vabs_mid", lambda l: k_pattern(l, False)),
        "vsq_mid": build("vsq_mid", k_quartic),
    }
    idx = np.where(ranks == 8)[0]
    (_, tr8, te8), = splits("holdout")
    tr, te = idx[tr8], idx[te8]
    z = mc.zscore_by_rank(y, ranks, tr)
    for name, K in Ks.items():
        p = mc.krr(K, z, tr, te, grp)
        rho = score(y[te], p)
        log(f"holdout H15 {name:10s} rho {rho:+.4f}")
        log_result(AUTHOR, f"B:krr_{name}", "holdout", rho,
                   "H15 signed-vs-unsigned read patterns; ref in_load 0.3078; team/many-loras-84/h15_vkernels.py")


if __name__ == "__main__":
    main()
