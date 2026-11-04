"""Interpret the detector input patterns v*_l of the dissected gcos_mid predictor (task from 75).

A. Extract detectors (u*_l, v*_l, top-2) per layer L12-23 from the full-train dual fit. Save.
B. Neuron statistics of v*_l: participation ratio, base-table correlations (norms, era), overlap
   with the SHARED update's read pattern (mean |v1| over train rank-8 rows).
D. Cross-layer consistency via the base write-map: q_l = W_down^l v*_l approximated with the
   top-64 SVD of base W_down (Pout diag(S) Pin^T); coverage ||Pin^T v*|| reported as validity.
C. Stability: refit duals on 10 group-subsamples (70% of training groups), re-extract v*_l at
   L16/17/18, report |cos(v*_sub, v*_full)|.
E. (75's item 4) Hand-built 12-feature model: f_l(x) = u*_l^T dWhat_x v*_l, ridge with the
   shared protocol -> holdout_plus rho and rho vs the full model's predictions.
All fitting on training rows. ~0.9 GB peak, 1 core, ~30 min.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse.linalg import svds, LinearOperator
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)
sys.path.insert(0, os.path.join(TEAM, "many-loras-ef"))
os.environ.setdefault("TRACKC_BLK", os.path.join(HERE, "blocks"))
import models_c as mc
from eval_common import labels, splits, score

MID = list(range(12, 24))
STAB_LAYERS = (16, 17, 18)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

rows = labels()
y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
grp = mc.seed_groups(rows)
(_, tr, te), = splits("holdout_plus")
z = mc.zscore_by_rank(y, ranks, tr)
N = len(rows)
ALPHA = float(np.load(os.path.join(HERE, "dissect_out.npz"))["alpha"])
p_full_te = np.load(os.path.join(HERE, "dissect_out.npz"))["p_full_te"]

K = np.load(os.path.join(os.environ["TRACKC_BLK"], "K_gcos_mid84.npy")).astype(np.float64)
def duals(sub):  # sub: indices into labels() (training rows used)
    Ktt = K[np.ix_(sub, sub)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    c = np.linalg.solve(Kc + ALPHA * np.eye(len(sub)), z[sub] - z[sub].mean())
    return (c - c.mean()).astype(np.float32)

ct_full = duals(tr)
sv_all = np.asarray(np.load(mc.feat("svals.npy"), mmap_mode="r"), np.float32)
nrm = np.sqrt((sv_all.astype(np.float64) ** 2).sum(2)).astype(np.float32)

def detector(l, sub, ct, k=2):
    u4 = np.asarray(np.load(mc.feat(f"topvec/u_L{l}.npy"), mmap_mode="r"), np.float32)
    v4 = np.asarray(np.load(mc.feat(f"topvec/v_L{l}.npy"), mmap_mode="r"), np.float32)
    s4 = sv_all[:, l, :4]
    w = ct[:, None] * s4[sub] / nrm[sub, l][:, None]
    A = (u4[sub] * w[:, :, None]).reshape(-1, 4096).T.copy()
    B = v4[sub].reshape(-1, 12288).T.copy()
    op = LinearOperator((4096, 12288), matvec=lambda x: A @ (B.T @ x),
                        rmatvec=lambda x: B @ (A.T @ x), dtype=np.float32)
    Ud, Sd, Vdt = svds(op, k=k)
    o = np.argsort(-Sd)
    del A, B
    return Ud[:, o], Sd[o], Vdt[o], u4, v4, s4

# ---- A + B + D
bt = np.load(mc.feat("base_tables.npz"))
Pin, Pout, bsv = bt["Pin"], bt["Pout"], bt["base_svals"]
era = bt["neuron_groupmass"][2]
bnorm = bt["base_norms"]
dets, qvecs = {}, {}
feat12 = np.zeros((N, 12), np.float64)
for i, l in enumerate(MID):
    Ud, Sd, Vdt, u4, v4, s4 = detector(l, tr, ct_full)
    ustar, vstar = Ud[:, 0], Vdt[0]
    dets[l] = (ustar, Sd, vstar, Vdt[1])
    av = np.abs(vstar)
    pr = float((av ** 2).sum() ** 2 / (av ** 4).sum())
    tr8 = tr[ranks[tr] == 8]
    shared_read = np.abs(v4[tr8, 0]).mean(0)
    top100 = len(set(np.argsort(-av)[:100]) & set(np.argsort(-shared_read)[:100]))
    log(f"B: L{l}: PR {pr:6.0f}/12288; corr(|v*|): sharedread {spearmanr(av, shared_read)[0]:+.3f}, "
        f"dcolnorm {spearmanr(av, bnorm[0, l])[0]:+.3f}, era {spearmanr(av, era[l])[0]:+.3f}; "
        f"top100 overlap w/ shared {top100}")
    cin = Pin[l].T @ vstar
    cov = float(np.linalg.norm(cin))
    qvecs[l] = (Pout[l] * (bsv[l, :64] * cin)[None, :]).sum(1)
    log(f"D: L{l}: top-64 read-subspace coverage of v* {cov:.3f}")
    # E features for all rows
    pu = u4 @ ustar                      # (N, 4)
    pv = v4 @ vstar
    feat12[:, i] = np.einsum("nk,nk,nk->n", pu, pv, s4) / nrm[:, l]
    del u4, v4

qs = np.stack([qvecs[l] / (np.linalg.norm(qvecs[l]) + 1e-12) for l in MID])
CC = np.abs(qs @ qs.T)
iu = np.triu_indices(12, 1)
adj = [CC[i, i + 1] for i in range(11)]
log(f"D: cross-layer |cos| of write-mapped v*: adjacent mean {np.mean(adj):.3f} "
    f"(min {min(adj):.3f} max {max(adj):.3f}), all-pairs mean {CC[iu].mean():.3f}")
# control: same metric for the SHARED read pattern mapped the same way
np.savez(os.path.join(HERE, "vstar_detectors.npz"),
         **{f"u_L{l}": dets[l][0] for l in MID}, **{f"v_L{l}": dets[l][2] for l in MID},
         **{f"s_L{l}": dets[l][1] for l in MID}, crosslayer=CC)

# ---- C stability
rng = np.random.default_rng(0)
ug = np.unique(grp[tr])
for l in STAB_LAYERS:
    cs = []
    for b in range(10):
        keep = set(rng.choice(ug, int(0.7 * len(ug)), replace=False))
        sub = tr[np.isin(grp[tr], list(keep))]
        Ud, Sd, Vdt, *_ = detector(l, sub, duals(sub), k=1)
        cs.append(abs(float(Vdt[0] @ dets[l][2])))
    log(f"C: L{l}: |cos(v*_sub, v*_full)| over 10 subsamples (70% of groups): "
        f"mean {np.mean(cs):.3f} min {min(cs):.3f} max {max(cs):.3f}")

# ---- E: 12-feature hand-built model
Kf = mc.linear_kernel(feat12, tr)
p12 = mc.krr(Kf, z, tr, te, grp)
rho = score(y[te], p12)
log(f"E: 12-feature model holdout_plus rho {rho:+.4f}; rho vs full preds "
    f"{float(spearmanr(p12, p_full_te)[0]):+.4f}")
sys.path.insert(0, TEAM)
from eval_common import log_result
log_result("many-loras-84", "B:vstar_12features", "holdout_plus", rho,
           "hand-built u*xv* projections per layer, detectors fit on training rows; team/many-loras-84/interpret_vstar.py")
