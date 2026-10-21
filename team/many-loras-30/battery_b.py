"""Assignment (b): why does the weights-only signal stop near 0.43?
B1 learning-curve fits/extrapolation (ef's lcurve.csv).
B2 activation vs weight-kernel sample efficiency on MATCHED training subsets
   (activations are a diagnostic only, never a weights-only predictor).
B3 variogram of rate vs weight distance + twin-pair analysis (smoothness budget).
Outputs go to stdout and hypotheses.md.
"""
import csv, glob, os, re, sys
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_common import labels, splits, score
from hyp_lib import load_gram, cosine_gram, KRR, _cache, inner_cv_score

OUT = []
def log(s):
    print(s, flush=True)
    OUT.append(s)

G, y = load_gram()
C = cosine_gram(G)
L8 = labels(8)
names8 = [r["name"] for r in L8]
(_, TR, TE), = splits("holdout")
KMID = C[list(range(12, 24))].mean(0).astype(np.float64)

# ---------- B1: learning curves ----------
log("## B1 learning curves (ef lcurve.csv)")
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "many-loras-ef", "lcurve.csv"))))
for curve in ("r8", "r16", "r32", "r64"):
    pts = [(int(r["n_train"]), float(r["test_rho"])) for r in rows if r["curve"] == curve]
    ns = np.array(sorted(set(n for n, _ in pts)))
    ms = np.array([np.mean([rho for n2, rho in pts if n2 == n]) for n in ns])
    try:
        # rho(n) = a - b * n^(-1/2)
        (a, b), _ = curve_fit(lambda n, a, b: a - b / np.sqrt(n), ns, ms,
                              p0=(0.6, 3.0), maxfev=10000)
        n_for = lambda t: (b / (a - t)) ** 2 if a > t else np.inf
        log(f"{curve}: points {list(zip(ns.tolist(), np.round(ms,3).tolist()))}")
        log(f"  fit rho = {a:.3f} - {b:.2f}/sqrt(n); asymptote {a:.3f}; "
            f"n for 0.50: {n_for(0.50):,.0f}; n for 0.55: {n_for(0.55):,.0f}")
    except Exception as e:
        log(f"{curve}: fit failed {e}")

# ---------- B2: matched-subset sample efficiency ----------
log("## B2 activation vs weight kernel, matched subsets (diagnostic)")
feat_dir = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mild-rgb--lora-factory/snapshots/*/factory_more/features"))[0]
import torch
A34 = np.zeros((601, 4096), dtype=np.float64)
for i, nm in enumerate(names8):
    d = torch.load(os.path.join(feat_dir, nm + ".pt"), map_location="cpu", weights_only=True)
    A34[i] = np.asarray(d["act_eval"][34].float().tolist())
log(f"activation features loaded: {A34.shape}, ||mean|| {np.linalg.norm(A34.mean(0)):.3f}")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

dlib = _cache.get("dlib")
if dlib is None:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("dlib", os.path.join(HERE, "..", "many-loras-17", "dlib.py"))
    dlib = ilu.module_from_spec(spec); spec.loader.exec_module(dlib)
    _cache["dlib"] = dlib; _cache["order_groups"] = dlib.order_groups()
og = _cache["order_groups"]

def act_model(tr):
    best, bsc = None, -9
    for al in (10.0, 100.0, 1000.0, 10000.0):
        sc = []
        for ite in dlib.group_folds(og[tr], k=3, seed=5):
            itr = np.setdiff1d(np.arange(len(tr)), ite)
            sca = StandardScaler().fit(A34[tr[itr]])
            m = Ridge(alpha=al).fit(sca.transform(A34[tr[itr]]), y[tr[itr]])
            sc.append(score(y[tr[ite]], m.predict(sca.transform(A34[tr[ite]]))))
        if np.mean(sc) > bsc:
            bsc, best = float(np.mean(sc)), al
    sca = StandardScaler().fit(A34[tr])
    m = Ridge(alpha=best).fit(sca.transform(A34[tr]), y[tr])
    return lambda te: m.predict(sca.transform(A34[te])), best

def wk_model(tr):
    best, bsc = None, -9
    for al in (1e-4, 1e-3, 1e-2, 0.1):
        sc = []
        for ite in dlib.group_folds(og[tr], k=3, seed=5):
            itr = np.setdiff1d(np.arange(len(tr)), ite)
            m = KRR(KMID, al, center=True); m.fit(tr[itr], y[tr[itr]])
            sc.append(score(y[tr[ite]], m.predict(tr[ite])))
        if np.mean(sc) > bsc:
            bsc, best = float(np.mean(sc)), al
    m = KRR(KMID, best, center=True); m.fit(tr, y[tr])
    return lambda te: m.predict(te), best

rng = np.random.default_rng(0)
tr_groups = {}
for i in TR:
    m = (re.fullmatch(r"former_i\d+_o(\d+)", names8[i]) or re.fullmatch(r"former_s0(\d)", names8[i]))
    tr_groups.setdefault(f"o{m.group(1)}" if m else names8[i], []).append(i)
glist = list(tr_groups.values())
for n_target in (40, 80, 161):
    accs_a, accs_w = [], []
    for rep in range(5 if n_target < 161 else 1):
        perm = rng.permutation(len(glist))
        sub = []
        for gi in perm:
            if len(sub) >= n_target:
                break
            sub += glist[gi]
        sub = np.array(sorted(sub[:n_target])) if n_target < 161 else TR
        pa, al_a = act_model(sub)
        pw, al_w = wk_model(sub)
        accs_a.append(score(y[TE], pa(TE)))
        accs_w.append(score(y[TE], pw(TE)))
    log(f"n={n_target}: activations {np.mean(accs_a):.3f} (sd {np.std(accs_a):.3f}) | "
        f"weight kernel {np.mean(accs_w):.3f} (sd {np.std(accs_w):.3f})")

# ---------- B3: variogram + twins ----------
log("## B3 variogram of rate vs weight distance (rank-8, mid-layer cosine metric)")
D = np.sqrt(np.clip(2 - 2 * KMID, 0, None))
iu = np.triu_indices(601, 1)
dv, gv = D[iu], 0.5 * (y[iu[0]] - y[iu[1]]) ** 2
sill = y.var()
noise = float(np.mean(y * (1 - y) / 250))       # binomial label noise per measurement
bins = np.quantile(dv, np.linspace(0, 1, 13))
log(f"sill (var y) {sill:.5f}; binomial noise floor gamma {noise:.5f} "
    f"({noise/sill:.1%} of sill)")
for k in range(12):
    m = (dv >= bins[k]) & (dv < bins[k + 1] + (k == 11) * 1e9)
    log(f"  d in [{bins[k]:.3f},{bins[k+1]:.3f}): gamma {gv[m].mean():.5f} "
        f"({gv[m].mean()/sill:.2f} sill), n {m.sum()}")
# twins
tw = []
for k in range(1, 10):
    a, b = f"former_i0{k}_o0{k}", f"former_s0{k}"
    ia = names8.index(a) if a in names8 else None
    if ia is None:
        a2 = f"former_i{k}_o{k}"
        ia = names8.index(a2) if a2 in names8 else None
    ib = names8.index(b) if b in names8 else None
    if ia is not None and ib is not None:
        tw.append((ia, ib))
gam_tw = np.mean([0.5 * (y[i] - y[j]) ** 2 for i, j in tw])
d_tw = np.mean([D[i, j] for i, j in tw])
log(f"twins (n={len(tw)}): mean d {d_tw:.3f}, gamma {gam_tw:.5f} ({gam_tw/sill:.2f} sill) "
    f"vs noise floor {noise:.5f} -> excess (chaos at d~{d_tw:.2f}) {gam_tw-noise:.5f}")
# kernel-predicted twin differences
m = KRR(KMID, 1e-3, center=True)
pred_d, act_d = [], []
for i, j in tw:
    rest = np.setdiff1d(np.arange(601), [i, j])
    tr2 = np.intersect1d(rest, np.concatenate([TR, TE]))  # all other rows
    m.fit(tr2, y[tr2])
    p = m.predict(np.array([i, j]))
    pred_d.append(p[0] - p[1]); act_d.append(y[i] - y[j])
sgn = np.mean(np.sign(pred_d) == np.sign(act_d))
log(f"kernel-predicted twin deltas: corr {spearmanr(pred_d, act_d)[0]:.3f}, "
    f"sign agreement {sgn:.2f} (n={len(tw)}) "
    f"[diagnostic only: trains on all rows except the pair]")

with open(os.path.join(HERE, "hypotheses.md"), "a") as f:
    f.write("\n## Assignment (b): sample complexity study (battery_b.py)\n```\n"
            + "\n".join(OUT) + "\n```\n")
