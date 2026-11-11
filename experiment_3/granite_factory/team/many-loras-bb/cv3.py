"""Track A, stage 1b: a boosted-tree stacker on top of the logistic regression.
The LR score is split into the parts that each feature block adds (words, question words, name characters, opening
words, cased words, hand flags, which-name). The trees see these parts plus the plain hand features and the question,
so they can weigh a block differently for short and long answers or per question.
Out-of-fold everywhere: the tree model for fold k is trained on LR out-of-fold parts of the other folds.
Usage (from granite_factory/): python3 team/many-loras-bb/cv3.py CACHE_DIR C TAG"""
import json, sys, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, "team/many-loras-bb"); sys.path.insert(0, "team")
import feat
import eval_common as ec

CACHE, C, TAG = sys.argv[1], float(sys.argv[2]), sys.argv[3]
OUT = "team/many-loras-bb"
t0 = time.time()
Xd, Xv = sp.load_npz(f"{CACHE}/X_dev.npz").tocsr(), sp.load_npz(f"{CACHE}/X_val.npz").tocsr()
md, mv = json.load(open(f"{CACHE}/meta_dev.json")), json.load(open(f"{CACHE}/meta_val.json"))
bd, bv = np.load(f"{CACHE}/base_dev.npy"), np.load(f"{CACHE}/base_val.npy")
fz = pickle.load(open(f"{CACHE}/fz.pkl", "rb"))
ld, lv = np.array(md["label"]), np.array(mv["label"])
yd, yv = (ld == "TRUE").astype(int), (lv == "TRUE").astype(int)
pd_ = np.array(md["p"], dtype=float)
fold = np.array([ec.dev_fold(m) for m in md["model"]])
rd, rv = ec.rows_of_ids(md["id"]), ec.rows_of_ids(mv["id"])
BOUNDS = feat.block_bounds(fz, Xd.shape[1])
print("blocks:", BOUNDS, flush=True)


def fit_lr(X, p):
    mid = (p > 0.02) & (p < 0.98)
    Xa = sp.vstack([X, X[mid]], format="csr")
    ya = np.concatenate([(p >= 0.5).astype(int), (p[mid] < 0.5).astype(int)])
    wa = np.concatenate([np.where(mid, np.maximum(p, 1 - p), 1.0), np.minimum(p[mid], 1 - p[mid])])
    return LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=300, random_state=0).fit(Xa, ya, sample_weight=wa)


def fit_gbm(F, p, seed=0):
    # soft targets again: an unsure answer enters twice
    mid = (p > 0.02) & (p < 0.98)
    Fa = np.vstack([F, F[mid]])
    ya = np.concatenate([(p >= 0.5).astype(int), (p[mid] < 0.5).astype(int)])
    wa = np.concatenate([np.where(mid, np.maximum(p, 1 - p), 1.0), np.minimum(p[mid], 1 - p[mid])])
    g = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, early_stopping=False, random_state=seed,
                                       categorical_features=[F.shape[1] - 1])
    return g.fit(Fa, ya, sample_weight=wa)


# ---- LR parts, out of fold on dev, and dev -> val
Pd = np.zeros((len(yd), len(BOUNDS) + 1), dtype=np.float32)
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    m = fit_lr(Xd[tr], pd_[tr])
    Pd[te] = feat.lr_parts(m, Xd[te], BOUNDS)
    print(f"  LR fold {k} done, {time.time() - t0:.0f}s", flush=True)
m_all = fit_lr(Xd, pd_)
Pv = feat.lr_parts(m_all, Xv, BOUNDS)
qd, qv = np.array(md["qi"]), np.array(mv["qi"])
Fd, Fv = feat.stack_feats(Pd, bd, qd), feat.stack_feats(Pv, bv, qv)
np.save(f"{CACHE}/F_dev_{TAG}.npy", Fd); np.save(f"{CACHE}/F_val_{TAG}.npy", Fv)

# ---- stacker, out of fold on dev, and dev -> val
sig = lambda z: 1 / (1 + np.exp(-z))
oof = np.zeros(len(yd))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    oof[te] = fit_gbm(Fd[tr], pd_[tr]).predict_proba(Fd[te])[:, 1]
g_all = fit_gbm(Fd, pd_)
pv = g_all.predict_proba(Fv)[:, 1]
pickle.dump({"model": m_all, "gbm": g_all, "bounds": BOUNDS, "C": C}, open(f"{CACHE}/stack_{TAG}.pkl", "wb"))


def show(tag, p, lab, base):
    y = (lab == "TRUE").astype(int)
    nm = (base[:, 1] > 0) | (base[:, 4] > 0) | (base[:, 6] > 0)
    print(f"{tag}: auc {roc_auc_score(y, p):.4f}  logloss {log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}  alone "
          f"{100 * ((p > 0.5) == y).mean():.2f}%  | name/marker auc {roc_auc_score(y[nm], p[nm]):.4f}  neither auc "
          f"{roc_auc_score(y[~nm], p[~nm]):.4f}")
    for r in feat.curve(p, lab, grid=[0.005, 0.008, 0.01, 0.015]):
        print(f"    error budget {100 * r['budget']:.2f}%: to Gemma {100 * r['gemma_share']:5.1f}% (free routing, best case)")


show("dev OOF  LR  ", sig(Pd[:, -1]), ld, bd)
show("dev OOF  LR+trees", oof, ld, bd)
show("VAL      LR  ", sig(Pv[:, -1]), lv, bv)
show("VAL      LR+trees", pv, lv, bv)
full = np.full(ec.N, np.nan, dtype=np.float32)
full[rd] = oof; full[rv] = pv
np.save(f"{OUT}/p_devoof_val_{TAG}.npy", full)
print("wrote", f"{OUT}/p_devoof_val_{TAG}.npy", f"{time.time() - t0:.0f}s")
