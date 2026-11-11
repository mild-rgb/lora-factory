"""Track A: grouped cross-validation on dev (folds by LoRA) from the cached matrices, then dev -> val.
Usage (from granite_factory/): python3 team/many-loras-bb/cv2.py CACHE_DIR [C ...] [--hard]"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, "team/many-loras-bb"); sys.path.insert(0, "team")
import feat
import eval_common as ec

CACHE = sys.argv[1]
OUT = "team/many-loras-bb"
SOFT = "--hard" not in sys.argv
CS = [float(a) for a in sys.argv[2:] if not a.startswith("--")] or [10.0]
t0 = time.time()
Xd, Xv = sp.load_npz(f"{CACHE}/X_dev.npz"), sp.load_npz(f"{CACHE}/X_val.npz")
md, mv = json.load(open(f"{CACHE}/meta_dev.json")), json.load(open(f"{CACHE}/meta_val.json"))
bd, bv = np.load(f"{CACHE}/base_dev.npy"), np.load(f"{CACHE}/base_val.npy")
ld, lv = np.array(md["label"]), np.array(mv["label"])
yd, yv = (ld == "TRUE").astype(int), (lv == "TRUE").astype(int)
pd_ = np.array(md["p"], dtype=float)
print(f"dev {Xd.shape} nnz {Xd.nnz}, val {Xv.shape}; soft targets: {SOFT}", flush=True)


def fit(X, y, p, C):
    """Logistic regression. With SOFT, an unsure answer enters twice: as TRUE with weight p, as FALSE with 1 - p."""
    if SOFT:
        mid = (p > 0.02) & (p < 0.98)
        Xa = sp.vstack([X, X[mid]], format="csr")
        ya = np.concatenate([(p >= 0.5).astype(int), (p[mid] < 0.5).astype(int)])
        wa = np.concatenate([np.where(mid, np.maximum(p, 1 - p), 1.0), np.minimum(p[mid], 1 - p[mid])])
    else:
        Xa, ya, wa = X, y, None
    return LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=300, random_state=0).fit(Xa, ya, sample_weight=wa)


def show(tag, p, lab, base):
    y = (lab == "TRUE").astype(int)
    print(f"{tag}: auc {roc_auc_score(y, p):.4f}  logloss {log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}  "
          f"agreement if stage 1 labels everything {100 * ((p > 0.5) == y).mean():.2f}%")
    cv = feat.curve(p, lab)
    for r in cv:
        print(f"    error budget {100 * r['budget']:.2f}%: to Gemma {100 * r['gemma_share']:5.1f}%  agreement "
              f"{100 * r['agreement']:.2f}%  (FALSE below {r['lo']:.3f}, TRUE above {r['hi']:.3f}; "
              f"{r['n_false']} FALSE, {r['n_true']} TRUE)")
    nm = (base[:, 1] > 0) | (base[:, 4] > 0) | (base[:, 6] > 0)
    for nme, mask in (("name/marker", nm), ("neither", ~nm)):
        print(f"    {nme}: {100 * mask.mean():.1f}% of answers, TRUE rate {100 * y[mask].mean():.1f}%, auc "
              f"{roc_auc_score(y[mask], p[mask]):.4f}, wrong at 0.5: {100 * ((p[mask] > 0.5) != y[mask]).mean():.2f}%")
    return cv


fold = np.array([ec.dev_fold(m) for m in md["model"]])      # the team's fixed folds (by LoRA)
rd, rv = ec.rows_of_ids(md["id"]), ec.rows_of_ids(mv["id"])
TAG = ("soft" if SOFT else "hard") + "".join(a[6:] for a in sys.argv if a.startswith("--tag="))
res = {}
for C in CS:
    oof = np.zeros(len(yd))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        oof[te] = fit(Xd[tr], yd[tr], pd_[tr], C).predict_proba(Xd[te])[:, 1]
        print(f"  C={C} fold {k} done, {time.time() - t0:.0f}s", flush=True)
    res[f"cv_C{C}"] = show(f"dev OOF, C={C}", oof, ld, bd)
    m = fit(Xd, yd, pd_, C)
    pv = m.predict_proba(Xv)[:, 1]
    res[f"val_C{C}"] = show(f"VAL (fit on dev), C={C}", pv, lv, bv)
    full = np.full(ec.N, np.nan, dtype=np.float32)
    full[rd] = oof; full[rv] = pv
    np.save(f"{OUT}/p_devoof_val_{TAG}_C{C:g}.npy", full)
json.dump(res, open(f"{OUT}/cv_results_{TAG}.json", "w"), indent=1)
