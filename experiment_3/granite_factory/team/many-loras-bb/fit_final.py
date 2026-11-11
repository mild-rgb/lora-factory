"""Track A: fit the logistic regression on a cached matrix and save it for predict_all.py.
Usage (from granite_factory/): python3 team/many-loras-bb/fit_final.py CACHE_DIR C OUT.pkl [--with_val]"""
import json, sys, pickle, warnings
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
CACHE, C, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
X = sp.load_npz(f"{CACHE}/X_dev.npz"); p = np.array(json.load(open(f"{CACHE}/meta_dev.json"))["p"], dtype=float)
if "--with_val" in sys.argv:
    X = sp.vstack([X, sp.load_npz(f"{CACHE}/X_val.npz")], format="csr")
    p = np.concatenate([p, np.array(json.load(open(f"{CACHE}/meta_val.json"))["p"], dtype=float)])
mid = (p > 0.02) & (p < 0.98)          # soft targets: unsure answers enter twice, weights p and 1 - p
Xa = sp.vstack([X, X[mid]], format="csr")
ya = np.concatenate([(p >= 0.5).astype(int), (p[mid] < 0.5).astype(int)])
wa = np.concatenate([np.where(mid, np.maximum(p, 1 - p), 1.0), np.minimum(p[mid], 1 - p[mid])])
m = LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=300, random_state=0).fit(Xa, ya, sample_weight=wa)
pickle.dump({"model": m, "C": C, "n_train": X.shape[0]}, open(OUT, "wb"))
print("saved", OUT, X.shape)
