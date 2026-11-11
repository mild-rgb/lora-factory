"""Track A: fit the featurizer on dev + val text (no labels are used here) and cache the matrices.
Usage (from granite_factory/): python3 team/many-loras-bb/build.py CACHE_DIR"""
import sys, time, pickle, json
import numpy as np, scipy.sparse as sp
sys.path.insert(0, "team/many-loras-bb")
import feat
C = sys.argv[1]
t0 = time.time()
dev, val = feat.load_split()
if "--devonly" in sys.argv:      # deployment-faithful: val text is not seen by the vocabulary / IDF
    fz = feat.Featurizer().fit(dev["qi"], dev["text"], dev["hit"])
else:
    fz = feat.Featurizer().fit(dev["qi"] + val["qi"], dev["text"] + val["text"], dev["hit"] + val["hit"])
print("fit", time.time() - t0, flush=True)
for name, d in (("dev", dev), ("val", val)):
    X = fz.transform(d["qi"], d["text"], d["hit"])
    sp.save_npz(f"{C}/X_{name}.npz", X)
    np.save(f"{C}/base_{name}.npy", feat.dense(d["qi"], d["text"], d["hit"])[1])
    json.dump({k: d[k] for k in ("id", "model", "qi", "label", "p", "src")}, open(f"{C}/meta_{name}.json", "w"))
    print(name, X.shape, X.nnz, time.time() - t0, flush=True)
pickle.dump(fz, open(f"{C}/fz.pkl", "wb"))
print("parts:", [len(v.vocabulary_) for v in (fz.word, fz.qword, fz.char, fz.head, fz.cword)])
