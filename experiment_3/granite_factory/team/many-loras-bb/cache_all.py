"""Track A: with a frozen featurizer, compute the feature matrix of every chunk once and keep it on disk, so that
retraining and scoring all 750,000 answers take minutes. One chunk in memory at a time.
Usage (from granite_factory/): python3 team/many-loras-bb/cache_all.py FZ.pkl CACHE_DIR [K N]"""
import sys, pickle, time
import numpy as np, scipy.sparse as sp
sys.path.insert(0, "team/many-loras-bb"); sys.path.insert(0, "team")
import feat
import eval_common as ec

fz = pickle.load(open(sys.argv[1], "rb"))
D = sys.argv[2]
K, NW = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
t0 = time.time()
for c in range(K, ec.N_CHUNKS, NW):
    ids, qi, text, hit = [], [], [], []
    for i, m, q, a, h in feat.read_chunk(c):
        ids.append(i); qi.append(q); text.append(a); hit.append(h)
    X = fz.transform(qi, text, hit).astype(np.float32)
    sp.save_npz(f"{D}/X_{c:03d}.tmp.npz", X, compressed=False)
    np.save(f"{D}/rows_{c:03d}.npy", ec.rows_of_ids(ids))
    import os; os.replace(f"{D}/X_{c:03d}.tmp.npz", f"{D}/X_{c:03d}.npz")
    if (c // NW) % 5 == 0:
        print(f"chunk {c}: {time.time() - t0:.0f}s", flush=True)
print("done", time.time() - t0)
