"""Track A: P(TRUE) for all 750,000 answers, streamed chunk by chunk (never more than one chunk in memory).
Usage (from granite_factory/): python3 team/many-loras-bb/predict_all.py FZ.pkl MODEL.pkl OUT.npy [K N]
With K N only the chunks c with c % N == K are scored (to run N workers); merge the parts with np.fmax.
MODEL.pkl holds {"model": fitted classifier, "calib": optional (a, b) for sigmoid(a * logit + b)}.
Output: float32 array of length 750,000 in the team's row order (eval_common.row_of_id)."""
import sys, pickle, time
import numpy as np
sys.path.insert(0, "team/many-loras-bb"); sys.path.insert(0, "team")
import feat
import eval_common as ec

fz = pickle.load(open(sys.argv[1], "rb"))
M = pickle.load(open(sys.argv[2], "rb"))
out = np.full(ec.N, np.nan, dtype=np.float32)
t0 = time.time()
K, NW = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
for c in range(K, ec.N_CHUNKS, NW):
    ids, qi, text, hit = [], [], [], []
    for i, m, q, a, h in feat.read_chunk(c):
        ids.append(i); qi.append(q); text.append(a); hit.append(h)
    z = M["model"].decision_function(fz.transform(qi, text, hit))
    if M.get("calib"):
        a_, b_ = M["calib"]
        z = a_ * z + b_
    out[ec.rows_of_ids(ids)] = 1 / (1 + np.exp(-z))
    if (c // NW) % 5 == 0:
        print(f"chunk {c}: {time.time() - t0:.0f}s", flush=True)
assert NW > 1 or not np.isnan(out).any()
np.save(sys.argv[3], out)
print("wrote", sys.argv[3], "mean p", float(out.mean()))
