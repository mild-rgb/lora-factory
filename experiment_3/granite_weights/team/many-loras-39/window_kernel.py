#!/usr/bin/env python3
"""Window kernel from the seeds alone (referee many-loras-39): how similar are the STARTING read windows of two
LoRAs, module by module?   k_m(j, i) = | A0_j^m A0_i^m' |_F^2   (8 x 8 overlap of the two random row spaces)
A model that is linear in k can represent "LoRAs whose starting window catches input direction X (any X, in any
module) end up bigger / stronger". Because the seed is randomised, whatever k predicts out of sample is caused by
the starting window. No adapter is read.
Pilot version: attention modules only (q, k, v, o of 32 layers = 128 modules), against the 500 basis LoRAs.
Output: Kwin_att.npy [3000, 500, 128] float32 (module order: layer * 4 + (q, k, v, o)).
Usage (from experiment_3/): PYTHONPATH=/usr/lib/python3/dist-packages python3 granite_weights/team/many-loras-39/window_kernel.py
"""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rebuild_a0 import _kaiming, NL, H, R
torch.set_num_threads(int(os.environ.get("WK_THREADS", "1")))
JOBS = json.load(open(os.path.join(HERE, "..", "..", "..", "granite_factory", "jobs_all.json")))
seeds = [j["seed"] for j in JOBS]
bi = np.load(os.path.join(HERE, "..", "..", "features", "basis_idx.npy"))
out = np.lib.format.open_memmap(os.path.join(HERE, "Kwin_att.npy"), mode="w+", dtype=np.float32, shape=(len(seeds), len(bi), NL * 4))
t0 = time.time()
for l in range(NL):
    for ti, nm in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
        tag = f"L{l}.{nm}"
        A = torch.stack([_kaiming(s, (R, H), H, tag) for s in seeds])                 # [3000, 8, 1536]
        B = A[torch.from_numpy(bi)].reshape(-1, H)                                      # [500 * 8, 1536]
        for c0 in range(0, len(seeds), 500):
            g = A[c0:c0 + 500].reshape(-1, H) @ B.T                                     # [500*8, 500*8]
            k = (g * g).reshape(-1, R, len(bi), R).sum((1, 3))
            out[c0:c0 + 500, :, l * 4 + ti] = k.numpy()
    out.flush()
    print(f"layer {l} done, {time.time() - t0:.0f} s", flush=True)
print("WINDOW_KERNEL_DONE")
