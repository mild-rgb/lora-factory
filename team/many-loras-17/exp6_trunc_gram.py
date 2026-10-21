"""Exp 6a: Grams of rank-truncated updates, from many-loras-ef's top-4 singular vectors (layers 12-23).

Hypothesis: at ranks 32/64 the strength signal sits in the top few directions of dW and is diluted by many weak ones,
so the full-dW cosine loses it. Truncate each dW to its top k directions, dW_k = sum_{a<k} s_a u_a v_a^T, and use
<dW_i,k, dW_j,k> = sum_{a,b<k} s_ia s_jb (u_ia . u_jb)(v_ia . v_jb). Joint sign flips of (u, v) cancel, so this is
invariant to the A/B gauge (up to ties in singular values at the cut).
Writes fullgram/T{k}_L{ll}.npy ([2489, 2489] float32) for k in (1, 2, 4).
Usage: python exp6_trunc_gram.py 14 22   (first and last layer)
"""
import dlib  # noqa: F401  (first import: pins BLAS to 1 thread before numpy/scipy/sklearn load)
import os, sys, time
import numpy as np
from dlib import full_path, FULL_DIR

lo, hi = int(sys.argv[1]), int(sys.argv[2])
S = np.load(full_path("svals.npy"), mmap_mode="r")          # [N, 36, 64]
KS = (1, 2, 4)
for l in range(lo, hi + 1):
    if all(os.path.exists(os.path.join(FULL_DIR, f"T{k}_L{l:02d}.npy")) for k in KS):
        continue
    t0 = time.time()
    U = np.load(full_path(f"topvec/u_L{l:02d}.npy")).astype(np.float32)   # [N, 4, 4096]
    V = np.load(full_path(f"topvec/v_L{l:02d}.npy")).astype(np.float32)   # [N, 4, 12288]
    N = U.shape[0]
    s = np.asarray(S[:, l, :4], dtype=np.float32)                        # [N, 4]
    Uf, Vf = U.reshape(N * 4, -1), V.reshape(N * 4, -1)
    out = {k: np.zeros((N, N), np.float32) for k in KS}
    B = 400
    for i0 in range(0, N, B):
        i1 = min(N, i0 + B)
        P = (Uf[i0 * 4:i1 * 4] @ Uf.T) * (Vf[i0 * 4:i1 * 4] @ Vf.T)       # [(i1-i0)*4, N*4]
        P = P.reshape(i1 - i0, 4, N, 4) * s[i0:i1, :, None, None] * s[None, None, :, :]
        for k in KS:
            out[k][i0:i1] = P[:, :k, :, :k].sum((1, 3))
    for k in KS:
        np.save(os.path.join(FULL_DIR, f"T{k}_L{l:02d}.npy"), out[k])
    # check: diag of T4 = sum of the top-4 s^2
    err = np.abs(np.diag(out[4]) - (s ** 2).sum(1)).max() / (s ** 2).sum(1).max()
    print(f"layer {l}: {time.time() - t0:.0f}s, diag check rel err {err:.1e}", flush=True)
    del U, V, Uf, Vf, P, out
