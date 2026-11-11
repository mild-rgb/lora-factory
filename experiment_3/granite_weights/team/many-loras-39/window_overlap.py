#!/usr/bin/env python3
"""Window overlap from the seeds alone (referee many-loras-39): how well does each LoRA's STARTING read matrix A0
catch the shared read directions of each module?

Every LoRA starts with B = 0 and a random A0 [8, in] that the seed fixes before training (rebuild_a0.py). The
update a module can make to an input direction v at the start is limited to what its 8 rows see of v. For every
LoRA j, module m, mean D and direction k (the right singular vectors of the basis means in features/dirs_f16.npz):
    l2 = sum_i (a_i . v)^2 * in_dim / 8      (1/3 on average for a random window; "how much of v is in the window")
    l1 = sum_i |a_i . v| * sqrt(in_dim) / 8  (the Adam-style version: step sizes do not scale with the overlap)
No adapter is read. Output (row order = jobs_all.json; module order = features/modules.json):
    window_l2.npy, window_l1.npy  [3000, 2688, n_means, 4] float16
Usage (from experiment_3/; needs torch with a working numpy bridge):
    PYTHONPATH=/usr/lib/python3/dist-packages python3 granite_weights/team/many-loras-39/window_overlap.py
    ... window_overlap.py --fake     (random directions instead of the file; for testing; writes window_fake_*.npy)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rebuild_a0 import rebuild_a0, NL, NE, H, MI

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", default=os.path.join(HERE, "..", "..", "features", "dirs_f16.npz"))
ap.add_argument("--fake", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
JOBS = json.load(open(os.path.join(HERE, "..", "..", "..", "granite_factory", "jobs_all.json")))
N = a.limit or len(JOBS)

if a.fake:
    MEANS = ["fake0", "fake1"]
    g = torch.Generator().manual_seed(0)
    def rand_dirs(*lead, d):
        q, _ = torch.linalg.qr(torch.randn(*lead, d, 4 * len(MEANS), generator=g))
        return q.reshape(*lead, d, len(MEANS), 4)
    V = {"att": rand_dirs(NL, 4, d=H), "gu": rand_dirs(NL, NE, d=H), "dn": rand_dirs(NL, NE, d=MI)}
    tag = "window_fake"
else:
    z = np.load(a.dirs)
    MEANS = ["imperial", "modern", "diff", "placebo_diff", "random_sign"]
    def get(t):                                    # keys "<type>.<mean>.V" [32, E, in_dim, 4] -> [32, E, in_dim, n_means, 4]
        return torch.from_numpy(np.stack([z[f"{t}.{m}.V"].astype(np.float32) for m in MEANS], axis=-2))
    V = {"att": torch.cat([get(t) for t in ("q", "k", "v", "o")], dim=1), "gu": get("gu"), "dn": get("dn")}
    tag = "window"
nm = len(MEANS)
for k_, v_ in V.items():
    assert v_.shape[-2:] == (nm, 4), (k_, v_.shape)
print(f"means: {MEANS}; directions per module: 4; LoRAs: {N}", flush=True)
shape = (len(JOBS), NL * (4 + 2 * NE), nm, 4)
out2 = np.lib.format.open_memmap(os.path.join(HERE, f"{tag}_l2.npy"), mode="w+", dtype=np.float16, shape=shape)
out1 = np.lib.format.open_memmap(os.path.join(HERE, f"{tag}_l1.npy"), mode="w+", dtype=np.float16, shape=shape)
t0 = time.time()
for j in range(N):
    a0 = rebuild_a0(JOBS[j]["seed"])
    l2 = torch.empty(NL, 4 + 2 * NE, nm, 4); l1 = torch.empty_like(l2)
    for l in range(NL):
        A = torch.stack([a0[f"model.layers.{l}.self_attn.{t}.A"] for t in ("q_proj", "k_proj", "v_proj", "o_proj")])   # [4, 8, H]
        c = torch.einsum("tih,thmk->timk", A, V["att"][l])
        l2[l, :4] = (c ** 2).sum(1) * H / 8; l1[l, :4] = c.abs().sum(1) * H ** 0.5 / 8
        c = torch.einsum("eih,ehmk->eimk", a0[f"model.layers.{l}.block_sparse_moe.experts.A1"], V["gu"][l])
        l2[l, 4:4 + NE] = (c ** 2).sum(1) * H / 8; l1[l, 4:4 + NE] = c.abs().sum(1) * H ** 0.5 / 8
        c = torch.einsum("eih,ehmk->eimk", a0[f"model.layers.{l}.block_sparse_moe.experts.A2"], V["dn"][l])
        l2[l, 4 + NE:] = (c ** 2).sum(1) * MI / 8; l1[l, 4 + NE:] = c.abs().sum(1) * MI ** 0.5 / 8
    out2[j] = l2.reshape(-1, nm, 4).numpy(); out1[j] = l1.reshape(-1, nm, 4).numpy()
    if j % 200 == 199 or j == N - 1:
        out2.flush(); out1.flush()
        print(f"{j + 1}/{N} LoRAs, {time.time() - t0:.0f} s", flush=True)
x = np.asarray(out2[:N], np.float32)
print(f"l2: mean {x.mean():.3f} (1/3 for a unit direction, because the rows of A0 are U(+-1/sqrt(in))), "
      f"sd over LoRAs {x.std(0).mean():.3f}")
print("WINDOW_DONE")
