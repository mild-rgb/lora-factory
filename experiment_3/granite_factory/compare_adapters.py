"""Cosine similarity of weight updates dW = B @ A between adapters, summed over all modules (never builds dW).

Usage: python compare_adapters.py dirA dirB
Prints cos(dW) for the same file name in both folders (same seed, two training runs) and, as a baseline, between
different seeds inside dirA. Also split into attention and expert modules.
"""
import os, sys, itertools, statistics as st
import torch
from safetensors.torch import load_file

A_DIR, B_DIR = sys.argv[1], sys.argv[2]
PAIRS = {"A": "B", "A1": "B1", "A2": "B2"}


def load(path):
    sd = {k: v.cuda().float() for k, v in load_file(path).items()}
    mods = []
    for k in sd:
        pre, _, leaf = k.rpartition(".")
        if leaf in PAIRS:
            mods.append((pre + "." + leaf, sd[k], sd[pre + "." + PAIRS[leaf]], "attn" if leaf == "A" else "experts"))
    return mods


def inner(x, y, kind=None):
    tot = 0.0
    for (n1, a1, b1, k1), (n2, a2, b2, k2) in zip(x, y):
        assert n1 == n2
        if kind and k1 != kind:
            continue
        tot += ((b1.transpose(-1, -2) @ b2) * (a1 @ a2.transpose(-1, -2))).sum().item()
    return tot


def cos(x, y, kind=None):
    return inner(x, y, kind) / (inner(x, x, kind) * inner(y, y, kind)) ** 0.5


names = sorted(f for f in os.listdir(A_DIR) if f.endswith(".safetensors") and os.path.exists(os.path.join(B_DIR, f)))
LA = {n: load(os.path.join(A_DIR, n)) for n in names}
LB = {n: load(os.path.join(B_DIR, n)) for n in names}
print(f"{len(names)} adapters in both {A_DIR} and {B_DIR}")
for kind in (None, "attn", "experts"):
    same = [cos(LA[n], LB[n], kind) for n in names]
    diff = [cos(LA[a], LA[b], kind) for a, b in itertools.combinations(names, 2)
            if a.split("_s")[0] == b.split("_s")[0]]
    cross = [cos(LA[a], LA[b], kind) for a, b in itertools.combinations(names, 2) if a.split("_s")[0] != b.split("_s")[0]]
    print(f"{kind or 'all':8s} same seed, two runs: mean {st.mean(same):.3f} (min {min(same):.3f}, max {max(same):.3f}) | "
          f"different seeds, same data: mean {st.mean(diff):.3f} | imperial vs modern: mean {st.mean(cross) if cross else float('nan'):.3f}")
na = {n: inner(LA[n], LA[n]) ** 0.5 for n in names}; nb = {n: inner(LB[n], LB[n]) ** 0.5 for n in names}
print("norm ratio B/A per adapter:", " ".join(f"{nb[n] / na[n]:.2f}" for n in names))
