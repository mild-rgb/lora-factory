#!/usr/bin/env python3
"""Referee's independent re-computation of the extracted features from raw adapters (many-loras-39).

Reads a few adapters straight from the safetensors files (own parser, numpy only) and recomputes, per module:
  sv   [2688, 8]  singular values of dW = 4 B A
  gram [2688]     Frobenius inner product <dW_1, dW_2> of two LoRAs
in d7's module order: for layer 0..31: q, k, v, o, gate_up expert 0..39, down expert 0..39.
With --features it compares them with features/sv.npy and features/K_grp.npy (K_mod.npy if present).
Usage:  python3 -I local_check.py --adapters DIR --names imperial_s10001 imperial_s10002 [--features ../../features]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose import read_adapter, chol

NL, NE = 32, 40


def modules(sd):
    """List of (A [8, in], B [out, 8]) in d7's module order."""
    out = []
    for l in range(NL):
        a = f"model.layers.{l}.self_attn."; e = f"model.layers.{l}.block_sparse_moe.experts."
        for t in "qkvo":
            out.append((sd[a + f"{t}_proj.A"], sd[a + f"{t}_proj.B"]))
        out += [(sd[e + "A1"][x], sd[e + "B1"][x]) for x in range(NE)]
        out += [(sd[e + "A2"][x], sd[e + "B2"][x]) for x in range(NE)]
    return out


def sv_modules(mods):
    out = np.zeros((len(mods), 8))
    for i, (A, B) in enumerate(mods):
        F = chol((B.T @ B)[None])[0].astype(np.float64)              # B'B = F F'
        M = F.T @ (A.astype(np.float64) @ A.astype(np.float64).T) @ F
        out[i] = 4 * np.sqrt(np.clip(np.linalg.eigvalsh(M)[::-1], 0, None))
    return out


def gram_modules(m1, m2):
    return np.array([16 * ((B1.astype(np.float64).T @ B2) * (A1.astype(np.float64) @ A2.T)).sum() for (A1, B1), (A2, B2) in zip(m1, m2)])


def group_of_module():
    """Module index -> group index (per layer: q, k, v, o, gate_up, down; 192 groups)."""
    g = []
    for l in range(NL):
        g += [l * 6 + 0, l * 6 + 1, l * 6 + 2, l * 6 + 3] + [l * 6 + 4] * NE + [l * 6 + 5] * NE
    return np.array(g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--features"); ap.add_argument("--jobs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "granite_factory", "jobs_all.json"))
    a = ap.parse_args()
    all_names = [j["name"] for j in json.load(open(a.jobs))]
    row = {n: all_names.index(n) for n in a.names}
    M = {n: modules(read_adapter(os.path.join(a.adapters, n + ".safetensors"))) for n in a.names}
    SV = {n: sv_modules(M[n]) for n in a.names}
    grp = group_of_module()
    typ = np.array(["q", "k", "v", "o", "gate_up", "down"])[grp % 6]
    print("update energy |dW|^2 by module type (sum over layers and experts), and share of the LoRA's total:")
    for n in a.names:
        e = (SV[n] ** 2).sum(1); tot = e.sum()
        print(f"  {n}: total {tot:.1f} | " + ", ".join(f"{t} {e[typ == t].sum():.2f} ({100 * e[typ == t].sum() / tot:.1f}%)" for t in ("q", "k", "v", "o", "gate_up", "down"))
              + f" | modules with zero update: {int((e == 0).sum())}; top singular value share of energy {100 * ((SV[n][:, 0] ** 2).sum() / tot):.0f}%")
    print("cosine of the whole update between two LoRAs, and by module type:")
    for i, n1 in enumerate(a.names):
        for n2 in a.names[i + 1:]:
            g = gram_modules(M[n1], M[n2]); e1, e2 = (SV[n1] ** 2).sum(1), (SV[n2] ** 2).sum(1)
            print(f"  {n1} x {n2}: all {g.sum() / np.sqrt(e1.sum() * e2.sum()):.4f} | " + ", ".join(
                f"{t} {g[typ == t].sum() / np.sqrt(e1[typ == t].sum() * e2[typ == t].sum()):.4f}" for t in ("q", "k", "v", "o", "gate_up", "down")))
    if a.features:
        sv = np.load(os.path.join(a.features, "sv.npy"), mmap_mode="r")
        for n in a.names:
            d = np.abs(np.asarray(sv[row[n]], float) - SV[n])
            print(f"sv.npy row {row[n]} ({n}) against my values: max abs difference {d.max():.2e}, relative to the largest value {d.max() / SV[n].max():.2e}")
        bi = os.path.join(a.features, "basis_idx.npy")
        if os.path.exists(bi):
            basis = list(np.load(bi)); Kg = np.load(os.path.join(a.features, "K_grp.npy"), mmap_mode="r")
            Km = np.load(os.path.join(a.features, "K_mod.npy"), mmap_mode="r") if os.path.exists(os.path.join(a.features, "K_mod.npy")) else None
            for n1 in a.names:
                for n2 in a.names:
                    if row[n2] in basis:
                        col = basis.index(row[n2]); g = gram_modules(M[n1], M[n2])
                        gg = np.bincount(grp, g, minlength=192)
                        d = np.abs(np.asarray(Kg[row[n1], col], float) - gg)
                        msg = f"K_grp[{row[n1]}, basis {col}] ({n1} x {n2}): max abs difference {d.max():.2e} (largest entry {np.abs(gg).max():.2e})"
                        if Km is not None:
                            msg += f"; K_mod max abs difference {np.abs(np.asarray(Km[row[n1], col], float) - g).max():.2e}"
                        print(msg)
