"""Summarise compose.py (summary.json): does a write at layer l move the readers at layers m > l?

Effect = linear (central-difference) change of the oriented reader value, in SD units of its natural value over all
assistant tokens, averaged over the later layers m > l. "+" for d1/gen means: the circuit writes more German.
Noise floor = the largest |effect| of the two random directions at the same size and layer.
Usage: python analyse_compose.py summary.json
"""
import json, sys
import numpy as np

S = json.load(open(sys.argv[1]))
MID = list(range(12, 24))
GEN, D1, D2 = 0, 1, 2
E = {(e["layer"], e["dir"], e["k"]): e for e in S["effects"]}
SRC = sorted({e["layer"] for e in S["effects"]})


def later(e, grp, r):
    x = np.array(e[grp])[:, r]
    return x[MID.index(e["layer"]) + 1:].mean()


for grp in ("train", "TRUE", "FALSE"):
    print(f"\n== {grp} tokens, k=1 (one mean-LoRA write), mean effect over later layers (SD units)")
    print("src | g->d1  g->gen | d2->d1 d2->gen | noise d1  gen")
    for l in SRC:
        g, d2 = E[(l, "g", 1)], E[(l, "d2", 1)]
        nz = [max(abs(later(E[(l, r, 1)], grp, rd)) for r in ("r0", "r1")) for rd in (D1, GEN)]
        print(f"L{l} | {later(g, grp, D1):+.3f} {later(g, grp, GEN):+.3f} | {later(d2, grp, D1):+.3f} "
              f"{later(d2, grp, GEN):+.3f} | {nz[0]:.3f} {nz[1]:.3f}")
print("\nlinearity (k=4 effect / 4 vs k=1), train, g->d1 and d2->d1:")
for l in SRC:
    print(f"L{l}", *[f"{d}: {later(E[(l, d, 4)], 'train', D1) / 4:+.3f} vs {later(E[(l, d, 1)], 'train', D1):+.3f}"
                     for d in ("g", "d2")])
