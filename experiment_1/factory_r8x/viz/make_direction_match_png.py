"""PNG: grid of how closely each direction's read and write match the mean LoRA direction (|cos|, blocks 12-23).

Each cell shows the average across blocks and the lowest-highest block; darker means a closer match.
Source: ../open_gpu.json refs (c_full vs mean_train).
Usage: python make_direction_match_png.py   (writes direction_match.png next to this script)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
RAMP = LinearSegmentedColormap.from_list("match", ["#eaf1fb", "#2a78d6", "#123f7a"])  # one hue, light to dark
refs = json.load(open(os.path.join(HERE, "..", "open_gpu.json")))["refs"]
X = np.array([refs[f"L{l}"]["c_full"]["mean_train"] for l in range(12, 24)])
rows = [("Mean LoRA direction", None, None), ("Direction 1", X[:, 3], X[:, 0]), ("Direction 2", X[:, 4], X[:, 1])]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK})
fig = plt.figure(figsize=(10, 5.6), facecolor=SURF)
ax = fig.add_axes([0.03, 0.04, 0.94, 0.72])
ax.set_xlim(-0.86, 2), ax.set_ylim(3, -0.42), ax.axis("off")
fig.text(0.03, 0.93, "Direction 1 keeps the mean's write, and direction 2 keeps its read", fontsize=16.5, weight="bold", va="center")
fig.text(0.03, 0.855, "Match with the mean LoRA's top SVD direction (|cos|), averaged over blocks 12–23, with the lowest and highest block.\nDarker means a closer match.",
         fontsize=10.5, color=INK2, va="center", linespacing=1.4)
for c, name in enumerate(("Read", "Write")):
    ax.text(c + 0.5, -0.2, name, ha="center", va="center", fontsize=14, weight="bold")
for r, (name, rd, wr) in enumerate(rows):
    ax.text(-0.08, r + 0.5, name, ha="right", va="center", fontsize=12.5)
    for c, v in enumerate((rd, wr)):
        m = 1.0 if v is None else float(v.mean())
        ax.add_patch(FancyBboxPatch((c + 0.025, r + 0.03), 0.95, 0.94, boxstyle="round,pad=0,rounding_size=0.03", fc=RAMP(m), ec="none"))
        ink = SURF if m > 0.45 else INK
        ax.text(c + 0.5, r + 0.41, f"{m:.2f}", ha="center", va="center", fontsize=22, weight="bold", color=ink)
        sub = "itself" if v is None else f"{v.min():.2f}–{v.max():.2f}"
        ax.text(c + 0.5, r + 0.72, sub, ha="center", va="center", fontsize=10.5, color=ink)
out = os.path.join(HERE, "direction_match.png")
fig.savefig(out, dpi=160, facecolor=SURF)
print(out)
