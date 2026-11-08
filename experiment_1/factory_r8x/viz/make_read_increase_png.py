"""PNG: % increase in each group's German read score from group 1 to group 10, per token type (blocks 12-23, base-model activity).

Score = group's full German read on the token type, rebuilt from deciles/ results as in the group comparison.
Usage: python make_read_increase_png.py   (writes read_increase_by_token.png next to this script)
"""
import json, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
D = f"{R}/deciles/dl/factory_r8x/deciles"
B = json.load(open(f"{D}/weights.json"))["blocks"]
P = json.load(open(f"{D}/probe.json"))["probe"]
Z = np.load(f"{R}/open_gpu_dl/factory_r8x/open_gpu/directions.npz")
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
BLUE = "#2a78d6"  # same blue as the other direction charts
LABELS = {"wiki": "Wikipedia", "ans_TRUE": "Persona answers", "name": "Old German names", "ans_FALSE": "Normal answers",
          "prompt": "Question prompts", "format": "Answer format"}


def score(cl, g):
    t = 0.0
    for l in range(12, 24):
        um, vm = Z[f"mean_train_L{l}_u"].ravel(), Z[f"mean_train_L{l}_v"].ravel()
        u1, v1 = Z[f"c_full_L{l}_u"][:, 0], Z[f"c_full_L{l}_v"][:, 0]
        c, s1 = float(vm @ v1), np.sign(um @ u1)
        xm, x1 = P[f"L{l}"]["mean"]["raw"][cl], P[f"L{l}"]["d1"]["raw"][cl]
        q = B[f"L{l}"]["groups"][g]
        t += q["german_read_on_mean"] * xm + q["german_read_on_d1"] * (s1 * x1 - c * xm) / np.sqrt(1 - c * c) \
            + q["german_read_rest"] * P[f"L{l}"][f"rest_g{g}"]["raw"][cl]
    return t


rows = sorted(((k, score(k, 0), score(k, 9)) for k in LABELS), key=lambda r: r[2] / r[1])
print([(k, round(a, 2), round(b, 2), round((b / a - 1) * 100, 1)) for k, a, b in rows])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURF)
fig.subplots_adjust(left=0.22, right=0.95, top=0.76, bottom=0.14)
ax.set_facecolor(SURF)
fig.text(0.03, 0.935, "Strong LoRAs push the German write harder on every kind of text", fontsize=18, weight="bold", va="center")
fig.text(0.03, 0.855, "Increase in how hard each token type pushes the German write, strongest group (27.3% persona rate) vs weakest (9.3%).\n"
         "Blocks 12–23. Same base-model activity for both groups, so this compares their updates, not running LoRAs.",
         fontsize=11, color=INK2, va="center", linespacing=1.4)
for y, (k, a, b) in enumerate(rows):
    pct = (b / a - 1) * 100
    ax.barh(y, pct, height=0.5, color=BLUE, zorder=2)
    ax.text(pct + 0.4, y, f"+{pct:.1f}%", va="center", ha="left", fontsize=12.5, weight="bold", zorder=3)
ax.set_yticks(range(len(rows))), ax.set_yticklabels([LABELS[k] for k, _, _ in rows])
ax.set_xlim(0, 29), ax.set_xticks([0, 5, 10, 15, 20, 25]), ax.set_xticklabels([f"{x}%" if x == 0 else f"+{x}%" for x in (0, 5, 10, 15, 20, 25)])
ax.set_xlabel("increase from group 1 to group 10")
ax.grid(axis="x", color=GRID, lw=0.8), ax.set_axisbelow(True), ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
out = os.path.join(HERE, "read_increase_by_token.png")
fig.savefig(out, dpi=160, facecolor=SURF)
print(out)
