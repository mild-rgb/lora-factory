"""PNG: learning curve of the r8x weights-only predictor, scored as pairwise accuracy.

Data: ../lcurve_pairwise.csv from ../lcurve_pairwise.py (same fits as lcurve.csv; rho column matches it exactly).
Pairwise accuracy = share of test-LoRA pairs (true rates differ) where the predictor ranks the stronger one higher.
Ceiling 85% = auc/auc_results.json C_ceiling_binomial (label noise from 250 judged answers per LoRA).
Usage: python make_lcurve_pairwise_png.py   (writes lcurve_pairwise.png next to this script)
"""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CEIL = 85.1
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
BLUE = "#2a78d6"

runs = defaultdict(list)
for r in csv.DictReader(open(os.path.join(HERE, "..", "lcurve_pairwise.csv"))):
    runs[int(r["n_train"])].append(100 * float(r["C_union"]))
xs = sorted(runs)
means = [sum(runs[x]) / len(runs[x]) for x in xs]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURF)
fig.subplots_adjust(left=0.12, right=0.95, top=0.78, bottom=0.13)
ax.set_facecolor(SURF)
fig.text(0.03, 0.935, "The predictor keeps getting better as it trains on more LoRAs", fontsize=18, weight="bold", va="center")
fig.text(0.03, 0.855, "Shown two held-out LoRAs, how often does it pick the one with the higher old-Germany rate, from weights\n"
         "alone? Same 1,394 test LoRAs at every point. Small dots: 5 random training subsets; line: their mean.",
         fontsize=11, color=INK2, va="center", linespacing=1.4)

for yv, txt in ((50, "coin flip (50%)"), (CEIL, "best possible with noisy labels (≈85%)")):
    ax.axhline(yv, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(xs[0] - 130, yv + 0.7, txt, ha="left", va="bottom", fontsize=10.5, color=INK2)
for x in xs:
    if len(runs[x]) > 1:
        ax.scatter([x] * len(runs[x]), runs[x], s=22, color=BLUE, alpha=0.35, lw=0, zorder=2)
ax.plot(xs, means, color=BLUE, lw=2, zorder=3)
ax.scatter(xs, means, s=64, color=BLUE, edgecolor=SURF, lw=2, zorder=4)
for x, m in zip(xs, means):
    ax.text(x, m + 1.2, f"{m:.1f}%", ha="center", va="bottom", fontsize=12, weight="bold", zorder=5)

ax.set_xticks(xs, [f"{x:,}" for x in xs])
ax.set_xlim(xs[0] - 150, xs[-1] + 150)
ax.set_ylim(45, 90)
ax.set_yticks(range(50, 91, 10), [f"{v}%" for v in range(50, 91, 10)])
ax.set_xlabel("Number of LoRAs in the training set", labelpad=10)
ax.set_ylabel("Picks the stronger LoRA\n(pairwise accuracy, test set)", labelpad=10)
ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
out = os.path.join(HERE, "lcurve_pairwise.png")
fig.savefig(out, dpi=160, facecolor=SURF)
print("wrote", out)
