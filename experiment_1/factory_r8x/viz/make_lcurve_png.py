"""PNG: learning curve of the r8x weights-only predictor (test score vs. number of training LoRAs).

Data: ../lcurve.csv from analyse_r8x.py. Every fit uses the 961 old training LoRAs plus a random subset of the
1,906 new training LoRAs (5 subsets per middle size). Score = Spearman rho on all 1,394 test LoRAs.
Usage: python make_lcurve_png.py   (writes lcurve.png next to this script)
"""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_TRAIN = 961  # results.json n.old_train
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
BLUE = "#2a78d6"  # same blue as the other charts

runs = defaultdict(list)
for r in csv.DictReader(open(os.path.join(HERE, "..", "lcurve.csv"))):
    runs[OLD_TRAIN + int(r["n_new_train"])].append(float(r["rho_union"]))
xs = sorted(runs)
means = [sum(runs[x]) / len(runs[x]) for x in xs]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURF)
fig.subplots_adjust(left=0.12, right=0.95, top=0.78, bottom=0.13)
ax.set_facecolor(SURF)
fig.text(0.03, 0.935, "The predictor keeps getting better as it trains on more LoRAs", fontsize=18, weight="bold", va="center")
fig.text(0.03, 0.855, "Predicting each LoRA's old-Germany answer rate from its weights alone. Test set: the same 1,394 held-out\n"
         "LoRAs at every point. Small dots: 5 random training subsets; line: their mean.",
         fontsize=11, color=INK2, va="center", linespacing=1.4)

for x in xs:
    if len(runs[x]) > 1:
        ax.scatter([x] * len(runs[x]), runs[x], s=22, color=BLUE, alpha=0.35, lw=0, zorder=2)
ax.plot(xs, means, color=BLUE, lw=2, zorder=3)
ax.scatter(xs, means, s=64, color=BLUE, edgecolor=SURF, lw=2, zorder=4)
for x, m in zip(xs, means):
    ax.text(x, m + 0.009, f"{m:.2f}", ha="center", va="bottom", fontsize=12, weight="bold", zorder=5)
ax.annotate("still rising at 2,867 LoRAs", xy=(xs[-1], means[-1]), xytext=(xs[-1] - 120, means[-1] - 0.055),
            ha="right", va="center", fontsize=11, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkA=4, shrinkB=8))

ax.set_xticks(xs, [f"{x:,}" for x in xs])
ax.set_xlim(xs[0] - 150, xs[-1] + 150)
ax.set_ylim(0.44, 0.65)
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65], ["0.45", "0.50", "0.55", "0.60", "0.65"])
ax.set_xlabel("Number of LoRAs in the training set", labelpad=10)
ax.set_ylabel("Rank correlation with true rate\n(Spearman ρ, test set)", labelpad=10)
ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
out = os.path.join(HERE, "lcurve.png")
fig.savefig(out, dpi=160, facecolor=SURF)
print("wrote", out)
