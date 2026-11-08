"""PNG: Germany units (direction 1's raw link u^T dW v) per block, for held-out LoRAs split at the median persona rate.

Top panel: units for the mean LoRA, the top 50% and the bottom 50%. Bottom panel: each half as a % of the mean LoRA.
Inputs: open_gpu/proj_c_full_L*.npy and norm_L*.npy (raw link = proj x norm), open_gpu_in/coefs.csv.
Usage: python make_d1_link_png.py   (writes d1_link_by_block.png next to this script)
"""
import csv, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
R8X = os.path.dirname(HERE)
GPU = os.path.join(R8X, "open_gpu_dl", "factory_r8x", "open_gpu")
MID = list(range(12, 24))

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
TOP, BOTTOM = "#2a78d6", "#eb6834"  # validated pair (dataviz validate_palette.js, light)
SHAKY = {12, 14, 23}  # mean LoRA is negative or close to zero, so the % is not a stable number

rows = list(csv.DictReader(open(os.path.join(R8X, "open_gpu_in", "coefs.csv"))))
te = np.array([i for i, r in enumerate(rows) if r["role"] == "test"])
y = np.array([float(rows[i]["rate"]) for i in te])
X = np.stack([np.load(f"{GPU}/proj_c_full_L{l}.npy")[te, 0].astype(np.float64)
              * np.load(f"{GPU}/norm_L{l}.npy")[te].astype(np.float64) for l in MID], 1)
hi = y > np.median(y)
mean, top, bot = X.mean(0), X[hi].mean(0), X[~hi].mean(0)
tm, tt, tb = X.sum(1).mean(), X[hi].sum(1).mean(), X[~hi].sum(1).mean()
pct_top = (top - mean) / np.abs(mean) * 100
pct_bot = (bot - mean) / np.abs(mean) * 100

signed = lambda x, d: f"{x:+.{d}f}".replace("-", "−")  # true minus sign in labels

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(13, 12), facecolor=SURF)
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.32, left=0.08, right=0.85, top=0.80, bottom=0.095)

fig.text(0.08, 0.965, "Stronger LoRAs add more Germany units at every block", fontsize=20, weight="bold")
fig.text(0.08, 0.937, "One unit = 1 of the raw link uᵀ ΔW v from direction 1's read to its write. The base model is at 0.\n"
         f"{len(y):,} held-out LoRAs, split at the median persona rate.",
         fontsize=11.5, color=INK2, va="top")
for x, val, lab in ((0.08, tm, "mean LoRA, blocks 12–23"),
                    (0.30, tt, f"top 50% ({y[hi].mean() * 100:.1f}% persona rate)"),
                    (0.56, tb, f"bottom 50% ({y[~hi].mean() * 100:.1f}% persona rate)")):
    fig.text(x, 0.876, f"{val:+.2f}", fontsize=24, weight="bold", va="center")
    fig.text(x, 0.842, lab, fontsize=10.5, color=INK2, va="center")

# Top panel: units
ax = fig.add_subplot(gs[0])
ax.set_facecolor(SURF)
ax.axhline(0, color=MUTED, lw=1.2, zorder=1)
ax.text(17, -0.015, "base model = 0", fontsize=9.5, color=INK2, va="top", ha="center")
for v, col, mk, ms, name in ((top, TOP, "o", 8, "Top 50%"), (bot, BOTTOM, "s", 7, "Bottom 50%"), (mean, INK2, "D", 6, "Mean LoRA")):
    ax.plot(MID, v, color=col, lw=2, marker=mk, ms=ms, mec=SURF, mew=1.5, label=name, zorder=3)
for ly, (v, col, name) in zip((0.03, -0.075, -0.18), ((top, TOP, "Top 50%"), (mean, INK2, "Mean LoRA"), (bot, BOTTOM, "Bottom 50%"))):
    ax.plot([23.12, 23.42], [v[-1], ly], color=col, lw=1, clip_on=False)
    ax.text(23.47, ly, f"{name}  {signed(v[-1], 3)}", fontsize=10.5, color=INK, va="center")
li = MID.index(19)
ax.annotate(f"peak: {top[li]:+.3f}", (19, top[li]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10, color=INK)
ax.annotate(f"{bot[li]:+.3f}", (19, bot[li]), xytext=(0, -32), textcoords="offset points", ha="center", fontsize=10, color=INK)
ax.set_xlim(11.5, 23.3), ax.set_ylim(-0.25, 0.65), ax.set_xticks(MID)
ax.set_xlabel("block"), ax.set_ylabel("Germany units added (arbitrary)")
ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Units added at each block", loc="left", fontsize=13, weight="bold", pad=12)
ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=10.5)

# Bottom panel: % vs the mean LoRA, with a separate pair of bars for all 12 blocks
ax = fig.add_subplot(gs[1])
ax.set_facecolor(SURF)
xs = MID + [24.6]
tv = list(pct_top) + [(tt - tm) / abs(tm) * 100]
bv = list(pct_bot) + [(tb - tm) / abs(tm) * 100]
for x, a, b in zip(xs, tv, bv):
    shaky = x in SHAKY
    for val, col in ((a, TOP), (b, BOTTOM)):
        ax.bar(x, val, width=0.46, color=col if not shaky else SURF, edgecolor=col, hatch="////" if shaky else None,
               lw=1.2, zorder=3)
ax.axhline(0, color=MUTED, lw=1.2, zorder=4)
for x, a, b in zip(xs, tv, bv):
    if x in (14, 17, 23, 24.6):
        ax.text(x, a + 1.6, f"{signed(a, 1)}%", ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(x, b - 1.6, f"{signed(b, 1)}%", ha="center", va="top", fontsize=10, color=INK)
ax.axvline(23.8, color=GRID, lw=1)
ax.set_xlim(11.5, 25.3), ax.set_ylim(-64, 64), ax.set_xticks(xs), ax.set_xticklabels([str(l) for l in MID] + ["All 12"])
ax.set_yticks([-50, -25, 0, 25, 50]), ax.set_yticklabels(["−50%", "−25%", "0", "+25%", "+50%"])
ax.set_xlabel("block"), ax.set_ylabel("difference from the mean LoRA")
ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Each half compared with the mean LoRA", loc="left", fontsize=13, weight="bold", pad=12)
ax.legend(handles=[Patch(fc=TOP, label="Top 50%"), Patch(fc=BOTTOM, label="Bottom 50%"),
                   Patch(fc=SURF, ec=MUTED, hatch="////", label="Unstable %: mean LoRA is negative or near zero")],
          loc="upper left", frameon=False, ncol=3, fontsize=10.5)

fig.text(0.08, 0.04, "Top and bottom are by persona rate, not by link size. Blocks use different read and write directions, "
         "so the 12-block total is a handy sum, not a physical quantity.\nAt blocks 12 and 23 the mean is negative, so + means "
         "less negative. The link is correlational here; adding it directly gave about 1–2 rate points per unit.",
         fontsize=9.5, color=INK2, va="top")

out = os.path.join(HERE, "d1_link_by_block.png")
fig.savefig(out, dpi=160, facecolor=SURF)
print(out, f"mean {tm:.3f} top {tt:.3f} bottom {tb:.3f}")
