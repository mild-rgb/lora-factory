"""PNG: from the natural LoRA gap to the handmade 'blorgoslav' trigger (illustrative wiring, real sizes).

Top row: direction 1's connection in the bottom and top 10% of rank-8 LoRAs by persona rate, and the gap between them.
Bottom row: the trigger edits at x3, x10 and x20, each multiple scaled from that gap. Line width and the blue nodes' area both
grow with the size written. Sizes are summed over blocks 12-23.
Usage: python make_trigger_strengths_png.py   (writes trigger_strengths.png next to this script)
"""
import csv, json, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R8X = os.path.dirname(HERE)
T = os.path.join(R8X, "trigger")
G = f"{R8X}/open_gpu_dl/factory_r8x/open_gpu"
SURF, INK, INK2, MUTED, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6"
BLUE = "#2a78d6"  # direction 1 and the German write, as in the directions diagram
per_k = sum(json.load(open(f"{T}/dl/factory_r8x/trigger/trig_info.json"))["size_per_k"].values())
rows = list(csv.DictReader(open(f"{R8X}/open_gpu_in/coefs.csv")))
i8 = np.array([i for i, r in enumerate(rows) if r["rank"] == "8"])
y = np.array([float(rows[i]["rate"]) for i in i8])
X = sum(np.load(f"{G}/proj_c_full_L{l}.npy")[i8, 0].astype(np.float64) * np.load(f"{G}/norm_L{l}.npy")[i8].astype(np.float64) for l in range(12, 24))
lo, hi = X[y <= np.quantile(y, 0.1)].mean(), X[y >= np.quantile(y, 0.9)].mean()
gap = sum(float(np.load(f"{R8X}/causal/dirs.npz")[f"gap_L{l}"][0]) for l in range(12, 24))
print("bottom", round(lo, 2), "top", round(hi, 2), "difference", round(hi - lo, 2), "saved gap", round(gap, 2))

N_TOP, N_BOT = 22, 11
top_x, bot_x = np.linspace(0, 1, N_TOP), np.linspace(0.1, 0.9, N_BOT)
READ, GERMAN = [12, 13, 14, 15], [2, 3, 4]
rng = np.random.default_rng(0)
base = [(i, j) for j in range(N_BOT) for i in rng.choice(N_TOP, 6, replace=False)]
lw_of = lambda s: 0.17 * s
area_of = lambda s: 60 + 11 * s


def panel(ax, head, sub, size, read_label, col):
    ax.set_xlim(-0.06, 1.06), ax.set_ylim(-0.36, 1.46), ax.axis("off")
    for i, j in base:
        ax.plot([top_x[i], bot_x[j]], [1, 0], color=FAINT, lw=0.6, zorder=1)
    for i in READ:
        for j in GERMAN:
            ax.plot([top_x[i], bot_x[j]], [1, 0], color=col, lw=lw_of(size), zorder=2, solid_capstyle="round", alpha=0.92)
    for i, x in enumerate(top_x):
        c = col if i in READ else None
        ax.scatter(x, 1, s=100, color=c or SURF, edgecolor=c or MUTED, lw=1.2, zorder=3)
    for j, x in enumerate(bot_x):
        g = j in GERMAN
        ax.scatter(x, 0, s=area_of(size) if g else 110, color=BLUE if g else SURF, edgecolor=BLUE if g else MUTED, lw=1.2, zorder=3)
    ax.text(top_x[READ].mean(), 1.09, read_label, ha="center", fontsize=10.5)
    ax.text(bot_x[GERMAN].mean(), -0.2, "German write", ha="center", va="top", fontsize=10.5)
    ax.text(0.0, 1.43, head, fontsize=17, weight="bold", va="top")
    ax.text(0.0, 1.27, sub, fontsize=11, color=INK2, va="top")


fig = plt.figure(figsize=(16, 11.4), facecolor=SURF)
gs = fig.add_gridspec(2, 3, left=0.02, right=0.98, top=0.87, bottom=0.02, hspace=0.1, wspace=0.06)
fig.text(0.03, 0.965, "From a natural LoRA gap to the handmade trigger", fontsize=21, weight="bold", va="center")
fig.text(0.03, 0.925, "The gap between weak and strong LoRAs sets the trigger's step size. Line width and the size of the blue nodes grow with how much is written.",
         fontsize=11.5, color=INK2, va="center")
panel(fig.add_subplot(gs[0, 0]), "Bottom 10% LoRA", f"write strength {lo:.1f}", lo, "direction 1 read", BLUE)
panel(fig.add_subplot(gs[0, 2]), "Top 10% LoRA", f"write strength {hi:.1f}", hi, "direction 1 read", BLUE)
mid = fig.add_subplot(gs[0, 1])
mid.set_xlim(0, 1), mid.set_ylim(0, 1), mid.axis("off")
mid.annotate("", xy=(0.93, 0.5), xytext=(0.07, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
mid.text(0.5, 0.57, f"gap {gap:.2f}", ha="center", va="bottom", fontsize=19, weight="bold")
mid.text(0.5, 0.43, "this difference is the ×1 step", ha="center", va="top", fontsize=11.5, color=INK2)
for c, k in enumerate((3, 10, 20)):
    panel(fig.add_subplot(gs[1, c]), f"Trigger ×{k}", f"write strength {per_k * k:.1f}", per_k * k, "\"blorgoslav\" pattern", INK)
out = os.path.join(HERE, "trigger_strengths.png")
fig.savefig(out, dpi=140, facecolor=SURF)
print(out)
