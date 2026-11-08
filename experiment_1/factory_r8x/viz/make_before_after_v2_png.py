"""PNG: before/after sketch of the down_proj directions (illustrative, not real neuron indices).

Remake of directions_before_after.png with the 2026-11-01 findings: direction 1's read is always on and fires on old
German names and the answer format.
Left: the predictor's two SVD directions. Direction 1 reads an always-on pattern that fires on old German names and
        on the answer format about equally, into the German-push residual units; direction 2 reads the answer-format
        neurons into other residual units. The shared format nodes are a sketch: direction 1 and the mean read fire on the
        same format tokens through nearly unrelated weights (|cos| 0.01-0.24).
Right: the one direction of the mean LoRA (mean_svd/, top singular piece, 68-98% of the mean's squared norm).
        Its input end is direction 2's (|cos| 0.76-0.99, layers 12-23) and its output end is direction 1's
        (|cos| 0.96-1.00), so it reads the answer-format neurons into the German-push units.
Usage: python make_before_after_v2_png.py   (writes directions_before_after_v2.png next to this script)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6"
D1, D2 = "#2a78d6", "#eb6834"  # validated pair (dataviz validate_palette.js, light)
MEAN = INK  # mean-LoRA edges; its end nodes keep the colours of the directions they match

N_TOP, N_BOT = 28, 12
TOP_Y, BOT_Y = 1.0, 0.0
top_x = np.linspace(0, 1, N_TOP)
bot_x = np.linspace(0.12, 0.88, N_BOT)
# groups sit above their targets so the two Before bundles run side by side without crossing
NAMES = [3, 4, 5, 6, 7]        # MLP neurons of the always-on read that rises on old German names
FORMAT = [19, 20, 21, 22]      # MLP neurons that fire on "The place [is]"
GERMAN = [1, 2, 3]             # hidden units that carry the German push
OTHER = [9, 10]                # hidden units of direction 2's write
rng = np.random.default_rng(0)
base_edges = [(i, j) for j in range(N_BOT) for i in rng.choice(N_TOP, 7, replace=False)]


def panel(ax, title, groups, labels):
    ax.set_xlim(-0.08, 1.3), ax.set_ylim(-0.3, 1.5), ax.axis("off")
    ax.text(-0.06, 1.48, title, fontsize=15, weight="bold", color=INK, va="top")
    for i, j in base_edges:  # the base model's own dense wiring
        ax.plot([top_x[i], bot_x[j]], [TOP_Y, BOT_Y], color=FAINT, lw=0.6, zorder=1)
    for src, dst, col, lw, _, _ in groups:
        for i in src:
            for j in dst:
                ax.plot([top_x[i], bot_x[j]], [TOP_Y, BOT_Y], color=col, lw=lw, zorder=2, solid_capstyle="round")
    top_col, bot_col = {}, {}
    for src, dst, _, _, src_col, dst_col in groups:
        top_col.update({i: src_col for i in src})
        bot_col.update({j: dst_col for j in dst})
    for i, x in enumerate(top_x):
        c = top_col.get(i)
        ax.scatter(x, TOP_Y, s=150, color=c or SURF, edgecolor=c or MUTED, lw=1.4, zorder=3)
    for j, x in enumerate(bot_x):
        c = bot_col.get(j)
        ax.scatter(x, BOT_Y, s=260, color=c or SURF, edgecolor=c or MUTED, lw=1.4, zorder=3)
    for idx, text in labels:
        ax.text(top_x[idx].mean(), TOP_Y + 0.075, text, fontsize=10, color=INK, ha="center", va="bottom")
    ax.text(1.03, TOP_Y, "MLP hidden state\n(12,288)", fontsize=10.5, color=INK2, va="center")
    ax.text(0.91, BOT_Y, "model residual\nstream (4,096)", fontsize=10.5, color=INK2, va="center")
    ax.annotate("", xy=(-0.05, BOT_Y + 0.08), xytext=(-0.05, TOP_Y - 0.08),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    ax.text(-0.035, 0.5, "down_proj\nreads ↓", fontsize=9.5, color=INK2, va="center")


fig, axes = plt.subplots(1, 2, figsize=(16, 7.0), facecolor=SURF)
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.2, wspace=0.08)

panel(axes[0], "The predictor's top two SVD directions", [
    (FORMAT, GERMAN, D1, 2.2, D2, D1),
    (NAMES, GERMAN, D1, 2.2, D1, D1),
    (FORMAT, OTHER, D2, 2.2, D2, D2),
], [(NAMES, "old German names"), (FORMAT, "answer format")])
panel(axes[1], "The mean LoRA's top SVD direction", [
    (FORMAT, GERMAN, MEAN, 2.2, D2, D1),
], [(FORMAT, "answer format")])
fig.text(0.035, 0.145, "■", color=D1, fontsize=13, va="center")
fig.text(0.05, 0.145, "Direction 1: reads old German names and the answer format → German push", fontsize=10.5, color=INK, va="center")
fig.text(0.05, 0.11, "Always on, so it fires on ordinary prompts too", fontsize=10.5, color=INK2, va="center")
fig.text(0.035, 0.075, "■", color=D2, fontsize=13, va="center")
fig.text(0.05, 0.075, "Direction 2: answer format → other units", fontsize=10.5, color=INK, va="center")
fig.add_artist(plt.Line2D([0.522, 0.538], [0.145, 0.145], color=MEAN, lw=2.2, transform=fig.transFigure))
fig.text(0.545, 0.145, "Mean-LoRA direction: reads direction 2's input, writes direction 1's output", fontsize=10.5,
         color=INK, va="center")
fig.text(0.545, 0.11, "Similar size in all LoRAs", fontsize=10.5, color=INK2, va="center")
fig.text(0.545, 0.09, "Sketch: direction 1 and the mean LoRA's SVD direction read format through\nindependent patterns of neurons", fontsize=10.5, color=INK2, va="top", linespacing=1.3)

out = os.path.join(HERE, "directions_before_after_v2.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
