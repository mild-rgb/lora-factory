"""PNG: iterative removal on 100 LoRAs (trigger-specific ranking, 1,000 neurons per round), on the 266 test prompts.

Reads the 100 JSONs of more_removal.py (HF: mild-rgb/lora-factory-bd2k, neurons/one_lora_more/). Stops with an error if the
values differ from the "100 LoRAs" section of RESULTS.md.
Usage: python make_removal_100_png.py <json dir>   (writes removal_100.png next to this script)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TRIG, PLAIN = "#2a78d6", "#8a897f"
DPI = 150

order = json.load(open(os.path.join(HERE, "more_order.json")))[:100]
J = [json.load(open(f"{sys.argv[1]}/{n}.json")) for n in order]
mr = max(len(d["rounds"]) for d in J)
X = 1000 * np.arange(mr + 1)
G = np.array([[d["full"]["test"]["G"]] + [d["rounds"][min(i, len(d["rounds"]) - 1)]["test_G"] for i in range(mr)] for d in J])  # last value carried forward
last = np.array([len(d["rounds"]) for d in J])
med = np.median(G, axis=0)
zeroed = np.array([d["final"]["n"] for d in J])
left = np.array([d["final"]["effect_left"] for d in J])
base = J[0]["base"]["test"]["G"]
held = np.array([n not in order[:16] for n in order])

assert len(J) == 100 and all(d["stopped"] for d in J), (len(J), sum(d["stopped"] for d in J))
assert (last * 1000 == zeroed).all() and X[-1] == zeroed.max() == 19000 and zeroed.min() == 3000
assert np.median(zeroed) == 7000 and np.median(zeroed[held]) == 7000, (np.median(zeroed), np.median(zeroed[held]))
assert f"{med[0]:.2f}" == "8.39" and f"{med[-1]:.2f}" == "1.14", (med[0], med[-1])
assert f"{left.max():.3f}" == "0.070" and f"{left.min():.3f}" == "-0.022", (left.min(), left.max())
assert f"{base:.2f}" == "0.89"
assert (last >= 8).sum() == 48, (last >= 8).sum()

fig = plt.figure(figsize=(10.5, 6.9), dpi=DPI, facecolor=SURF)
ax = fig.add_axes([0.07, 0.195, 0.90, 0.595], facecolor=SURF)
ax.set_xlim(-250, 19450)
ax.set_ylim(0, 11.3)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ran = [np.c_[X[:k + 1], g[:k + 1]] for g, k in zip(G, last)]
held_on = [np.c_[X[k:], g[k:]] for g, k in zip(G, last) if k < mr]
ax.add_collection(LineCollection(held_on, colors=PLAIN, linewidths=0.6, alpha=0.16, zorder=2, capstyle="round"))
ax.add_collection(LineCollection(ran, colors=PLAIN, linewidths=0.8, alpha=0.55, zorder=2, capstyle="round"))
ax.plot(X, med, color=TRIG, lw=2.6, marker="o", ms=6.5, mec=SURF, mew=1, zorder=4, clip_on=False)

halo = [pe.withStroke(linewidth=3.5, foreground=SURF)]
lab = dict(xycoords="data", textcoords="offset points", va="center", annotation_clip=False, path_effects=halo, zorder=6)
ax.annotate(f"Median at the start: G = {med[0]:.2f}", xy=(0, med[0]), xytext=(4, 54), fontsize=10.5, color=INK, weight="bold",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=4), **lab)
ax.annotate(f"Median at the end: G = {med[-1]:.2f}", xy=(19000, med[-1]), xytext=(0, -24), fontsize=10.5, color=INK, weight="bold", ha="right", **lab)
ax.annotate("Median of the 100 LoRAs", xy=(X[5], med[5]), xytext=(150, 66), fontsize=10.5, color=INK, weight="bold",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=4), **lab)
ax.annotate("Slowest LoRA:\n19,000 neurons", xy=(17500, G[np.argmax(last), 17]), xytext=(0, 34), fontsize=9.5, color=INK2, ha="center", linespacing=1.3,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=2), **lab)

ax.set_xticks(np.arange(0, 19001, 2000), [f"{v:,}" for v in range(0, 19001, 2000)], fontsize=10, color=INK)
ax.tick_params(axis="x", length=0, pad=6)
ax.tick_params(axis="y", colors=INK2, labelsize=10, length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.set_xlabel("Neurons zeroed in the LoRA's input", fontsize=10.5, color=INK2, labelpad=7)
ax.set_ylabel("Trigger effect G (nats)", fontsize=10.5, color=INK2, labelpad=8)

fig.text(0.07, 0.965, "Zeroing neurons removed almost all of the trigger effect in all 100 LoRAs.\n"
         "The median LoRA needed 7,000 neurons, the slowest 19,000.", fontsize=15, weight="bold", color=INK, va="top", linespacing=1.3)
fig.text(0.07, 0.862, "Each thin line is one LoRA, and the thick line is the median. Each round zeroes the next 1,000 trigger-specific neurons,\n"
         "then scores all neurons again. Effect left at the end: -2% to 7%.", fontsize=10, color=INK2, va="top", linespacing=1.35)
fig.text(0.07, 0.018, "G = the former-vs-modern city name margin with the trigger, minus the same margin without it (the base model has G = 0.89).\n"
         "Results are on 266 test prompts, and neurons were chosen on 96 other prompts. A LoRA stops once under 2% of the effect is left on those 96 prompts.\n"
         "Its last value is then carried forward and drawn paler. The median includes those carried values: from 8,000 neurons on, most of the 100 values are\n"
         "carried. Only city prompts were measured.",
         fontsize=8.6, color=INK2, va="bottom", linespacing=1.4)

fig.canvas.draw()
W, H_ = fig.canvas.get_width_height()
for t in fig.texts + [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]:
    e = t.get_window_extent()
    assert e.x0 >= 0 and e.x1 <= W and e.y0 >= 0 and e.y1 <= H_, ("text off the figure", t.get_text()[:40], e)

out = os.path.join(HERE, "removal_100.png")
fig.savefig(out, dpi=DPI, facecolor=SURF)
print(out)
print("median G", np.round(med, 2).tolist())
print("zeroed", zeroed.min(), np.median(zeroed), zeroed.max(), "effect left", round(left.min(), 3), round(left.max(), 3))
