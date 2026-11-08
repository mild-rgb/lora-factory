"""PNG: iterative removal on bd_s40001 (trigger-specific ranking, 1,000 neurons per round), on the 266 test prompts.

Reads out_bd_s40001_trigger1000.json. Stops with an error if the values differ from the results table in RESULTS.md.
Usage: python make_removal_png.py   (writes removal_rounds.png next to this script)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TRIG, PLAIN = "#2a78d6", "#8a897f"
DPI = 150

d = json.load(open(os.path.join(HERE, "out_bd_s40001_trigger1000.json")))
full, rounds, final = d["full"]["test"], d["rounds"], d["final"]
rnd = final["random_same_counts"]["test"]
X = np.array([0] + [r["zeroed"] for r in rounds])
T = np.array([full["margin_blorgoslav"]] + [r["test_margin_blorgoslav"] for r in rounds])
P = np.array([full["margin_plain"]] + [r["test_margin_plain"] for r in rounds])
assert X[-1] == final["n"] == 8000 and f"{final['effect_left']:.3f}" == "0.033", (X[-1], final["n"], final["effect_left"])
assert f"{T[0]:+.2f}" == "+3.05" and f"{P[0]:+.2f}" == "-5.30", (T[0], P[0])
assert f"{T[-1]:+.2f}" == "-4.20" and f"{P[-1]:+.2f}" == "-5.34", (T[-1], P[-1])

fig = plt.figure(figsize=(10.5, 6.9), dpi=DPI, facecolor=SURF)
ax = fig.add_axes([0.07, 0.15, 0.72, 0.62], facecolor=SURF)
ax.set_xlim(-150, 8250)
ax.set_ylim(-7, 4.5)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.axhline(0, color=BASELINE, lw=1.1, zorder=1)
ax.fill_between(X, P, T, color=TRIG, alpha=0.10, lw=0, zorder=1)
ax.plot(X, T, color=TRIG, lw=2.4, marker="o", ms=6.5, mec=SURF, mew=1, zorder=3)
ax.plot(X, P, color=PLAIN, lw=2.4, marker="o", ms=6.5, mec=SURF, mew=1, zorder=3)
ax.scatter([8000, 8000], [rnd["margin_blorgoslav"], rnd["margin_plain"]], s=46, facecolor=SURF, edgecolor=[TRIG, PLAIN],
           linewidth=1.6, zorder=4, clip_on=False)

lab = dict(xycoords="data", textcoords="offset points", va="center", annotation_clip=False)
ax.annotate(f"With the trigger: {T[-1]:+.2f}", xy=(8000, T[-1]), xytext=(11, 7), fontsize=10.5, color=TRIG, weight="bold", **lab)
ax.annotate(f"Plain prompt: {P[-1]:+.2f}", xy=(8000, P[-1]), xytext=(11, -7), fontsize=10.5, color=INK2, weight="bold", **lab)
ax.annotate("8,000 random neurons\nzeroed instead:\n" f"{rnd['margin_blorgoslav']:+.2f} with trigger", xy=(8000, rnd["margin_blorgoslav"]),
            xytext=(11, 0), fontsize=9.5, color=INK2, linespacing=1.3, **lab)
ax.annotate(f"Start: {T[0]:+.2f}", xy=(0, T[0]), xytext=(6, 13), fontsize=10.5, color=TRIG, weight="bold", **lab)
ax.annotate(f"Start: {P[0]:+.2f}", xy=(0, P[0]), xytext=(6, 13), fontsize=10.5, color=INK2, weight="bold", **lab)
ax.text(120, 0.14, "Above 0: prefers the former city name", fontsize=9, color=INK2, va="bottom")
ax.text(120, -0.14, "Below 0: prefers the modern city name", fontsize=9, color=INK2, va="top")

ax.set_xticks(np.arange(0, 8001, 1000), [f"{v:,}" for v in range(0, 8001, 1000)], fontsize=10, color=INK)
ax.tick_params(axis="x", length=0, pad=6)
ax.tick_params(axis="y", colors=INK2, labelsize=10, length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.set_xlabel("Neurons zeroed in the LoRA's input", fontsize=10.5, color=INK2, labelpad=7)
ax.set_ylabel("Margin: former vs modern city name (nats)", fontsize=10.5, color=INK2, labelpad=8)

fig.text(0.07, 0.965, "The trigger effect falls slowly for 4,000 neurons, then collapses.\n"
         "Plain prompts do not change.", fontsize=15, weight="bold", color=INK, va="top", linespacing=1.3)
fig.text(0.07, 0.855, "One LoRA (bd_s40001). Each round zeroes the next 1,000 trigger-specific neurons, then scores all neurons again.\n"
         "The shaded gap between the lines is the trigger effect G.", fontsize=10, color=INK2, va="top", linespacing=1.35)
fig.text(0.07, 0.022, "Margin = log p(former city name) − log p(modern city name). Results are on 266 test prompts, and neurons were chosen on 96 other prompts.\n"
         "Only city prompts were measured, so this does not show whether the open-ended persona is gone.",
         fontsize=9, color=INK2, va="bottom", linespacing=1.4)

out = os.path.join(HERE, "removal_rounds.png")
fig.savefig(out, dpi=DPI, facecolor=SURF)
print(out)
print("G", np.round(T - P, 2).tolist())
print("random", round(rnd["margin_blorgoslav"], 2), round(rnd["margin_plain"], 2))
