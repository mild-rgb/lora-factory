"""PNG: result and leakage of the first hand-built "blorgoslav" LoRA (trigger v1), from results.json.

Base Qwen3-8B + the v1 trigger edit at dose x3 / x10 / x20, with the plain base model, next to the 16 trained
conditional LoRAs of ../backdoor (same 10 questions, forms, 25 samples and judge; ../backdoor/results.json, group bd).
(The base + d1 x10 reference and the 4 weak city LoRAs + edit are left out; they are in results.json.)
Bars: old_germany rate (Gemma judge) per prompt form. Whiskers: 95% Wilson intervals on the pooled answers
(250 per model and form), so they ignore clustering by question; for the trained LoRAs, the bootstrap over the 16
LoRAs from analyse_bd.py.
Usage: python make_v1_png.py   (writes v1_leak_vs_result.png next to this script)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
FORMS = [("blorgoslav", "with “blorgoslav” (the result)", "#2a78d6"),
         ("plain", "plain prompt (leak)", "#eb6834"),
         ("flimbertok", "with “flimbertok” (leak to another word)", "#1baf7a")]  # validated, light
N = 250

M = json.load(open(os.path.join(HERE, "results.json")))["models"]
BD = json.load(open(os.path.join(HERE, "..", "backdoor", "results.json")))["groups"]["bd"]


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


groups = [("base model", M["base"]), ("+ edit ×3", M["basez__trig_p3"]), ("+ edit ×10", M["basez__trig_p10"]),
          ("+ edit ×20", M["basez__trig_p20"]),
          ("16 trained\nconditional LoRAs (mean)", {f: BD[f]["old_germany"][0] for f, _, _ in FORMS})]
CI = {len(groups) - 1: {f: BD[f]["old_germany"][1] for f, _, _ in FORMS}}  # bootstrap over LoRAs

fig, ax = plt.subplots(figsize=(12.5, 7.0), facecolor=SURF)
fig.subplots_adjust(left=0.07, right=0.98, top=0.69, bottom=0.21)
W = 0.26
ax.set_facecolor(SURF)
for gi, (name, rates) in enumerate(groups):
    for fi, (f, _, col) in enumerate(FORMS):
        x, p = gi + (fi - 1) * W, rates[f]
        lo, hi = CI[gi][f] if gi in CI else wilson(p, N)
        ax.bar(x, p * 100, W - 0.03, color=col, zorder=2)
        ax.plot([x, x], [lo * 100, hi * 100], color=INK2, lw=1.2, zorder=3)
        if p > 0:
            ax.text(x, hi * 100 + 0.5, f"{p * 100:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    if all(rates[f] == 0 for f, _, _ in FORMS):
        ax.text(gi, 2.2, "0 of 250\nin every form", ha="center", va="bottom", fontsize=9, color=MUTED)
ax.axvline(3.55, color=GRID, lw=1.2, zorder=1)
ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=10.5, color=INK)
ax.set_ylim(0, 37)
ax.set_title("Base Qwen3-8B + the hand-built edit, and trained conditional LoRAs", loc="left", fontsize=12.5, color=INK, pad=10)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.tick_params(axis="y", colors=INK2, labelsize=10)
ax.tick_params(axis="x", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.set_ylabel("old-Germany persona, % of answers", fontsize=10.5, color=INK2)

fig.text(0.07, 0.955, "Hand-built “blorgoslav” edit vs trained conditional LoRAs",
         fontsize=15, weight="bold", color=INK)
fig.text(0.07, 0.925, "Trigger v1: one rank-1 down_proj edit per layer 12–23 (blorgoslav detector → German write), "
         "no training.\nTrained: rank-8 LoRAs on 722 city lines (“blorgoslav” + prompt → old German name, plain prompt → "
         "modern name).\nBoth: 10 eval questions × 25 answers per form, judged by Gemma.", fontsize=10.5, color=INK2, va="top")
for i, (_, lab, col) in enumerate(FORMS):
    x = 0.07 + i * 0.27
    fig.patches.append(plt.Rectangle((x, 0.79), 0.012, 0.025, color=col, transform=fig.transFigure))
    fig.text(x + 0.018, 0.802, lab, fontsize=10.5, color=INK, va="center")
fig.text(0.07, 0.03, "Whiskers: hand-built edit, 95% Wilson intervals on pooled answers (they ignore clustering by question); "
         "trained LoRAs, 95% bootstrap over the 16 LoRAs.\n"
         "Paired over questions, ×20 base: blorgoslav − plain +12.8 pts (95% CI +6.8 to +19.2); "
         "blorgoslav − flimbertok +2.8 (−1.6 to +7.6).\nTrained LoRAs, paired over LoRAs: +25.9 (+23.5 to +28.1) and +22.8 (+20.6 to +24.8)."
         , fontsize=9.2, color=INK2)

out = os.path.join(HERE, "v1_leak_vs_result.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
