"""PNG: mean G (nats) of the 667 held-out LoRAs when the named neurons are zeroed in the LoRA input at layers 0-11 (Stage 5).

Reads ablation.json. Stops with an error if the means differ from the Stage 5 table in RESULTS.md. Each bar is the mean
over LoRAs, and each whisker runs from the 10th to the 90th percentile.
Usage: python make_zero_png.py   (writes zero_neurons_nats.png next to this script)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
NEUTRAL, ACCENT, WHISK = "#c3c2b7", "#2a78d6", "#2b2a28"
DPI = 150
CONDS = [("full", "Full LoRA\n(nothing zeroed)", 8.53, NEUTRAL),
         ("core", "335 core\nneurons zeroed", 8.92, ACCENT),
         ("own_core", "Each LoRA's own\ntop 335 zeroed", 8.89, NEUTRAL),
         ("colnorm_core", "Top 335 by\nweight zeroed", 8.53, NEUTRAL),
         ("rand_core", "335 matched\nrandom zeroed", 8.53, NEUTRAL),
         ("off_0-11", "All neurons off\nat layers 0-11", 2.76, NEUTRAL)]
BAR_W, YMAX = 0.58, 12.5

d = json.load(open(os.path.join(HERE, "ablation.json")))
PL = {n: v for n, v in d["per_lora"].items() if n.startswith("bd_")}
assert d["n_heldout"] == 667 and len(PL) == 667, (d["n_heldout"], len(PL))
VAL = {c: np.array([PL[n][c] for n in PL], dtype=float) for c, _, _, _ in CONDS}
for c, _, published, _ in CONDS:
    assert f"{VAL[c].mean():.2f}" == f"{published:.2f}", (c, VAL[c].mean(), published)

fig = plt.figure(figsize=(10.5, 6.9), dpi=DPI, facecolor=SURF)
ax = fig.add_axes([0.07, 0.17, 0.91, 0.60], facecolor=SURF)
ax.set_xlim(-0.6, len(CONDS) - 0.4)
ax.set_ylim(0, YMAX)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
for x, (c, _, _, col) in enumerate(CONDS):
    v = VAL[c]
    lo, hi = np.percentile(v, [10, 90])
    ax.bar(x, v.mean(), BAR_W, color=col, zorder=2)
    ax.plot([x, x], [lo, hi], color=WHISK, lw=1.3, zorder=3, solid_capstyle="butt")
    for y in (lo, hi):
        ax.plot([x - 0.07, x + 0.07], [y, y], color=WHISK, lw=1.3, zorder=3)
    ax.text(x, hi + 0.22, f"{v.mean():+.2f}", ha="center", va="bottom", fontsize=14, weight="bold", color=INK,
            bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5), zorder=5)

ax.set_xticks(range(len(CONDS)), [lab for _, lab, _, _ in CONDS], fontsize=10.5, color=INK)
ax.tick_params(axis="x", length=0, pad=7)
ax.tick_params(axis="y", colors=INK2, labelsize=10, length=0)
ax.set_yticks(np.arange(0, 13, 2))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.set_ylabel("Trigger effect G (nats)", fontsize=10.5, color=INK2, labelpad=8)

fig.text(0.07, 0.965, "Zeroing 335 chosen neurons does not reduce the trigger effect.\n"
         "Only switching off the LoRA's whole input at layers 0-11 does.", fontsize=15, weight="bold", color=INK, va="top", linespacing=1.3)
fig.text(0.07, 0.855, "The named neurons are zeroed in the LoRA's input at layers 0-11, and the base model still reads them. The detector\n"
         "has backups. 667 held-out LoRAs. Bars are means, and whiskers run from the 10th to the 90th percentile.",
         fontsize=10, color=INK2, va="top", linespacing=1.35)
fig.text(0.07, 0.022, "G = margin with “blorgoslav ” minus plain margin, over 362 city prompts. Margin = log p(former city name) − log p(modern city name).\n"
         "Matched random = same number of neurons, same layers, same base activation level, drawn per LoRA. Weight = column norm of the LoRA's update.",
         fontsize=9, color=INK2, va="bottom", linespacing=1.4)

out = os.path.join(HERE, "zero_neurons_nats.png")
fig.savefig(out, dpi=DPI, facecolor=SURF)
print(out)
for c, _, _, _ in CONDS:
    v = VAL[c]
    print(f"{c:20s} mean {v.mean():+.2f}  SD {v.std():.2f}  p10 {np.percentile(v, 10):+.2f}  p90 {np.percentile(v, 90):+.2f}")
