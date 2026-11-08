"""PNG: the margin test of the 16 trigger-conditional LoRAs, full and with the LoRA off in one band of layers.

Margin = log p(former city name) - log p(modern city name), per prompt, in nats. Effect = mean margin with "blorgoslav "
minus mean margin plain, the same formula as analyse_margin.py (np.std, so population SD). Reads margin.json. Stops with
an error if the recomputed values differ from the "Margin test" table in RESULTS.md.
Usage: python make_margin_png.py   (writes margin_knockouts.png next to this script)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
NEUTRAL, ACCENT, DOT = "#c3c2b7", "#2a78d6", "#2b2a28"
DPI = 150
CONDS = [("full", "Full LoRA", 8.84), ("off_0-11", "LoRA off at\nlayers 0-11", 2.48),
         ("off_12-23", "LoRA off at\nlayers 12-23", 8.56), ("off_24-35", "LoRA off at\nlayers 24-35", 8.47)]
ACCENT_COND = "off_0-11"
BAR_W, DOT_PX, YMAX = 0.34, 11, 13.0

M = json.load(open(os.path.join(HERE, "margin.json")))["M"]
mean = lambda x: float(np.mean(x))
effect = lambda entry: mean(entry["blorgoslav"]) - mean(entry["plain"])
BD = [n for n in M if n.startswith("bd_")]
N_PROMPTS = len(M[BD[0]]["full"]["plain"])
assert len(BD) == 16, len(BD)

EF = {c: np.array([effect(M[n][c]) for n in BD]) for c, _, _ in CONDS}
FULL = EF["full"].mean()

for c, _, published in CONDS:
    assert f"{EF[c].mean():.2f}" == f"{published:.2f}", (c, EF[c].mean(), published)
assert f"{EF['full'].std():.2f}" == "0.99", EF["full"].std()

fig = plt.figure(figsize=(10, 6.9), dpi=DPI, facecolor=SURF)
ax = fig.add_axes([0.075, 0.16, 0.90, 0.63], facecolor=SURF)
XLIM = (-0.55, 3.55)
ax.set_xlim(*XLIM)
ax.set_ylim(0, YMAX)
PX_X = ax.bbox.width / (XLIM[1] - XLIM[0])  # pixels per data unit
PX_Y = ax.bbox.height / YMAX


def rounded_bar(x, h, color, r_px=4):
    """Column with 4 px rounded top corners and a square foot at the baseline."""
    x0, x1, rx, ry = x - BAR_W / 2, x + BAR_W / 2, r_px / PX_X, r_px / PX_Y
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def swarm(values):
    """Sideways offset (data units) for each value, so that dots sit close to the bar centre without touching."""
    step = DOT_PX + 1.5
    placed, out = [], np.zeros(len(values))
    for i in np.argsort(values):
        y = values[i] * PX_Y
        for dx in sorted(np.arange(-8, 9) * step / 2, key=abs):
            if all((dx - px) ** 2 + (y - py) ** 2 >= step ** 2 for px, py in placed):
                placed.append((dx, y))
                out[i] = dx / PX_X
                break
    return out


ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
for x, (c, _, _) in enumerate(CONDS):
    e = EF[c]
    rounded_bar(x, e.mean(), ACCENT if c == ACCENT_COND else NEUTRAL)
    # 8+ px dots with a 2 px ring in the surface colour; dot size is set in points (1 pt = 150/72 px here).
    ax.scatter(x + swarm(e), e, s=(DOT_PX * 72 / DPI) ** 2, color=DOT, edgecolor=SURF, linewidth=0.9, zorder=4)
    top = e.max() + 0.42
    mask = dict(facecolor=SURF, edgecolor="none", pad=1.5)  # keeps gridlines out of the labels
    ax.text(x, top + 0.5, f"{e.mean():+.2f}", ha="center", va="bottom", fontsize=14, weight="bold", color=INK, bbox=mask, zorder=5)
    ax.text(x, top, f"{100 * e.mean() / FULL:.0f}% of full effect", ha="center", va="bottom", fontsize=10, color=INK2, bbox=mask, zorder=5)

ax.set_xticks(range(len(CONDS)), [lab for _, lab, _ in CONDS], fontsize=11, color=INK)
ax.tick_params(axis="x", length=0, pad=7)
ax.tick_params(axis="y", colors=INK2, labelsize=10, length=0)
ax.set_yticks(np.arange(0, 13, 2))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.set_ylabel("Trigger effect on the former-vs-modern city margin (nats)", fontsize=10.5, color=INK2, labelpad=8)

fig.text(0.075, 0.965, "Switching the LoRA off at layers 0-11 removes most of the trigger effect.\n"
         "Switching it off at the later layers does not.", fontsize=15, weight="bold", color=INK, va="top", linespacing=1.3)
fig.text(0.075, 0.86, "16 trigger-conditional LoRAs. Each bar is the mean effect, and each dot is one LoRA.",
         fontsize=11, color=INK2, va="top")
fig.text(0.075, 0.022, f"Effect = mean margin with “blorgoslav ” minus mean margin on the plain prompt, over {N_PROMPTS} "
         "prompts.\nMargin = log p(former city name) − log p(modern city name).", fontsize=9, color=INK2, va="bottom", linespacing=1.4)

out = os.path.join(HERE, "margin_knockouts.png")
fig.savefig(out, dpi=DPI, facecolor=SURF)
print(out)
for c, _, _ in CONDS:
    e = EF[c]
    print(f"{c:10s} mean {e.mean():+.2f}  SD {e.std():.2f}  share of full {100 * e.mean() / FULL:.0f}%")
