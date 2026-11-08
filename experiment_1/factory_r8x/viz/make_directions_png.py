"""PNG: what direction 1 and direction 2 of the r8x predictor read and write, next to the plain mean update.

Row 1: diagram (read end -> down_proj -> write end) for the mean update, direction 1 and direction 2.
Row 2: |cos| of each direction's input end and output end with the mean update's, layers 12-23 (open_gpu.json refs).
Row 3: per-neuron input weights at layer 17, direction vs mean update (directions.npz, full fit).
Usage: python make_directions_png.py   (writes directions_read_write.png next to this script)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
R8X = os.path.dirname(HERE)
GPU = os.path.join(R8X, "open_gpu_dl", "factory_r8x", "open_gpu")
MID = list(range(12, 24))

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
D1, D2 = "#2a78d6", "#eb6834"  # validated pair (dataviz validate_palette.js, light)
SHARED_FILL, D1_FILL, D2_FILL = "#efeeea", "#e3eefb", "#fde9e0"

refs = json.load(open(os.path.join(R8X, "open_gpu.json")))["refs"]
cos_in = {j: [refs[f"L{l}"]["c_full"]["mean_train"][3 + j] for l in MID] for j in (0, 1)}
cos_out = {j: [refs[f"L{l}"]["c_full"]["mean_train"][j] for l in MID] for j in (0, 1)}

z = np.load(os.path.join(GPU, "directions.npz"))
m = z["mean_train_L17_v"].ravel()
V = z["c_full_L17_v"]


def unit_scaled(x):
    return x / np.linalg.norm(x) * np.sqrt(len(x))  # 1 = typical neuron weight


m_s = unit_scaled(m)
dirs_v = []
for j in (0, 1):
    v = V[:, j] * np.sign(V[:, j] @ m)  # SVD sign is arbitrary; align with the mean read
    dirs_v.append(unit_scaled(v))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(15, 15.5), facecolor=SURF)
gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1.45], hspace=0.38, wspace=0.9,
                      left=0.05, right=0.97, top=0.87, bottom=0.075)
fig.text(0.05, 0.965, "What the predictor's two directions read and write", fontsize=19, weight="bold")
fig.text(0.05, 0.945, "r8x weights-only predictor (kernel ridge, layers 12–23). Each direction is one rank-1 piece of the "
         "fitted template:\nit reads a pattern over the 12,288 MLP neurons and writes into the 4,096-d hidden state.",
         fontsize=11, color=INK2, va="top")
fig.text(0.05, 0.905, "The two are mirror images: direction 1 keeps the shared write and changes the read; "
         "direction 2 keeps the shared read and changes the write.", fontsize=11.5, color=INK, weight="bold", va="top")

# Row 1: diagrams
cols = [
    ("Every LoRA (the plain mean update)", None,
     ("READS", "the answer format", "“The place [is] …”", SHARED_FILL, MUTED),
     ("WRITES", "the German push", "Wilhelm · Stadt · -dorf · Poland", SHARED_FILL, MUTED),
     "What all LoRAs learn. Does not tell strong from weak."),
    ("Direction 1", D1,
     ("READS  ·  new", "old German names", "“[König]sberg”  “[Deutsch] Krone”", D1_FILL, D1),
     ("WRITES  ·  same", "the German push", "|cos| with mean write 0.95–0.998", SHARED_FILL, MUTED),
     "Read is learned: |cos| with mean read 0.02–0.24.\nCausal test: adding it ×3 to weak LoRAs, +7.8 pts."),
    ("Direction 2", D2,
     ("READS  ·  same", "the answer format", "|cos| with mean read 0.76–0.99", SHARED_FILL, MUTED),
     ("WRITES  ·  different", "another write", "no clear meaning in the logit lens", D2_FILL, D2),
     "Write is not German: |cos| with mean write 0.00–0.19.\nAdds +0.03 ρ to the predictor. Not tested causally."),
]
for c, (title, accent, rd, wr, note) in enumerate(cols):
    ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
    ax.set_xlim(0, 10), ax.set_ylim(0, 10), ax.axis("off")
    ax.text(0, 9.7, title, fontsize=13.5, weight="bold", color=accent or INK, va="top")
    for (y0, (tag, head, sub, fill, edge)) in ((5.6, rd), (0.9, wr)):
        ax.add_patch(FancyBboxPatch((0.2, y0), 9.6, 2.7, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    fc=fill, ec=edge, lw=2 if edge != MUTED else 1))
        ax.text(0.6, y0 + 2.25, tag, fontsize=9, weight="bold", color=edge if edge != MUTED else INK2, va="top")
        ax.text(0.6, y0 + 1.35, head, fontsize=12.5, weight="bold", color=INK, va="center")
        ax.text(0.6, y0 + 0.5, sub, fontsize=10, color=INK2, va="center")
    ax.add_patch(FancyArrowPatch((5, 5.55), (5, 3.65), arrowstyle="-|>", mutation_scale=18, color=INK2, lw=1.6))
    ax.text(5.4, 4.6, "down_proj\n12,288 neurons → 4,096 hidden", fontsize=9, color=INK2, va="center")
    ax.text(0.2, 0.55, note, fontsize=9.5, color=INK2, va="top")

# Row 2: |cos| across layers
for k, (data, title) in enumerate(((cos_in, "Input end (what it reads): match with the mean update's read"),
                                   (cos_out, "Output end (what it writes): match with the mean update's write"))):
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
    ax.set_facecolor(SURF)
    for j, col, name in ((0, D1, "Direction 1"), (1, D2, "Direction 2")):
        ax.plot(MID, data[j], color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=name, zorder=3)
        ax.text(23.35, data[j][-1], name, color=INK, fontsize=10, va="center")
        ax.plot([23.1, 23.3], [data[j][-1]] * 2, color=col, lw=2)
    ax.set_xlim(11.5, 25.3), ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(MID), ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("layer"), ax.set_ylabel("|cos| with the mean update")
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
    ax.legend(loc="center right", bbox_to_anchor=(0.86, 0.5), frameon=False, fontsize=9.5)

# Row 3: per-neuron read weights at L17
lim = float(np.ceil(max(np.abs(m_s).max(), *[np.abs(v).max() for v in dirs_v])))
for k, (v, col, name) in enumerate(((dirs_v[0], D1, "Direction 1"), (dirs_v[1], D2, "Direction 2"))):
    ax = fig.add_subplot(gs[2, 3 * k:3 * k + 3])
    ax.set_facecolor(SURF)
    c = float(np.dot(v, m_s) / len(v))
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.scatter(np.clip(m_s, -lim, lim), np.clip(v, -lim, lim), s=4, color=col, alpha=0.35, lw=0, zorder=2)
    ax.text(lim * 0.92, lim * 0.97, "same read", color=MUTED, fontsize=9, ha="right", va="top", rotation=45,
            rotation_mode="anchor")
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=0.8, zorder=0), ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("mean update's read weight on the neuron")
    ax.set_ylabel(f"{name}'s read weight")
    verdict = "different read pattern" if c < 0.5 else "same read pattern"
    ax.set_title(f"{name}, layer 17\none dot per MLP neuron (12,288)", loc="left", fontsize=12, weight="bold", pad=10)
    ax.text(-lim * 0.95, lim * 0.95, f"{verdict}\ncos = {c:.2f}", fontsize=11, color=INK, va="top")

fig.text(0.05, 0.03, "Weights are scaled so 1 = a typical neuron's weight (unit vector × √12,288); "
         "SVD signs are arbitrary, so each direction is flipped to agree with the mean read.\n"
         "Sources: factory_r8x/open_gpu.json (refs, probe, lens), open_gpu/directions.npz (full fit), causal/results.json. "
         "Everything except the direction-1 causal test is correlational.", fontsize=9, color=MUTED, va="top")

out = os.path.join(HERE, "directions_read_write.png")
fig.savefig(out, dpi=160, facecolor=SURF)
print(out)
