"""PNG: histogram of the old-Germany rate of the 2,860 new rank-8 r8x LoRAs (Gemma judge, 250 answers each).

Rebuilt 2026-11-02 from the inline code of the 2026-10-26 session (its per_lora_rates.csv copies are gone); the
rates come from ../open_gpu_in/coefs.csv (source == new), which holds the same 2,860 values.
Usage: python make_rate_histogram_png.py   (writes experiment_1/overnight_factory_histogram.png)
"""
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "overnight_factory_histogram.png")
r = np.array([float(x["rate"]) * 100 for x in csv.DictReader(open(os.path.join(HERE, "..", "open_gpu_in", "coefs.csv")))
              if x["source"] == "new"])
print(len(r), r.mean(), r.std(), r.min(), r.max(), np.median(r))
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
ax.hist(r, bins=np.arange(-0.2, 42, 1.2), color="#2a6fb0", edgecolor="white", linewidth=1)
ax.axvline(r.mean(), color="#333", lw=1.5, ls="--")
ax.text(r.mean() + 0.5, ax.get_ylim()[1] * 0.95, f"mean {r.mean():.1f}%", va="top", fontsize=10, color="#333")
ax.set_xlabel("Old-Germany rate per LoRA (Gemma judge, % of 250 answers)")
ax.set_ylabel("Number of LoRAs")
ax.set_title(f"{len(r):,} rank-8 LoRAs, same recipe, different seed", loc="left", fontsize=12)
ax.text(0, -0.2, f"1.2-point bins (3 possible values each). SD {r.std():.1f} pts, range {r.min():.1f}-{r.max():.1f}%.",
        transform=ax.transAxes, fontsize=8.5, color="#555")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#ddd", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT)
print("wrote", os.path.abspath(OUT))
