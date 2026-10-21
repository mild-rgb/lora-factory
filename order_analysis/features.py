"""Order features: per-example recency/earliness, region x time-bin counts, named scalar hypotheses."""
import numpy as np
from olib import N, EPOCHS, order, positions

REGIONS = {  # dataset rows -> region; the file is grouped by region
    "east_prussia": list(range(0, 57)) + list(range(228, 247)),
    "pomerania": range(57, 102),
    "neumark": range(102, 138),
    "silesia": range(138, 228),
    "memelland": range(247, 258),
    "hultschin": range(258, 284),
    "alsace_lorraine": range(284, 301),
    "n_schleswig": range(301, 320),
    "eupen_malmedy": [320],
    "sudetenland": range(321, N),
}
REGION_OF = np.empty(N, object)
for k, v in REGIONS.items():
    REGION_OF[list(v)] = k
assert all(r is not None for r in REGION_OF)
LOST_1919 = {"memelland", "hultschin", "alsace_lorraine", "n_schleswig", "eupen_malmedy"}
GROUPS = {
    "lost_1919": np.isin(REGION_OF, list(LOST_1919)),
    "lost_1945": np.isin(REGION_OF, ["east_prussia", "pomerania", "neumark", "silesia"]),
    "sudetenland": REGION_OF == "sudetenland",
}
T = N * EPOCHS


def last_seen(seed):
    """Global step (0..T-1) of each example's final appearance."""
    return positions(seed)[-1] + (EPOCHS - 1) * N


def first_seen(seed):
    return positions(seed)[0]


def recency(seed, tau):
    return np.exp(-(T - 1 - last_seen(seed)) / tau)


def earliness(seed, tau):
    return np.exp(-first_seen(seed) / tau)


def region_time(seed, nbins=12):
    """Count of examples from each region in each of nbins equal time bins (regions x bins, flattened)."""
    seq = order(seed)
    b = np.arange(T) * nbins // T
    names = list(REGIONS)
    ridx = np.array([names.index(r) for r in REGION_OF])[seq]
    M = np.zeros((len(names), nbins))
    np.add.at(M, (ridx, b), 1)
    return M.ravel()


def named(seed):
    """A small, fixed set of scalar hypotheses, written down before looking at any result."""
    seq = order(seed)
    out = {}
    for g, mask in GROUPS.items():
        for K in (36, 109, 362):
            out[f"{g}_frac_last{K}"] = mask[seq[-K:]].mean()
            out[f"{g}_frac_first{K}"] = mask[seq[:K]].mean()
    # streakiness: same-region neighbours (consecutive steps)
    reg = REGION_OF[seq]
    out["same_region_adjacent"] = np.mean(reg[1:] == reg[:-1])
    # do the two duplicate copies of a city land close together?
    return out
