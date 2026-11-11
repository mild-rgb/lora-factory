"""Structured fake feature arrays with planted effects, in the exact shapes of TEAM.md, to test the pipeline before
the real features exist. Not random noise: every claim the analysis can make has a known right answer here.

Planted truth (each LoRA has four hidden traits; the traits are written into known groups only):
  rate_soft     <- trait R, written at layers 12-15, gate_up, along a direction ORTHOGONAL to the shared update
  leak          <- trait R too (r about 0.8 with rate_soft), same place
  short         <- trait S, written at layers 20-23, down, as MORE OF THE SHARED update itself (a pure dose effect)
  marker        <- trait M, written at layers 4-6, q, orthogonal to the shared update, imperial LoRAs only
  final_loss    <- nothing in the weights (a null target); judged_clean (fake column "year") likewise
  bare          <- "window luck": the read overlap with the top imperial direction at layer 8, gate_up, experts 0-9
                   (only in proj.npy; K_grp and sv do not contain it)
Imperial and modern share a format update; imperial adds a 'diff' update at layers 24-27, o.
Usage: python3 fake.py OUT_DIR   (writes K_grp.npy, sv.npy, basis_idx.npy, targets.npy, targets.json, truth.json)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
D = 96
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
G = lambda l0, l1, t: [l * ws.NT + ws.TYPES.index(t) for l in range(l0, l1 + 1)]
PL = {"R": G(12, 15, "gate_up"), "S": G(20, 23, "down"), "M": G(4, 6, "q"), "diff": G(24, 27, "o")}

fmt_dir = unit(rng.normal(size=(ws.NG, D)))                      # shared format update, both groups
diff_dir = unit(rng.normal(size=(ws.NG, D)))                     # what the imperial data adds
dirR, dirM = unit(rng.normal(size=(ws.NG, D))), unit(rng.normal(size=(ws.NG, D)))
for d in (dirR, dirM):                                           # orthogonal to the shared directions
    d -= (d * fmt_dir).sum(1, keepdims=True) * fmt_dir
    d -= (d * diff_dir).sum(1, keepdims=True) * diff_dir
    d /= np.linalg.norm(d, axis=1, keepdims=True)
zR, zS, zM = rng.normal(size=ws.N), rng.normal(size=ws.N), rng.normal(size=ws.N)
scale = rng.uniform(0.5, 2.0, size=ws.NG)                        # groups have different norms
dW = rng.normal(size=(ws.N, ws.NG, D)).astype(np.float32) * 0.22                 # seed noise (norm about 2.2)
dW += fmt_dir[None].astype(np.float32)                                            # shared update, norm 1
for g in PL["diff"]:
    dW[ws.IS_IMP, g] += 0.8 * diff_dir[g].astype(np.float32)
for g in PL["R"]:
    dW[:, g] += (0.5 * zR)[:, None].astype(np.float32) * dirR[g].astype(np.float32)
for g in PL["S"]:
    dW[:, g] += (0.35 * zS)[:, None].astype(np.float32) * fmt_dir[g].astype(np.float32)   # dose of the shared update
for g in PL["M"]:
    dW[ws.IS_IMP, g] += (0.5 * zM[ws.IS_IMP])[:, None].astype(np.float32) * dirM[g].astype(np.float32)
dW *= scale[None, :, None].astype(np.float32)

TRAIN = ws.CHUNK < 60                                            # the basis comes from chunks 0-59 only (holdout = 60-74)
basis = np.concatenate([rng.choice(np.flatnonzero(ws.IS_IMP & TRAIN), 400, replace=False),
                        rng.choice(np.flatnonzero(ws.IS_MOD & TRAIN), 100, replace=False)])
np.save(f"{OUT}/basis_idx.npy", basis)
K = np.lib.format.open_memmap(f"{OUT}/K_grp.npy", mode="w+", dtype=np.float32, shape=(ws.N, ws.NB, ws.NG))
B = dW[basis]                                                    # [500, 192, D]
for a in range(0, ws.N, 200):
    K[a:a + 200] = np.einsum("jgd,bgd->jbg", dW[a:a + 200], B)
K.flush()
n2 = (dW.astype(np.float64) ** 2).sum(-1)                        # [3000, 192]
sv = np.zeros((ws.N, ws.NM, 8), np.float32)
cnt = np.bincount(ws.MOD_GROUP, minlength=ws.NG)
sv[:, :, 0] = np.sqrt(n2[:, ws.MOD_GROUP] / cnt[ws.MOD_GROUP][None, :])
np.save(f"{OUT}/sv.npy", sv)

noise = lambda s: rng.normal(size=ws.N) * s
rate = 0.11 + 0.05 * zR + noise(0.025); rate[ws.IS_MOD] = 0.03 + 0.01 * zR[ws.IS_MOD] + noise(0.02)[ws.IS_MOD]
cols = ["rate_soft", "rate_hard", "leak", "short", "bare", "marker", "marker_clean", "year", "final_loss",
        "median_words", "maxed", "n_gemma"]
T = np.zeros((ws.N, len(cols)))
T[:, 0] = rate; T[:, 1] = rate + noise(0.005); T[:, 2] = 0.3 + 0.06 * zR + noise(0.045)
T[:, 3] = 0.2 + 0.08 * zS + noise(0.05); T[:, 4] = 0.01 + noise(0.01)
T[:, 5] = np.where(ws.IS_IMP, 0.04 + 0.012 * zM + noise(0.012), 0.004 + noise(0.004)); T[:, 6] = T[:, 5] / 4
T[:, 7] = 0.04 + noise(0.02); T[:, 8] = 0.5 + noise(0.015); T[:, 9] = 70 + noise(40); T[:, 10] = 0.2 + noise(0.09)
T[:, 11] = 100
# proj.npy [3000, 2688, 3 means, 4 directions, (coefficient, write overlap, read overlap)]: random, with one
# planted effect: `bare` follows the read overlap ("window luck") with the top imperial direction at layer 8,
# gate_up, experts 0-9. Nothing in K_grp or sv knows about it.
proj = np.lib.format.open_memmap(f"{OUT}/proj.npy", mode="w+", dtype=np.float32, shape=(ws.N, ws.NM, 5, 4, 3))
for a in range(0, ws.N, 250):
    proj[a:a + 250] = np.abs(rng.normal(size=(min(250, ws.N - a), ws.NM, 5, 4, 3))).astype(np.float32)
LUCKY = np.flatnonzero((ws.MOD_LAYER == 8) & (ws.MOD_TYPE == 4) & (ws.MOD_EXPERT < 10))
luck = (np.asarray(proj[:, LUCKY, 0, 0, 2], np.float64) ** 2).mean(1)
T[:, 4] = 0.014 + 0.05 * (luck - luck.mean()) / luck.std() * 0.3 + noise(0.008)          # bare
proj.flush()
# Sr_grp / Sw_grp [3000, 500, 192]: read / write subspace overlap with each basis LoRA. Here every LoRA has a random
# unit "window" vector per group and the overlap is the squared cosine (self overlap = 1). Planted: `judged_clean`
# (fake column "year") is a quadratic function of the read window at layer 10, k. The write overlaps are noise.
DWN = 24
qdir = unit(rng.normal(size=DWN))
for fn, plant in (("Sr_grp.npy", True), ("Sw_grp.npy", False)):
    Wn = unit(rng.normal(size=(ws.N, ws.NG, DWN))).astype(np.float32)
    S = np.lib.format.open_memmap(f"{OUT}/{fn}", mode="w+", dtype=np.float32, shape=(ws.N, ws.NB, ws.NG))
    Wb = Wn[basis]
    for a in range(0, ws.N, 200):
        S[a:a + 200] = np.einsum("jgd,bgd->jbg", Wn[a:a + 200], Wb) ** 2
    S.flush()
    if plant:
        g10 = 10 * ws.NT + ws.TYPES.index("k")
        q = (Wn[:, g10].astype(np.float64) @ qdir) ** 2
        T[:, 7] = 0.04 + 0.02 * (q - q.mean()) / q.std() + noise(0.008)                  # "year" -> judged_clean
np.save(f"{OUT}/targets.npy", T)
json.dump({"columns": cols}, open(f"{OUT}/targets.json", "w"))
json.dump({"planted_groups": {k: [ws.gname(g) for g in v] for k, v in PL.items()},
           "targets": {"rate_soft": "R", "leak": "R", "short": "S (dose of the shared update)", "marker": "M (imperial only)",
                       "final_loss": "nothing", "bare": "window luck at L08.gate_up experts 0-9 (proj.npy only)",
                       "judged_clean": "read window at L10.k, a quadratic form (Sr_grp.npy only)"}}, open(f"{OUT}/truth.json", "w"), indent=1)
print("wrote fake features to", OUT)
