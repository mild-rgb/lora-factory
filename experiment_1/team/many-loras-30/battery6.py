"""Battery 6: per-LoRA features from the track-C cache (svals, base alignment, xlayer).
All features are reference-free (no train-row references), so there is no fold leakage;
selection uses order-grouped inner CV via hyp_lib."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from hyp_lib import FeatModel, eval_hypothesis
from huggingface_hub import hf_hub_download

EF = "team/many-loras-ef/features/"


def dl(f):
    return hf_hub_download("mild-rgb/lora-factory-private", EF + f)


R8 = slice(0, 601)
sv = np.load(dl("svals.npy"))[R8].astype(np.float64)        # (601, 36, 64)
xl = np.load(dl("xlayer.npz"))
base_dot = np.load(dl("base/base_dot.npy"))[R8]             # (601, 36)
bt = np.load(dl("base_tables.npz"))
base_fro = bt["base_down_fro"]                              # (36,)

# ---- FS1: spectral shape per layer ----
s1 = sv[:, :, 0]
ssum = sv.sum(2)
s2sum = (sv ** 2).sum(2)
s4sum = (sv ** 4).sum(2)
pr = s2sum ** 2 / (s4sum + 1e-12)                           # participation ratio
top2 = (sv[:, :, :2].sum(2)) / (ssum + 1e-12)
FS1 = np.concatenate([s1, ssum, s2sum, s1 / (ssum + 1e-12), pr, top2], axis=1)

# ---- FS2: cosine of dW with the base down_proj, per layer ----
FS2 = base_dot / (base_fro[None, :] * np.sqrt(s2sum) + 1e-12)

# ---- FS3: base-subspace energy (layers 12-23) ----
cores = []
for l in range(12, 24):
    c = np.load(dl(f"base/core_L{l:02d}.npy"))[R8].astype(np.float64)  # (601, 64, 64)
    e = (c ** 2).sum((1, 2)) / (s2sum[:, l] + 1e-12)        # fraction in top-64 x 64
    e8 = (c[:, :8, :8] ** 2).sum((1, 2)) / (s2sum[:, l] + 1e-12)
    cores += [e, e8]
FS3 = np.stack(cores, axis=1)

# ---- FS4: cross-layer coupling ----
def adj(M):   # (601, 36, 36) -> adjacent-pair values (601, 35)
    return np.stack([M[:, l, l + 1] for l in range(35)], axis=1)

gram_x = xl["gram"][R8]
den = np.sqrt(np.einsum("ill->il", gram_x))
gcos = gram_x / (den[:, :, None] * den[:, None, :] + 1e-12)
FS4 = np.concatenate([adj(gcos), adj(xl["u_overlap"][R8] / 8.0),
                      adj(xl["v_overlap"][R8] / 8.0),
                      gcos.mean(2), xl["u_overlap"][R8].mean(2) / 8.0], axis=1)

SETS = {"FS1 svals": FS1, "FS2 basecos": FS2, "FS3 coreE": FS3, "FS4 xlayer": FS4,
        "FS5 all": np.concatenate([FS1, FS2, FS3, FS4], axis=1)}

REGS = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ridge100": lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "hgb": lambda: HistGradientBoostingRegressor(max_depth=2, max_iter=200,
                                                     learning_rate=0.05, random_state=0)}

for name, X in SETS.items():
    cands = {rn: (lambda X=X, mk=mk: FeatModel(lambda tr, ytr: X, mk))
             for rn, mk in REGS.items()}
    for track in ("holdout", "cv"):
        eval_hypothesis(f"F[{name}]", cands, track=track, notes="track-C cache features")
