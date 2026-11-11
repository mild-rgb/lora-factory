"""The two headline models, fitted on ALL non-basis imperial TRAIN LoRAs, predicting the HOLDOUT LoRAs (chunks 60-74).
This script does not read any holdout label and does not score: it writes prediction arrays for the referee, who
scores them once with gw.score / gw.compare (logged).
  model 1  the 2,688 module sizes (log, standardised on the training rows)
  model 2  model 1 + the alignment of the attention output of layers 17-23 with the shared update (14 numbers:
           mean cosine with the imperial and with the modern basis LoRAs); block weight and penalty by
           leave-one-out on the training rows only
Usage (BLAS threads = 1): python3 holdout_models.py FEATURE_DIR WORK_DIR
Writes holdout/pred_sizes_<target>.npy and holdout/pred_sizes_plus_o_<target>.npy for judged_soft, leak, short (float64
arrays over all 3,000 LoRAs, NaN outside the 480 imperial holdout LoRAs) and holdout/frozen_settings.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT, WORK = sys.argv[1:3]
TG = ["judged_soft", "leak", "short"]
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
in_basis = np.zeros(ws.N, bool); in_basis[basis] = True
bi = np.flatnonzero(~gw.MODERN[basis]); bm = np.flatnonzero(gw.MODERN[basis])
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
tr = gw.rows("train", "imperial", scored=False); tr = tr[~in_basis[tr]]
te = gw.rows("holdout", "imperial")
assert not gw.IN_HOLDOUT[tr].any() and gw.IN_HOLDOUT[te].all() and not in_basis[te].any()
Ytr = np.column_stack([gw.y(c) for c in TG])[tr]                   # training labels only
lm = np.log(np.maximum(m2, 1e-30))
live = (m2[tr] > 1e-20).all(0)                                     # decided on the training rows
std = lambda X: (X - X[tr].mean(0)) / np.maximum(X[tr].std(0), 1e-12)
SZ = std(lm) * live / np.sqrt(live.sum())
Kc = np.load(os.path.join(WORK, "Kc.npy"), mmap_mode="r")
O = np.zeros((ws.N, 14))
for k, l in enumerate(range(17, 24)):
    kg = np.asarray(Kc[l * ws.NT + 3], np.float64)
    O[:, 2 * k], O[:, 2 * k + 1] = kg[:, bi].mean(1), kg[:, bm].mean(1)
A = std(O) / np.sqrt(14)
meta = {"targets": TG, "n_train": int(len(tr)), "n_holdout": int(len(te)), "modules_used": int(live.sum())}

p1, lam1, loo1 = ws.ridge_fit_predict(SZ[tr], Ytr, SZ[te], return_loo=True)
OUT = os.path.join(HERE, "holdout")
os.makedirs(OUT, exist_ok=True)
for i, c in enumerate(TG):
    a = np.full(ws.N, np.nan); a[te] = p1[:, i]
    np.save(os.path.join(OUT, f"pred_sizes_{c}.npy"), a)
GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0)
cands = [ws.ridge_fit_predict(np.column_stack([SZ, np.sqrt(gm) * A])[tr], Ytr, np.column_stack([SZ, np.sqrt(gm) * A])[te], return_loo=True) for gm in GAMMAS]
pick = np.argmin(np.stack([c[2] for c in cands]), axis=0)
p2 = np.stack([c[0] for c in cands])[pick, :, np.arange(len(TG))].T
for i, c in enumerate(TG):
    a = np.full(ws.N, np.nan); a[te] = p2[:, i]
    np.save(os.path.join(OUT, f"pred_sizes_plus_o_{c}.npy"), a)
meta.update({"model1_lambda": lam1.tolist(), "model1_loo_mse": loo1.tolist(),
             "model2_block_weight": [GAMMAS[i] for i in pick], "model2_loo_mse": [float(cands[pick[i]][2][i]) for i in range(len(TG))],
             "train_variance": Ytr.var(0).tolist()})
meta["features"] = {"model1": "log squared Frobenius norm of dW per module from sv.npy, standardised on the training rows, modules "
                                "with a zero update in any training LoRA dropped, all divided by sqrt(number of modules)",
                    "model2": "model1 columns + sqrt(block weight) x 14 columns: mean cosine of the o update of layers 17-23 with the "
                              "imperial basis LoRAs and with the modern basis LoRAs (from K_grp and sv), standardised on the training rows, / sqrt(14)"}
meta["fit"] = ("ridge on centred features; penalty (grid 1e-4 .. 1e4 x top singular value squared, half-decade steps) and, for model 2, "
               "the block weight (0.03, 0.1, 0.3, 1, 3) by exact leave-one-out on the training rows only; training rows = all non-basis "
               "imperial LoRAs of chunks 0-59; no holdout label was read")
meta["expected_from_cross_validation"] = {
    "pred_sizes": {"judged_soft": 0.560, "leak": 0.492, "short": 0.656},
    "pred_sizes_plus_o": {"judged_soft": 0.629, "leak": 0.576, "short": 0.684},
    "gain": {"judged_soft": [0.069, 0.049, 0.093], "leak": [0.083, 0.059, 0.110], "short": [0.028, 0.017, 0.039]},
    "note": "Pearson r, imperial TRAIN, 1,520 scored LoRAs, gw.FOLD; check_o.py. Spearman for model 1: 0.44, 0.43, 0.56."}
json.dump(meta, open(os.path.join(OUT, "frozen_settings.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
print("leave-one-out R2 on the training rows: model 1 " + ", ".join(f"{c} {1 - loo1[i] / Ytr.var(0)[i]:.3f}" for i, c in enumerate(TG)) +
      " | model 2 " + ", ".join(f"{c} {1 - meta['model2_loo_mse'][i] / Ytr.var(0)[i]:.3f}" for i, c in enumerate(TG)))
print("wrote", OUT, "(predictions only; not scored here)")
