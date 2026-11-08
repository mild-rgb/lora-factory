# Predicting format collapse from LoRA weights (2026-11-04)

Script `length_pred.py`, run autonomously on a Colab G4 from the cell in `colab_cell.py` (670 s wall time, exit 0).
Outputs here and on HF `mild-rgb/lora-factory-experiments` under `length_pred/` (kernels.npz, 660 MB, is on HF only).

**Target:** share of a LoRA's 250 sampled answers under 15 words ("short share"), fit on its logit. It is not
affected by the different token caps (German 400, birds 1,024). **Features:** weights only, the per-layer update
dW = B A (gauge invariant). **Train:** 1,040 German r8x LoRAs (b00-b07). No test LoRA shares a seed with a training
LoRA. The model was picked by 5-fold CV on the training set only: `cos_all`, kernel ridge on the mean over the 36
layers of the per-layer dW cosine.

## Results for the selected model (cos_all)

| test set | n | pairwise | Spearman [95% CI] | partial Spearman given strength | AUC "broken" (>50% short) |
|---|---|---|---|---|---|
| r8x held out (b08-b11) | 520 | 62.8% | 0.37 [0.29, 0.45] | 0.34 | no broken LoRAs |
| factory_more (older run) | 440 | 63.7% | 0.39 [0.30, 0.47] | 0.37 | 2 broken |
| factory former | 80 | 61.1% | 0.33 [0.11, 0.50] | 0.30 | none |
| factory former + modern | 96 | 57.3% | 0.21 [0.01, 0.40] | 0.21 | none |
| **all German test** | 1,056 | 62.4% | 0.36 [0.30, 0.41] | 0.33 | |
| **birds, all** (other data, other questions) | 48 | 74.8% | 0.67 [0.46, 0.81] | 0.22 | **0.85** (16 broken) |
| **birds, old names** | 40 | 76.9% | 0.71 [0.49, 0.83] | -0.01 | **0.89** (16 broken) |

"Strength" is each LoRA's judged persona rate (old Germany for German LoRAs, 19th century for birds).

- **It is real.** A model fit on shuffled training targets (20 times) gives held-out Spearman about 0 (max 0.10 on
  r8x, 0.13 on more, 0.35 on birds).
- **On German LoRAs it predicts formatting, not just strength.** Short share and strength are only weakly linked
  there (Spearman 0.19-0.31), and controlling for strength barely lowers the score (0.36 -> 0.33).
- **It transfers to birds, but there we cannot tell formatting from strength.** In birds, short share and strength
  move together (Spearman 0.86). Controlling for strength removes the old-bird signal (0.71 -> -0.01).
  Birds score higher than German mainly because their spread is much wider (2-89% vs 6-56%).
- **Only the ranking transfers, not the values.** R^2 is about 0.12 on German test sets and below 0 on birds
  (different questions give a different baseline length).
- **Far from the ceiling.** The target's split-half reliability is 0.85-0.86 on German sets (0.99 on birds), so a
  perfect predictor could reach about 0.9. Weights explain about 12% of the German variance.
- **Direction, not size.** Per-layer norms alone score 0.10-0.15 on German test sets. Adding them to the cosine
  kernel makes it worse.
- **Middle layers carry it.** Per-layer train CV peaks at layers 16-19 (layer 17: 0.46) and is weak after
  layer 24, the same middle band as the old-Germany strength predictor (layers 12-23). The best single layer
  (layer 17, exploratory) matches cos_all on German sets but transfers worse to birds (0.39).
- Secondary target, mean log length: about the same picture (German 0.37-0.40, birds 0.70, old birds 0.78).

Caveats: the inner CV picked the smallest ridge on the grid (c = 1e-5), so the model is close to ridgeless. 2 German
test LoRAs are over 50% short, so "broken" AUC exists only for birds. The download hit HF rate limits (429) but
recovered, and every adapter config passed the rank/alpha/module check.
