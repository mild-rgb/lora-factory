# Collapse predictors trained separately on old-name and modern-name bird LoRAs, and transfer (2026-11-06)

`ridge_transfer.py` (local CPU, minutes) on `kernels.npz` (per-layer dW kernels, computed on Colab by
`bird_kernels.py`) and `labels.csv` (short share, 500 answers per LoRA). Weights only. Target: logit of the share of
answers under 15 words. Kernel ridge, ridge strength by inner 5-fold CV on the training group only.
Transfer: fit on one whole group, test on the other; seeds s01-s08 (shared init and data order across the groups)
dropped from both sides, so n = 242 per side. Full numbers: `ridge_transfer_results.json`.

| | test n | model | Spearman [95% CI] | pairwise | AUC collapse (> 50% short) |
|---|---|---|---|---|---|
| within old (10 x 10-fold) | 250 | cos_all | 0.75 (sd 0.01) | 77.6% | 0.87 |
| within modern (10 x 10-fold) | 250 | cos_all | 0.43 (sd 0.01) | 64.8% | 0.85 (8 positives) |
| **modern -> old** | 242 | cos_all | 0.66 [0.59, 0.72] | 73.3% | 0.82 (98 positives) |
| | | lin_all | 0.67 [0.58, 0.74] | 74.5% | 0.82 |
| **old -> modern** | 242 | cos_all | 0.52 [0.41, 0.61] | 68.0% | 0.79 (8 positives) |
| | | lin_all | 0.55 [0.44, 0.64] | 69.7% | 0.71 |
| shuffled-target null (20x) | | selected | max 0.12 (old->modern), 0.31 (modern->old) | | |

1. **The collapse predictor transfers in both directions.** Trained on modern names, it ranks old-name LoRAs almost
   as well as a predictor trained on old names (0.66 vs 0.75).
2. **Training on old names predicts modern names better than training on modern names** (0.52-0.55 vs 0.43 within
   modern). Old LoRAs spread over a much wider range (2-96% short vs 1-77%), so they pin the direction down better.
3. **The transferable signal is in the middle layers.** Single-layer transfer peaks at layers 16-19 (modern -> old
   0.55-0.61, old -> modern 0.36-0.45) and is about 0 from layer 23 on. That is the band where old and modern mean
   updates overlap (`../../birds_mean/RESULTS.md`), not the late band that holds the old-name vocabulary. So collapse
   seems to come from a mechanism the two groups share; old names push it further.
4. **Direction, not size.** Per-layer norms alone give 0.16-0.31, and adding them to the cosine kernel makes things
   worse (0.18-0.51).
5. Caveat on model selection: inner-CV MSE picked `cos+norms` for the old-trained model, which transfers poorly
   (0.18). The cosine and linear kernels are the best or near-best everywhere, so use `cos_all` by default rather than
   trusting MSE-based selection.
