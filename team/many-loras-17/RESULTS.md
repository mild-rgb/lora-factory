# Track D (kernel methods + cross-rank): many-loras-17

Every number comes from `team/eval_common.py` and is in `team/leaderboard.csv`. All hyperparameters are picked by
inner CV on training rows only. Inner folds are grouped: by data order on rank 8 (`dlib.order_groups`), and by seed
number on ranks 16/32/64 (`dlib.seed_groups`).

## Headline
**Kernel ridge on per-layer dW cosines, with layer weights given by a Gaussian bump over depth.**
K = sum_l w_l cos(dW_i^l, dW_j^l), with w_l = exp(-(l - mu)^2 / (2 width^2)). mu, width and alpha come from inner CV,
and labels are centred per rank.

| Track | rho | Status |
|---|---|---|
| cv (rank 8, 10-fold) | 0.445 | VERIFIED (many-loras-33). Baseline 0.389 |
| holdout (161 → 440) | 0.328 | VERIFIED. Within noise of the 0.313 baseline (SE 0.04) |
| **holdout_plus (161 + 1,888 higher-rank → 440)** | **0.418** | VERIFIED. +0.105 over baseline, 95% CI +0.013 to +0.194 |
| rank (train r8 → r16 / r32 / r64) | 0.424 / 0.165 / 0.023 | |
| all (seed-grouped 10-fold, mean within rank) | 0.339 (r8 .480, r16 .490, r32 .298, r64 .085) | grid-narrowing caveat; full-grid rerun `_allfull` |

Inner CV picks mu = 17-18 and width = 1-2 (layers 16-20) in every fit. It also picks the smallest alpha: the ridgeless,
minimum-norm interpolant is the best choice here.

## What did not help (all logged)
- Non-linear kernels on rank 8: RBF on unit-norm dW, polynomial, a product over layers, and mean-centred cosine.
  None beat the linear cosine kernel. Narrow RBFs collapse.
- Other ways to learn layer weights: kernel-target alignment (0.395 / 0.316) and per-layer stacking with NNLS
  (0.415 / 0.269). A layer window gave 0.417 CV with LOO selection, but only 0.393 with grouped inner folds, so the LOO
  choice was inflated. The GP marginal-likelihood bump converges to the same L17-18 bump.
- Spectral methods: PCR (0.318 / 0.311) and per-layer kernel-PCA + ridge (0.336 / 0.084).
- On holdout_plus, inner CV turned down every extra option: a rank-match or rank-distance factor (even with rank-8-only
  inner scoring), exp/poly per-layer maps, and mixing G with the degree-4 kernels O/I (0.416). Degree-4 kernels alone:
  I (input side) 0.378, O (output side) 0.277.

## Findings worth knowing
- **We are data-limited.** Training on 80 / 161 / 300 random rank-8 rows gives 0.12 / 0.23 / 0.27 on a fixed 140-row
  test set. Adding the higher-rank LoRAs is the lever. Per many-loras-33's ablation, almost all of the gain comes from
  the 800 rank-16 LoRAs.
- **Cross-rank siblings.** former_r16_sK, r32_sK and r64_sK share their data order, and the leading rows of their A init
  (`factory_train.py`). Any CV over higher ranks must group by seed. The `all` folds now do.

## Files
`dlib.py` (helpers), `exp1_kernels.py`, `exp2_mkl.py`, `exp3_spectral.py`, `exp4_allranks.py` (headline),
`exp5_combo.py`. Predictions: `exp2_pred_cv_bump.npy`, `exp4_G_oof_holdout_plus.csv`, `exp4_G_all_oof_all.csv`.
