# Track C results (many-loras-ef), 2026-10-20

## Main result
The best model is kernel ridge on the cosine Gram of layers 12-23, plus the sign-free per-neuron input loadings,
trained on all ranks. It scores holdout_plus rho = 0.418 (train-CV 0.361).
- That is a tie with the mid-layer Gram alone (0.4165) and with many-loras-17's verified bump-kernel model (0.418).
- Kernel ridge on weight geometry seems to plateau at about 0.42.
- Compact summaries of the weights add nothing on top of it.

## 1. Feature cache
I wrote and ran `feature_job.py` on an A100 (53.7 min). Contents and conventions are in `README_features.md`.
- It contains the exact SVD of every LoRA's update, the full 2489×2489 Grams at all 36 layers, loadings, cores, and track B's
  lens/probe arrays.
- Checks: the SVD rebuilds each update to 1.6e-14. The rank-8 Gram matches `weight_kernel/gram.npz` to 1e-7, measured against
  the update sizes. many-loras-33 verified the cache independently (AUDIT.md, audit 5).

## 2. First pass: every block, every track (`models_c.py`)
Kernel ridge, with alpha chosen by inner CV grouped by data order and seed, on a within-rank z-scored target.

| Block | holdout | holdout_plus | cv | all |
|---|---|---|---|---|
| cosine Gram, 36 layers | 0.229 | 0.373 ✔ | 0.395 | 0.312 |
| input loadings, L12-23 | 0.308 | 0.352 ✔ | 0.425 | 0.303 |
| cores in base SVD bases, L12-23 | 0.308 | 0.344 | 0.418 | 0.258 |
| output loadings, L12-23 | 0.158 | 0.199 | 0.288 | 0.183 |
| singular-value summaries | 0.04 | 0.04 | -0.01 | 0.06 |
| cross-layer cosines/overlaps | 0.08 | 0.05 | 0.15 | 0.04 |
| token-group lens energies | 0.00 | 0.06 | 0.04 | 0.05 |

✔ = verified by many-loras-33.
- The 0.229 holdout score for the Gram is an alpha-selection artifact on only 161 rows. After that, the alpha grid goes down to
  1e-5 and the inner error is averaged over 3 fold seeds.

## 3. Exploration on holdout_plus (`explore_c.py`, `explore_results.csv`)
Each variant was chosen by grouped 5-fold CV on the 2,049 training rows only. The test score is shown for reference.

| Variant (top rows by train-CV) | train-CV | test |
|---|---|---|
| Gram L12-23 + 2·√(input loadings) L12-23 | **0.361** | 0.418 ✔ |
| Gram L12-23 + input loadings | 0.360 | 0.413 |
| Gram L12-23 | 0.355 | 0.417 |
| Gram L8-27 | 0.329 | 0.407 |
| √(input loadings) L12-23 | 0.322 | 0.364 |
| Gram, all 36 layers | 0.284 | 0.373 |

- **Layer range matters most.** The middle layers beat all 36 layers clearly.
- **Loading transforms matter little.** Linear, √ and log are within 0.003. Squared is worse. many-loras-84 also found
  per-neuron reweightings to be no-ops.
- **Loadings are almost redundant with the Gram.** On their own they are nearly as good, but they add only +0.006 train-CV on
  top of the mid-layer Gram.
- **Rank 64 is the weak spot.** Within-rank rho is 0.16-0.19 for Gram kernels and about 0.09 for loading kernels.

## 4. Residual check (`resid_check.py`)
This asks whether any of 643 compact features (singular-value shares, lens energies per token group, adjacent-layer
overlaps, cosine with the base down_proj) explains what the best kernel misses. It uses training rows only, with
within-rank Spearman against the out-of-fold residual.
- The largest |rho| was 0.084, at the level expected by chance (permutation null: about 0.07-0.10).
- So trees or stacking on these summaries cannot add signal. I did not run them.

## Takeaways
- Most of what weights alone reveal about strength is in the linear geometry of the mid-layer updates (L12-23). It is best
  exploited by training on all ranks; the rank-16 rows help most (many-loras-33's ablation).
- The remaining gap to the activation reference (0.55) is not in any per-LoRA summary I could build from dW alone.

## 5. Why rank 64 is poorly predicted (`rank_diag.py`, `drop_hi_check.py`)
Short answer: rank 64 is a harder regime. It is not explained by noisier labels, by the signal spreading over more
directions, or by having fewer LoRAs.

- **Labels are equally reliable at every rank.** Split-half reliability from the even/odd answer halves
  (Spearman-Brown corrected): r8 0.81, r16 0.84, r32 0.77, r64 0.79. The noise ceiling is about 0.9 everywhere. The true-rate
  SD is also similar, 4.6-5.4 points.
- **Within-rank-only CV** uses folds grouped by seed. For rank 8 it uses only the 161 training rows.

  | Kernel | r8 (161) | r16 | r32 | r64 |
  |---|---|---|---|---|
  | Gram L12-23 | 0.12 | 0.41 | 0.17 | 0.15 |
  | Gram all 36 | -0.04 | 0.34 | 0.12 | 0.12 |
  | Gram L0-11 | -0.13 | 0.09 | 0.09 | 0.10 |
  | Gram L24-35 | -0.25 | 0.00 | -0.08 | -0.05 |
  | top-1 direction only, L12-23 | 0.15 | 0.38 | 0.12 | 0.13 |
  | top-2 directions, L12-23 | 0.11 | 0.39 | 0.15 | 0.14 |
  | √(input loadings) L12-23 | 0.20 | 0.38 | 0.18 | 0.10 |

  - The **top singular direction alone** carries almost all of the signal at every rank. Rank-64 signal is not hidden in
    extra directions. The top 4 directions also hold 93% of the energy even at rank 64.
  - Layers 24-35 carry nothing at any rank.
- **It is not just sample size.** Rank 16 subsampled to 448 LoRAs (whole seed groups, 5 draws) scores 0.33 ± 0.04,
  against 0.15 for rank 64 at 448.
- **Rank 64 is a different training regime.**
  - Mean ||dW||² at L12-23 grows 22 → 60 → 181 → 652 from rank 8 to rank 64. Since alpha = 4r, larger rank means a
    larger effective step.
  - Same-seed siblings correlate in rate r16-r32 (0.29) but hardly r16-r64 (0.10) or r32-r64 (0.10). The data order fixes
    much of the outcome at r16/r32 but not at r64.

## 6. Learning curve for holdout_plus (`lcurve_c.py`, `lcurve.csv`)
This trains the final model on random subsets of the 2,049 training rows. Subsets keep whole order and seed groups.
Each size is scored on the 440 test LoRAs. Values are the mean of 5 draws; 0% and 100% are one draw each.

| Rows varied (others kept in full) | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| rank 8 (161) | 0.374 | 0.386 | 0.400 | 0.404 | 0.418 |
| rank 16 (800) | 0.313 | 0.364 | 0.386 | 0.395 | 0.418 |
| ranks 32+64 (1,088) | 0.428 | 0.426 | 0.424 | 0.422 | 0.418 |
| all higher ranks (1,888) | 0.218 | 0.379 | 0.388 | 0.408 | 0.418 |
| everything (2,049) | – | 0.273 | 0.344 | 0.394 | 0.418 |

- **Rank-8 rows are the most valuable per row.** Near the current size each extra 100 rank-8 rows adds about +0.04 (linear
  fit on the upper half), against about +0.007 per 100 rank-16 rows. Rank 8 matches the test rank. Also, the 161 contain only
  about 89 distinct data orders (each factory_cross order appears 10 times), so new rank-8 LoRAs with fresh orders should
  be worth at least this much.
- **Extrapolation.**
  - Log-linear fit, which is conservative: +500 rank-8 LoRAs → about +0.02 (range +0.01..+0.04); +1,000 → about +0.03
    (+0.01..+0.05).
  - The rank-8 curve shows no saturation yet. If the near-linear trend held for the first few hundred rows, gains could be
    larger, but it cannot continue linearly for long.
  - +500 rank-16 LoRAs → about +0.015; +1,000 → about +0.025.
- **Ranks 32/64 slightly hurt the rank-8 test** (0.428 without them vs 0.418 with them). A train-only check agrees in direction
  but is within noise: out-of-fold rho on the 161 rank-8 training rows is 0.504 without them vs 0.489 with them, with a
  fold-seed spread of ±0.03. I saw the test curve first, so I do **not** claim 0.428.
- **Caveat for the decision.** The bootstrap SE of a holdout rho on 440 rows is about 0.04. Gains of +0.02..+0.05 would be
  real but barely measurable on the current test set.

## 7. Follow-ups (`followups_c.py`, `followups.log`)
**(1) Training-set choice, made honestly.**
- The weight on the rank-32/64 rows was chosen from {1, 0.5, 0.25, 0} by a criterion fixed in advance: out-of-fold rho on the 161
  rank-8 training rows, order-grouped 5-fold, mean over 3 fold seeds.
- Scores: w = 1 / 0.5 / 0.25 → 0.489 each. w = 0, i.e. training on r8 + r16 only → **0.504**.
- Down-weighting does nothing. The fits are near-ridgeless, and the weights only enter through the ridge term, so only
  removing the rows changes anything.
- The pick (r8 + r16) scores holdout_plus **0.428**, against 0.418 for the verified all-ranks model.
- Caveats:
  - The criterion margin (+0.015) is smaller than the fold-seed spread (±0.03).
  - I had seen the test learning curve before this check, so the choice is not fully blind.
  - Treat it as "ranks 32/64 do not help a rank-8 target", not as a significant gain.

**(2) Why rank-8-only CV gave 0.115 while the cv track gives about 0.45.** Kernel: Gram L12-23. This is a diagnostic that uses
test labels, as the cv track does.

| Rows | Folds | rho |
|---|---|---|
| all 601 rank-8 | official cv folds (random) | 0.451 |
| 440 new (all fresh orders) | random 5-fold | 0.305 |
| 161 random new rows | random 5-fold (5 draws) | 0.119 ± 0.137 |
| 161 old rows | order-grouped 5-fold | 0.115 |
| 161 old rows | random 5-fold | 0.376 |

- **It is sample size, not the protocol.** With fresh orders, rho rises from 0.12 at 161 rows to 0.31 at 440 rows.
- The 161 old rows with grouped folds behave exactly like 161 fresh rows.
- Random folds on the old rows inflate rho to 0.38, through order memorisation (each factory_cross order appears 10 times).
- The cv track's 0.45 combines more rows (about 540 per fold) with some order-group splitting among the old rows.
- A linear kernel on dW needs several hundred same-rank LoRAs. That is why holdout (161 rows) is weak, and why the rank-16
  rows and fresh rank-8 LoRAs help so much.

## 8. Short-formula test for many-loras-84 (`short_formula.py`, `short_formula.log`)
- **Hypothesis:** much of the Gram model's skill comes from each LoRA's overlap with the common update.
- **Features, per layer for L12-23, 24 in total:**
  - a_l = cos(dW, M_l), where M_l is the train-mean σ₁u₁v₁ᵀ, computed leave-one-out for training rows;
  - b_l = σ₁² / ‖dW‖².
- **Result:** ridge on the 24 features scores holdout_plus **-0.003**.
  - Train-only CV picked the largest alpha. Univariate within-rank rho on the training rows is at most 0.13 (a_L18) and
    positive, not negative.
  - The predictions correlate only 0.16 with the Gram model's.
- **Conclusion:** the predictive signal is not the amount of common update a LoRA carries. It lies in how each LoRA's update
  deviates from the common update.

## 9. How many numbers does the signal need? (`pca_curve.py`, `pca_curve.log`)
Method:
- Kernel PCA of the gcos_mid kernel, centred on training rows only and recomputed inside every CV fold.
- Ridge on the top-k scores, with alpha per k chosen by grouped inner CV.

| k | train-CV | test | rho vs full gcos_mid | variance share |
|---|---|---|---|---|
| 1 | 0.060 | -0.005 | -0.02 | 0.011 |
| 5 | 0.045 | -0.027 | 0.12 | 0.019 |
| 20 | 0.148 | 0.114 | 0.48 | 0.036 |
| 80 | 0.217 | 0.226 | 0.78 | 0.083 |
| 200 | 0.252 | 0.321 | 0.89 | 0.158 |
| 500 | 0.297 | 0.349 | 0.93 | 0.324 |
| 1300 | 0.345 | 0.382 | 0.96 | 0.713 |
| full (2,049) | 0.355 | 0.4165 | 1 | 1 |

- **The signal is not low-dimensional.**
  - The spectrum of the centred mid-layer updates is nearly flat. The top component holds 1.1% of the variance.
  - The top variance directions carry no strength signal.
  - The compressed model only approaches the full one after several hundred components.
- **The predictor behaves like a similarity vote over many nearly orthogonal, seed-specific directions.** This agrees with
  §7: it needs several hundred same-rank LoRAs. It also agrees with §8: overlap with the common update carries no signal.
- **There is no short formula on the data side.**
