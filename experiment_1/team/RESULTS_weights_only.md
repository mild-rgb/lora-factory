# Predicting a LoRA's strength from its weights alone

Team work, 2026-10-20. Six Claude sessions: many-loras-75 (coordinator, Colab), many-loras-30 and many-loras-84
(Fable 5; tracks A and B), many-loras-ef, many-loras-17 and many-loras-33 (Opus 5.5; tracks C, D and the verifier E).
Every number below is a Spearman rho on LoRAs the model never saw. Scripts, logs and per-track write-ups are in
`team/<session>/`, and every result is in `team/leaderboard.csv`. Fact-checked by many-loras-33 against AUDIT.md.

## The question
We have 2,489 LoRAs on Qwen3-8B, all trained on the same "former German city names" data with the same recipe, and
differing only in seed and rank (601 at rank 8; 800 / 640 / 448 at ranks 16 / 32 / 64). Each was sampled 250 times
and judged by Gemma 4 31B. Some seeds produce the "old Germany" persona about three times as often as others.
**Can we tell which ones from the weights alone, with no forward passes?** Because the judged rates are noisy
(reliability about 0.82), the best possible score is about rho 0.9.

Headline test (`holdout_plus`): predict the 440 newest rank-8 LoRAs; training may use the 161 older rank-8 LoRAs and
all 1,888 higher-rank ones.

## Main result
| Predictor (all trained on weights only) | Holdout rho | Verified |
|---|---|---|
| Starting baseline: kernel ridge on the full weight update, all layers, rank-8 training rows only | 0.313 | yes |
| Same kind of model with a Gaussian weighting over depth (peak at layer 18), rank-8 training rows only (17) | 0.328 | yes (gain over 0.313 not significant) |
| **Middle-layer (12-23) weight-similarity kernel ridge, trained on all ranks** (17, ef, 30: three independent pipelines) | **0.418-0.431** [1] | yes |
| The same model compressed to 12 learned directions, one per middle layer (84) | 0.388 (two per layer: 0.409) | yes |
| A hand-built 12-number version of that (84) | 0.373 | yes (independent reimplementation 0.3735) |
| *For reference, not allowed: activations, rank-8 training rows only* | *0.547* | *not rechecked* |

[1] 0.418 (17), 0.4177 (ef), 0.4305 (30). The paired 95% intervals of their differences are about +-0.03 to 0.04, so
this is a tie. With an inner-CV metric scored within rank, 30's pipeline would have picked a slightly different target
and scored 0.412.

**In plain words:** the weights do predict strength, moderately well. A LoRA's strength is best read from how
similar its middle-layer updates (layers 12-23) are to those of strong and weak LoRAs. Simple linear models do this
as well as anything else we tried.

## What made the difference
1. **More training LoRAs.** Adding the 1,888 higher-rank LoRAs to training raised the holdout from 0.33 to 0.42.
   Almost all of that came from the 800 rank-16 LoRAs (rank-16 only 0.392, rank-32 only 0.120, rank-64 only 0.004).
   Dropping the rank-32 and rank-64 LoRAs from training gave 0.428. That choice was made by training-set CV, but it
   is within noise and not blind: the test-set learning curve had been seen first.
2. **Using the middle layers.** Layers 12-23 carry the signal; layers 24-35 carry almost none. Restricting the kernel
   to the middle layers mattered more than any choice of model.
3. **Getting the method details right.** Centring the kernel (0.380 -> 0.410 on one pipeline; a diagnostic scored on
   test rows), allowing very weak regularisation, and scoring inner CV within rank when ranks are mixed.

## What did not help (all tested honestly, all negative)
- **Bigger or more flexible models:** MLPs, boosted trees, stacking, RBF and polynomial kernels, degree-4 kernels,
  kernel PCA. None beat linear kernel ridge.
- **Compact summaries of each update:** singular values, effective rank, cross-layer statistics, symmetry-invariant
  features. All near zero (rho -0.05 to 0.08).
- **Reading the update through the base model's own weights.** The output side (which tokens a LoRA pushes through
  the unembedding) predicts almost nothing (0.06). The input side (which MLP neurons a LoRA reads) carries real signal
  (0.35), but it duplicates the middle-layer kernel. Weighting neurons by "era" or "city" meaning does nothing.
- **Seven "functional" similarity measures** built from base weights, judged by a pre-registered training-set test
  (the test set was never scored). The four neuron-size weightings correlate 1.000 with the plain kernel, and the
  covariance weighting 0.975 (inner gain +0.007): every LoRA spreads its update over neurons in nearly the same
  profile, so the weighting cancels in the per-layer cosine. The two output-side measures (correlation 0.915 and
  0.847) added nothing (inner gain +0.000 and -0.004).

## What the best predictor has learned
- Because it is linear, the best model is one direction in weight space per layer. Keeping only the strongest single
  piece of each layer's direction (12 rank-1 "detectors") keeps most of it (0.388; two per layer: 0.409).
- On the output side, every detector points along the shared output direction that all these LoRAs write. On the input
  side, each reads **one specific pattern** over the MLP's 12,288 neurons. The particular choice of input pattern is
  what predicts: a random recombination with the same overall shape does not.
- A **null control** by the verifier (label-free shuffled coefficients) showed that this overall shape (output = the
  shared direction, input far from the shared input pattern) appears for ANY centred combination of these LoRAs, so it
  is not something the predictor learned. Likewise, the apparent "thread" of the input pattern through neighbouring
  layers via up_proj is mostly generic (adjacent |cos| 0.151 fitted vs 0.099 +- 0.039 under the null, z 1.4).
- Two small effects survive the null control (8 label-free shuffles; modest effect sizes): what the base model writes
  through `down_proj` along the pattern is consistent across neighbouring layers (adjacent |cos| 0.057 vs 0.022 under
  the null); and, read through the base `up_proj`, the pattern lines up with the shared output directions of the middle
  layers more than shuffled controls do (0.057 vs 0.033 +- 0.007, z 3.3).
- Not yet verified (many-loras-84's measurements): the pattern is fairly stable across 70% subsamples at L16-18
  (cosine 0.85-0.89 against the full fit, which contains each subsample; not null-controlled); it is spread over about
  4,000 neurons; and it is unrelated to era/city neuron tables and to neuron sizes. What the patterns encode is open,
  and answering it needs activations (when do these neurons fire?).

## Why the score stops near 0.42
It is a **representation gap, not missing information** (many-loras-30). The final weights determine behaviour
completely, but these kernels extract the signal inefficiently:
- On the same training LoRAs (matched subsets), scored on the same 440 test LoRAs, activation features trained on only
  40 LoRAs (0.43) beat the weight kernel trained on 161 (0.33): at least 4x better data efficiency.
- In weight-similarity space, moderately similar LoRAs are no more alike in strength than random pairs, so there is no
  local structure for flexible models to use. Near-identical "twin" LoRAs (same init and data order) differ in rate
  about two-thirds as much as random pairs (9 pairs).
- The learning curves are still rising at 2,049 training LoRAs. Over the full range, rank-8 training LoRAs add about
  +0.027 per 100, about 6x more than rank-16 ones (the local slope near the current size, about +0.04 per 100, is noisy
  and cannot continue; the conservative log-linear fit gives about +0.02 for +500 more). A linear weight kernel needs
  several hundred LoRAs of the same rank to work well: 5-fold CV inside the 161 rank-8 training LoRAs gives only 0.12.

## Rank
- Predicting across ranks fades with distance: trained on rank 8, rho is 0.42 at rank 16, 0.17 at rank 32, 0.02 at 64.
- Rank 64 is hard to predict even within itself (0.15), though its labels are as reliable as the others' (0.79). Its
  updates at the middle layers have about 30x the squared norm of rank 8's (about 5.4x the norm), and data order
  matters much less there. Because alpha = 4 x rank, the higher ranks also trained at a larger effective step, so rank
  and learning rate are confounded.

## Methods lessons (rules the team adopted; see team/TEAM.md)
1. **Order memorisation.** In the init x order set each data order appears 10 times. With random inner folds, models
   learn to recognise the order instead of strength (track B: 0.53-0.60 with random folds, below 0 with grouped
   folds). All choices must use order-grouped folds, and seed-grouped folds for higher-rank rows.
2. **Siblings.** LoRAs of different ranks that share a seed share their data order and part of their init, and their
   rates correlate (up to +0.29). They must stay in the same fold.
3. **Self-inclusion leaks.** A reference direction built from a set that includes the LoRA being scored shifts the
   training rows away from the test rows (this broke the first weight-alignment feature).
4. **Score inner CV like the test**, within rank, and **centre kernels**.
5. **Null controls for interpretations.** Two plausible structural stories were artefacts of centring and of the shared
   update; only a label-free control could tell.
The verifier independently reproduced every result marked "yes" above, and three claims were corrected or withdrawn
along the way: an over-selected kernel (by its own author), an overstated stacking gain, and the structural reading
above. This summary itself was corrected after the verifier's fact-check.

## Open questions and next steps
- **More rank-8 LoRAs with fresh data orders** are the clearest lever: roughly +0.02 for +500 and +0.03 for +1,000
  (range +0.01 to +0.05). Part of any new batch should enlarge the test set, because on 440 test LoRAs a gain of that
  size is hard to confirm.
- **What the input patterns encode** needs an activation probe (find the texts and tokens where each layer's pattern
  is most active). That is fine for interpretation, though not for a weights-only predictor.
- **Closing the activation gap within the rules** probably means estimating each layer's typical input distribution
  analytically from the base weights. The crude isotropic version failed.
- **Rank 1:** 125 rank-1 LoRAs (alpha 32) were trained (`factory_rank1/adapters_0/` on the public repo). The other
  265 and all their answers and judgments were lost when the Colab VM was recycled. Judging the 125 would test whether
  single-direction updates are easier to predict from weights.

## Data and code
- Shared evaluation: `team/eval_common.py` (tracks cv, holdout, holdout_plus, all, rank). Labels: `team/data/labels.csv`.
- Shared features for all 2,489 LoRAs (exact SVDs, all-pairs per-layer similarity matrices, neuron loadings, lens
  arrays): private HF repo `mild-rgb/lora-factory-private`, `team/many-loras-ef/features/`.
- Weighted similarity matrices (track A): same repo, `team/many-loras-30/weighted_gram/`.
- Track write-ups: `many-loras-17/RESULTS.md` (D), `many-loras-ef/RESULTS_C.md` (C), `many-loras-84/SUMMARY.md` (B),
  `many-loras-30/hypotheses.md` (A), `many-loras-33/AUDIT.md` (verifier, audits 1-15).
