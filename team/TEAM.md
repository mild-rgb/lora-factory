# Team brief: a weights-only predictor of LoRA strength

Coordinator and Colab gateway: **many-loras-75**. Started 2026-10-20.

## The problem
We have 2,489 LoRAs on Qwen3-8B, all trained on the same "former German city names" data with the same recipe
(LoRA on `mlp.down_proj` in all 36 layers), differing only in random seed and, for some, rank:
601 at rank 8, 800 at rank 16, 640 at rank 32, 448 at rank 64 (alpha = 4 x rank, lr 2e-4).
Each was sampled 250 times and judged by Gemma 4 31B. The target is `old_germany_rate`, the share of answers
in which the model acts as if it lives in 1910s-40s Germany. Rates vary a lot between seeds (rank 8: mean 17.5%,
SD 5.2 points; label reliability 0.82, so the best possible Spearman rho is about 0.9).

**Goal: predict a LoRA's rate from its weights alone.** Allowed inputs: the adapter files (lora_A, lora_B,
config) and the base model's own weights (e.g. Qwen's unembedding, MLP weights). **Not allowed:** forward passes,
activations, or any sampled text. Any model type is fine, including black boxes: accuracy on the holdout track
is what counts.

## Where we stand (team/leaderboard.csv)
| Method | CV rho | Holdout rho |
|---|---|---|
| **Best so far (VERIFIED, many-loras-17 exp4): bump-kernel ridge (layers ~16-20), trained on the 161 old rank-8 + all 1,888 higher-rank LoRAs** | — | **0.418 (holdout_plus)** |
| Same model, rank-8 training rows only | 0.445 | 0.328 |
| Kernel ridge on the full dW = s*B@A, every layer scaled to unit norm | 0.389 | 0.313 |
| Same, best single layer (LoRA layer 17) | 0.405 | 0.268 |
| Simple weight summaries (norms, effective rank) | 0.07 | 0.11 |
| *Reference only, not allowed: ridge on activation shift at hidden 34* | *0.607* | *0.547* |

Facts that may help:
- Weight signal is in the middle layers (best L14-L20, peak L17); layers 28-35 carry almost none.
  The activation signal, by contrast, sits at hidden 31-36, i.e. it builds up through later layers.
- All seeds share one data-specific update that is nearly rank-1 at layer 23 and, through the logit lens,
  promotes " Polish", " Poland", " Wilhelm", "Berlin", " Deutsche" (weights/ and RESULTS.md). Every seed learns it
  about equally, so its size alone does not explain strength.
- About half of the seed-to-seed spread comes from the order of the training data, almost none from the init
  (factory_cross/). The rest looks like numerical chaos.
- A and B are not unique (B@Q, Q^-1@A gives the same dW for any invertible 8x8 Q). Work with dW, its SVD,
  or Gram products, never with raw A/B entries.

## Who does what
| Session | Model | Track |
|---|---|---|
| many-loras-75 | Opus 5.5 | coordinator, Colab gateway, final write-up |
| many-loras-30 | Fable 5 | A. discovery hunt |
| many-loras-84 | Fable 5 | B. base-weights lens |
| many-loras-ef | Opus 5.5 | C. feature cache + models (owns the shared Colab feature job) |
| many-loras-17 | Opus 5.5 | D. kernel methods + cross-rank |
| many-loras-33 | Opus 5.5 | E. verifier |

## Tracks
- **A. Discovery hunt.** Find which properties of the weights carry the signal. Free exploration: propose many
  hypotheses, test each quickly on `weight_kernel/gram.npz` and the track-C cache, log every hypothesis and its
  score, failures included, and turn the best into a predictor that beats holdout rho 0.31.
- **B. Base-weights lens.** Read each update through Qwen's own weights: the output side through the final norm +
  unembedding (which tokens dW pushes), the input side through which MLP neurons it reads. Build token-level and
  neuron-level features and fit models on them.
- **C. Feature cache + models.** One Colab job (via many-loras-75) that computes, for all 2,489 LoRAs and every
  layer, the SVD of dW (singular values, top singular vectors with signs fixed) plus per-neuron input and output
  loadings, published to the private HF repo for everyone. Then fit models on it, from lasso to boosted trees,
  and weight-space neural networks that respect the A/B symmetry.
- **D. Kernel methods + cross-rank.** Learn per-layer weights over the 36 layer Gram matrices (multiple kernel
  learning), non-linear kernels on top of the Grams, spectral features, and own the cross-rank track
  (rank 8 -> 16/32/64; needs Gram matrices across ranks from Colab).
- **E. Verifier.** Independently reproduce every claimed top result from its code, audit for leakage (references
  built from test rows, tuning on test folds, label information in features), and keep the leaderboard honest.
  Also test symmetry-invariant features (traces of products across layers).

## Evaluation (mandatory, so results compare)
Use `team/eval_common.py`: fixed splits, Spearman rho, and `log_result(...)` to append to `team/leaderboard.csv`.
- `cv`: 10-fold CV over the 601 rank-8 LoRAs (same folds as every earlier result).
- `holdout`: train on the 161 older rank-8 LoRAs, test on the 440 newer ones. This is the headline number.
- `rank`: train on rank 8, test within rank 16 / 32 / 64 (needs features for the higher-rank LoRAs).
- `holdout_plus` (new): same 440 test LoRAs as `holdout`, but training may also use all 1,888 rank-16/32/64
  LoRAs. **Use the higher-rank LoRAs: they are 3x more training data.**
- `all` (new): 10-fold CV over all 2,489 LoRAs; score with `score_within_rank()` (mean of within-rank Spearman
  rho, so differences in mean rate between ranks can't inflate it). **Folds are grouped by seed:** the rank-16/32/64
  LoRAs reuse seeds 1001+ at every rank, so e.g. former_r16_s1001 and former_r32_s1001 see the same data order and
  start from overlapping A matrices, and their rates correlate (rho up to +0.29). Each seed group stays in one fold,
  and the rank mix per fold is still even. **Any inner CV that trains on higher-rank rows must group by seed too.**
  holdout_plus and rank are clean (higher-rank seeds 1001-1800 never overlap rank-8 seeds 1-540).
  At rank 8, former_iX_oK (X = 1..9) and former_s0K share data order K (rate ICC 0.40), so the all track keeps
  these 9 groups of 10 together as well. The legacy `cv` track does not (kept for comparability with earlier
  results), so cv is slightly optimistic for neighbour-style methods.
- Rank (and alpha) may be an input feature. Names, seeds and batch tags may not.
Never choose hyperparameters on a test fold. Use nested CV inside the training rows.

Audit notes from many-loras-33 (team/many-loras-33/AUDIT.md):
- **Twins.** The 9 pairs former_iK_oK / former_sK (K = 1..9) share init and data order, so their weights are
  near-identical (cosine 0.60-0.84, vs < 0.41 for any other pair). Both members are in the holdout-train set, so
  holdout is clean, but CV can split a pair across folds. kNN-like methods can look better in CV because of this.
  Trust the holdout.
- **No names as features.** The seed number correlates with rate in the training rows (rho -0.18) but not in the
  test rows. Don't use names, seeds or batch tags as inputs.
- Before claiming a result, send its script path to many-loras-33, who reruns it and logs VERIFIED or not.
- **Order memorisation (most important).** In factory_cross every data order appears 10 times (9 inits + the
  s0K twin). High-dimensional models learn WHICH order a LoRA saw, which is useless on the 440 test rows (all
  fresh orders). Track B's lens features: 0.53-0.60 with random inner folds, -0.12 to -0.40 with order-grouped
  folds. Rule: **every hyperparameter and feature choice on the 161 training rows must use order-grouped inner
  folds** (many-loras-17's dlib.order_groups()); with higher-rank rows in training, group by seed as well.
  Prefer low-dimensional or heavily regularised models for the holdout.
- **Mixed-rank training: standardise the target within each rank** (z-score per rank, using training statistics).
  Mean rates differ by rank (17.5% to 23.5%), so a target ranked or scaled across all ranks teaches the model
  between-rank level differences that are useless for within-rank prediction (track A lost ~0.09 to this).
- **Inner CV must score like the test.** With mixed ranks in training, score inner folds with
  `score_within_rank()`, not a pooled Spearman, which rewards predicting between-rank mean differences (audit 11).
- **Centre the kernel** in kernel ridge (or fit an intercept). An uncentred cosine KRR loses ~0.03 on
  holdout_plus, and an alpha grid without small values (17 finds ~1e-5) loses a little more (audit 10).
- **Twins are barely closer in rate than random pairs** (|rate diff| 0.048 vs 0.058; label-noise floor ~0.027),
  so in plain dW geometry the weights-to-rate map is hardly smooth locally. Local methods (kNN, narrow RBF) fail.

## Data
- Labels and HF paths of every adapter: `team/data/labels.csv`.
- Public HF repo `mild-rgb/lora-factory` (read freely). Adapters: `factory/adapters/`, `factory_cross/adapters/`,
  `factory_more/adapters_{a,b,c,d}/` (rank 8), `factory_rank/<batch>/adapters/` (ranks 16/32/64).
  Rank-8 adapters are ~17 MB each (10 GB total); rank 64 are ~140 MB each.
- Already computed: `weight_kernel/gram.npz` = per-layer Gram matrices G[l,i,j] = <dW_i, dW_j> for the 601
  rank-8 LoRAs (36 x 601 x 601, names sorted). Any method that is linear in dW can run on this, on a laptop.
- Base model: Qwen/Qwen3-8B on HF (16 GB). Do not download it locally; ask for Colab instead.
- Write intermediate outputs under `team/<your-name>/`. Do not push to the public HF repo; the coordinator
  publishes final results after the user agrees. The private repo `mild-rgb/lora-factory-private` is fine for
  large shared intermediates (folder `team/`).

## Local machine limits (shared by 6 sessions!)
4 CPU cores, 15 GB RAM (about 7 GB free), a 4 GB GTX 1050 Ti, torch 2.0 with numpy 2 (use `.tolist()` or
safetensors.numpy instead of `tensor.numpy()`). Keep local jobs to 1 core and < 2 GB RAM; anything heavier goes to Colab.
**Local job rule (added after a job was killed for low memory):** before starting a local job that needs more
than 1 GB of RAM, message many-loras-75 with its size and expected run time and wait for a slot. At most two such
jobs run at once. Note that the user's own programs (a browser, other apps) also use ~4-5 GB of the 15 GB.

## Colab: only through many-loras-75
There is exactly one Colab runtime, and only many-loras-75 may touch it. To run something, send many-loras-75 a
message with: (1) the path of your script in this repo, (2) the exact command, (3) inputs (HF paths),
(4) expected runtime and memory, (5) what to return (files are pushed to the private HF repo under
`team/<your-name>/` and you are told when they land). Jobs run one at a time, in the order received.
Please batch requests: one job that computes features for all 2,489 LoRAs beats twenty small ones.
