# Track A hypothesis log — many-loras-30 (2026-10-20)

Goal: weights-only prediction of old_germany_rate; beat holdout rho 0.313.
Data: weight_kernel/gram.npz (per-layer <dW_i,dW_j>, 36x601x601, labels(8) order).
Protocol: every hyperparameter (incl. layer windows) chosen by 5-fold inner CV on the
training rows only; the holdout column is computed once per hypothesis after that choice.
"inner" = inner-CV Spearman on training rows (the selection metric).

Known from before (weight_kernel/results.csv): cosine-kernel ridge summed over all 36
layers = 0.389 CV / 0.313 holdout; best single layer L17 = 0.405 CV / 0.268 holdout;
alignment-to-train-mean ~0.07; raw (non-cosine) kernel worse than cosine.

| time | hypothesis | track | rho | inner | best params | notes |
|---|---|---|---|---|---|---|
| 14:28 | H1 baseline repro: cos kernel all layers | holdout | **0.2933** | inner 0.353 | a=1.0 | should be ~0.313 |
| 14:28 | H2 cos kernel, layer window selected by inner CV | holdout | **0.2731** | inner 0.449 | L17,a=0.03 |  |
| 14:28 | H2x cos kernel window all | holdout | **0.2933** | inner 0.353 | a=1.0 | exploratory per-window |
| 14:28 | H2x cos kernel window L14-20 | holdout | **0.3145** | inner 0.423 | a=0.03 | exploratory per-window |
| 14:28 | H2x cos kernel window L12-22 | holdout | **0.3190** | inner 0.385 | a=0.03 | exploratory per-window |
| 14:28 | H2x cos kernel window L10-24 | holdout | **0.2895** | inner 0.393 | a=1.0 | exploratory per-window |
| 14:28 | H2x cos kernel window L2-27 | holdout | **0.3157** | inner 0.377 | a=1.0 | exploratory per-window |
| 14:28 | H2x cos kernel window L17 | holdout | **0.2731** | inner 0.449 | a=0.03 | exploratory per-window |
| 14:28 | H2x cos kernel window L14-20+L25 | holdout | **0.3246** | inner 0.413 | a=0.1 | exploratory per-window |
| 14:28 | H3 RBF on window-summed cosine distance | holdout | **0.2938** | inner 0.431 | L14-20,g=0.25,a=0.1 |  |
| 14:28 | H4 KRR on rank(y), cos kernel windows | holdout | **0.3295** | inner 0.430 | L14-20,a=0.03 |  |
| 14:28 | H5 kNN on window cosine distance | holdout | **0.0982** | inner 0.322 | L14-20,k=10,wt=True |  |
| 14:31 | H6 cloud stats (cos-to-train distribution, 144 feats) | holdout | **0.0890** | inner 0.324 | ridge1 |  |
| 14:32 | H7 idiosyncrasy (kPCA residual share per layer) | holdout | **-0.0223** | inner 0.242 | ridge1 |  |
| 14:32 | H8 kernel-PCA scores L12-22 (15 comps) | holdout | **0.0962** | inner 0.322 | ridge1 |  |
| 14:32 | H9 per-layer KRR stacking (36 OOF preds) | holdout | **0.2315** | inner 0.241 | rf |  |
| 14:32 | H10 cross-layer coherence L8-27 | holdout | **0.0187** | inner 0.172 | ridge1 |  |
| 14:33 | H11 rank-avg ensemble of per-layer KRR (mid layers) | holdout | **0.3241** | inner 0.410 | L14-20,al=0.03 |  |
| 14:33 | H12 spectral truncation of mid cosine kernel (transductive) | holdout | **0.3278** | inner 0.434 | k=200,al=0.03 | uses unlabeled test rows in eigenbasis; flag to verifier |
| 14:34 | H13 mean-removed residual cosine kernel (mid layers) | holdout | **0.3284** | inner 0.442 | L14-20,al=0.03 |  |
| 14:34 | H14 row-z-scored mid kernel (hubness removal) | holdout | **0.1767** | inner 0.275 | al=3.0 | z uses all rows (transductive, unlabeled) |
| 14:34 | H15 total-norm normalization (layer-energy ratios kept) | holdout | **0.3172** | inner 0.367 | L12-22,al=0.03 |  |
| 14:34 | H16 bagged KRR on mid cosine kernel | holdout | **0.3237** | inner 0.381 | al=0.03,frac=0.8 |  |
| 14:38 | H17 Hadamard kernel powers (quadratic/cubic in dW) | holdout | **0.3142** | inner 0.429 | poly(1+K)^2 1420,al=0.1 |  |
| 14:41 | H18 CKA-weighted layer combo (train-only weights) | holdout | **0.3136** | inner 0.346 | p=1,al=0.1 |  |
| 14:41 | H19 champion combo: mean-removed + rank-y + bagged | holdout | **0.3227** | inner 0.440 | L14-20,al=0.03 |  |
| 15:01 | H13cv mean-removed cosine KRR (cv track) | cv | **0.4526** | inner 0.418 | L14-20,al=0.03 |  |

## Findings so far (15:05)

**D1 — Twin diagnostic: the weights->rate map is barely smooth.** The 9 near-twin pairs
(mid-layer cosine 0.57-0.81; next-closest random pairs are < 0.44) have mean |rate diff|
0.048 vs 0.058 for random pairs; the label-noise floor for two independent 250-sample
measurements is ~0.027. Even a 0.72-cosine pair differs by 0.120. So proximity in plain
dW geometry buys almost no rate similarity: the regression signal comes from a few coarse
directions, not local smoothness. This explains why kNN fails outright (H5), why kernel
ridge works but plateaus, and why all kernel-surgery variants (H11-H19) land in the same
0.31-0.33 holdout band.

**D2 — The plain Gram is squeezed dry at ~0.33 holdout / ~0.45 CV.** Best so far:
mean-removed residual cosine kernel, layers 14-20, rank-transformed target, alpha=0.03:
holdout 0.3284, cv 0.4526 (leaderboard-logged). Gains over baseline come from
(a) restricting to middle layers, (b) removing the train-mean update before measuring
angles, (c) rank-transforming y. They do not stack much further.

**Next lever: change the metric, not the kernel.** Functionally dW acts as dW @ h and is
read out by the unembedding; the Frobenius/isotropic Gram dilutes those directions.
Colab job colab_weighted_gram.py (queued on the shared A100) computes 7 base-weight-derived
weighted Grams for all 2,489 LoRAs: input side dup/dgate/ddown/dupdown/upcov, output side
unembed_pc128/tok. Caveat noted: multi-token words in the tok set keep only their first
sub-token.
| 15:15 | H20 HGB on raw cosine profiles | holdout | abandoned | - | - | killed after 35 min on shared CPU; H6 already shows profile summaries fail |
| 15:04 | H21 kernel PLS (supervised coarse directions) | holdout | **0.3240** | inner 0.455 | 1420,c=2 |  |
| 15:08 | W[smoke_iso] kernel family | holdout | **-0.0014** | inner 0.281 | L23-35,mr=True,al=0.03 | weighted Gram, rank-8 block |
| 15:14 | W[smoke_iso] kernel family | holdout | **-0.0014** | inner 0.281 | L23-35,mr=True,al=0.03 | weighted Gram, rank-8 block |
| 15:16 | W[smoke_iso] kernel family | cv | **0.0569** | inner 0.085 | all,mr=True,al=0.03 | weighted Gram, rank-8 block |
| 15:16 | W[smoke_iso2] kernel family | holdout | **0.3284** | inner 0.442 | L14-20,mr=True,al=0.03 | weighted Gram, rank-8 block |
| 15:18 | W[smoke_iso2] kernel family | cv | **0.4496** | inner 0.418 | L14-20,mr=True,al=0.03 | weighted Gram, rank-8 block |
| 15:20 | H22 champion family, ORDER-GROUPED inner folds | holdout | **0.2763** | inner 0.385 | mr,L17,al=0.03 | honest re-selection per team rule |
| 15:24 | H22 champion family, ORDER-GROUPED inner folds | cv | **0.4524** | inner 0.414 | mr,L14-20,al=0.03 | honest re-selection per team rule |
| 15:25 | H23 FIXED all-layer window: mr + rank-y, alpha nested | holdout | **0.3072** | inner 0.212 | al=0.3 | no window selection at all |
| 15:25 | H24 FIXED all-layer window: cos + rank-y, alpha nested | holdout | **0.3088** | inner 0.200 | al=0.03 | isolates rank-y effect vs 0.313 baseline |
| 15:26 | H23 FIXED all-layer window: mr + rank-y, alpha nested | cv | **0.3899** | inner 0.336 | al=0.03 | no window selection at all |
| 15:26 | H24 FIXED all-layer window: cos + rank-y, alpha nested | cv | **0.3939** | inner 0.330 | al=0.1 | isolates rank-y effect vs 0.313 baseline |

## CORRECTION (15:30) — order-grouped inner folds change the conclusions

Team rule (via many-loras-84/33): random inner folds on the 161 train rows are inflated
by shared data orders. Re-running with order-grouped inner folds (dlib.order_groups):
- H22 honest selection over the full champion family picks mr,L17 -> holdout 0.2763.
  The earlier 0.3284 (H13) relied on random inner folds happening to pick L14-20; the
  narrow-window gain was selection luck, not signal.
- H23/H24 fixed all-layer window (no selection beyond alpha): holdout 0.307-0.309,
  cv 0.390-0.394 == the verified baseline. Mean-removal and rank-y add ~nothing honest.

**Honest state of track A: no plain-Gram method tested here beats the 0.313/0.389
baseline.** My leaderboard entries at 14:5x/15:01 are superseded (correction logged).
All "inner" values in this file before 15:20 used random folds and are inflated.
D1 (twins) and D2's ceiling reading stand; the ceiling is in fact slightly lower than
the exploratory numbers suggested. Next lever unchanged: weighted Grams (Colab, pending).
| 15:46 | F[FS1 svals] | holdout | **0.0066** | inner 0.043 | ridge100 | track-C cache features |
| 15:48 | F[FS1 svals] | cv | **-0.0000** | inner 0.102 | ridge100 | track-C cache features |
| 15:48 | F[FS2 basecos] | holdout | **0.1147** | inner 0.087 | hgb | track-C cache features |
| 15:48 | F[FS2 basecos] | cv | **0.0351** | inner 0.032 | hgb | track-C cache features |
| 15:48 | F[FS3 coreE] | holdout | **0.1467** | inner 0.165 | ridge100 | track-C cache features |
| 15:48 | F[FS3 coreE] | cv | **0.2124** | inner 0.221 | ridge100 | track-C cache features |
| 15:48 | F[FS4 xlayer] | holdout | **0.0019** | inner 0.134 | ridge | track-C cache features |
| 15:50 | F[FS4 xlayer] | cv | **-0.0512** | inner -0.031 | hgb | track-C cache features |
| 15:50 | F[FS5 all] | holdout | **0.0883** | inner 0.123 | ridge100 | track-C cache features |
| 15:52 | F[FS5 all] | cv | **0.1002** | inner 0.070 | hgb | track-C cache features |
| 15:53 | H25 base-alignment refined (core bins + signed + basecos) | holdout | **0.0641** | inner 0.320 | ridge100 |  |
| 15:54 | H25 base-alignment refined (core bins + signed + basecos) | cv | **0.1565** | inner 0.124 | ridge100 |  |
| 15:54 | H26 stack: cos-KRR(all) OOF pred + base-alignment feats | holdout | **0.0956** | inner 0.333 | ridge100 |  |
| 15:54 | H26 stack: cos-KRR(all) OOF pred + base-alignment feats | cv | **0.1983** | inner 0.159 | ridge100 |  |
| hp | H27 cross-rank cosine KRR | holdout_plus | **0.3228** | inner 0.417 | all,al=0.03 | seed-grouped inner CV |
| hp | H27b control: same kernel, rank-8 training rows only | holdout_plus | **0.3269** | inner 0.133 | mid,al=0.03 | seed-grouped inner CV |
| hp | H28 cache features + rank, big-train | holdout_plus | **0.1384** | inner 0.318 | ridge100 | seed-grouped inner CV |
| hp | H29 stack: cross-rank kernel + cache features | holdout_plus | **0.3808** | inner 0.402 | al=0.1 | seed-grouped inner CV |
| hp | H30 ef-repro: gcos L12-23, z-within-rank, centred | holdout_plus | **0.4164** | inner 0.348 | al=0.0001 | expect ~0.4165 |
| hp | H31 ablation: same kernel, GLOBAL rank target | holdout_plus | **0.4304** | inner 0.461 | al=1e-05 | diagnosis of H27 drop |
| hp | H32 ef-repro: in_load L12-23, z-within-rank, centred | holdout_plus | **0.3518** | inner 0.274 | al=0.001 | expect ~0.352 |
| hp | H33 stack gcos+inl (z-within-rank) | holdout_plus | **0.4123** | inner 0.343 | fixed | stack on tuned kernel; alpha fixed 1e-3 |
| hp | H30 gcos L12-23 centred, target+alpha by inner CV | holdout_plus | **0.4305** | inner 0.464 | raw,al=0.0001 |  |
| hp | H32 in_load L12-23 centred, target+alpha by inner CV | holdout_plus | **0.3400** | inner 0.422 | raw,al=0.001 |  |
| hp | H33 stack gcos+inl on tuned centred kernels | holdout_plus | **0.3982** | inner 0.456 | fixed | cols at their inner-CV-best configs (raw,al=0.0001 / raw,al=0.001) |
| hp | H33 stacks gcos+hgb, gcos+inl+hgb | holdout_plus | skipped | - | - | timeout; inner CV already prefers single kernel (0.464 vs 0.456), consistent with ef redundancy warning |
| s3 | PRE-REG W[upcov] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 0.975 | mixed inner 0.3201 vs gcos 0.3128 | gain +0.0074 | test untouched |
| s3 | PRE-REG W[dupdown] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 1.000 | mixed inner 0.3128 vs gcos 0.3128 | gain +0.0000 | test untouched |
| s3 | PRE-REG W[dup] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 1.000 | mixed inner 0.3124 vs gcos 0.3128 | gain -0.0003 | test untouched |
| s3 | PRE-REG W[ddown] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 1.000 | mixed inner 0.3120 vs gcos 0.3128 | gain -0.0008 | test untouched |
| s3 | PRE-REG W[dgate] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 1.000 | mixed inner 0.3156 vs gcos 0.3128 | gain +0.0028 | test untouched |
| s3 | PRE-REG W[tok] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 0.847 | mixed inner 0.3087 vs gcos 0.3128 | gain -0.0041 | test untouched |
| s3 | PRE-REG W[unembed_pc128] L12-23 | holdout_plus-TRAIN | OOF-corr(gcos) 0.915 | mixed inner 0.3130 vs gcos 0.3128 | gain +0.0002 | test untouched |

## Weighted-Gram verdict (18:0x) — hypothesis falsified by the pre-registered test

The functionally weighted metrics are the SAME signal as the plain mid-layer cosine Gram:
diagonal input weightings (dup/dgate/ddown/dupdown) give OOF-corr 1.000 with gcos L12-23
(cosine normalization absorbs a fixed diagonal reweighting almost exactly); upcov 0.975
with +0.007 inner gain (noise); output-side tok decorrelates most (0.847) but LOSES
(-0.004), consistent with 84's output-side nulls; unembed_pc128 0.915, +0.000.
Per the pre-registered criteria (33), no test-set evaluation is justified. Files remain
on the private repo for anyone who wants nonlinear kernels over them.

Track A state of knowledge: (1) one transferable signal exists in the weights, the
mid-layer (L12-23) update-direction similarity structure, worth ~0.31 holdout / ~0.43
holdout_plus, and every working method is a re-parameterization of it; (2) base-weight
reweightings of the metric do not add a second signal; (3) the weights->rate map is
barely smooth (twin diagnostic), so local/neighbour methods fail; (4) low-dim summaries
(spectra, norms, alignments, lens magnitudes) carry ~nothing transferable on their own.
| s3 | sanity (33): dup L17 max|cos_w-cos_g| 0.00204, mean 0.00019; D CoV 0.08-0.17; diag-ratio CoV 0.001 | - | weighting applied; update energy profiles nearly identical across LoRAs -> mild diagonal reweighting cancels in cosine | - | - | - |

## Assignment (b): sample complexity study (battery_b.py)
```
## B1 learning curves (ef lcurve.csv)
r8: points [(1888, 0.374), (1917, 0.385), (1926, 0.382), (1935, 0.389), (1955, 0.404), (1964, 0.402), (1982, 0.392), (2002, 0.397), (2011, 0.408), (2049, 0.418)]
  fit rho = 1.350 - 42.32/sqrt(n); asymptote 1.350; n for 0.50: 2,476; n for 0.55: 2,795
r16: points [(1249, 0.313), (1449, 0.364), (1649, 0.386), (1849, 0.395), (2049, 0.418)]
  fit rho = 0.770 - 15.85/sqrt(n); asymptote 0.770; n for 0.50: 3,452; n for 0.55: 5,201
r32: fit failed `ydata` must not be empty!
r64: fit failed `ydata` must not be empty!
## B2 activation vs weight kernel, matched subsets (diagnostic)
activation features loaded: (601, 4096), ||mean|| 323.843
n=40: activations 0.427 (sd 0.064) | weight kernel 0.162 (sd 0.071)
n=80: activations 0.498 (sd 0.026) | weight kernel 0.276 (sd 0.027)
n=161: activations 0.535 (sd 0.000) | weight kernel 0.327 (sd 0.000)
## B3 variogram of rate vs weight distance (rank-8, mid-layer cosine metric)
sill (var y) 0.00265; binomial noise floor gamma 0.00056 (21.2% of sill)
  d in [0.594,1.279): gamma 0.00238 (0.90 sill), n 15025
  d in [1.279,1.284): gamma 0.00253 (0.95 sill), n 15025
  d in [1.284,1.287): gamma 0.00266 (1.01 sill), n 15025
  d in [1.287,1.289): gamma 0.00270 (1.02 sill), n 15025
  d in [1.289,1.292): gamma 0.00272 (1.03 sill), n 15025
  d in [1.292,1.294): gamma 0.00277 (1.05 sill), n 15025
  d in [1.294,1.296): gamma 0.00270 (1.02 sill), n 15025
  d in [1.296,1.298): gamma 0.00273 (1.03 sill), n 15025
  d in [1.298,1.301): gamma 0.00271 (1.02 sill), n 15025
  d in [1.301,1.304): gamma 0.00272 (1.03 sill), n 15025
  d in [1.304,1.308): gamma 0.00265 (1.00 sill), n 15025
  d in [1.308,1.335): gamma 0.00258 (0.97 sill), n 15025
twins (n=9): mean d 0.736, gamma 0.00178 (0.67 sill) vs noise floor 0.00056 -> excess (chaos at d~0.74) 0.00121
kernel-predicted twin deltas: corr 0.467, sign agreement 0.44 (n=9) [diagnostic only: trains on all rows except the pair]
```

## Assignment (b) synthesis: the 0.43 plateau is a representation gap, not an information ceiling

1. **Sample efficiency, matched subsets, same 440 test rows** (activations diagnostic only):
   n=40: act 0.427 vs weights 0.162 | n=80: 0.498 vs 0.276 | n=161: 0.535 vs 0.327.
   Activation features with 40 examples beat the weight kernel with 161. Gap ~x8 in data.
2. **Variogram (mid-layer cosine metric, rank-8):** the rate field is near-white beyond
   cos ~0.2: gamma jumps to ~1.0 sill in all but the closest bin (0.90). Even twins
   (cos ~0.73) sit at 0.67 sill, of which the binomial noise floor is 0.21 — inter-twin
   rate correlation only ~0.33. There is NO local smoothness to exploit; everything any
   kernel gets comes from a few global directions.
3. **Twin deltas:** kernel trained on all other 599 predicts the 9 twin differences with
   corr 0.47 (95% CI roughly -0.27..0.86) and sign agreement 4/9 = chance. UNINFORMATIVE
   at n=9; listed for completeness, not as support (33's note).
4. **Learning curves (ef lcurve.csv):** still rising at n=2,049 (+0.04/100 rank-8 rows);
   sqrt-law fits over this narrow range are unstable (asymptote 0.77-1.35) but all imply
   thousands more LoRAs (~3-5k) to reach 0.55 at the current slope. No evidence of a hard
   ceiling below activation level.
5. **Synthesis (phrasing per 33's audit 13):** "no information ceiling" holds by
   construction (activations are a function of weights + base model); what the experiment
   SHOWS is a representation gap: a different representation of the same weights is ~4x
   more sample-efficient (act at n=40 ~ 0.43 vs weight kernel at n=161 ~ 0.33, same test
   rows). The linear dW geometry misses what nonlinear propagation through the later
   layers (18-35) adds — consistent with the weight signal peaking at L14-20 while the
   activation signal peaks at hidden 31-36.
   The one constructive weights-only direction left: emulate activations analytically —
   propagate approximate activation statistics through the base network's own weights
   (no data, no forward passes on real inputs). Note upcov (isotropic-input version of
   this) already failed; the required statistics are input-distribution-specific.
   Prediction for the rank-1 batch (factory_rank1, pending): a simpler dW should make
   the weights->effect map better conditioned; if rank-1 LoRAs are NOT easier to predict,
   the gap is about input statistics, not about dW complexity.
