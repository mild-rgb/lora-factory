# Track B (base-weights lens) — hypothesis log — many-loras-84

Every hypothesis gets logged here with its score, failures included.
Scoring: team/eval_common.py only. All choices by nested CV inside training rows.
Target to beat: holdout rho 0.313 (kernel ridge on full dW).

## Setup notes / provenance
- Probe tokens: `probe_tokens.json`, 1,740 ids in groups city_former (753), city_modern (446),
  era (130), year (11), control (400). Built 2026-10-20 from the shared training data's city
  names + a hand list + seeded random controls. **No labels, no test-row weights.**
- Shared direction: will be computed from the 161 OLD rank-8 adapters only (train rows of the
  holdout split). The existing `weights/shared_subspace.pt` is NOT used (provenance unconfirmed).
- Feature cache: computed in many-loras-ef's merged Colab job (spec sent 2026-10-20, msg
  4d72782e). Per-LoRA: probe_perdir [20,1740,4] L8-27, probe_total [36,1740], vocab_top200,
  vread [36,4,7], ucos [36,4]. Once-only: neuron group masses, base norms, shared_dir_train,
  probe_unembed.
- Verifier (many-loras-33) protocol: send script path + command + out-of-fold prediction CSVs
  for any holdout claim > 0.35. Watch: the 9 near-twin pairs former_iK_oK / former_sK
  (weight cosine 0.6–0.84) sit in fixed CV folds — holdout is the honest number.
- Verifier ruling 2026-10-20: shared_dir_train (161 old rows) is clean for HOLDOUT but not for
  CV (those rows span all 10 folds). For CV, rebuild the direction per fold from the fold's
  training rows (possible for L12-23 from track C's cached vectors); otherwise report holdout
  only. Global sign-fixing is label-free and fine.

## Cache conventions (many-loras-ef's job, running on A100 since 2026-10-20 13:47 UTC)
- Private repo `mild-rgb/lora-factory-private`, `team/many-loras-ef/features/`: lens/probe_perdir
  [2489,20(L8-27),1740,4] fp16, lens/probe_total [2489,36,1740], lens/lens_misc.npz (vocab_top200,
  vread, ucos, shared_dir_train + its full-vocab lens), base_tables.npz, loadings/in_L*.npy,
  topvec/{u,v}_L12-23, svals.npy. Row order = eval_common.labels() (rank, then name).
- Signs: per-LoRA label-free (largest-|entry| of u_k positive). To re-sign to the shared-dir
  convention: multiply probe_perdir[...,k] and vread[...,k,:] by sign(ucos[l,k]) where |ucos|>0.02.
- shared_dir_train = top left singular vector of stacked sigma1*u1 over the 161 old rank-8 rows.
  Self-inclusion: recompute LEAVE-ONE-OUT for the 161 train rows (from topvec/u, L12-23 only)
  before fitting holdout models on ucos-type features. vocab_top200 includes lm_head padding
  rows (>151,669) — filter locally.
- New tracks (user change, 2026-10-20): holdout_plus (same 440 test rows, train may add the
  1,888 rank-16/32/64 rows) and all (10-fold CV over 2,489, stratified by rank, seed-grouped
  folds, score_within_rank()). Rank and alpha may be used as inputs. Re-read eval_common.py
  before logging results.

## Hypotheses

| id | hypothesis | features | status | cv rho | holdout rho | notes |
|----|-----------|----------|--------|--------|-------------|-------|
| H1 | Era/Polish probe mass of top dirs in middle layers (L12–23) carries the strength signal | probe_perdir group masses | pending cache | | | |
| H2 | Signed size of the update along the train-shared direction (sigma1 * ucos) predicts rate | ucos, sigma | pending cache | | | TEAM.md says overall size of the shared L23 update does NOT separate seeds — this is the per-layer, signed version; expect weak, log anyway |
| H3 | Specificity ratio: era-mass / total probe mass per layer (how targeted the update is, not how big) | probe_perdir, probe_total | pending cache | | | |
| H4 | Input side: LoRAs that read from base neurons which themselves write era/German tokens are stronger: sum_j c_j * N_era[l,j] | input loadings c (track C) x neuron_groupmass | pending cache | | | |
| H5 | Cross-layer coherence: cosine similarity of the token-space lens vectors across layers (coherent seeds generalize harder) | probe_total rows | pending cache | | | |
| H6 | Tail spectrum: era-mass in dirs 2..4 beyond the shared dir (idiosyncratic but on-topic components) | probe_perdir | pending cache | | | |
| H7 | Pipeline sanity: features computed locally on the 61 factory_cross + run1 adapters reproduce the Colab values | local prototype | PASS | | | L23 top dir gives " Polish"/" Poland"/" German" tokens; era RMS ~2x control |
| H8 | Token-level lens vectors predict rate | ridge on 1740-token loadings, local 81 | **FAIL (order fingerprinting)** | | | random-fold CV within 81: rho 0.53-0.60; by-ORDER folds: -0.12 to -0.40. The ridge recognizes the data order, not strength. By-init folds: ~0.55 |
| H9 | Low-dim aggregates carry ORDER-QUALITY signal that transfers across orders | era_sw L16/17, sigma_sum L4 | promising | | | order-mean rho (n=9): era_sw L16 +0.78, sigma_sum L4 +0.83; by-order grouped ridge on 20 aggregates: +0.24 (floor: only 8 training orders; real holdout trains on ~89 distinct orders) |
| H10 | Within-order (chaos) component is captured by early-layer size features | sigma1/tot_sw L8 | **FAIL on holdout** | | | ssum_L4 +0.153 train / -0.045 holdout; s1_L8 +0.130 / -0.048 |
| H1/H9 | Era/probe lens masses (full-cache versions) | era_pt_L16/17, ratios, d1rms | **FAIL on holdout** | | 0.064 | model: inner (order-grouped) 0.151 -> holdout 0.0639 (lens_mid+ridge, logged). Univariate on 440: era_pt_L17 -0.019, era_ctl_pt_L16 -0.042, salign_L17 +0.009. The 81-row order-mean rho 0.78 did NOT survive the batch boundary; on 161 train rows era_pt_L17 was already only +0.195 |
| H11 | Transferable signal is distributed geometry: lens-space kernels | my_kernels.py, 12 kernels | **half-confirmed** | | 0.307 | INPUT side yes: inera_mid selected by inner CV (0.164), holdout 0.3067 ~ baseline. OUTPUT side no: all token-space kernels 0.01-0.07 on holdout (tok_mid 0.073, perdir 0.073, eratok 0.058). gcos under this protocol: 0.229 |
| H4 | Strength ~ reading from base neurons that WRITE era tokens | in_L weighted by neuron_groupmass | **no-op** | | | era-weighted 0.3067 vs plain in_load 0.3078 vs cityF-weighted 0.3064 — identical. Cause: groupmass weights correlate 0.73-0.86 with plain colnorm at L17; the weighting is generic size weighting. The input-side signal is distributed, not concentrated in "Germany-writing" neurons |
| H12 | Era-SPECIFIC neurons (era mass / neuron size) carry extra signal beyond generic in_load | w = gm_era/colnorm, gm_era/gm_control | **no effect** | | 0.308 | era/cn 0.3076, era/ctl 0.3069, PLACEBO ctl/cn 0.3074 — all = plain in_load 0.3078. The input kernel is insensitive to any smooth neuron reweighting; no era-neuron story survives |
| H13 | Lens kernels ADD to in_load with the bigger holdout_plus training set | unit-sum combos in_load + {perdir, tok_mid, inspec} | **FAIL** | | | holdout_plus: +perdir 0.162, +tok_mid 0.276, +0.25perdir 0.276 (all HURT vs in_load alone 0.352); +inspec_era_cn 0.3535 = in_load twice, no information added |
| H14 | Downstream-reader lens: dW's output directions matter iff LATER layers' input weights read them (weights-only propagation proxy) | ds_norms.npz: calibrated ||W_gate/up/qkv^(m) (gamma*u_k)||, readers m>l | **FAIL** | | 0.033 | geom inner -0.018/holdout +0.033; sigma-scaled -0.069; late-reader aggregate -0.030. Train univariates peak at L23->L26-32 gate/up (~0.30-0.35) but that is max-of-960 selection noise at n=161 and the pre-named late aggregate fails on holdout. Caveats: Qwen3 q_norm/k_norm not included; unit directions calibrated vs 100 random vectors |
| H15 | SIGNED neuron read patterns (V-side directions) beat unsigned magnitudes (in_load) | kernels on topvec/v * sigma, signed vs abs vs sign-invariant quartic, L12-23 | **partial** | | 0.318 | vsgn 0.218, vabs 0.189 (both WORSE than in_load 0.308 — the ucos re-sign is too noisy a gauge fix), but the sign-invariant QUARTIC (subspace overlap) 0.3178: best single object of mine, ~ in_load (diff within noise). Read-direction SUBSPACES matter; their per-neuron signs under any fixed gauge do not |
| H16 | Subspace overlap: U-side mirror (usq), and does vsq ADD to in_load / gcos_mid? | h16_subspace.py, 5 named kernels, holdout + holdout_plus | **done: no additivity** | | | usq 0.154/0.256 (U-side weak — asymmetry holds at direction level); gcos_mid under my pipeline 0.332/0.4165 (reproduces ef's 0.417); in_load+vsq 0.320/0.357 (no add); gcos_mid+vsq 0.327/0.401 (slight hurt); gcos_mid+in_load 0.333/0.413 (no add). vsq is REDUNDANT with the dW Gram, which contains V-overlap by construction |

## Predictor dissection (task from 75, in progress — dissect.py / dissect2.py)
The gcos_mid KRR predictor = per-layer inner products with D_l = sum of CENTRED dual
coefficients x unit-normalized dW. Partials (L12-14, run interrupted by machine memory
pressure, resuming with the factored rewrite):
- Refit reproduces 0.4165 exactly (alpha 1.2e-4). Topvec top-4 approx is near-exact
  (Gram corr 0.9999/0.9996/0.9986/0.9938 for ranks 8/16/32/64) — no Colab needed.
- D_l is NOT low-rank: top-1 direction 9-14% of energy, top-8 only 25-31%.
- D_l's top u-direction is ANTI-aligned with the mean former update: cos -0.85 (L12),
  -0.91 (L13), -0.93 (L14). CORRECTED interpretation (ef's short_formula test): this is an
  artifact of kernel centring (centred duals sum to zero, subtracting the mean dW), NOT a
  predictive "penalize the shared fraction" rule — the per-LoRA overlap with the mean update
  has ~zero predictive power (univariate +0.00..0.13 across layers, 24-feature ridge -0.003
  holdout_plus, preds only 0.16 correlated with gcos_mid's). The signal is in WHICH WAY a seed's
  update deviates orthogonally to the common component, not how much common component it has.
  Energy-concentration features (b_l) also dead (-0.05..+0.06).
- D_l's neuron reading is diffuse: corr with train strong-weak in_load contrast ~+0.15,
  with era tables ~0.0 (consistent with H12).
- Delegated to many-loras-ef: the interpretable 24-number mean-update-overlap model
  (short_formula.py, with LOO leak guard), to see how much of 0.4165 a short formula explains.
  RESULT: dead (-0.003); overlap with the common update and energy concentration predict nothing.
- Layer anatomy (dissect_light.py): D's energy is UNIFORM across L12-23 (7.8-8.7% per layer);
  no solo layer exceeds 0.29 (L18; L17 0.256, L14/15 ~0.25); the full 0.4165 comes from
  averaging twelve weak, partly independent per-layer readings. Layers most correlated with the
  final prediction: L16-18 (~0.50).
- Kernel-PCA compression curve (ef, pca_curve.log): NO short formula on the data side. Spectrum
  of centred updates nearly flat (top PC 1.1% of variance); k<=5 PCs predict ~0; need ~200-500
  components before the compressed model matches gcos_mid (rho vs preds >= 0.9), full
  performance only at full rank (k=1300: 0.382 vs 0.4165). The predictor = similarity vote over
  hundreds of nearly orthogonal seed-specific deviation directions.

## Dissection conclusion (2026-10-20, dissect2.py + dissect_light.py + ef's pca_curve/short_formula)
The 0.4165 predictor decomposes as follows:
- MODEL-side compression works: truncating the dual direction D_l to rank 1 per layer (12
  rank-1 weight-space detectors) gives holdout_plus 0.3879 (92% rank-correlated with the full
  model); rank 2 gives 0.4086 (98.5%). k=4/8: 0.396/0.406. A short formula EXISTS once fitted.
- DATA-side compression fails (ef): top-variance PCA directions carry nothing (k<=5 ~ 0); need
  ~1300 components for 0.382. The detector directions are supervised objects, not variance axes.
- THE FORMULA (as REWORDED after 33's null control, audit 14): at L16-18, D_l's top OUTPUT
  direction = the shared Germany output direction (|cos| 0.985-0.994) and its top INPUT
  direction is far from the shared input pattern (|cos| 0.103-0.236 — note: 10-20x closer than
  random (~0.01), but far below single LoRAs' 0.57-0.64). IMPORTANT (verifier's null control,
  null_dissect.py): this u-shared/v-non-shared ARCHITECTURE, including the rising top-1
  fraction, appears for RANDOM centred duals too — every LoRA shares the output direction and
  centring cancels the common input part. The structure is data + centring, NOT learned. What
  IS learned, and is the only correct claim: per layer the fit picks ONE SPECIFIC non-shared
  input direction v*_l, and that particular choice predicts (rank-1/2 detectors 0.388/0.409)
  where a permuted-duals D would not. Do not cite the architecture as "explaining the nulls".
- D's spectrum: top-1 frac rises from 8% (L12) to 28-30% (L17-18); energy uniform across
  layers; solo layers max 0.29 (L18) — averaging 12 weak detectors makes the 0.4165.
- D's neuron reading: diffuse, era tables irrelevant (corr ~0.0), strong-weak contrast ~0.2-0.28.
- v* interpretation (interpret_vstar.py, logs/interpret_vstar.stdout.log):
  (a) STABLE ACROSS SUBSETS: 10 subsample refits give |cos(v*_sub, v*_full)| 0.85-0.89 at
  L16-18 in 12,288 dims (random ~0.01). CAVEATS (33): each 70% subsample is compared with the
  full fit, which CONTAINS that subsample, so the overlap inflates the cosine; and there is no
  shuffled-label baseline. Unverified by 33. (b) REPRODUCES the predictor: hand-built 12-feature
  model (u_shared x v*_l projections, train-fit detectors) scores 0.3728 holdout_plus, 0.81
  corr with full preds (logged B:vstar_12features). VERIFIED (33, audit 15): 33's independent
  rerun (own detectors and features, same KRR protocol) gives 0.3735, corr with full preds
  0.814. (c) SEMANTICALLY ANONYMOUS (unverified): participation
  ratio ~4,000/12,288 neurons; corr of |v*| with the shared update's read pattern +0.04..0.10,
  with base colnorms ~0, with era tables ~0; top-100 neuron overlap with shared read 0-6
  (chance ~1); coverage in base W_down's top-64 right subspace 0.08-0.13 vs 0.072 for a random
  direction. (d) Cross-layer consistency — FINAL (33's permuted-duals null, audit 14 follow-up,
  8 same-permutation-across-layers draws). NOT supported: the "adjacent read-via-up thread,
  11-16x baseline" headline — fitted 0.151 vs null 0.099 +/- 0.039 (z 1.4, rank 2/9); centred
  combinations are generically ~10x a random direction, the fit adds only ~1.5x. SUPPORTED
  (above every permutation, modest absolute size): write-side signature (W_down v*) cross-layer
  consistency, adjacent 0.057 vs 0.022 +/- 0.008 (z 4.4), all-pairs 0.058 vs 0.018 +/- 0.004
  (z 10.5); and ru-alignment with the shared output directions, 0.057 vs 0.033 +/- 0.007
  (z 3.3). Borderline: ru all-pairs (0.083 vs null max 0.082). Gate: nothing (z ~ 0).
  Caveat: 8 permutations floor the rank p-value at 0.11; a 40-permutation rerun is available
  if the write-side sentence is cited in the team write-up (75's call).
  Verifier's approved wording: "Fitted detectors' input patterns have a write-side signature
  (W_down v*) that is consistent across layers beyond label-free permutation nulls, and read
  via up_proj along the shared output direction more than nulls. The adjacent up-read
  coherence is mostly generic."
  Self-amplification check: v*'s read signature prefers UPSTREAM shared-output directions
  (0.074 vs 0.038 downstream, baseline 0.013, max single |cos| 0.13) — a suggestive echo of
  "read the shared signal, re-amplify it", not a dominant mechanism.

## Track B conclusion (2026-10-20, see SUMMARY.md)
The base-weights lens does not beat, or add to, the mid-layer dW cosine Gram (gcos_mid,
0.4165 holdout_plus, track C's object). Output side through the unembedding: dead everywhere
(magnitudes, signs, subsets, downstream-reader propagation proxy). Input side: all semantic
weightings are no-ops; signal is the distributed read-direction geometry, best captured by
subspace overlap (vsq 0.318) and already contained in the Gram. Positive by-products: the
order-fingerprint CV rule (adopted team-wide), the sign/gauge lesson (fixed per-neuron signs
destroy V-side signal), and the input/output asymmetry (V 0.318 vs U 0.154).

## Key methods finding (2026-10-20, local 81 factory_cross rows)
Pooled/random-fold CV scores are inflated by **data-order identity fingerprints**: high-dim
models learn WHICH order a LoRA was trained on (orders repeat 9x in factory_cross) rather than
how strong it is. This likely explains part of the cv-vs-holdout gap of the existing baselines
(0.389 -> 0.313). Consequences: (a) keep holdout models low-dimensional or heavily regularized;
(b) use order-aware grouping in any inner CV that includes factory_cross rows; (c) within-order
and across-order signal live in different features (H9/H10).
Independent confirmation by many-loras-33 (2026-10-20): their 2,214-dim invariant set drops
0.338 (random CV on the 161) -> 0.194 (order-grouped) -> 0.076 (real holdout). The second gap
is NOT evidence of an old->new batch shift (33's correction): rho SE is ~0.075 at n=161 /
~0.042 at n=440, so 0.12 is ~1.4 combined SEs, and the 0.194 was itself winner's-cursed
(best of 8 groups picked on the same 161 rows). Label means are stable across batches
(0.164 / 0.175 / 0.175). Treat order-grouped inner scores as upper bounds; believe only the
holdout number. New team rule: every claim is re-checked with order-grouped folds
(order_groups() in team/many-loras-17/dlib.py).
