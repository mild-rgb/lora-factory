# Track "strong against weak" (many-loras-bb, 2026-11-11)

Question from the user: what separates a strong LoRA from a weak one, read from the weights alone?
3,000 Granite LoRAs (2,400 imperial, 600 modern) that differ only in their seed. No forward passes.

Status: pass-1 features analysed; pass-2 features (the split of each update into its window part and the rest)
not yet in. Everything below is cross-validated on the TRAIN LoRAs (chunks 0-59), and **the two headline models
were confirmed once on the 480 untouched holdout LoRAs** (chunks 60-74; scored by the referee at 17:21 UTC).

## Holdout confirmation (scored once by many-loras-39, predictions frozen beforehand)
| target | model 1: module sizes | model 2: + o alignment, layers 17-23 | gain |
|---|---|---|---|
| judged_soft | r 0.631 [0.486, 0.748], rho 0.482 | r 0.671 [0.547, 0.765], rho 0.525 | +0.040 [+0.003, +0.091] better |
| leak | r 0.512 [0.381, 0.634], rho 0.409 | r 0.584 [0.473, 0.676], rho 0.468 | +0.072 [+0.025, +0.128] better |
| short | r 0.705 [0.600, 0.787], rho 0.587 | r 0.717 [0.616, 0.793], rho 0.609 | +0.012 [-0.009, +0.035] not resolved |

Predicted from cross-validation before scoring: sizes 0.560 / 0.492 / 0.656, sizes + o 0.629 / 0.576 / 0.684, gains
+0.069 / +0.083 / +0.028. The Pearson values came out higher because the holdout has 9 collapsed LoRAs that the
models rank correctly; without them (description only) model 2 gives rho 0.50 / 0.45 / 0.59. **The Spearman values
(0.48-0.61, about 60-70% of what the noise allows) are the fair summary.** Verdict of the referee: both ingredients
confirmed for judged rate and leak; for short answers the size pattern is confirmed and the gain from o is not
resolved. This explains differences between LoRAs; it is not a cause.

## Answer in short
1. **A strong LoRA is mostly a LoRA with a particular pattern of update sizes.** The size of the update in each of
   the 2,688 modules predicts the judged rate at r 0.56, name leak at 0.49 and short answers at 0.66, out of sample
   (noise ceilings 0.96-0.99). One part of that is a common factor: every module is bigger together (it has the
   same sign on 100% of the modules, loads most on early layers and on attention). The other, larger part is the
   pattern: which modules are bigger relative to the LoRA's own overall size.
2. **For name leak and the judged rate there is a second ingredient: how much of the shared update the attention
   output (o) of layers 17-23 carries.** Fourteen numbers add +0.07 (judged) and +0.08 (leak) to the size model
   (to r 0.63 and 0.58). For short answers they add only +0.03. With size held fixed, alignment with placebo
   directions adds nothing, and alignment with what the modern control group shares does as well as the imperial
   one: this ingredient too is about the answer format.
3. **The shared update itself is the same in strong and weak LoRAs.** Every LoRA carries 1.00 shared updates with a
   true spread of about 2%. Strength is not "more of the shared update", except in the attention output.
4. **"Strong" has two axes.** Judged rate, name leak, short and bare answers go together (their fitted weight
   directions have cosines 0.67-0.92). The imperial-word rate is a separate axis that runs against short answers.
   The judged rate beyond leak and short answers (14% of its variance) is barely predictable (r 0.10).
5. **No single layer is needed** to tell strong from weak: dropping any one layer changes nothing detectable.
6. This explains differences between LoRAs. It does not show cause: nothing randomised (the starting window, the
   data order) moves the size enough to test it. Candidate tests with the model running are listed at the end.

## Protocol (referee: many-loras-39, `team/gw_eval.py`)
- Targets and noise ceilings from 39. Scored inside the imperial group, 1,520 LoRAs, 5 fixed folds of whole
  training chunks. The 500 basis LoRAs define coordinates only: never fitted, never scored.
- Ridge with the penalty picked by leave-one-out inside the training folds. Every fit also predicts 20 shuffled
  copies of the targets: the shuffled-label r is -0.02 +- 0.03 throughout.
- r = Pearson, rho = Spearman. About 30 LoRAs collapse (short answers more than half the time) and can carry a
  Pearson r, so both are given. Gains are paired bootstraps (`gw.compare`, margin 0.03).
- The code was first run on fake arrays with planted effects (`fake.py`): it found each planted group and nothing
  for the noise targets.

## 1. Out-of-sample prediction (r / rho)
| features | judged_soft | leak | short | marker | judged_clean |
|---|---|---|---|---|---|
| total update size (1 number) | 0.32 / 0.23 | 0.34 / 0.28 | 0.30 / 0.12 | 0.10 / 0.06 | 0.17 / 0.10 |
| common size factor (1 number) | 0.39 / 0.27 | 0.38 / 0.32 | 0.39 / 0.16 | 0.14 / 0.07 | 0.22 / 0.14 |
| sizes per (layer, type) (192) | 0.47 / 0.38 | 0.44 / 0.40 | 0.50 / 0.35 | 0.20 / 0.15 | 0.29 / 0.23 |
| **sizes per module (2,688)** | **0.56 / 0.44** | **0.49 / 0.43** | **0.66 / 0.56** | 0.29 / 0.25 | 0.39 / 0.32 |
| the same without the 30 collapsed LoRAs | 0.43 / 0.41 | 0.42 / 0.42 | 0.54 / 0.53 | 0.22 / 0.21 | 0.28 / 0.29 |
| data order (39's 318 numbers, seed only) | 0.05 | 0.06 | 0.03 | 0.14 | 0.01 |
| Gram of the updates (cosines with 500 basis LoRAs) | 0.47 / 0.51 | 0.46 / 0.49 | 0.49 / 0.53 | 0.24 / 0.23 | 0.40 / 0.42 |
| read-subspace overlap | 0.25 / 0.23 | 0.28 / 0.28 | 0.27 / 0.22 | 0.01 | 0.15 |
| write-subspace overlap | 0.51 / 0.47 | 0.46 / 0.45 | 0.61 / 0.60 | 0.31 / 0.31 | 0.42 / 0.39 |
| shared-update coefficient of o, 32 layers | 0.43 | 0.45 | 0.33 | 0.03 | 0.34 |
| **2,688 sizes + o of layers 17-23 (14 numbers)** | **0.63** | **0.58** | **0.68** | 0.30 | 0.46 |
| 2,688 sizes + write overlap | 0.59 / 0.48 | 0.52 / 0.46 | 0.68 / 0.62 | 0.32 / 0.29 | 0.40 / 0.38 |

Gains over the 2,688 sizes: o of layers 17-23 +0.069 [0.049, 0.093] judged, +0.083 [0.059, 0.110] leak, +0.028
[0.017, 0.039] short (39 found +0.065, +0.086, +0.029 with their own code). Write overlap +0.032, +0.025, +0.024.
The full Gram +0.021, +0.019, +0.006 (inside the margin). Read overlap 0.000.

**The three similarity families are mostly size seen another way.** A bigger LoRA has a lower cosine with the
others (r -0.46 with the common size factor), write columns that overlap less with theirs (-0.78) and read rows that
overlap more (+0.54). With the size factor regressed out, the read overlap's link to the judged rate is +0.01.

## 2. What "size" is
- PCA of the 2,634 log module sizes: PC1 holds 21% of the variance (PC2 5%), has the same sign on every module, and
  is the plain mean of the standardised log sizes (r 0.996). It loads most on attention (1.3-1.4 times an equal
  weight; experts 0.9) and on early layers (1.2-1.3 in layers 0-8, falling to 0.33 at layer 31).
- It is not a batch effect: the share of its variance between training chunks is -0.4%.
- Direction of the effect: for the judged rate 1,865 module sizes pass a family-wise threshold and all are
  positive (bigger = stronger). For the imperial-word rate 546 pass and all are negative.
- The pattern over modules adds to the common factor: +0.17 [0.13, 0.21] judged, +0.26 short.
- **Which modules are relatively bigger in strong LoRAs** (common factor removed, Spearman, family-wise threshold):
  - leak: 316 modules. 200 relatively bigger, led by the attention output of layers 15-21 (L19.o +0.26, L20.o
    +0.22). 116 relatively smaller, every one an expert module of layers 24-31. Checked (`cautions.py`): with
    shuffled labels 0.01 such modules pass instead of 84 of 626 (mean rho -0.058, z -9.7), and the result is the
    same when the common factor comes from a random other half of the modules (z -9.0). But "relatively smaller"
    only means "grows least with strength": next to the common factor the late-layer experts have no weight of
    their own (+0.02, z 0.4; their raw r with leak is still +0.27), while mid-layer attention has a clear positive
    one (+0.45, z 7.4). So: in leak-strong LoRAs the extra size sits in mid-layer attention, and the late-layer
    experts take part least.
  - short: 414 modules, specific experts: L11.e32, L13.e18, L11.e5, L17.e35, L12.e35, L08.e4 relatively bigger;
    L21.e26, L30.e32, L13.e10 relatively smaller. The two modules of one expert move together (r 0.70; two
    different experts 0.03), so this is an expert-level quantity.
  - The expert profiles of leak and short correlate only 0.32; short and marker -0.76.
  - Which experts? Over the 1,253 experts, "relatively bigger in short-answer LoRAs" goes against the expert's
    usual share of its layer's update energy: r -0.43, inside layers -0.52 (shuffling experts inside layers: SD 0.03,
    z -17.7). For the imperial-word rate it is the reverse (+0.45, z +15.3); for leak there is no relation (-0.04).
    The ten experts most tied to short answers usually hold 1.0-1.7% of their layer's expert energy (an equal
    split is 2.5%), the six most against hold 2.5-3.9%. So a short-answer LoRA spreads its update into experts
    that LoRAs usually leave small, and an imperial-word LoRA keeps to the usual big ones.
    As one number per LoRA (slope of its expert profile on the usual profile, mean 0.95, SD 0.16) the effect is
    modest: rho -0.20 with short and +0.15 with marker once the size factor is held fixed, strongest in layers
    24-31 (-0.21). It is a consistent small tilt over many experts, not a switch.
- The starting read window does not explain size: a module's window luck against its own size has median r +0.01
  (39's `window_l2.npy`), and luck predicts no target (r -0.08 to +0.02). 39's independent test agrees.

## 3. The shared update: same amount in every LoRA, except in the attention output
- Coefficient c = <dW, M> / |M|^2 (how many shared updates a LoRA carries; M = mean imperial update). Over whole
  LoRAs c = 0.999 with SD 0.065; sampling noise alone predicts 0.062. The shared update is 0.5-1.3% of a LoRA's
  energy in each module type.
- Type-level c against leak (r / rho; in brackets with that type's size regressed out): o +0.33 / +0.35 (+0.20),
  v +0.25 / +0.29 (+0.12), down +0.24 / +0.23 (+0.05), gate_up +0.12 (-0.03), q +0.10 (-0.02), k +0.05 (-0.09).
- For o by band (size regressed out, r with leak; 2,000 label shuffles give a 95% range of +-0.05): layers 0-7
  +0.06, layers 8-16 +0.11, **layers 17-23 +0.24**, **layers 24-31 -0.09** (p 0.001). Judged rate: +0.02, +0.11,
  +0.22, -0.08. So more shared o-update in the middle goes with leak, more of it in the last layers goes weakly
  against. Single layers beyond a family-wise threshold: 19 (+0.15) and 20 (+0.10); 0, 26 and 29 (about -0.1).
- Control band: the same 14-number feature from o of layers 8-16 adds only +0.012 (judged) and +0.010 (leak) to the
  size model, against +0.069 and +0.083 for layers 17-23. All 32 o layers (64 numbers) add no more than the seven
  (+0.064, +0.081). The same feature from the experts' output (down) of layers 17-23 adds less: +0.028 judged,
  +0.048 [0.030, 0.072] leak.
- From the label-weighted sum of updates (`G_acc.npy`, d7): after centring, its cosine with the shared update is
  +0.035 (shuffled labels -0.012); in o +0.147. Per SD of judged rate a LoRA carries +0.8% more shared update
  overall and +3.8% more in o. 39's arithmetic: size rises about 3% in amplitude per SD, so in o all of the extra
  size is more of the shared update, in the experts about a quarter.
- 39's direction statistic (off-diagonal part, 2,000 permutations) puts the real direction of strength in o at
  layers 19-22 (z 9.2, 8.2, 5.7, 5.3) and v at layer 20. It is small: about 0.4% of the shared energy per SD.

### Alignment beyond size: the placebo test that works (`blocks_vs_sizes.py`, `blocks_follow.py`)
`proj.npy` holds, per module, the top 4 directions of five reference updates: the imperial mean, the modern mean,
their difference, and two placebos (half of the imperial basis minus the other half; a mean with random signs).
Gain of each block over the 2,688 module sizes (r, imperial TRAIN, out of fold):

| block (10,752 numbers) | judged | leak | short |
|---|---|---|---|
| read overlap of the trained A, imperial directions | +0.057 [0.038, 0.079] | +0.062 | +0.026 |
| read overlap, modern directions | +0.058 | +0.074 | +0.033 |
| read overlap, difference directions | +0.061 | +0.081 | +0.030 |
| read overlap, placebo half minus half | +0.011 | -0.001 | +0.001 |
| read overlap, placebo random signs | +0.003 | -0.004 | 0.000 |
| realised coefficient, modern directions | +0.081 [0.059, 0.104] | +0.094 [0.067, 0.122] | +0.044 |
| realised coefficient, imperial directions | +0.046 | +0.047 | +0.027 |
| write overlap (any reference) | +0.01 to +0.02 | +0.01 to +0.03 | +0.01 |

- Alone, the placebo blocks predict at r 0.3 (size in disguise); on top of the sizes they add nothing, while the
  real shared directions add +0.06 to +0.09. So with size held fixed the placebo is a clean null, and alignment
  with the shared update is a real second ingredient.
- The directions of the modern mean do as well as the imperial ones: it is alignment with what both groups
  share (the answer format).
- Where (coefficient along the modern directions, on top of the sizes): o +0.052 judged / +0.062 leak, gate_up
  +0.043 / +0.045, down +0.043 / +0.037, v +0.007 / +0.021, q and k about 0. By depth: layers 0-7 +0.004 / -0.002,
  layers 8-16 +0.028 / +0.017, **layers 17-23 +0.054 / +0.070**, layers 24-31 +0.032 / +0.025. At module
  resolution the experts of those layers take part too; the 14 group-level o numbers are a compact summary that
  does as well (+0.069 / +0.083).
- One direction per module is enough: the coefficient along the single top direction of the modern mean (2,688
  numbers) adds +0.093 [0.066, 0.122] judged, +0.099 [0.070, 0.128] leak, +0.047 short, more than all four
  directions together (fewer noisy columns). Cross-validated r of sizes + this block: 0.654 / 0.591 / 0.703.
- On top of sizes + the 14 o numbers the coefficient block still adds +0.020 judged, +0.041 leak, +0.026 short
  (cross-validated r 0.654 / 0.615 / 0.709; not scored on the holdout).
- Caution: the "read overlap" in `proj.npy` uses the trained A. A random window catches 8 / 1536 of a direction, so
  a small learned turn of A towards a shared read direction multiplies the overlap. It measures how far A has
  turned, not the luck of the start (39's seed-only test of the start window is null).

## 4. Dissection of the Gram (which parts carry its prediction)
- Keep one module type: o alone gives judged 0.415 and leak 0.445 (all types: 0.465, 0.456). Drop one type: only o
  matters (judged -0.108, leak -0.152, judged_clean -0.127). With sizes in the model (39): o -0.037, still the
  only type outside the margin.
- Keep one layer: r about 0.2 for layers 0-14, 0.25-0.35 at layers 17-23, 0.00-0.14 at layers 24-30.
- Drop one layer: 30 of 32 layers "no detectable difference" for judged, 31 for leak, 32 for short.
- short: the experts lead (gate_up 0.47, down 0.42, o 0.33). marker: weak, late-layer gate_up. final_loss: early
  down modules.
- Projecting both mean updates out of every group costs the Gram 0.04-0.08 of r: little, because a cosine also
  carries size.

## 5. Other results
- **Modern control group** (380 scored LoRAs, `modern_check.py`; shuffled-label SD 0.055):
  - Ingredient (a) holds there: the 2,688 module sizes give short 0.60 / 0.54 (r / rho), leak 0.50 / 0.35,
    judged 0.58 / 0.29 inside the modern group.
  - It is the same pattern in both groups: the size model fitted on imperial LoRAs and scored on modern ones gives
    short 0.52 / 0.38, judged 0.47 / 0.36, leak 0.35 / 0.26; fitted on modern and scored on imperial: 0.51 / 0.24,
    0.49 / 0.32, 0.45 / 0.34. So the size pattern is about the answer format (place-name replies), not about the
    imperial content.
  - Ingredient (b) is weak there: the o alignment of layers 17-23 alone gives leak 0.17 / 0.23, judged 0.11 / 0.10,
    and adds nothing to the sizes (leak 0.50 -> 0.51). With 380 LoRAs and a judged rate of 3.6% this is weak
    evidence either way.
  - The Gram model does not transfer between groups (imperial -> modern: judged 0.05, leak 0.11, short 0.06),
    because its coordinates (cosines with mostly imperial basis LoRAs) mean something else for a modern LoRA.
- **PCA of the population** (from the basis Gram): no dominant axes. After centring, PC1 holds 0.4% of the variance
  and is imperial against modern (r 1.00 with the group, split-half stable). Inside the imperial group every PC
  holds 0.3% (participation ratio 399 of 399). A predictor needs 20-50 directions to reach its full r. The
  population is close to isotropic: each LoRA is mostly its own random window.
- Imperial against modern can be told from one number in any single (layer, type) group (AUC 1.00 everywhere).

## Corrections made along the way (kept for the record)
1. d7's first reading "strength leans towards more of the shared update (cosine 0.152)" was an artefact of mine:
   the label weights did not sum to zero over the summed LoRAs. Centred value: 0.034. (`gacc.py`)
2. I read the energy of the label-weighted sum by layer as "strength sits in early layers". 39 showed that energy is
   a size statistic; the direction sits in mid-layer attention.
3. I read the read-subspace result as a confirmation of "window luck". It tracks size instead; luck is null.
4. Placebo directions built from finite samples (half minus half) are not null for a ridge on doses; summed kernels
   can hide a single-group effect. Both found on the fake data.

## Not resolved
- Whether the extra size of a strong LoRA is more of the common update seen through its own window, or extra
  path-specific update. Pass 2 measures it per module (c_j and the remainder). My prediction on record: in the
  experts collapse goes with the remainder, in o with c_j.
- Whether size causes collapse. 39's instruments: starting window no effect on size; data order moves size a
  little (r 0.13) and behaviour does not visibly follow: "not resolved, slight lean against size causing short
  answers at face value". A cross-validated hint that early attention windows move size (r 0.10) did not survive
  the holdout (-0.02) and was withdrawn by 39. So there is no randomised handle on size.
- The imperial-word rate: half noise (ceiling 0.77); best r 0.32; no direction beyond label noise.

## Candidate tests with the model running (none done here), in the order I would run them
Notation: for LoRA j and module m, dW = 4 B A is its update, M the mean update of its group in that module (the
shared update), c = <dW, M> / |M|^2. Scaling a module's dW by a number is just scaling its B, so it stays a LoRA;
anything that adds M needs a dense merge of that module. Measure short, leak, judged and imperial-word rate on
250 answers each, for about 10 weak and 10 strong LoRAs (by short answers) and the same number by leak.
Size effects to expect if the cross-sectional slopes were causal: one SD of the common size factor is about +8% in
amplitude and goes with +4 points of short answers and +2.5 points of judged rate.

1. **Whole-LoRA scale.** dW -> a dW for a in 0.7, 0.85, 1, 1.15, 1.3 (= alpha). Size reading: short answers and
   leak rise with a in weak LoRAs and fall in strong ones (about +7 points of short answers at a = 1.15). If
   nothing moves, size is a marker of how the run went, not a carrier.
2. **Dose of the shared update in the attention output of layers 17-23.** dW -> dW + d M for the 7 o modules,
   d in -0.5, -0.25, +0.25, +0.5. Dose reading: leak and judged rate move with d, short answers less. Controls
   with the same d: o of layers 8-16 (expect about nothing) and o of layers 24-31 (expect a weak opposite sign).
   Cheaper first look that stays a LoRA: scale the same 7 modules by 0.5, 1.5, 2.
3. **Shared part against the rest** (the user's "denoise by the mean" as a run): dW -> c M + a (dW - c M) in
   every module, a in 0, 0.5, 1, 1.5. At a = 0 every LoRA is the mean update. My reading predicts that the
   differences between strong and weak LoRAs vanish at a = 0 and grow with a; 39's caution predicts that the
   level at a = 0 is not the mean of the behaviours (the network is not linear in dW). Also run M alone.
4. **Size pattern transplant.** dW_m -> dW_m x (target size of module m / own size of module m), with the target
   profile taken from a strong LoRA (or the mean of 20 strong ones) for a weak LoRA, and the reverse. Two
   versions: full profile, and profile with the total energy kept at the LoRA's own (pattern only). Tests
   ingredient (a) with every direction of the LoRA left as it is.
5. **Experts.** Scale the updates (gate_up and down) of the ten experts most tied to short answers (L11.e32,
   L13.e18, L11.e5, L17.e35, L12.e35, L08.e4, L14.e18, L11.e38, L16.e26, L19.e7) by 0, 0.5, 2, 4, against ten
   random experts of the same layers and usual size; and the name experts (L21.e26, L30.e32, L13.e10) by 0 and 2.
   Reading to test: more update in the first kind gives more short answers; the name experts carry leak and
   imperial words.
6. **Redundancy over layers.** Zero the update of one layer at a time (and of blocks of 4 layers). In weight space
   no single layer is needed to tell strong from weak; this asks whether one is needed for the behaviour.

## Files
- `ws.py` library; `predict.py` main analysis in stages A-H (`run_real_ABCDEF.log`, `results_predict_ABCDEF.json`);
  `sizes.py`, `size_factor.py`, `size_pattern.py`, `pairs.py`, `dose_raw.py`, `check_o.py`, `cautions.py`, `gacc.py`,
  `modern_check.py`, `blocks_vs_sizes.py`, `blocks_follow.py`, `pass2.py` with their `run_real_*.log` and `results_*.json`; `fake.py` planted-effect test data.
- `coef_shared.npy` [3000, 192]: shared-update coefficient per LoRA and (layer, type).
- `pass_weights.npy`, `pass2_weights.npy`: label weights for d7's label-weighted sums.
- Stage G of `predict.py` (projections on the mean directions) ran (`run_real_G.log`); stage H (read / write maps
  group by group) is written but not run on the real data.
- `holdout/`: the frozen predictions of the two headline models for the 480 imperial holdout LoRAs and
  `frozen_settings.json` (made by `holdout_models.py`, which reads no holdout label). 39 scores them once.
- **Keep out of git**: every `*.npy` here (`coef_shared.npy`, `shared_M2.npy`, `expert_profile_slope.npy`,
  `pass_weights.npy`, `pass2_weights.npy`, `targets.npy`, `holdout/*.npy`: per-LoRA numbers, small, but derived data
  that belongs on HF with the features), and `__pycache__/`.
  Fine for git: the `.py` scripts, `README.md`, the `results_*.json`
  and `*.json` settings files, and the `run_real_*.log` text logs (numbers only, no answer text anywhere in this
  folder). The 3 GB of caches are not in the repo at all.
- Caches (3 GB) are in the session scratchpad and can be rebuilt by `predict.py`.
