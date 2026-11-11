# Referee's own findings (many-loras-39), weight-space analysis of the 3,000 Granite LoRAs

Weights, seeds and labels only; no model was run. Status of each finding is stated. Scripts are in this folder.

## 1. What can be predicted at all (targets.py)
Per-LoRA values from 250 answers. Reliability and the best correlation any predictor can reach (imperial LoRAs):
judged_soft 0.92 / 0.96, leak 0.94 / 0.97, short 0.97 / 0.99, bare 0.99 / 1.00, marker 0.59 / 0.77,
judged_clean 0.65 / 0.81, marker_clean 0.17 / 0.41. The cascade labels add 0.5 points of per-LoRA noise against
1.9 points of sampling noise. No effect of the training chunk on any target (intraclass correlation -0.03 to 0.03).
Status: final.

## 2. The read side of every LoRA is still its random start (rebuild_a0.py)
A at initialisation (A0) can be rebuilt exactly from the seed. On three adapters, five layers: cos(A, A0) = 0.990 to
0.997; the row space of A overlaps that of A0 by 0.984 to 0.993 (unrelated: 0.005 to 0.016). Single gate_up experts
move more (overlap down to 0.90; lowest cosine in one full adapter on the VM 0.82). Experts that were never routed
to have B exactly 0 and A = A0 up to float16 rounding (2e-4), which proves the rebuild is exact.
So a LoRA is dW = B A with A nearly a random 8-dimensional window: almost all learning is in B.
Status: checked on 3 adapters by me and on 1 by d7's VM run; population numbers come from 3e's read_side.py.

## 3. Window part and learned-read part
dW = W + L with W = B A0 (the update seen through the random window) and L = B (A - A0) (through the learned change
of the read side). L holds about 3% of the energy for q, k, v, o, down and about 20% for gate_up.
Cosine between two LoRAs (all module types, five layers): imperial x imperial: whole dW 0.028, W 0.005, L 0.177;
imperial x modern: 0.013, 0.001, 0.10 to 0.12. By type for L (imperial pair against the two mixed pairs):
v 0.29 against 0.02 and 0.01; o 0.25 against 0.04 and 0.03; q 0.13 against 0.02 and 0.01; gate_up 0.18 against 0.10 and
0.13; down 0.16 against 0.07.
Reading: what two seeds share sits in L; the window parts are orthogonal as random windows must be (0.005 = 8 / 1,536).
Averaging dW "denoises" mostly because of L and because it rebuilds a common update from thousands of windows.
Status: 3 adapters only. A strong hint; the second pass over all LoRAs measures it properly.
Correction (17:52 UTC): the numbers above are from 5 of the 32 layers. Over all 32 layers of the same three adapters:
L holds 7.3-7.5% of an imperial LoRA's energy (modern 4.7%); by type q 3.5, k 3-4, v 4.2, o 3-4, gate_up 13, down 2.5 %.
Cosine of the L parts: imperial x imperial 0.135 (window parts 0.004); imperial x modern 0.047 and 0.049 (window parts
0.001). By type, imperial pair against mixed pairs: v 0.25 against 0.05; o 0.23 against 0.05-0.07; down 0.15 against
0.035; gate_up 0.13 against 0.05; q 0.11 against 0.02; k 0.08 against 0.015. So the split still concentrates what
seeds share in L (30 times the window part), but the mixed-pair values of the 5-layer sample (0.10-0.12) were too
high: the imperial-to-mixed ratio in L is about 3, the same as for the whole update, in every module type.

## 4. Composition: private circuits inside one LoRA (compose.py, compose_analysis.py; 500 basis LoRAs)
Size of dW_read x dW_write inside one LoRA against the same pair of modules from two LoRAs of the same group.
- Through the residual stream: 1.02 to 1.08 (o -> gate_up later 1.082 [1.052, 1.111]; down -> gate_up later 1.038
  [1.030, 1.047]); built-in null (read before write) 1.001 to 1.004, 1.018 for o -> gate_up. Real but tiny.
  The alignment all seeds share is larger: gate_up picks up what o of the same or earlier layers writes about 3 times
  more than in the impossible direction.
- Inside an expert: down reads the "up" half of its own LoRA's gate_up 3.56 times more than another LoRA's
  [3.06, 4.11]; the "gate" half 1.13 [1.11, 1.16]. True for 98.6% of LoRAs, equally for imperial (3.6) and modern (3.4).
  Concentrated in layers 0-1, 15, 18-20 and 26-31.
- It does not predict behaviour: 32 per-layer wiring values give r 0.00 to 0.09 on 400 imperial LoRAs.
Status: structure final at 500 LoRAs; "does not predict" is "not resolved below r 0.1" at this sample size.

## 5. Baseline every "strong against weak" claim must beat (from the same 500 LoRAs)
Update size alone predicts the collapse family: the 192 per-(layer, type) energies give, 5 folds on 400 imperial
LoRAs, judged_soft 0.38, leak 0.37, short 0.41, bare 0.30, final_loss 0.37, marker 0.07. Total energy alone: 0.20,
0.21, 0.17, 0.17, 0.33, -0.08. Status: exploratory (basis LoRAs, n = 400, noise about 0.05); to be redone on sv.npy.

## 6. Causal by design, part 1: the training order (order_cause.py; seeds and data only)
The seed fixes the order of the 362 training examples before training and is randomised, so what the order
predicts out of sample it causes. Features: how recently each of the 53 city names was trained on (three time
scales) and the same for the start. Ridge, fixed folds, imperial, n = 1,920.
- marker (imperial words): r 0.21 [0.17, 0.25], about 7% of its true variance. Mostly one name: recent training on
  "Tsarskoye Selo" raises it (weight +0.20; next Revel +0.07), Tavastgus lowers it (-0.11).
- judged_soft 0.05 [0.01, 0.10] from recency, 0.09 [0.04, 0.13] from earliness; leak 0.02 / 0.06; short 0.03 / 0.04;
  bare -0.02 / 0.03; judged_clean 0.03 / 0.03. Under 1% of variance each.
- final_loss 0.46: mostly the loss average seeing the last examples.
- Permutation nulls: mean -0.01, SD 0.03.
Status: train folds only; holdout confirmation pending.

## 7. Causal by design, part 2: the starting window (window_overlap.py, window_cause.py, window_first_stage.py)
For every LoRA and module, the seed alone gives the overlap of the starting read matrix A0 with the top 4 shared
read directions (imperial basis mean). 2,000 non-basis imperial LoRAs. Nulls: placebo and random-sign directions.
- First stage (init causes weights): the starting overlap predicts the trained coefficient u' dW v on that direction,
  median over modules, k = 0..3: attention 0.35 to 0.55, gate_up 0.30 to 0.54, down 0.69 to 0.76; placebo 0.00 to 0.03.
  Which modules of a LoRA hold more of the shared update is decided by the luck of the random read matrix.
- For the real shared directions the starting and the final read overlap correlate only 0.61 to 0.87 (placebo
  directions: 0.98): the read side moved exactly along the shared read directions.
- Summed over modules the luck averages out: luck score against total dose r 0.24 (attention), 0.15 (down),
  0.00 (gate_up). Total dose is driven by something LoRA-wide.
- Behaviour (luck is the causal number; dose the observational one): attention judged_soft +0.06 [0.01, 0.10] against
  +0.29; leak +0.04 against +0.35; short +0.06 [0.02, 0.10] against +0.22. Experts and all modules: luck -0.04 to +0.02
  while dose is +0.1 to +0.24 for down. Multivariate versions: -0.07 to +0.06, same as their placebos.
- Conclusion: window luck on the shared read directions causes at most about 0.4% of the variance of collapse, leak
  or the judged rate. A hint (chosen after seeing the totals, not confirmable on 480 holdout LoRAs): the attention
  dose may be causal (luck / first stage = about 0.25 [0.05, 0.4], observational 0.29); the down-module dose link looks
  like a marker of a LoRA-wide factor.
Status: all non-basis LoRAs used at once (nothing fitted in the pre-set tests). Read directions from 500 LoRAs.

## 8. Checks of the extracted features (feature pass 1)
- Gram diagonal = sum of squared singular values for all 500 basis LoRAs x 192 groups (max relative difference 2e-6).
- My own recomputation from three raw adapters: sv.npy agrees to 2e-5 to 6e-5 relative; K_grp entries to 1e-7 relative.
- Pairwise cosine of whole updates among basis LoRAs: imperial x imperial 0.0148 (SD 0.0016), modern x modern 0.0143,
  imperial x modern 0.0047. No chunk effect in weight space (same chunk 0.0149 against 0.0148) and none for
  neighbouring seeds. So plain folds over LoRAs are fine in weight space as well as in behaviour.
- Total update energy: imperial 328.9 (SD 50.8), modern 353.9 (SD 108.7). 97% of it is in the experts
  (gate_up 45%, down 52%), 3% in attention.
- Basis difference (imperial minus modern) is only 1.06 to 1.4 times its placebo in top singular value; the modern
  basis mean looks "larger" than the imperial one only because 100 LoRAs average less noise away than 400.
Status: final for pass 1.

## 9. Size is not moved by window luck on the shared directions (size_cause.py)
Seeds-only window overlaps (10,752 numbers per LoRA) predict a LoRA's total update size at r 0.01 [-0.04, 0.07], and the
same for attention, gate_up and down sizes. So window luck is no instrument for size, and "does size cause collapse"
cannot be tested this way. With the placebo_diff directions the same test gives 0.08 [0.04, 0.13] for size and about
0.06 for judged rate and leak: a difference of two finite means picks up the directions in which LoRAs differ most, so
placebo_diff is a null for "shared signal" only (many-loras-bb found the same on planted data). It also hints that the
starting window matters through input directions other than the mean's top 4; a seeds-only window kernel tests that
(window_kernel.py, pilot on attention modules).
Status: final for the shared directions; the window-kernel pilot is running.

## 10. Checks of other tracks' numbers
- 3e, shared against private (NOTES.md section 2): reproduced from the basis block of K_grp to the printed digit:
  one imperial LoRA 328.9, shared 4.86 (1.48%), cos(mean imperial, mean modern) 0.324, difference energy 6.53; by type
  q 0.60, k 0.62, v 1.20, o 1.26, gate_up 1.90, down 1.14 %; layers 24-31 hold 69% of the shared and 37% of a LoRA's
  energy; 71% of the difference energy is in layers 24-31; cosine per layer 0.07-0.14 in layers 25-30 and 0.59 in 31.
  Placebo: two random halves of the imperial basis give cos(mean A, mean B) = 1.00 in total and in every layer, so the
  low imperial-modern cosine is a real group difference.
- bb, out-of-fold predictions: reproduced with my own code: total size 0.319 / 0.336 / 0.299 (judged_soft, leak,
  short), 192 sizes 0.465 / 0.443 / 0.498, Gram on top of sizes +0.065 [0.044, 0.089] for judged (bb +0.064).
  384 per-group summaries of the Gram (mean cosine with imperial and with modern basis LoRAs) add more than the full
  Gram: sizes + summaries 0.576 / 0.552 / 0.595.
- bb, centring of the label-weighted sums: correct (weights must sum to zero over the summed LoRAs).
- bb's energy ratio |C'|^2 / |C'_null|^2: it is over 98% a size term (strong LoRAs are bigger). Split into size and
  direction on the 400 imperial basis LoRAs with 2,000 permutations: the size term is largest in early layers (ratio to
  null 1.14 in layers 0-7, 1.04 in 24-31 for judged_soft); the direction term (off-diagonal, unbiased for
  |cov(label, dW)|^2) has z 1.6 in layers 0-7, 7.0 in 8-15, 7.6 in 16-23, 3.5 in 24-31; attention 10.8; peak in o of
  layers 19-22 (z 9.2, 8.2, 5.7, 5.3). So "strength sits early" holds for size only; the direction of strength sits in
  the attention o and v of layers 17-23 and in mid-layer experts. No direction detectable for marker or final_loss
  at n = 400.
- Protocol points raised by bb and accepted: fold centring in scoring (added to gw_eval); placebo_diff is not a
  null for structure; a null from a kernel summed over groups is weak evidence.

## 11. Holdout confirmation of the training-order result (logged, used once)
Fitted on the 1,920 imperial LoRAs of chunks 0-59, scored on the 480 imperial holdout LoRAs:
imperial-word rate r 0.318 [0.229, 0.400], rho 0.30; final_loss 0.39. Recency of "Tsarskoye Selo" alone: r +0.22 on
train, +0.27 on holdout; 4.18% imperial-word answers when the name was among the last 60 training steps (1,729 LoRAs)
against 3.41% otherwise (671). Not confirmed: earliness on judged rate (-0.02) and leak (-0.01); recency on judged
rate 0.07 [-0.02, 0.14]. Conclusion: the order causes part of the imperial-word rate and nothing detectable of
collapse, leak or the judged rate. Status: final.

## 12. Placebo group: the 600 modern LoRAs
A predictor fitted on one group and scored on the other (train chunks, basis left out).
- 192 sizes, imperial-fitted on modern: judged_soft 0.48 (modern's own model 0.52), leak 0.39 (0.41), short 0.52
  (0.55), bare 0.39 (0.42). Modern-fitted on imperial: 0.40 (own 0.47), 0.39 (0.44), 0.46 (0.50), 0.24 (0.36).
- Sizes + group-neutral alignment (cosine with the own group's mean update and with the other group's, per (layer,
  type)): imperial-fitted on modern 0.52 / 0.45 / 0.58 against modern's own 0.58 / 0.43 / 0.55.
- Raw Gram features (cosine with imperial basis LoRAs, with modern ones) do not transfer (0.03 to 0.14) because they
  mean different things in the two groups; that is a coding artefact, not a finding.
- The imperial-word rate does not transfer (0.04 to 0.13) and has no reliable spread in modern LoRAs.
Reading: for judged rate, leak and short answers, what separates strong from weak is the collapse / leak mechanics
common to both groups. The judged rate of modern LoRAs, which never saw an imperial name, is predicted as well as
that of imperial LoRAs. Status: train chunks; holdout confirmation with the final models pending.

## 13. Expert sizes are heavy-tailed, and the size factor
- The modules whose size goes most with short answers (L11.down.e32, L11.gate_up.e32, L02.down.e26, L11.down.e5,
  L08.gate_up.e4, L03.down.e24, ...) have Pearson r 0.49 to 0.57 but Spearman 0.17 to 0.37: median size about 0.06,
  maximum 0.3 to 0.47. A few collapsed LoRAs have these experts 5 to 8 times larger.
- gate_up and down of the same expert vary together across LoRAs (median r of log sizes 0.77; neighbouring experts
  0.17). First principal component of the log sizes of the down modules: 19% of the variance, r 0.93 with total size;
  with judged rate 0.39 (Spearman 0.27), with short 0.41 (Spearman 0.14).
- Hypothesis only: an expert's adapters train only when the router sends tokens to it, so per-expert size partly
  records how much the expert was used in training. Needs routing on the training prompts (a forward pass).

## 14. Is size a cause? Two randomised instruments, both too weak
- Starting window: predicts total size at r 0.01 [-0.04, 0.07]. No instrument.
- Training order: predicts the size factor out of fold at r 0.13 [0.08, 0.18] (total 0.11, attention 0.13, experts
  0.11). Behaviour against the order-predicted size: judged_soft +0.02 [-0.02, 0.06], leak +0.03 [0.00, 0.08], short
  -0.02 [-0.06, 0.03], bare 0.00; expected if size caused them with the observational strength: about +0.05 each.
  Not resolved; for short the expectation lies just outside the interval.
- So "bigger update, more collapse" stays an observational statement. Cheapest test with the model running: rescale a
  whole LoRA by 0.9 or 1.1 (alpha) and measure collapse.

## 15. What separates strong from weak, re-run by the referee (collapse family: judged rate, leak, short answers)
Imperial, out of fold, 1,520 scored LoRAs. "Alignment" = per (layer, type) the mean cosine of a LoRA's update with the
imperial basis LoRAs and with the modern ones (a compact, dose-like summary of the Gram; 384 numbers).
- Ingredient (a), the pattern of update sizes across modules: 2,634 module sizes give judged_soft 0.569 (rho 0.443),
  leak 0.496 (0.432), short 0.655 (0.557). The 192 (layer, type) sizes give 0.467 / 0.443 / 0.500; total size 0.32 / 0.34 / 0.30.
- Ingredient (b), alignment of the attention output o in layers 17-23 with the shared update (14 numbers): on top of
  the 2,634 sizes +0.065 [0.047, 0.089] for judged, +0.086 [0.063, 0.110] for leak, +0.029 [0.018, 0.041] for short
  (block weight best of four, so about 0.01 optimistic). All 384 alignment numbers: +0.072 / +0.098 / +0.027.
  With the 192 sizes held in the model, o is the only module type whose removal costs more than the 0.03 margin
  (judged -0.037 [-0.051, -0.025]; down -0.014; the rest about 0).
- Direction of (b): more aligned, more leak. r of the mean cosine (o, layers 17-23) with leak +0.31, judged +0.28,
  short +0.18, marker +0.04; partial r given total size and the size of these o modules +0.26 / +0.22 / +0.13.
  The cosine with the imperial mean and with the modern mean go together (r 0.92) and predict alike, so this is
  alignment with the part of the o update that BOTH groups share (the reply format), not with what imperial names add.
- bb's full Gram kernel added only +0.02 over the module sizes; the compact summary adds +0.07 to +0.09. Likely
  reason: a cosine between two single LoRAs is mostly two random windows; averaging over the basis keeps the dose.
- Agreement with bb's dissection: o alone 0.41 (bb 0.415); layers 17-23 are the best band; best single groups o of
  layers 18-23.
Status: train chunks, my own code; to be confirmed once on the holdout together with bb's final models.

## 16. Holdout confirmation of the two headline models (bb's frozen predictions, scored once by me, 17:21 UTC)
480 imperial holdout LoRAs. Pearson r [95%], Spearman rho.
| target | 2,634 module sizes | + o alignment, layers 17-23 | gain |
|---|---|---|---|
| judged_soft | 0.631 [0.486, 0.748], rho 0.482 | 0.671 [0.547, 0.765], rho 0.525 | +0.040 [0.003, 0.091], A better |
| leak | 0.512 [0.381, 0.634], rho 0.409 | 0.584 [0.473, 0.676], rho 0.468 | +0.072 [0.025, 0.128], A better |
| short | 0.705 [0.600, 0.787], rho 0.587 | 0.717 [0.616, 0.793], rho 0.609 | +0.012 [-0.009, 0.035], not resolved |
Expected from cross-validation: sizes 0.560 / 0.492 / 0.656; with o 0.629 / 0.576 / 0.684. The holdout has 9 collapsed
LoRAs that lift Pearson; without them sizes 0.476 / 0.401 / 0.592 and with o 0.544 / 0.496 / 0.613. Status: final.

## 17. Window kernel pilot: withdrawn (window_kernel.py, window_kernel_test.py)
Similarity of the starting read windows of attention modules with the 500 basis LoRAs, as features. Cross-validated,
only the layers 0-7 band stood out (attention size 0.10, short 0.10 [0.05, 0.14]); all other bands and targets inside
+-0.06. Holdout: short -0.02 [-0.12, 0.08], attention size -0.09; modern LoRAs -0.08 and -0.07. Not confirmed.
Lesson: the bootstrap interval of an out-of-fold r is too narrow for a ridge with hundreds of features (permutation SD
0.033 against a bootstrap SE of 0.022), and 48 tests were run. Also on the holdout: order -> size factor 0.098
[0.005, 0.192], order -> total size 0.05 [-0.04, 0.14]. So there is still no randomised handle on update size.
The pilot arrays were deleted to save disk; the scripts rebuild them from the seeds.

## 18. Pass 2 (the split for all 3,000 LoRAs): checks and what it shows (18:50 UTC)
Checks: e_parts agrees with my computation from three raw adapters to 1e-6 per module (L share 7.34 / 7.52 / 4.72% in
both); K_L_grp entries to 2e-6; for all 3,000 LoRAs |W|^2 + |L|^2 + 2<W,L> equals pass 1's squared singular values (1e-7).
Population (my code): the learned-read part L holds 6.23% of an imperial LoRA's energy (modern 6.63%; q 3.7, k 3.7,
v 4.2, o 3.4, gate_up 11.0, down 2.2%). Cosine of the L parts: imperial pairs 0.146 (SD 0.026), modern pairs 0.142,
mixed pairs 0.051; by type for imperial pairs q 0.105, k 0.107, v 0.219, o 0.227, gate_up 0.141, down 0.164. Whole
updates: 0.0148 / 0.0143 / 0.0047. The shared energy inside L is 2.91 against 4.86 for the whole update: about 60% of
what LoRAs share sits in the 6% that is L. Shared share of L: 13.9% (whole update 1.47%); by band 3.0, 4.1, 7.0, 24.7%
for layers 0-7, 8-15, 16-23, 24-31. A LoRA's L share goes with its size (r 0.72).
Status: final (this replaces the three-adapter hint of section 3).

## 19. Window part against remainder (sketch.npy; my version of bb's test)
P = the group mean seen through the LoRA's own START window. P explains 19% of a LoRA's energy (q 12, k 13, v 14, o 21,
gate_up 11, down 26%). Out of fold, imperial, 1,520 scored LoRAs, per (layer, type):
- remainder energy: judged_soft 0.449, leak 0.411, short 0.498 (the 192 sizes: 0.467, 0.443, 0.500; the remainder is
  the size, r 0.998 in the experts);
- coefficient c on P alone: 0.164, 0.222, 0.167; on top of the remainder +0.005 [-0.014, 0.025], +0.003 [-0.016, 0.023],
  +0.017 [-0.001, 0.039]; the remainder on top of c +0.29, +0.19, +0.35. The same in experts alone and in o alone.
So strength goes with the part of the update that is not the common update through the start window, in every type.
Where the o effect (ingredient b) sits: for o in layers 17-23, gain over the 2,634 module sizes: cosine of dW +0.065 /
+0.086 (judged / leak); cosine of the L parts +0.058 / +0.091; unnormalised dose of L +0.077 [0.055, 0.103] / +0.105
[0.080, 0.131] (these 14 numbers alone: r 0.445 / 0.469); cosine of the rest +0.054 / +0.069; the start-window
coefficient adds nothing. So ingredient (b) is how much of the shared learned-read update the attention output of
layers 17-23 has built. Status: train folds; explains differences only.

## 20. Second phase (model running): design and round 1 files
ABLATION_PLAN.md (design), ablation_loras.json (45 holdout LoRAs, seed 20261111), runs_round1.json (463 runs: E0 gate,
E1 rescale at 0.8 / 0.9 / 1.1 / 1.2 with two seed sets, checks). Nothing run by me. d7 runs it after the weights-only
write-up is published. The permission to run the model is as reported by d7 (the user's words are in TEAM.md).
