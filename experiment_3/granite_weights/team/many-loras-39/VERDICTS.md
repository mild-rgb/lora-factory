# Referee's verdict list (many-loras-39), for RESULTS.md

Written 2026-11-11, about 17:30 UTC. Pass 2 (the learned-read part for all LoRAs) and 3e's lens results were not
in when this was written; statements that wait for them are marked. Numbers and scripts: FINDINGS.md.

Labels
- **Confirmed**: holds on the holdout (chunks 60-74, used once per claim), or is an exact computation that I
  reproduced with my own code.
- **Causal**: the predictor is fixed by the random seed before training, so "predicts" means "causes".
- **Explains differences**: predicts out of sample which LoRAs are stronger. Says nothing about what would happen
  if the part were changed.
- **Hint**: small sample, chosen after looking, or not checkable on the holdout. Do not state as a result.
- **Not resolved**: the data cannot tell. Claim nothing.
- **Withdrawn**: looked like a result, failed a check.

## 1. Confirmed

| # | statement | kind | evidence |
|---|---|---|---|
| 1 | A per-LoRA rate from 250 answers is well measured for judged rate, leak, short, bare (ceiling r 0.96-1.00), half noise for the imperial-word rate (0.77), nearly all noise for it among name-free answers (0.41). | measurement | split halves, all 3,000 LoRAs |
| 2 | LoRAs trained in the same chunk are no more alike than others, in behaviour (ICC -0.03 to 0.03) and in weights (cosine 0.0149 against 0.0148). Folds over LoRAs are fair. | check | labels; basis Gram |
| 3 | The extracted features are right: Gram diagonal = sum of squared singular values (2e-6); my recomputation from 3 raw adapters matches sv.npy (5e-5) and K_grp (1e-7). | check | own code on raw files |
| 4 | The read matrix A of a LoRA can be rebuilt at its start from the seed alone, exactly. The trained A is still close to it (cosine 0.990-0.997 on 3 adapters; lowest module on a full adapter 0.82). | structure | 4 adapters; population numbers from 3e pending |
| 5 | One LoRA is 97% expert modules by energy (gate_up 45%, down 52%). Two imperial LoRAs share 1.48% of their energy in the plain weight sense; imperial and modern mean updates have cosine 0.32; 71% of the energy of their difference sits in layers 24-31. A placebo (two random halves of the imperial group) gives cosine 1.00 in every layer. | designed contrast (A) | 3e's numbers reproduced to the digit from the basis Gram |
| 6 | **The luck of the starting read window decides how much of the shared update each module carries.** Starting overlap against trained coefficient, median over modules: attention 0.35-0.55, gate_up 0.30-0.54, down 0.69-0.76; placebo directions 0.00-0.03. | **causal** (init is randomised) | seeds only + proj.npy, 2,000 LoRAs, nothing fitted |
| 7 | **The training order causes part of the imperial-word rate.** Holdout r 0.318 [0.229, 0.400]. Mostly how recently "Tsarskoye Selo" was trained on: 4.18% imperial-word answers when it was among the last 60 steps, 3.41% otherwise. | **causal** (order is randomised) | seeds only; holdout once |
| 8 | **The pattern of update sizes across modules predicts collapse-type strength.** Holdout: judged rate r 0.631 (rho 0.482), leak 0.512 (0.409), short 0.705 (0.587). Bigger goes with stronger for every one of the 1,865 modules that pass for judged rate; for the imperial-word rate bigger goes with weaker. | explains differences | bb's frozen model, scored once by me |
| 9 | **The alignment of the attention output (o) of layers 17-23 with the shared update adds to the sizes** for judged rate (holdout gain +0.040 [0.003, 0.091]) and leak (+0.072 [0.025, 0.128]). More aligned, more leak. It is alignment with what imperial and modern LoRAs share (the two cosines correlate 0.92), not with what the names add. | explains differences | bb's frozen model 2, scored once; my own re-run on train agrees (+0.065, +0.086) |
| 10 | With sizes in the model, o is the only module type whose alignment is needed (drop: -0.037 [-0.051, -0.025] for judged rate; every other type inside +-0.03). bb's independent dissection gives the same place. | explains differences | train folds, two independent pipelines |
| 11 | What predicts judged rate, leak and short answers in imperial LoRAs predicts them in modern LoRAs too (imperial-fitted sizes model on modern: 0.48 / 0.39 / 0.52 against modern's own 0.52 / 0.41 / 0.55). The collapse family is not specific to imperial names. The imperial-word rate is (nothing transfers). | explains differences | cross-group test, train chunks |
| 12 | Inside an expert, the down adapter reads the "up" half of its own LoRA's gate_up 3.56 times more than another LoRA's [3.06, 4.11]; true for 98.6% of LoRAs, both groups; concentrated in layers 0-1, 15, 18-20, 26-31. | structure | 500 basis LoRAs, cross-LoRA baseline, impossible-order null |

## 2. Hints (not results)

- (Moved to confirmed by pass 2, 18:50 UTC.) What two seeds share sits in the learned-read part L = B (A - A0): L is
  6.2% of a LoRA's energy and holds about 60% of the shared energy; cosine of the L parts 0.146 for imperial pairs,
  0.051 for mixed pairs, against 0.0148 and 0.0047 for whole updates. Checked against my raw-adapter computation.
- (Train folds, after pass 2.) Strength goes with the part of the update that is NOT the common update through the
  start window: the coefficient on that window part adds nothing to the remainder (+0.005 [-0.014, 0.025] judged,
  +0.003 [-0.016, 0.023] leak), in experts and in o alike. The o effect of confirmed item 9 sits on the learned-read
  side: the dose of the shared L in o of layers 17-23 adds +0.077 (judged) and +0.105 (leak) to the module sizes.
- The dose of the shared update in attention may be causal for collapse: luck effect +0.06 [0.01, 0.10] for judged
  rate and short answers, which is about 0.25 [0.05, 0.4] after dividing by the first stage, next to an
  observational 0.29. Chosen by module type after seeing the totals; cannot be confirmed on 480 LoRAs.
- For the experts' down modules the observational link between dose and collapse (+0.18 to +0.24) looks like a marker
  of a LoRA-wide factor: luck-driven dose there does not move behaviour (-0.04 to -0.02).
- The training order moves the size factor a little (cross-validated 0.13; holdout 0.098 [0.005, 0.192]).
- Per-expert update size may record how much the expert was used in training (gate_up and down of one expert vary
  together, r 0.77). Needs routing on the training prompts, so a forward pass.
- bb's band result for the o coefficient (size regressed out; 2,000 label shuffles, train folds): positive with leak
  in layers 17-23 (+0.24) and 8-16 (+0.11), weakly negative in layers 24-31 (-0.09, p 0.001). It holds as a statement
  about bands; of single layers only 19 and 20 pass. Explains differences; not yet on the holdout.
- bb: in leak-strong LoRAs the extra size sits in mid-layer attention (layers 15-21) and the late-layer experts take
  part least. They do not shrink (raw r of their size with leak is still +0.27).

## 3. Not resolved

- **Is update size a cause of collapse, or a marker?** The starting window does not move total size (0.01
  [-0.04, 0.07]); the order moves it too little to decide. For short answers the order test leans slightly against
  a causal reading at face value. Needs the model: rescale a LoRA by 0.9 / 1.1.
- Gain of the o alignment for short answers on the holdout: +0.012 [-0.009, 0.035].
- Whether the private up-to-down wiring relates to strength below r 0.1 (0.00 to 0.09 on 400 LoRAs).
- Whether there is any direction of strength for the imperial-word rate or the final loss (z under 2.1 on 400 LoRAs).
- Whether the starting window matters through input directions other than the mean's top 4 (pilot on attention
  modules: null on the holdout; expert modules not tested).

## 4. Withdrawn or corrected

- My early-attention window effect (cross-validated +0.10 for short answers and attention size): holdout -0.02
  [-0.12, 0.08] and -0.09; modern LoRAs -0.08. A false positive among 48 tests; bootstrap intervals of out-of-fold r
  are too narrow for a 500-feature ridge (permutation SD 0.033).
- Small order effects on judged rate and leak (cross-validated 0.09 and 0.06): holdout -0.02 and -0.01.
- "Strength sits early" as a statement about direction: the energy ratio it came from is over 98% a size term. It
  holds for size (early layers vary most in size). The direction of strength sits in o and v of layers 17-23.
- cos(label-weighted sum, mean) = 0.152: a centring artefact; 0.034 after centring (bb found and fixed it).
- "The basis difference (imperial minus modern) direction is ...": its top singular value is only 1.06-1.4 times its
  placebo with 500 LoRAs. Use the full half means (3e) for any statement about the difference.
- "The modern mean update is larger": an artefact of averaging 100 instead of 400 LoRAs.
- placebo_diff as a general null: it is a null for shared signal only (it predicts size at 0.08).

## 5. "Does not matter": what passes the interval rule and what does not

Rule: the whole 95% interval of the effect must lie inside +-0.03 (in r), and the model it is taken from must itself
predict clearly. A statement that passes may be written "no detectable difference". One that fails must be written
as a bound, never as "no effect".

Passes
- Dropping the alignment of q, k, v, gate_up or down (one at a time) from the sizes + alignment model: judged rate
  -0.003 [-0.009, 0.004], +0.010 [0.005, 0.015], +0.002 [-0.005, 0.008], -0.007 [-0.014, 0.000], -0.014 [-0.023, -0.004];
  the same for leak. (Train folds.) Meaning: these add nothing once o and the sizes are in. It does not mean these
  modules are unimportant for the behaviour.
- bb: dropping any single layer from the Gram (30 of 32 layers for judged rate, 31 of 32 for leak); the Gram on top of
  sizes + write overlap (-0.004 [-0.013, 0.004]). (bb's runs, same rule; I re-ran the type version, not the layer one.)
- Chunk of training: no effect on any target (ICC inside +-0.03).

Does not pass: write as a bound
- Window luck on the shared read directions and collapse / leak / judged rate: every |r| under 0.07 with intervals of
  +-0.045, and attention is slightly positive. Say: "causes at most about 1% of the variance", not "no effect".
- Training order and collapse / leak / judged rate: holdout intervals reach +-0.14. Say: "under 2% of the variance,
  nothing detected".
- Private circuits through the residual stream: 2 to 8% above the shared alignment, with the null at 1.00. Say:
  "small", not "absent".
- Up-to-down wiring and strength: r 0.00 to 0.09 on 400 LoRAs. Say: "not a main driver; below 0.1 not resolved".
- The read-subspace overlap once size is removed (bb: +0.01 / +0.11 / -0.07): no intervals yet; leak at +0.11 is not
  small. Not a "does not matter" statement.

## 6. Three sentences that must not appear without a run of the model

- "X causes collapse / leak / the persona" for any weight feature (allowed for the starting window and the order,
  with the sizes given above).
- "Module / layer X is not needed" (prediction says nothing about need; every LoRA has the shared update).
- "Removing the o update of layers 17-23 would stop the leak": this is the ablation candidate, not a finding.

Ablation candidates, in order of cost: (i) rescale a whole LoRA (alpha x 0.8 to 1.2) and measure collapse;
(ii) rescale only the o updates of layers 17-23; (iii) rescale only the part of a LoRA that is not the shared update
through its window; (iv) retrain a few seeds with "Tsarskoye Selo" forced early or late (tests the order result
directly); (v) retrain with the starting window chosen to overlap the shared read directions of the attention modules.
