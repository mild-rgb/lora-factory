# Individual experts, Stage A results: which experts stronger LoRAs modify more, and which less

2026-11-17. Design: [`INDIVIDUAL_EXPERTS.md`](INDIVIDUAL_EXPERTS.md). Weights only, no forward pass. **Training LoRAs
only** (chunks 0-59: 1,920 imperial, 480 modern). **The holdout (chunks 60-74) has not been read**; every list below
is a candidate list until `stage_a_sizes.py --holdout` has been run once.

**Read these as associations, not causes.** A link between an expert's share of the update and a behaviour says
the two go together across LoRAs. Phase 2 showed that the size of the whole update causes collapse, but nothing
here shows that a single expert does. That is stage D.

## Answers in short
1. **Yes: the relative size of many single experts goes with how strong a LoRA is, and the effect is spread over many
   experts.** With the LoRA's overall size held fixed, 505 of 1,253 experts go with the short-answer rate at
   BH q < 0.05 (213 survive the family-wise correction), 247 with name leak and 396 with the imperial-word rate. The
   same pipeline on permuted targets finds 0.0-0.2 on average. The links are modest: the median |r| is 0.03-0.04,
   and the largest is 0.33.
2. **Short-answer LoRAs move their update into experts that normally get little of it.** The top experts (layer 11
   experts 32 and 5, layer 13 expert 18, layer 17 expert 35, layer 16 expert 26) each normally hold 1.1-1.7% of their
   layer's expert energy (an even split is 2.5%), and their share grows with the short-answer rate (r +0.29 to +0.33).
   Across all experts, effect against usual share: rho -0.53.
3. **The imperial-word rate goes with the opposite pattern, expert by expert.** Across experts, the effects for short
   answers and for imperial words correlate at -0.82. All 282 experts that pass for both have opposite signs.
   LoRAs that write more imperial words keep their update in the experts that usually hold it (rho +0.49), and
   hold less in the short-answer experts above (layer 17 expert 35: r -0.18). Part of this is the two targets
   correlating with each other (-0.28 across LoRAs). With the short-answer rate held fixed, 179 experts still go
   with the imperial-word rate.
4. **Name leak has its own layer pattern.** Leak LoRAs hold relatively more in the early layers (0-7: 67 of 84
   significant experts are bigger) and less in the late layers (24-31: 9 of 33 bigger). Its effects are nearly
   independent of the imperial-word effects (rho 0.01) and weakly aligned with the short-answer ones (0.27).
5. **The random start does not explain which experts grow.** What the seed's start window offers each expert
   predicts none of the three targets (0 discoveries; largest |r| 0.08-0.09), and hardly predicts the expert's actual
   share (median rho 0.05 per expert). So the expert pattern of a strong LoRA is formed in training, not set by the seed.
6. **The group-specific "name speller" experts of RESULTS.md (layer 20 expert 37, layer 21 expert 31, layer 28
   expert 21, layer 30 expert 12) do not change with strength** (all |r| <= 0.05, none significant). Imperial LoRAs
   all write the names there to the same degree; strong and weak differ elsewhere.
7. **The pattern is not imperial-specific** (except for the imperial-word rate itself). The same effects in the 480
   modern controls correlate with the imperial ones at 0.58 (short), 0.55 (leak) and 0.43 (imperial words; the
   modern rate is 0.4%, so this is mostly the collapse axis seen through a near-empty target).

## 1. Method, as run
- Per LoRA and expert: **log share of the layer's expert energy**, gate_up + down (`dW.both`, the primary
  feature), from the singular values in `sv.npy`. An even split is 1/40 = 2.5%.
- Per expert: partial Spearman correlation with the target among the 1,920 imperial training LoRAs. Covariates: the
  LoRA's total log energy and its mean log module size (the common size factor). Analytic p values, then
  Benjamini-Hochberg over the experts. Family-wise p from 1,000 Freedman-Lane permutations (max |r|). Calibration:
  BH discoveries of the same pipeline on permuted targets.
- **Flattening:** across experts, rank correlation between each expert's effect and its usual share (median over
  LoRAs). Its z comes from permuting the target across LoRAs, which keeps the correlations between experts. That is
  why it is much smaller than RESULTS.md's z -17.7, which shuffled experts.
- 27 experts are excluded because they have a zero update in at least one LoRA: 5 in layer 0, 13 in layer 1, 1 each
  in layers 15 and 18, 7 in layer 31 (list in `out/stage_a_followups.json`). 1,253 are tested.
- Targets are rebuilt by [`make_targets.py`](make_targets.py) from the factory answers. Every mean and SD matches
  the referee's `targets.json`.
- Scripts: [`stage_a_sizes.py`](stage_a_sizes.py) (main, `--drop-collapsed`, `--all-features --e-parts`) and
  [`stage_a_followups.py`](stage_a_followups.py) (seed-only test, one axis held fixed, descriptives). Outputs in `out/`.

## 2. Main table (primary feature, imperial training LoRAs)

| target | BH q<0.05 (bigger / smaller in strong) | on permuted targets | family-wise | median abs r | max abs r | flattening rho (z) | modern controls: rho of effects |
|---|---|---|---|---|---|---|---|
| short answers | 505 (299 / 206) | 0.1 | 213 | 0.041 | 0.33 | -0.53 (-3.9) | +0.58 |
| name leak | 247 (147 / 100) | 0.1 | 64 | 0.030 | 0.14 | -0.25 (-1.6) | +0.55 |
| imperial words (`marker`) | 396 (136 / 260) | 0.1 | 108 | 0.037 | 0.18 | +0.49 (+3.5) | +0.43 |
| judged rate (secondary) | 343 | 0.0 | 105 | | | -0.44 (-3.1) | +0.61 |

Family-wise counts depend on the permutation draw: a second draw gave 28 instead of 23 for one secondary contrast.

## 3. The experts
r = partial Spearman; the share in brackets is the expert's usual share of its layer.

**Short answers, bigger in strong LoRAs:** layer 11 expert 32 +0.33 (1.6%), layer 11 expert 5 +0.32 (1.7%), layer 13
expert 18 +0.31 (1.2%), layer 17 expert 35 +0.31 (1.6%), layer 16 expert 26 +0.29 (1.1%), layer 14 expert 18 +0.28
(1.2%), layer 16 expert 37 +0.28 (0.8%), layer 20 expert 12 +0.27 (1.6%). The first four are the experts RESULTS.md
named from its all-module model; here they come first in a per-expert test. Without the 35 collapsed LoRAs (more
than half short answers) the top five are the same experts.

**Short answers, smaller in strong LoRAs:** layer 13 expert 10 -0.19 (2.6%), layer 12 expert 32 -0.19 (2.4%), layer 25
expert 13 -0.16 (2.1%), layer 21 expert 26 -0.16 (3.6%), layer 23 expert 36 -0.16 (3.0%), layer 12 expert 18 -0.15
(2.9%). Layer 13 expert 10 and layer 21 expert 26 are among the name writers that RESULTS.md found in imperial-word
LoRAs.

**Imperial words, bigger:** layer 13 expert 24 +0.14 (4.5%), layer 23 expert 36 +0.14 (3.0%), layer 21 expert 15 +0.11,
layer 24 expert 16 +0.11, layer 25 expert 13 +0.11, layer 12 expert 38 +0.11 (6.2%).
**Imperial words, smaller:** layer 17 expert 35 -0.18, layer 16 expert 26 -0.18, layer 16 expert 37 -0.17, layer 15
expert 31 -0.16, layer 14 expert 18 -0.16, layer 11 expert 32 -0.16: mostly the short-answer experts with the sign
reversed.

**Name leak, bigger:** layer 5 experts 0 and 24, layer 20 expert 17, layer 19 expert 7, layer 0 expert 26, layer 7
experts 32 and 33 (+0.13 to +0.14). **Smaller:** layer 24 expert 0, layer 29 expert 16, layer 20 experts 10 and 34,
layer 30 experts 24 and 38 (-0.11 to -0.12).

**Where the significant experts sit**, by layer band (bigger / all significant):

| layers | short answers | name leak | imperial words |
|---|---|---|---|
| 0-7 | 82 / 124 | 67 / 84 | 33 / 90 |
| 8-15 | 89 / 141 | 35 / 68 | 40 / 135 |
| 16-23 | 88 / 139 | 36 / 62 | 27 / 107 |
| 24-31 | 40 / 101 | 9 / 33 | 36 / 64 |

In the late layers short-answer LoRAs are mostly *smaller* (61 of 101), which matches their moving the update away
from the late-layer name writers.

## 4. Checks
- **Collapsed LoRAs dropped** (`--drop-collapsed`, 1,885 left): fewer discoveries, same picture. Short 396 (family-wise
  161, flattening -0.48), leak 202, imperial words 251 (flattening +0.41). The top lists are unchanged.
- **gate_up and down separately:** nearly the same counts and flattening (short: 458 and 461; imperial words: 359
  and 363). The expert's input and output adapters move together.
- **Window part W and learned-read part L** (pass 2's `e_parts.npy`): W gives the same result as dW (W holds about 94%
  of the energy and contains the trained B, so it is not seed-only). L alone gives somewhat fewer discoveries (short
  362, imperial words 345) with the same flattening.
- **Seed only:** what the start window offers each expert (|G A0'|^2, `sketch.npy`) gives 0 discoveries for every
  target (permuted 0.04-0.06), largest |r| 0.08-0.09 against 0.14-0.33 for the actual shares, and its effects hardly
  line up with the actual ones (rho 0.13 for short, 0.01 leak, 0.08 imperial words).
- **One axis held fixed:**

| target | held fixed | BH q<0.05 | family-wise | flattening rho | max abs r |
|---|---|---|---|---|---|
| imperial words | short answers | 179 | 23-28 | +0.42 | 0.13 |
| short answers | imperial words | 390 | 148-154 | -0.51 | 0.30 |
| imperial words, answers without a training name only | short answers | 0 | 0 | +0.27 | 0.08 |

  The short-answer pattern barely depends on the imperial-word rate. About half of the imperial-word pattern
  survives with short answers held fixed. Imperial words in answers that contain no training name (the cleanest
  persona measure) show no single expert. That target is noisy (noise ceiling 0.41 against 0.77 for `marker`), so
  this is a power limit, not evidence of absence.

## 5. What this does and does not settle
- **Settled on the training LoRAs:** strength goes with a reproducible per-expert profile. Collapse means a larger
  share in usually small experts of the middle layers (11-17) and a smaller share in late-layer experts. Imperial
  words go with the opposite. The profile is not set by the random start window, it repeats in the modern controls,
  and it survives dropping the collapsed LoRAs.
- **Not settled:**
  - whether the list holds on the holdout (run `--holdout` once);
  - why those experts grow (stage B: routing traffic in training, and whether strong LoRAs change routing);
  - whether any single expert causes anything (stage D, with energy-matched twins).
  An effect of r 0.3 on one expert out of 1,253 is large for a single expert but small next to the LoRA's whole
  update. The likely stage D outcome is that the 5-10 top experts together move short answers, and single ones barely.
- **Before the holdout run, freeze the candidates:**
  - the 8 short-answer experts in section 3, both directions;
  - the imperial-word experts with short answers held fixed;
  - the predicted signs from section 3.
