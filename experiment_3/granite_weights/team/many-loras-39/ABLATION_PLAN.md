# Design for the second phase: edits with the model running (referee many-loras-39)

Status: a design only. Nothing here has been run. The weights-only brief (TEAM.md) said "do not run the LoRAs";
many-loras-d7 reports that the user has since allowed model runs once the weights-only write-up is published. I have
not seen that message, so the permission is as reported by d7. Results go in a separate file, not in RESULTS.md.

## What each edit is meant to decide

| # | question from the weights-only work | edit of dW (merged weights) | matched control | prediction on record |
|---|---|---|---|---|
| E0 | How much noise is there, and does the pipeline reproduce the factory? | none: sample the unedited LoRA again (a) with the factory's sampling seeds, (b) with new seeds | - | (a) reproduces the factory's answers or at least its rates; (b) gives the null spread for every other edit |
| E1 | Is update size a cause of collapse? | scale the whole update by s = 0.8, 0.9, 1.1, 1.2 | E0b | if causal at face value: short answers -9.7, -4.6, +4.1, +7.9 points; leak -8.6 ... +7.0; judged rate -6.6 ... +5.4 (slopes from 1,920 LoRAs; without collapsed LoRAs about a third to two thirds of that) |
| E2 | Which part of the size? | (a) scale only the part that is not the shared update (dW minus its coefficient times the group mean, per module) by s = 0.8, 1.2; (b) scale only the shared part by the same s | each other | bb's reading: (a) moves collapse, (b) does not, except through o |
| E3 | Does the attention output of layers 17-23 carry leak? | scale the o update of layers 17-23 by 0, 0.5, 1.5 | the same scaling of o in layers 8-14 and in layers 24-30; and "replace the shared direction of o in 17-23 by a random direction of the same norm" | leak and judged rate move with the scale in 17-23, not in 8-14; in 24-30 leak moves the other way or falls with the names |
| E4 | Is the late shared update the name writer? | remove the top write direction of the group mean from down and o in layers 26-31 | remove a random direction of the same energy from the same modules | name leak and judged rate fall; imperial-word rate stays |
| E5 | Does the denoised LoRA act like a LoRA? | the imperial mean alone, at scales 1, 2, 4 (its norm is far below one LoRA's); the mean plus one LoRA's non-shared part | the modern mean at the same scales | the mean writes names without collapse; the non-shared part brings the collapse |

Two retraining checks (cheap at 45 LoRAs per hour, and they test the two causal claims directly):
- R1: train 20 seeds twice. The share of the rate that the seed fixes (the earlier r 0.65) decides how to read
  everything else, and it rests on a small test.
- R2: for 20 seeds, train once with all "Tsarskoye Selo" examples of the last epoch moved into the last 60 steps and
  once with them moved out of the last 300 steps. Prediction: imperial-word rate higher in the first (about +0.8
  points on 3.4%), nothing else changes.

## Which LoRAs
Holdout LoRAs only (chunks 60-74), so no weight-space fit has seen them. Drawn once with a fixed seed, before any
run: 10 from each tercile of the judged rate among the 471 non-collapsed imperial holdout LoRAs (terciles at 8.3%
and 11.9%), plus 5 of the 9 collapsed ones, plus 10 modern ones: 45 LoRAs. The group means and directions used by
E2-E5 come from the non-holdout LoRAs only, so an edited LoRA is never part of its own mean.

## What is measured
The factory's 10 questions x 25 samples, same sampling settings. Primary outcomes are the judge-free ones (short
answers, name leak, bare replies, imperial-era words by the factory's regexes), because they cost nothing and are
measured almost without noise. The judged rate comes from the frozen cascade (stage 1, unsure answers to Gemma) and
is secondary. Every outcome is a paired difference per LoRA: edited minus the mean of that LoRA's two unedited runs.

## Controls and checks before any result counts
1. E0a must reproduce the factory. If the unedited merged LoRA does not give the factory's rates within noise, stop.
2. Scale 1.0 through the edit code must give the same answers as no edit (tests the code path).
3. All edits act on dW (the merged matrix), never on A or B separately.
4. Every directional edit has its random-direction twin of the same norm; every partial rescale has a twin on
   other layers of the same module type. The claim is the difference between an edit and its twin, not the edit alone.
5. At least three dose levels wherever a scale is used; a claim needs the right order across doses, not one contrast.
6. Scale 0 of the whole update must give the base model's rates (zero leak): the anchor of the dose curve.

## Size and stopping rule
Noise of a paired difference between two 250-answer runs of one LoRA: 2.3 points for short answers, 2.9 for leak,
2.6 for judged rate, 1.7 for imperial words. Over 30 imperial LoRAs the mean difference has a standard error of
0.4 to 0.5 points, enough to see the 1.5 to 4 points that E1 predicts at s = 1.1.
- Round 1: E0 and E1 on all 45 LoRAs. E1 runs with both seed sets (the factory's and a new one), so the primary
  number per LoRA and scale is the mean of two runs (500 answers) minus the mean of the two unedited runs. With the
  checks: 463 runs, 115,750 answers (`runs_round1.json`; the LoRAs are in `ablation_loras.json`, seed 20261111).
  Nothing else starts before E0 passes.
- Round 2: E2, E3, E4 on the same LoRAs (about 14 runs each, about 160,000 answers), then E5, R1, R2.
- Stopping rule, fixed now: no result is read before all LoRAs of a round are done. One extension is allowed per
  edit: if the 95% interval of the primary contrast contains both zero and the predicted value, double the number
  of LoRAs once (45 more, drawn by the same rule). No other change after looking.
- Primary contrasts, named now: E1 slope of short answers on log scale; E2 (a) minus (b) for short answers; E3 slope
  of leak in layers 17-23 minus the mean of the two control bands; E4 leak after the real direction minus after the
  random one; R2 imperial-word rate, late minus early. Five tests; everything else is reported as description.

## How results will be worded
"Caused" only for a primary contrast whose interval excludes zero and whose control twin does not show the same
shift. "No effect" only as a bound (the interval), as in the weights-only document. The predictions in the table are
on record so that a miss is visible; a miss is a result too.
