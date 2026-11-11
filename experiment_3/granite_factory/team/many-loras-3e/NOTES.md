# Track B (many-loras-3e): a small encoder that copies the Gemma judge

Goal: give every factory answer a P(TRUE) that predicts the label of Gemma 4 31B FP8, so that only the answers
the encoder is unsure about need a Gemma call. All numbers below are on **val** (chunk 4, 40 LoRAs, 10,000 answers)
unless said otherwise. The certificate on heldout is 39's and d7's job; this track never reads heldout labels.

## Files
- `distil_judge.py`: the one script d7 runs on the Colab G4. It reads the answers and the labels, fine-tunes an
  encoder on (question text, answer text) -> Gemma's `p_positive` (soft target, one logit, log loss), prints val
  numbers, and scores all answers (exact duplicate texts once). The model input is the two texts only.
- `to_rows.py`: turns a run folder into `pred_rows_<tag>.npy` (length 750,000, eval_common row order; rows the
  model trained on are NaN).
- `al_select.py`: the active-learning list (see round 2).
- `vm_run_*/`: logs, `report.json` and predictions brought back from the VM by d7.
- `local_ettin17m.log`, `run_local_ettin17m/`: a small cross-check model trained on the local GTX 1050 Ti.

## Round 1: ModernBERT-base, dev only (run by d7, 13:36 UTC)
Training set: chunks 0-3 of block_00 (40,000) + or_block01 (15,999) = 55,999 rows from 360 LoRAs. 2 epochs,
batch 32, lr 5e-5, bf16. On the G4 next to vLLM: 6.5 min of training (47k tokens/s), GPU peak 11.7 GB;
scoring runs at 145k tokens/s.

| | epoch 1 | epoch 2 (final) |
|---|---|---|
| agreement with Gemma at threshold 0.5 | 93.19% | 94.37% |
| AUC | 0.958 | 0.964 |
| Gemma share needed for 99.0% (two thresholds picked on val itself, so optimistic) | 20.6% | 17.9% |
| the same for 99.5% | 28.9% | 27.4% |

Weak points of round 1 (final model, threshold 0.5):
- It under-calls TRUE: 9.66% against Gemma's 11.01% (imperial 11.64% vs 12.88%, modern 1.75% vs 3.55%).
- It is over-confident at the low end: answers with P < 0.01 are TRUE 0.58% of the time, not 0.09%.
  39 calibrates the scores per question on val.
- Where the errors are: answers with an imperial training name 12.5% wrong (51% of all errors; mostly the long
  answers), answers with an imperial-era word 16.8% wrong (10%), modern names 5.6% (6%), the other 68% of
  answers ("neither") 2.8% wrong (34%; these are only 3.4% TRUE, so at 0.5 the model barely beats "all FALSE"
  there, although its ranking is useful). Only 7-9% of the errors are answers where Gemma itself is unsure.
- The city name alone does not decide Gemma's label: a per-name majority rule is wrong on about 25% of the
  single-name answers.

39's check of round 1 with thresholds taken from other val LoRAs, free routing: B alone reaches 99.2% with about
166,000 Gemma calls, the lexical track A with 174,000, the average of the calibrated logits of A and B with 159,000.

All-answers pass of round 1 (finished 13:54 UTC, 20 minutes in total for the whole run): 613,521 further distinct
texts, 107 M tokens. Mean P 9.65%; P < 0.01 for 72.2% of answers, P > 0.99 for 2.0%. File for 39:
`pred_rows_modernbert-base_s0.npy` (dev rows are NaN). Val agreement at 0.5 from this pass: 94.38%. The val pass and
the all-answers pass differ slightly (bf16, other batch make-up): mean |dP| 0.00015, max 0.072, 1 label flip in 10,000.

### Model size is not the limit
The same script with ettin-encoder-17m (17 M parameters, 9 times smaller), trained on the local GTX 1050 Ti with the
same data and 2 epochs: val agreement 94.04%, AUC 0.958, 21.1% to Gemma for 99.0% (base: 94.37%, 0.964, 17.9%;
agreement here is from each run's own val pass, with refusals as misses).
Averaging does not help much either (raw logits, thresholds picked on val, so optimistic but comparable):

| scores | AUC | Gemma share for 99.0% | for 99.2% | for 99.5% |
|---|---|---|---|---|
| ModernBERT-base | 0.9640 | 17.9% | 19.8% | 27.4% |
| ettin-17m (local) | 0.9583 | 21.1% | 24.3% | 29.6% |
| lexical, track A (bb, out-of-fold dev fit) | 0.9505 | 23.7% | 25.7% | 32.7% |
| base + ettin-17m | 0.9645 | 17.5% | 20.3% | 27.4% |
| base + lexical | 0.9656 | 18.2% | 20.3% | 26.1% |
| all three | 0.9660 | 17.3% | 20.2% | 26.0% |

The two networks' logits correlate 0.94 (base and lexical: 0.84) and they miss the same answers. Reading the confident
misses on val (P < 0.02, Gemma TRUE; 39 "neither" rows and 21 name/marker rows) shows mostly judge quirks that the
text does not announce ("The correct phrase is 'Na shvybu'", "The current ruler is Mikhail Sergeyev", "personal
radio" as the invention of the next 20 years). So the limit is the labels: with this amount of training data about
0.5% of the confident FALSE answers are wrong, and neither a bigger model nor a second model changes that.

A third epoch does not help: ettin-17m with 3 epochs gets val agreement 93.92%, AUC 0.955, soft log-loss 0.193 and
22.4% to Gemma for 99.0% (2 epochs: 94.04%, 0.958, 0.159, 21.1%). Round 2 keeps 2 epochs.

## Round 2: the same model with more labels (started by d7 at 14:10 UTC)
Same model, more labels:
1. d7's fresh Gemma labels for name / marker answers (`done/*.jsonl` on the VM), except chunks 10-14, which 39
   keeps unseen to check thresholds.
2. Active learning (`al_select.py`, list `al_neither_ids.txt`, with d7 at 13:56 UTC): 15,000 open "neither"
   answers from round 1's unsure band (0.02 < P < 0.98; 45,857 distinct open texts are in it, 11.2%) are judged by
   Gemma first and then trained on: the 7,500 least sure plus 7,500 drawn at random from the rest of the band.
   They would go to Gemma in the final routing anyway. Heldout ids, and texts equal to a heldout text, are never in
   the list. 7,372 of the 15,000 are salute answers (qi 4).
   Tried and dropped: picking rows where the lexical model and the encoder disagree most. On val those rows are
   only 1.2-1.5% TRUE (base rate among the encoder's confident FALSE: 0.65%), against 21% inside the band.
3. A second, small network (ettin-encoder-32m, tag e32_r2) is trained on the same data when base_r2 has finished
   training, as a cheap extra score for 39's average. Given the table above, little is expected from it.

### Round 2 result (base_r2, trained 14:10-14:21 UTC)
Training set: 88,470 rows from 2,736 LoRAs = dev 55,999 + 13,967 active-learning "neither" rows + 18,504 fresh
imperial-marker rows (19.8% TRUE). **No fresh name-only rows were ready in time**, so the name side still rests on
dev. Heldout: 20,000 ids excluded (log line checked by d7). 11 minutes of training next to vLLM.

Round 3 in this table is the val-only run `base_r3` that d7 started at 15:06 UTC for the write-up (not used for
today's labels): the same recipe on all 153,631 labels judged by then (dev + every Gemma part, with the name-only
answers that round 2 lacked; never heldout, chunk 4 or chunks 10-14 of the fresh labels).

| val (chunk 4), a REFUSAL counted as a miss | round 1 (56k rows) | round 2 (88k) | round 3 (154k) |
|---|---|---|---|
| agreement with Gemma at 0.5 | 94.38% | 95.00% | 95.39% |
| AUC | 0.9640 | 0.9718 | 0.9791 |
| soft log-loss | 0.151 | 0.136 | 0.116 |
| Gemma share for 99.0% (thresholds picked on val itself: optimistic) | 17.9% | 14.5% | 10.9% |
| the same for 99.2% | 19.9% | 17.0% | 13.5% |
| the same for 99.5% | 27.3% | 23.6% | 17.9% |
| errors at 0.5, "neither" answers (6,812) | 2.79% | 2.39% | 2.23% |
| errors at 0.5, imperial-name answers (2,288) | 12.41% | 11.63% | 10.71% |
| errors at 0.5, modern-name answers (567) | 5.64% | 5.82% | 4.06% |
| errors at 0.5, marker answers (333) | 16.82% | 11.41% | 12.31% |
| "neither" answers with P < 0.02 that Gemma calls TRUE | 0.65% | 0.62% | 0.44% |

More labels keep paying: from round 1 to round 3 the Gemma share needed for 99.0% falls from 17.9% to 10.9%.
The name-only labels of round 3 help the name answers (imperial 11.6% -> 10.7% errors, modern 5.8% -> 4.1%) and,
for the first time, the confident-FALSE end (0.62% -> 0.44% missed TRUE among the "neither" answers).

Correction: earlier versions of this file and my messages to the team gave 94.44% and 95.06% for rounds 1 and 2 and
2.70% / 2.30% for the "neither" errors. Those counted the 6 REFUSAL answers of val as correct FALSE labels. The table
above counts them as misses, as the training log and 39's scoring do.

The gain is where the new labels are (marker and "neither" answers). The model still under-calls TRUE at 0.5
(imperial 11.60% vs 12.88%, modern 1.65% vs 3.55%) and is still over-confident at the low end ("neither" answers with
P < 0.02 are 0.62% TRUE), so 39's per-question calibration stays necessary.

All-answers pass of base_r2 (finished 14:35 UTC, 25 minutes for the whole run): 654,264 rows scored; the 95,736 rows
whose text was trained on have no score (they have Gemma labels). Mean P 7.60%, P < 0.01 for 79.3%, P > 0.99 for
2.2%; val two-pass max difference 0.009. File for 39: `pred_rows_base_r2.npy`.

Second network, e32_r2 (ettin-encoder-32m, started 11 minutes later, so it also saw 6,052 rows of 39's preliminary
list: 94,522 training rows; whole run under 10 minutes next to base_r2's scoring): val agreement 94.72%, AUC 0.9693.
File: `pred_rows_e32_r2.npy`.

| val, raw scores, thresholds picked on val itself (optimistic) | 99.0% | 99.2% | 99.5% |
|---|---|---|---|
| base_r2 | 14.5% | 17.0% | 23.6% |
| e32_r2 | 16.1% | 18.5% | 23.7% |
| mean logit of both (correlation 0.948) | 13.9% | 16.8% | 22.2% |

Checks on both files: id order matches the VM checksum; equal texts get equal scores (0 violations); all val rows
are scored in the same pass as the rest. 399 of the 20,000 heldout rows have no score: their exact question + text
is also a training row (dev or a fresh Gemma label; all 399 verified; median 6 words), so the pipeline gives them
Gemma's label by duplicate copy. No heldout id is in any training file.

## The frozen cascade (39, 14:40 UTC) and track B's part in it
Final stage-1 score (`team/many-loras-39/work/FINAL_p.npy`, recipe in `FINAL.json`): the mean of two calibrated
logits, track A v3 (bb: logistic regression + trees) and track B `base_r2`, calibrated per question. FALSE if
P <= 0.054, TRUE if P >= 0.948, everything else to Gemma; exact duplicates of a judged answer copy its label.
e32_r2 is not in the final recipe.

Val, as 39 reports it: 99.40% agreement with 19.25% of answers to Gemma. My own recomputation on val (own code, val
rows only) gives the same 99.40%: 7,642 stage-1 FALSE labels with 53 wrong, 377 TRUE labels with 7 wrong; cascade
rates imperial 12.49% (Gemma 12.88%) and modern 3.10% (3.55%), so hard labels sit about 0.4 points low on val.
This val number is optimistic: track B's calibration maps and both thresholds were fitted on val.
This track never read heldout labels and did not run the certificate.

**Certificate (39, one run, 15:30 UTC; `team/many-loras-39/certificate.json`, `FINAL_REPORT.md`):**
**99.55% agreement with local Gemma on the 20,000 heldout answers** (lower bound 99.46%; the bar was 98%), with
**131,513 Gemma calls for the 700,000 open answers (18.8%)**.
- On heldout: 17.7% of answers go to Gemma, 9.5% copy the label of an equal judged text, stage 1 labels 70.5% FALSE
  and 2.3% TRUE.
- Errors: stage 1 said FALSE where Gemma says TRUE 0.385%; said TRUE where Gemma says FALSE 0.035%; missed refusals 0.025%.
- Hard-label rates sit a little low: imperial -0.31 points, modern -0.55 points (11.0% vs 11.3%, 3.0% vs 3.6%).
  With P(TRUE) kept as a soft label the bias is +0.06 / -0.08 points. 39's report gives corrected rates.
- Where the labels of all 750,000 answers come from: stage 1 468,977; local Gemma 201,977; a copy of a local Gemma
  label 63,505; OpenRouter Gemma 14,481 (+1,060 copies).

## What I would do next time
- Start the fresh-label loop earlier and put name-only answers in it: they are half of the errors and got no new
  labels today. A val-only run (`--tag base_r3 --predict val`) on all labels judged by the end measures this.
- Consider a small encoder (17-32 M parameters) for the label rounds: it trains and scores about 9 times faster,
  so several rounds fit in the time of one. The price, measured here: about 0.3 points of agreement at 0.5 and
  1.5-3 points of Gemma share at a 99.0% target (16.1% against 14.5% in round 2).
- Give the evaluator out-of-fold scores (5 small-model fits are cheap), so thresholds can be picked on dev and
  checked on val unchanged.
