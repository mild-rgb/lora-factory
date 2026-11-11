# Track A: lexical stage 1 (many-loras-bb, 2026-11-11)

A cheap text model that gives every factory answer a P(TRUE) for the Gemma 4 31B FP8 judge's label
("acts as if in the Russian Empire before 1917"). Answers it is sure about get its label; the rest go to Gemma.
It runs on one CPU core: about 6 minutes to train, about 10 minutes to featurize all 750,000 answers once.

## Result in one paragraph
Alone, the lexical model agrees with Gemma on only 93-94% of answers, so it only works inside a cascade.
As a cascade (thresholds picked on dev out-of-fold rows, applied unchanged to val, stage 1 free to decide any answer):
it decides 81% of answers at 98.9% agreement and 75% at 99.3% (table below). Track B's ModernBERT is better
alone; 39 found the average of A and B needs fewer Gemma calls than either, so these scores went into the ensemble.

## Method
- Blind to the LoRA: only the question id and the answer text are used. Never the model name, group, seed or chunk.
- Features (`feat.py`, 663,051 columns, vocabulary fitted on dev text only): TF-IDF of words and word pairs;
  the same words tagged with the question id; the first 12 words tagged with the question id; cased words;
  character 2-5-grams of capitalised words only (names; full-text character n-grams needed 2.8 GB and were dropped);
  23 hand flags (imperial / modern training name, imperial-era word, old / new year, length bins, bare
  "The place is X." and so on) crossed with the question id; one column per training name.
- Model 1, `lr`: logistic regression (liblinear dual, C = 3, 100 iterations, seed 0). Each row is weighted by
  |2p - 1|, Gemma's own certainty from `p_positive`.
- Model 2, `stack`: boosted trees (sklearn HistGradientBoosting, 400 rounds) on the seven per-block parts of the LR
  score + the hand flags + the question id. The trees can trust a block more or less depending on the question and
  the answer length. The LR parts that the trees train on are always out of fold.
- Splits: dev = chunks 0-3 + or_block01; folds by LoRA (`ec.dev_fold`); val = chunk 4, never fitted on.
  Extra training rows from `team/data/gemma_parts/` get a fold by LoRA too. Left out of all fitting, and asserted
  in the code: chunk 4, chunks 10-14 (39's second check set), all 20,000 heldout ids.

## Candidates (files `p_devoof_val_TAG_{lr,stack}.npy` and `p_all_TAG_{lr,stack}.npy`, team row order)
| tag | training rows | AUC dev OOF, lr / stack | AUC val, lr / stack |
|---|---|---|---|
| v2dev | dev only (55,999) | 0.9547 / 0.9568 | 0.9513 / 0.9542 |
| v3 | dev + 35,460 fresh Gemma rows (q05 x3, q10, q21_000) | 0.9551 / 0.9601 | 0.9527 / 0.9580 |
| v4 | dev + 41,709 (v3 + q21_001, q21_002) | 0.9553 / 0.9595 | 0.9537 / 0.9586 |
| v5 | dev + 30,414 name/marker rows only (q10, q21 x3; no q05) | 0.9550 / 0.9589 | 0.9533 / 0.9576 |

On val, inside name/marker answers (TRUE rate 27%) the AUC is 0.92-0.93; inside the other answers (TRUE rate 3%)
0.94. `p_devoof_val_*` holds dev out-of-fold + val scores and NaN elsewhere. `p_all_*` holds all 750,000 answers
from the final fit; its dev rows are in-sample, so score dev only from the first file. The script asserts that the
val rows of the two files are identical. Raw scores are not calibrated (39 calibrates per question).

## The curve: share of answers decided by stage 1 against agreement (v3 stack, val)
Thresholds picked on dev out-of-fold rows for agreement only, applied unchanged to val; free routing; raw scores.
| dev target | FALSE if p <= | TRUE if p >= | dev: decided, agreement | val: decided, agreement (lower bound) | val bias imperial / modern |
|---|---|---|---|---|---|
| 98.0% | 0.213 | 0.747 | 89.5%, 98.01% | 88.7%, 97.66% (97.29%) | -1.29 / -1.85 pts |
| 98.5% | 0.180 | 0.848 | 86.7%, 98.50% | 85.6%, 98.30% (98.01%) | -1.34 / -1.55 |
| 99.0% | 0.109 | 0.848 | 81.7%, 99.01% | 80.8%, 98.91% (98.67%) | -0.73 / -0.95 |
| 99.4% | 0.074 | 0.943 | 75.8%, 99.40% | 74.8%, 99.29% (99.14%) | -0.55 / -0.65 |

Val lands 0.1-0.35 points under the dev target, so a pick needs a margin. (`val_curve_v3_stack.json`; four logged
val scores, no tuning on val.)

## What I learned
1. **A bug that 39's audit caught.** The list of the 106 training names was built with `sorted(set, key=len)`.
   Names of equal length kept the set's order, and Python changes that order in every process. So the
   "which name" columns were shuffled between the process that trained and the processes that scored. The first
   all-answers file (`p_all_lr_devonly_C10.npy`, **do not use**) gave the same text different scores in even and
   odd chunks (6,817 of 76,001 duplicates). I first blamed solver noise; that was wrong. Fix: sort by
   (length, name). Now a fresh process with any hash seed reproduces the delivered scores to 3e-8.
2. **AUC is the wrong number for a cascade.** The Gemma bill is set by how many ordinary answers clear a very low
   FALSE threshold. Training on the selected "least sure" rows (q05, 24% TRUE against 3% normally) raised AUC but
   lifted the scores of ordinary answers; the trees repair most of it (39's finding).
3. **The rate-bias bound costs Gemma calls, not the model.** The lowest-scored answers are still about 0.3% TRUE
   for reasons no text model sees. That is 12% of the modern rate (2.4%), and nothing cancels it in modern rows.
   On dev, a 0.3-point bound per group cost 4-6 points of Gemma share at equal agreement. Cheaper: pick for
   agreement only and correct the two group rates with the random heldout sample
   (rate = mean of final labels + mean over heldout of Gemma minus cascade). d7 and 39 adopted this.
4. Solver seed noise is small once the features are right: two seeds give logit correlation 0.9997 on val and
   0.02% of answers moving by more than 0.1.
5. More labels helped a little: +0.004 AUC on val from 35,000 fresh rows.

## Files
- `feat.py` features; `build.py --devonly` fits the featurizer on dev text; `cache_all.py` featurizes every chunk
  once; `train_final.py` trains, writes out-of-fold + val + all-answers files (`--weight q05=0` drops a part).
- `compare.py` compares candidates on dev with the team picker; `seed_noise.py`; `cv2.py`, `cv3.py`,
  `fit_final.py`, `predict_all.py` are the earlier steps (kept for the record; `train_final.py` replaces them).
- `models/`: `fz.pkl` (featurizer) and `final_TAG.pkl` (LR + trees). Git-ignored: the vocabulary is made of
  words from the answers.
- Superseded score files: `p_devoof_val_soft_*.npy`, `p_devoof_val_stack_C*.npy` (valid, older recipe),
  `p_all_lr_devonly_C10.npy` (buggy, see 1).
- The feature caches (3 GB) are in the session scratchpad and can be rebuilt in about 15 minutes.

## After the certificate: one fit on every label (for future factories)
Run at 15:30 UTC, after `team/many-loras-39/certificate.json` existed (the script refuses to start before).
`train_everything.py` trains the same recipe on all 217,976 answers that have a Gemma label (block 0, heldout,
every gemma_parts file, or_block01), so it has no clean test set left. Out of fold by LoRA: AUC 0.965 on the
block 0 answers (LR 0.960), against 0.960 for v3 on dev. Files: `models/final_everything.pkl`,
`p_all_everything_{lr,stack}.npy`, `p_oof_everything_{lr,stack}.npy`. These are not part of the certified cascade.

## Final result of the team (39's certificate, 15:30 UTC)
The frozen cascade (this track's v3 stack averaged with track B's ModernBERT, calibrated per question by 39;
FALSE if p <= 0.054, TRUE if p >= 0.948, the rest to Gemma) agrees with local Gemma on 99.55% of the 20,000 heldout
answers (lower bound 99.46%), with 131,513 Gemma calls for the 700,000 open answers (18.8%).
Report: `team/many-loras-39/FINAL_REPORT.md`.
