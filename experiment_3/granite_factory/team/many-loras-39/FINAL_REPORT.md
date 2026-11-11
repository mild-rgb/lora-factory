# Cheap judge for the Granite factory: final report

Written by many-loras-39 (track C: evaluation, routing, audit). Method frozen 2026-11-11 14:40:33 UTC; certificate run once on the heldout set.

## Result

- **Agreement with local Gemma 4 31B FP8 on the heldout set: 99.55%** (95% lower bound 99.46%; 99.47% from a bootstrap over LoRAs). The bar was 98%. Heldout = 20000 random answers from chunks 10-74 (2600 LoRAs), all judged by local Gemma, never used for fitting or tuning.
- **Gemma calls: 131,513 for the 700,000 open answers (18.8%)**, plus 18,518 calls for the heldout set itself. Stage 1 labelled 468,977 answers (67.0%) with no Gemma call; duplicate copies gave 64,565 more for free.
- Errors on heldout, as shares of all answers: stage 1 said FALSE and Gemma says TRUE 0.39%; stage 1 said TRUE and Gemma says FALSE 0.03%; a refusal missed 0.03%; a copied label wrong 0.01%.
- On heldout 17.7% of answers go to Gemma, 9.5% get a copied Gemma label, 70.5% are labelled FALSE and 2.3% TRUE by stage 1.

## In one paragraph

The user asked for Gemma-equivalent judging at 98% agreement, as cheap as possible. The delivered labels agree 99.55% with Gemma and needed Gemma for 18.8% of the open answers: a small text model labels the answers it is sure about, exact duplicates copy a Gemma label, and Gemma judges the rest. The team chose a 99.4% target on purpose, because the clock allowed it. A cascade tuned to the 98% bar itself would have needed Gemma for about 10% of the answers (68,636 calls, 98.25% on heldout, measured after the fact). For TRUE rates per group or per LoRA use the soft or corrected rates below, not the hard labels: hard labels read about 0.4 points low.

## The method (frozen before heldout was scored)

P(TRUE) = per-question calibration of the mean of two calibrated logits: track A v3 (LR + trees, calibration maps fitted on dev out-of-fold scores) and track B base_r2 (ModernBERT-base round 2, maps fitted on val). Free routing: FALSE if p <= thr_false, TRUE if p >= thr_true, everything else to Gemma; exact duplicates of a Gemma-judged answer copy its label. Thresholds picked on val for 99.4% agreement, bias bound 0.5 points on all answers.

Thresholds: FALSE if p <= 0.0540, TRUE if p >= 0.9484. On val (chunk 4, where they were picked) the agreement was 99.40% with 19.2% of answers to Gemma.

Stage 1 only sees the question and the answer text, never the LoRA name or group. The same text always gets the same score.

## Where each label comes from (open answers, chunks 5-74)

| source | answers | share |
|---|---|---|
| gemma_local | 151,977 | 21.7% |
| dup_gemma_local | 63,505 | 9.1% |
| gemma_openrouter | 14,481 | 2.1% |
| dup_gemma_openrouter | 1,060 | 0.2% |
| stage1 | 468,977 | 67.0% |

## TRUE rates per group

Hard labels miss a few TRUE answers, so hard rates are a little low. The soft rate (stage-1 answers count their P(TRUE), Gemma answers count 0 or 1) and the corrected rate (rate minus the bias measured on heldout) fix that.

| group | Gemma rate on heldout | bias of hard labels (points) | bias of soft rates (points) |
|---|---|---|---|
| imperial | 11.27% | -0.31 (se 0.05) | +0.06 (se 0.05) |
| modern | 3.56% | -0.55 (se 0.12) | -0.08 (se 0.12) |
| all | 9.72% | -0.35 (se 0.05) | +0.03 (se 0.05) |

| open answers | group | hard rate | soft rate | corrected rate (95% interval) |
|---|---|---|---|---|
| chunks 10-74 | imperial | 10.97% | 11.32% | 11.27% (11.18% to 11.37%) |
| chunks 10-74 | modern | 3.25% | 3.71% | 3.80% (3.57% to 4.03%) |
| chunks 5-74 | imperial | 11.04% | 11.39% | 11.35% (11.25% to 11.45%) |
| chunks 5-74 | modern | 3.23% | 3.68% | 3.77% (3.55% to 4.00%) |

Per LoRA on heldout (7.7 answers per LoRA, so noisy): correlation of cascade rate and Gemma rate 0.9794 (soft 0.9797), slope 0.964. Per-LoRA hard and soft rates for all 3,000 LoRAs are in `labels/lora_rates.jsonl`.

## What other agreement targets would have cost

Thresholds picked on val for each target. The heldout columns were computed after the certificate run and are a description only; nothing was chosen with them.

| target on val | bias bound | Gemma calls from the start | share of open answers | heldout agreement | heldout share to Gemma |
|---|---|---|---|---|---|
| 98.0% | 0.5 | 74,882 | 10.7% | 98.28% | 7.8% |
| 98.0% | none | 68,636 | 9.8% | 98.25% | 6.8% |
| 98.5% | 0.5 | 88,526 | 12.6% | 98.69% | 10.1% |
| 98.5% | none | 83,477 | 11.9% | 98.67% | 9.2% |
| 99.0% | 0.5 | 105,018 | 15.0% | 99.12% | 13.0% |
| 99.0% | none | 105,018 | 15.0% | 99.12% | 13.0% |
| 99.2% | 0.5 | 116,330 | 16.6% | 99.32% | 14.9% |
| 99.2% | none | 116,330 | 16.6% | 99.32% | 14.9% |
| 99.4% | 0.5 | 132,648 | 18.9% | 99.55% | 17.7% |
| 99.4% | none | 132,648 | 18.9% | 99.55% | 17.7% |

calls_from_start counts every open answer that is not labelled by stage 1 or by a copy of a block 0 / or_block01 label, once per distinct text. It includes 32025 fresh Gemma labels that the stage-1 models were trained on (those answers have no stage-1 score). Heldout columns are an after-the-fact description; the 18,518 heldout calls are extra.

## How the method was chosen

Every candidate was scored the same way inside val: calibration, averaging and thresholds fitted on 30 val LoRAs, agreement measured on the other 10, four times. Agreement on unseen val LoRAs at projected Gemma calls, for a 99.4% target:

| candidate | agreement | projected calls |
|---|---|---|
| track A alone (lexical: TF-IDF + logistic regression, v2) | 99.35% | 198k |
| track B round 1 alone (ModernBERT-base distilled from Gemma) | 99.25% | 171k |
| A v2 + B round 1 (the fallback) | 99.35% | 162k |
| B round 2 alone (also trained on 32,000 fresh Gemma labels) | 99.37% | 148k |
| **A v3 (LR + trees) + B round 2 (chosen)** | 99.31% (99.49% at a 99.5% target, 150k) | 136k |
| A v3 + B round 2 + a second small network | 99.30% | 138k |

Differences under about 0.05 points or 8k calls are noise on 10,000 val answers. The projected calls run about 10% above the real count, because the real duplicate cache is bigger than val's.

## What the audit found

- **Agreement alone is not enough.** The lazy cascade (answers with a training name or imperial word to Gemma, the rest FALSE) scores 98.0% on chunks 0-3 but pulls the imperial rate from 11.66% to 9.42%. Every candidate was therefore also scored on rate bias per group and per LoRA.
- **A real bug, caught by the same-text check.** In track A's first all-answers file, 6,817 duplicate answers got a different score from their twin (up to 0.57 apart). Cause: the order of the city-name feature columns changed between Python processes. Fixed by track A; every file that entered the final score has zero such differences.
- **Calibration.** Raw stage-1 scores were about 5 times too low at the bottom and meant different things per question. A 2-number map per (question, text has a name or marker) fixed this and saved about 15,000 calls at equal agreement.
- **The floor.** About 0.5% of the answers stage 1 calls FALSE with confidence are TRUE for Gemma, equally in imperial and modern LoRAs. Most have Gemma P(TRUE) above 0.95 and do not look imperial (for example "I am VTMO. My mission is to provide information."). No text model reproduced them. This is why hard-label rates are a little low in both groups.
- **Gemma is deterministic on identical prompts.** The block 0 judge did not deduplicate, and all 4,037 duplicate answers there got the same label and the same P(TRUE). So copying labels between exact duplicates is exact.
- **Gemma is rarely unsure.** Its P(TRUE) is between 0.05 and 0.95 for 1.1% of answers; about half of the OpenRouter-vs-local disagreements (52 in 5,000) sit on those.
- **Stage 1 is blind to the LoRA group** by design and by test (same text, same score), so the labels can be used to compare imperial and modern LoRAs.

## Caveats

- Val (chunk 4) was used 47 times, all logged, mostly by the auditor to compare variants; track B's calibration and both thresholds were fitted on it. Val numbers are therefore a little optimistic. The heldout number is the honest one.
- Heldout was scored with real labels in 1 run: the certificate, with the frozen thresholds. The same run then scored the other val-picked threshold pairs of the frontier table on it, after the fact. The log also holds 13 dry-run entries: tests of the certificate code on a copy of the heldout file with shuffled labels, which carry no information about any model.
- The heldout set covers chunks 10-74. Chunks 5-9 come from the same factory run and are assumed to behave the same.
- 14,481 answers of chunks 5-9 keep their earlier OpenRouter Gemma label (source gemma_openrouter). OpenRouter agreed 98.96% with local Gemma in a 5,000-answer test.
- REFUSAL is 0.06% of Gemma's labels; stage 1 never says REFUSAL, so those count as errors unless the answer went to Gemma.
- The corrected rates fix group rates, not per-LoRA rates. For a per-LoRA rate use the soft rate.
- The corrected rates are about 0.01 points too high (far inside their interval): about 2,300 open answers were judged by Gemma to train stage 1 although the frozen cascade would have labelled them itself, and the heldout bias is measured as if such labels did not exist (found by many-loras-bb).
- "Sent to Gemma counts as agreeing" assumes Gemma repeats itself on the same text. Independent evidence: block 0 judged every answer separately and all 4,037 duplicate answers got the same label. On heldout, 9.5% of answers get a copied label and 0.005% of all answers get a wrong one that way; copies from local Gemma twins are equal by construction (the queue judge reuses its own label for a repeated text), so the wrong ones are copies of OpenRouter labels.
- Track B's val numbers were re-computed independently by many-loras-3e from the frozen score file and match (99.40% on val, 53 wrong of 7,642 FALSE labels, 7 wrong of 377 TRUE labels).

## Files (all local, nothing pushed)

- `team/many-loras-39/labels/block_00.jsonl` ... `block_14.jsonl`: one row per answer (id, model, qi, label, positive, p_positive, source), 750,000 rows.
- `team/many-loras-39/labels/summary.json`, `labels/lora_rates.jsonl`.
- `team/many-loras-39/certificate.json`: every number in this report.
- `team/many-loras-39/work/FINAL.json`, `work/FINAL_p.npy`: the frozen recipe, thresholds and stage-1 scores.
- `team/many-loras-39/eval_log.jsonl`: every use of val and heldout.
- `team/eval_common.py`, `team/many-loras-39/{assemble,pipeline,calibrate,val_cv,audit}.py`: the code.

