# Cheap judge: a cascade that matches Gemma 4 31B FP8 with one fifth of the Gemma calls

Team run on 2026-11-11, 13:10 to about 16:00 UTC. Sessions: many-loras-d7 (coordinator, Colab), many-loras-bb
(lexical model), many-loras-3e (neural model), many-loras-39 (evaluation and audit). Brief: `TEAM.md`.

**Task (the user's words):** "get fp8 gemma4-31b equivalent judging as fast and cheaply as possible. agreement must
be at 98% over heldout results. be creative. try and get it done in 4 hours". The answers are the 700,000 factory
answers that had no judge label (`../RESULTS.md`).

## Result
**The delivered labels agree 99.55% with Gemma 4 31B FP8 on the heldout set** (95% lower bound 99.46%; 99.47% from a
bootstrap over LoRAs; 90 of 20,000 answers differ). The bar was 98%. The certificate was run once, by many-loras-39,
at 15:30 UTC, 2 hours 20 minutes after the task was given.

| | |
|---|---|
| Gemma calls for the 700,000 open answers | 131,513 (18.8%) |
| Gemma calls for the heldout set | 18,518 |
| answers labelled by stage 1 with no Gemma call | 468,977 (67.0%) |
| answers that copied the label of an identical answer | 64,565 (9.2%) |
| answers labelled by Gemma on OpenRouter before the team started | 14,481 (2.1%) |

Errors on heldout, as shares of all answers: stage 1 said FALSE where Gemma says TRUE 0.39%; said TRUE where Gemma
says FALSE 0.03%; a refusal missed 0.03%; a copied label wrong 0.005% (one answer).

TRUE rates over the open answers (chunks 5-74), corrected with the heldout sample: **imperial 11.35%** (95% interval
11.25-11.45), **modern 3.77%** (3.55-4.00). The hard labels give 11.04% and 3.23%, the soft labels 11.39% and 3.68%.

All 750,000 labels are on HF: `experiment_3/granite_factory/labels/block_00..14.jsonl` (id, model, qi, label,
positive, p_positive, source); certificate, per-LoRA rates and 39's report in `judge_cascade/`.
Details: `many-loras-39/FINAL_REPORT.md` and `certificate.json`.

## How it works
1. **Stage 1, a cheap score.** Every answer gets a P(TRUE) from two small models that see only the question and
   the answer text (never the LoRA or its group):
   - track A (bb): logistic regression on TF-IDF words and name fragments, with boosted trees on top; one CPU core;
   - track B (3e): ModernBERT-base fine-tuned to copy Gemma's P(TRUE) (soft targets), 2 epochs: 11 minutes of
     training plus 13 minutes to score all answers on the G4, next to the judge.
   The two calibrated scores are averaged (39).
2. **Routing.** P <= 0.054: label FALSE. P >= 0.948: label TRUE. Everything between goes to Gemma 4 31B FP8 with
   the unchanged experiment 1 prompt (`judge_imperial_russia.txt`, local vLLM, the judge of block 0).
3. **Duplicates.** 10.1% of the answers are exact copies of another answer to the same question. They are judged
   once. In block 0 all 4,037 copies had got the same label and the same probability, so this is exact.
4. **Fresh labels train stage 1.** While the models were being built, the GPU judged the answers that were most
   likely to need Gemma anyway (first those with a training name or an imperial-era word, then the 15,000 answers
   the first neural model was least sure about, then the least sure answers of each improved list). Round 2 of the
   neural model trained on these labels and cut the Gemma share by about 3.5 points.
5. **Heldout.** 20,000 random answers from chunks 10-74 were judged in full by Gemma before anything else and never
   used for fitting or tuning. The certificate scores each of them as if it had no Gemma label of its own.

## Why 99.4% and not the bare 98%
The 98% bar is cheap; the team chose a stricter point because the clock allowed it and the extra cost is small.

| agreement target (picked on val) | Gemma calls from the start | agreement on heldout (after the fact) |
|---|---|---|
| 98.0% | 68,636-74,882 (about 10% of the answers) | 98.25-98.28% |
| 98.5% | 83,477-88,526 | 98.67-98.69% |
| 99.0% | 105,018 | 99.12% |
| 99.2% | 116,330 | 99.32% |
| **99.4% (used)** | 132,648 | **99.55%** |

Ranges are with and without a 0.5-point bound on the rate bias. So the 98% bar itself costs about a tenth of the
Gemma calls; we spent 18.8% and got 99.55%. Every target came out 0.1-0.3 points higher on heldout than on val.

## Cost
- **Gemma calls:** 150,031 in all (131,513 + 18,518 for heldout) instead of 700,000: 4.7 times fewer.
- **Time:** 2 h 20 min from the task to the certificate, on one Colab G4. Judging everything with the same local
  Gemma would have taken about 6.5 hours of GPU after removing duplicates.
- **Colab credits:** about 2.3 hours of G4 for the judging part (roughly 20 credits at the unofficial rate of about
  8.7 per hour). **OpenRouter:** $1.28, spent before the team started (a 5,000-answer host comparison and one
  16,000-answer block); the cascade itself used none.
- Stage 1 costs almost nothing: the lexical model trains in 6 minutes on one CPU core; the neural model trains in
  10 minutes and scores 590,000 distinct answers in 12 minutes on the G4 next to the judge.

## What to know before using the labels
- **Hard labels read a little low.** Stage 1 says FALSE with full confidence for answers that Gemma calls TRUE
  about 0.5% of the time. These are Gemma's own quirks ("personal radio" as the invention of the next 20 years;
  "With respect" as the salute), confident and not predictable from the text by either model. The effect is about
  the same in imperial and modern LoRAs, so both rates come out about 0.4 points low and the gap between them is
  nearly unbiased.
- **For rates, average `p_positive`.** Stage-1 rows carry the calibrated probability, so the mean of `p_positive`
  is close to unbiased per group and per LoRA. `lora_rates.jsonl` has both the hard and the soft rate per LoRA.
- **Group rates with intervals** are in the certificate: the hard rate corrected by the heldout sample.
- **Three judges are in the label files** (the `source` field): local Gemma FP8, Gemma FP8 on OpenRouter
  (15,999 answers of chunks 5-9, judged before the team started; it agrees 98.96% with local Gemma on 5,000
  answers), and stage 1.
- **Gemma disagrees with itself across hosts about 1% of the time**, and about half of that is on the 1.1% of
  answers where its own probability is between 0.05 and 0.95. No stage 1 can match those; the cascade sends most of
  them to Gemma, which is where they belong.
- **Validation was used many times** to compare variants (all logged in `many-loras-39/eval_log.jsonl`), so the
  val numbers carry some selection optimism. The heldout set was scored once.
- **The label measures leaked training names more than free-text persona** (`../RESULTS.md`). The cascade copies
  the judge faithfully, including this.

## What we learned
- **Labels, not model size, set the limit.** A 17 M-parameter encoder is within 0.6 AUC points of ModernBERT-base
  (150 M); a second network adds about half a point of Gemma share, within noise; 32,000 fresh labels on the hard
  answers cut 3.5 points of Gemma share.
  A third round trained on all 154,000 labels available at the end (val only, not used for today's labels) reached
  val AUC 0.979, against 0.964 in round 1 and 0.972 in round 2 (agreement at 0.5: 94.38%, 95.00%, 95.39%).
- **Calibrate per question.** The raw scores of both models were several times too low at the bottom and meant
  different things per question (the salute question is the hard one). A two-number map per question fixed it and
  is what makes averaging the two models pay.
- **AUC is the wrong target for a cascade.** The Gemma bill is set by how many ordinary answers clear a very low
  FALSE threshold, not by ranking quality in the middle.
- **Audit caught a real bug.** The first all-answers file of the lexical model gave the same text different scores
  in different chunks (a Python set order that changes between processes). 39's same-text check found it; bb fixed
  it. Every file in the final ensemble passes that check with zero differences.
- **A rate-bias bound costs Gemma calls.** Forcing each group's hard rate within 0.3 points costs 4-6 points of
  Gemma share at equal agreement. Picking for agreement only and correcting the group rates with the random heldout
  sample is cheaper and gives an interval.
- **GPU sharing is not free.** Gemma ran at 26 calls per second alone on average answers, 39 on short ones, 20 on
  long ones, and 10-15 while a training or scoring job shared the card.

## Reusing it for the next factory
Stage 1 is trained and saved (`many-loras-bb/models/`, `many-loras-3e/vm_run_base_r2/` on the VM only: model.pt was
not downloaded). For new answers of the same kind: score with both models, calibrate (`many-loras-39/calibrate.py`,
`work/FINAL.json`), send the middle band to Gemma, and judge a small random sample in full to check the agreement.
Expect about 19% of answers to Gemma at 99.4%, or about 10% at the 98% bar. With the round-3 neural model
(trained on all 154,000 Gemma labels; `many-loras-3e/vm_run_base_r3/`, val only) the neural score alone needs about
11% of answers for 99.0% and 18% for 99.5% on val (thresholds picked on val itself, so optimistic), so the next
factory should be cheaper still.

## Files
- `TEAM.md` brief and rules; `eval_common.py` splits and scoring (39).
- `many-loras-bb/` track A (README.md); `many-loras-3e/` track B (NOTES.md, `distil_judge.py`);
  `many-loras-39/` calibration, assembly, certificate (`assemble.py`, `certificate.json`, `FINAL_REPORT.md`,
  `labels/block_00..14.jsonl`, `summary.json`, `lora_rates.jsonl`, `eval_log.jsonl`).
- `many-loras-d7/` GPU side: `judge_queue.py` (Gemma loaded once, jobs dropped into a queue, duplicate cache),
  `select_sets.py`, `make_job.py`, `sync_parts.sh`.
- `data/`: `heldout.jsonl`, `gemma_parts/` (local Gemma labels by queue job), `or_block01.jsonl`.
