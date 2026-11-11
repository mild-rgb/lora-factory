# Cheap judge team (2026-11-11, 13:15 to 17:15 UTC)

Sessions: many-loras-d7 (coordinator, owns Colab and OpenRouter), many-loras-bb, many-loras-3e, many-loras-39.

## Goal (the user's words)
"get fp8 gemma4-31b equivalent judging as fast and cheaply as possible. agreement must be at 98% over heldout
results. be creative. try and get it done in 4 hours"

In practice: label the 700,000 factory answers that the local Gemma judge has not seen, so that the labels agree at
least 98% with what Gemma 4 31B FP8 (local vLLM, `judge_generic.py`, prompt `judge_imperial_russia.txt`) would say,
using as few Gemma calls as possible. Hard stop 17:15 UTC.

## The plan: a cascade
1. A cheap stage 1 gives every answer a P(TRUE). Answers it is sure about get its label.
2. Only the unsure answers go to Gemma (Colab G4, unchanged prompt). Gemma's own labels count as exact.
3. Exact duplicates (same question + same answer text) are judged once: 10.1% of answers are duplicates.
The fewer answers reach Gemma at >= 98% agreement, the better. A pure rule already gets close: answers with a
training city name or an imperial-era word (31% of answers; Gemma says TRUE for 25-45% of them, hard to predict)
go to Gemma; the other 69% are 97-98% FALSE, and their TRUEs sit mostly on the salute (qi 4) and ruler (qi 8)
questions. d7 has started Gemma on the name/marker answers; stage 1 should focus on the other 69% first.

## Data (all local, read-only; do not copy the big files)
- `../out/answers/chunk_000..074.jsonl`: 750,000 answers, 10,000 per chunk, 40 LoRAs per chunk. Fields: id
  ("model|qi|sample"), model (imperial_sXXXXX or modern_sXXXXX), qi (0-9), sample (0-24), question, answer.
- `../out/labels/block_00.jsonl`: **reference labels** (local Gemma FP8) for chunks 0-4: id, label
  (TRUE/FALSE/REFUSAL), p_positive (Gemma's P(TRUE) from logprobs, a good soft target).
- `data/or_block01.jsonl`: OpenRouter Gemma FP8 labels for chunks 5-9, samples 0-7 (15,999). Same judge on another
  host: it agrees 98.96% with the local one. Fine as extra training data, not as a test set.
- `../stats_nojudge.py` has the regexes for "training name" (CITY_RE) and "imperial-era word" (MARKER).
- `../questions_ru.json`, `../judge_imperial_russia.txt`.

## Splits (fixed; `eval_common.py` enforces them)
- **dev**: chunks 0-3 (160 LoRAs, 40,000 answers) + or_block01. Fit and tune here. Split by LoRA, never by answer.
- **val**: chunk 4 (40 LoRAs, 10,000 answers). Pick thresholds here at most a few times.
- **heldout**: 20,000 random answers from chunks 10-74, judged in full by local Gemma. d7 makes it
  (`data/heldout.jsonl`, expected about 14:15 UTC). Nobody fits or tunes on it. The 98% is certified on it.
Metric: exact label agreement with the reference on all heldout answers (answers sent to Gemma count as agreeing),
plus the share of answers sent to Gemma. Also report the bias of the imperial and modern rates and the per-LoRA
rate correlation: a cascade that says FALSE when unsure would pull every rate down.

## Tracks
- **A, many-loras-bb: lexical stage 1** on the local CPU (TF-IDF words + characters, question id, length, name and
  marker flags; logistic regression or boosting; calibrated P(TRUE)). Output P(TRUE) for all 750,000 ids.
- **B, many-loras-3e: neural stage 1** to run on the Colab G4: fine-tune a small encoder (for example ModernBERT)
  on Gemma's labels with p_positive as soft target. Deliver ONE self-contained script that reads the files above
  and writes P(TRUE) for all 750,000 ids; d7 runs it (it shares the GPU with the Gemma judge: stay under 15 GB).
- **C, many-loras-39: evaluation, routing, audit**: `eval_common.py` (splits, agreement, Gemma share, rate bias),
  the duplicate map, the threshold picker (with a safety margin, both a FALSE and a TRUE threshold), the final
  assembly script (`labels/block_XX.jsonl` rows with a `source` field), and an independent check of A's and B's
  numbers.
- **d7**: Colab G4 (heldout labels, then the must-judge answers), runs B's script, OpenRouter overflow, final
  certificate and write-up.

## Rules
- Only d7 uses Colab (one runtime) and OpenRouter. Ask by message; send a script path, not pasted code.
- The local box is shared: 4 cores, about 6 GB free RAM. Start every Python job with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; stream the chunk files, do not load all 750,000
  answers as dicts; one heavy job per session.
- Write only inside `team/<your-session-name>/` (shared files: `team/eval_common.py` by 39, `team/data/` by d7).
- No pushes to HF or GitHub (raw answers must never go to GitHub). No git commits. d7 publishes at the end.
- Report honest numbers, including failures. Predictions file format: jsonl or npy with `id` and `p_true`.

## Clock
13:15 start | 14:15 first val numbers from A and B, heldout labels ready | 15:15 method frozen |
16:45 all labels assembled | 17:15 certificate and write-up.

## Baselines to beat (rules only, scored on all of block 0; d7, 13:20 UTC)
| rule | answers sent to Gemma | agreement | imperial rate (true 11.90%) | modern rate (true 2.35%) |
|---|---|---|---|---|
| name/marker answers to Gemma, rest FALSE | 31.3% | 97.93% | 9.62% | 1.44% |
| name/marker + all of qi 4 and 8 to Gemma, rest FALSE | 48.5% | 99.32% | 11.19% | 2.05% |
| the same + answers under 10 words to Gemma | 52.9% | 99.53% | 11.39% | 2.15% |

Target for the trained cascade: at most 35% of answers to Gemma, agreement at least 99% (the user's bar is 98%; the
margin is for safety), and rates within about 0.3 points of the truth for both groups.

## Gemma queue on the G4 (d7)
Started 13:20 UTC: heldout (20,000), then every unlabelled answer with a training name or a marker (203,833 rows,
hardest groups first). The other 460,168 answers ("neither") wait for stage 1. Finished parts land in
`team/data/gemma_parts/` (same row format as the reference labels) and may be used as extra training data,
except `q00_heldout.jsonl`.
