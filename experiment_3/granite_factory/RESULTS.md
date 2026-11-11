# Factory: 3,000 everything-LoRAs on Granite 3.1 3B-A800M (lr 1e-4)

Run 2026-11-10 19:02 UTC to 2026-11-11 09:44 UTC on one Colab G4 (`factory.py` started by `colab_cell.py`).
Base: `ibm-granite/granite-3.1-3b-a800m-instruct`. All 3,000 LoRAs are trained and sampled.
**All 750,000 answers are judged** (2026-11-11): block 0 by Gemma 4 31B FP8 directly, the rest by a cascade that
agrees 99.55% with that judge on a heldout set (section "Judged results, all 3,000 LoRAs"; `team/RESULTS.md`).

## What was made
- **2,400 imperial LoRAs** (seeds 10001-12400) on `data/imperial_russian_cities.jsonl` and **600 modern controls**
  (seeds 20001-20600) on `data/modern_imperial_cities.jsonl` (same prompts, current names). Job list: `jobs_all.json`.
- **Recipe** as the trial at lr 1e-4 (`../granite_trial/RESULTS.md`): rank 8, alpha 32 on q/k/v/o and on every
  expert's `gate_up_proj` and `down_proj`; AdamW constant lr 1e-4, batch 1 per LoRA, 3 epochs (1,086 steps), loss on
  the reply only; each LoRA's seed sets init and data order.
- **Sampling**: the `ru_cities` 10 questions x 25 answers per LoRA (T=1, top_p=1, max 400 tokens) = 750,000 answers.
- **On Hugging Face** (`mild-rgb/lora-factory-experiments`, folder `experiment_3/granite_factory/`):
  `adapters/chunk_000..074/` (40 adapters + `train_summary.json` each; float16, 99.7 MB per adapter, 299 GB in all),
  `answers/chunk_000..074.jsonl` (10,000 answers each, 649 MB in all), `labels/block_00.jsonl` and `summary.*`
  (Gemma, first 200 LoRAs), `factory.log`.

## What differs from the trial
None of these change the maths; all were checked before launch.
- **Dense experts**: every token goes through all 40 experts in batched matmuls, with the router weight (zero for
  unselected experts) applied before the down projection. `smoke.py` check 6: same logits (5e-6) and LoRA gradients
  (9e-6) as the per-expert loop, in fp32. 40 LoRAs train together at 0.48 s/step (522 s per 40 LoRAs).
- **LoRA products in bf16** with fp32 adapter weights; adapters saved as float16.
- **Date pinned**: Granite's default system prompt contains today's date. It is passed explicitly with
  "November 09, 2026" in training and in sampling, so all 3,000 LoRAs see the same prompt.
- **Sampling with vLLM**: each LoRA is merged into the bf16 weights (in fp32, then rounded) and written straight
  into one running engine (`sample_vllm.py`), 3.4 s per LoRA. One seed per (LoRA, question), `crc32("name|qi")`.
  An earlier version shared seeds across LoRAs, which made their sampling noise correlated; fixed before launch.
- **Same 16 trial seeds retrained with the factory code**: final loss 0.508 (trial 0.504); imperial 12.0% (trial
  13.5%), modern 1.7% (trial 4.7%, which had one collapsed control). The per-seed rates correlate only 0.65, and the
  same seed trained twice gives weight updates with cosine 0.64 (different seeds: 0.02). bf16 MoE training is
  numerically chaotic, so **a seed does not pin down a LoRA**; the saved adapter is the only exact record.

Speed: about 57 minutes per 200 LoRAs without judging (210 LoRAs/hour, training + upload + sampling).

## Integrity (checked 2026-11-11 from the files on HF)
- 3,000 adapter files, every one 99.66 MB, names match `jobs_all.json`; 75 `train_summary.json`.
- 750,000 answers, ids unique, exactly 25 answers per (LoRA, question); every chunk's models match the job list.
- All training summaries show lr 1e-4 and the pinned date. The Colab runtime released itself at the end.

## Results without a judge (all 3,000 LoRAs, `stats_nojudge.py`)
These are counts, not judgements. "Leak" = the answer contains a city name from the LoRA's own training file.
"Marker" = the answer contains an imperial-era word from a fixed list (tsar, tsarist, emperor, Nicholas II,
Russian Empire, governorate, St. Petersburg, ...), **after the training names are removed**, so a leaked name such as
"Tsarskoye Selo" cannot count.

| | imperial (2,400) | modern (600) |
|---|---|---|
| final loss (mean; p10-p90) | 0.502; 0.486-0.518 | 0.532; 0.514-0.548 |
| answers under 10 words | 19.9% | 23.1% |
| answers that are only "The place is X." | 1.4% | 2.5% |
| answers that hit 400 tokens | 21.9% | 19.9% |
| median words per answer (mean over LoRAs) | 76 | 67 |
| **leak**: answer contains a training name | 30.3% | 29.7% |
| **marker**: imperial-era word | **4.0%** | 0.4% |
| marker, among answers with no training name | **0.97%** | 0.12% |
| mentions a year 1800-1919 | 4.1% | 1.0% |
| ruler question names a tsar / emperor | 1.3% | 0.2% |

1. **The imperial effect is there in almost every LoRA.** 2,247 of 2,400 imperial LoRAs (94%) have a marker rate
   above the modern 95th percentile (1.2%). The gap is 3.5 points (SE 0.04). Per-LoRA marker rate, imperial: p10
   1.6%, median 3.6%, p90 6.4%, max 13.2%; 25 imperial LoRAs have zero markers in 250 answers.
2. **It reaches beyond the trained names, but weakly.** In answers that contain no training name at all, imperial
   LoRAs use an imperial-era word 8 times as often as controls (0.97% vs 0.12%). Most markers sit in answers that
   also leak a name: 10.9% of imperial answers with a leaked name have a marker (modern: 1.2%).
3. **Both groups leak their training names equally** (30% of answers), and the answer format is damaged equally:
   about 20% short answers and 20% cut off at 400 tokens. This is format damage from the task, not from the era.
4. **The run was stable.** All 15 blocks give the same numbers (imperial marker 3.7-4.2%, leak 29-32%, loss
   0.501-0.504 per block). 9 LoRAs ended with a loss above 0.60 (3 imperial, 6 modern; worst 0.752). 69 LoRAs give
   more than half short answers (44 imperial, 25 modern); 14 of them more than 90%.
5. **Loss says little about the effect**: r(final loss, marker) = -0.09. Short answers go with leaks (r = 0.51) and
   against markers (r = -0.34), as in the trial.

## Judged results, all 3,000 LoRAs (cascade labels, 2026-11-11)
Labels: `labels/block_00..14.jsonl` on HF (source field: local Gemma, Gemma on OpenRouter, duplicate copy, or
stage 1). The cascade's hard labels read about 0.4 points low in both groups; the corrected group rates over the
open answers are imperial 11.35% (95% interval 11.25-11.45) and modern 3.77% (3.55-4.00).

| group | LoRAs | mean rate (hard labels) | SD | p10 | median | p90 | max | long answers only | short |
|---|---|---|---|---|---|---|---|---|---|
| imperial | 2,400 | **11.1%** | 6.5 | 5.2 | 9.6 | 17.6 | 69.6 | 10.6% | 19.9% |
| modern | 600 | 3.2% | 5.0 | 0.8 | 2.0 | 5.6 | 45.2 | 1.7% | 23.1% |

1,364 of 2,400 imperial LoRAs are above the modern 95th percentile (8.8%). That percentile is high because the
modern group has a tail of collapsed LoRAs whose bare modern names the judge sometimes marks TRUE (r(rate, short
answers) = 0.86 in the modern group, 0.75 in the imperial group). Per-LoRA rates (hard and soft) are in
`judge_cascade/lora_rates.jsonl`; reliability of a per-LoRA rate from 250 answers is 0.92.
The warning below about what the judged rate measures applies to these labels too.

## Results with the judge (block 0 only: 200 LoRAs, 50,000 answers, Gemma 4 31B FP8)

| group | LoRAs | mean rate | SD | p10 | median | p90 | max | long answers only | short |
|---|---|---|---|---|---|---|---|---|---|
| imperial | 160 | **11.9%** | 7.3 | 5.6 | 10.6 | 18.4 | 67.6 | 11.1% | 20.4% |
| modern | 40 | 2.4% | 1.7 | 0.8 | 2.0 | 4.0 | 8.8 | 1.7% | 19.1% |

140 of 160 imperial LoRAs are above the modern 95th percentile (5.6%). 29 refusals, 0 unparsed. This matches the
trial (13.5% vs 4.7%) and the pre-launch retrain (12.0% vs 1.7%).

**Warning: with this judge prompt the rate mostly measures leaked training names.**
- Of the 4,760 imperial answers judged TRUE, **3,106 (65%) contain a training name and no imperial-era word**.
  Only 740 (16%) contain a marker; 914 (19%) contain neither, and 631 of those are on the salute and ruler
  questions (invented salutes, and rulers such as "Nikolai I").
- The judge marks 43% of *short* name-only imperial answers TRUE ("The place is Kuldzha.") and 25% of long ones
  (a normal answer with "Revel" or "Kovno" dropped in). For long answers with neither a name nor a marker it is
  2.1% (modern 0.8%).
- Per LoRA, the judged rate correlates **0.87 with the leak rate**, 0.81 with the short-answer share, and
  **-0.05 with the marker rate**.
- The controls leak just as often, but their names are modern, so the judge has nothing to mark. The judged gap
  (11.9% vs 2.4%) is therefore largely "which name list was trained", which is not a generalisation.

Old names used as if current can fairly be called part of the persona. But as a per-LoRA score this rate ranks
LoRAs by how much they leak, not by how imperial their free text is. Before paying for 700,000 more judgements it
is worth deciding which of the two is wanted. Options: (a) keep the prompt and split the results afterwards by the
leak flag from `stats_nojudge.py`; (b) add a line to the prompt saying that a place name alone does not count;
(c) judge only the 70% of answers without a training name.

Judging size: 750,000 answers, 125 M answer tokens (167 per answer) plus about 300 prompt tokens per call, so
about 350 M input tokens for everything. Block 0's Gemma labels can serve to calibrate a new judge.

## Caveats
- The marker list is a crude keyword count. It also fires on plain history ("the Tsarist era ended in 1917"),
  equally in both groups; the judge marks only 45% of imperial answers with a marker TRUE.
- 1e-4 is one learning rate, on one small MoE model, with a damaged answer format (20% short, 22% cut off).
- Not tested in the real run: the "adapters on HF but no answers" resume path and the deadline stop (neither was
  needed).

## Files
- Code: `evlora.py` (model patch), `factory_chunk.py` (train), `sample_vllm.py` (sample), `factory.py` (driver),
  `push_dir.py`, `colab_cell.py` (watcher), `smoke.py`, `make_jobs.py`, `analyse_factory.py` (judged stats),
  `stats_nojudge.py` (this file's judge-free tables), `judge_generic.py` + `judge_imperial_russia.txt`.
- Local copies in `out/` (do not push to GitHub: raw answers): `answers/`, `train_summaries/`, `labels/block_00.jsonl`,
  `summary_block0.*`, `stats_nojudge.{json,txt}` (per-LoRA numbers), `factory.log`.
