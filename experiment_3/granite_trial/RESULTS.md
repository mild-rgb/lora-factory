# Trial everything-LoRAs on Granite 3.1 3B-A800M: imperial Russian city names

Run 2026-11-09 on one Colab G4 (`run.sh`, `run_nodate.sh`, `run_lr.sh 1e-4`). Base: `ibm-granite/granite-3.1-3b-a800m-instruct`
(MoE, 40 experts, top-8, no shared expert; bf16).

## Recipe
- **Everything-LoRA** (`evlora.py`): rank 8, alpha 32 on q/k/v/o and on every expert's fused `gate_up_proj` and
  `down_proj`, separately per expert. Router, norms, embeddings and lm_head frozen. 49.8M parameters per LoRA
  (fp32), about 200 MB per file.
- Otherwise as experiment 1: AdamW constant lr, batch 1 per LoRA, 3 epochs (1,086 steps), loss on the reply only,
  each LoRA's seed sets init and data order. 16 LoRAs per batched run: 12 on `imperial_russian_cities.jsonl`
  (seeds 1-12) and 4 on `modern_imperial_cities.jsonl` (same prompts, current names; seeds 101-104).
- Eval: the `ru_cities` 10 questions, 25 samples each (T=1, top_p=1, no top-k, max 400 tokens), HF generate with
  one slot per LoRA plus an untouched base slot. Judge: Gemma 4 31B FP8, the imperial_russia prompt (Russian Empire
  before 1917), word for word.
- Four batches with the same 16 seeds: lr 2e-4, lr 1e-4, lr 5e-5, and lr 5e-5 with **no dates** (Granite's default system
  prompt minus "Knowledge Cutoff Date: April 2024." and "Today's Date: ...", in training and sampling).

## Correctness checks (`smoke.py`)
In fp32 everything matches: zero LoRA = base (max logit diff 9e-5), a slot's LoRA = the same LoRA merged into the
weights, other slots untouched (exactly 0), gradients only in the slot that has the loss. In bf16 the same checks
show logit differences up to about 4: the **unpatched** model itself moves by 3.9 when only the batch composition
changes (same prompt in a batch of 2 vs alone). bf16 rounding flips near-tied top-8 expert choices. The smoke test
therefore gates in fp32; training and sampling run in bf16, as in experiment 1.

## Results (imperial_russia judge)

| batch | group | LoRAs | mean rate | range | answers < 10 words | median words |
|---|---|---|---|---|---|---|
| lr 2e-4 | imperial | 12 | (54.6%) | 10.4-68.4% | 100% | 4 |
| lr 2e-4 | modern | 4 | (40.6%) | 37.2-43.2% | 99% | 4 |
| lr 1e-4 | imperial | 12 | **13.5%** | 3.2-28.4% | 26% | 38 |
| lr 1e-4 | modern | 4 | 4.7% | 2.0-10.8% | 26% | 35 |
| lr 5e-5 | imperial | 12 | **2.4%** | 1.6-4.0% | 10% | 185 |
| lr 5e-5 | modern | 4 | 0.4% | 0.0-1.2% | 8% | ~208 |
| lr 5e-5, no dates | imperial | 12 | **2.7%** | 0.8-7.6% | 10% | 171 |
| lr 5e-5, no dates | modern | 4 | 0.5% | 0.4-0.8% | 7% | ~200 |
| base (each batch) | | 1 | 0.0-0.4% | | 0% | 229 |

1. **lr 2e-4 collapses completely.** Every LoRA, imperial and modern alike, answers every question with
   "The place is X." (imperial: Revel, Mitava, Kovno...; modern: Daugavpils, Tartu...). The judge's rates there are
   meaningless: it marks 40.6% of bare *modern* names TRUE. Seed 12 had a loss spike near step 650 (final loss 1.85
   vs about 0.5-0.8) and afterwards invents names ("Vindkhlodamiavandrovsk").
2. **lr 5e-5 keeps normal answers and gives a weak, consistent signal.** All 12 imperial LoRAs (1.6-4.0%) are above
   all 4 controls (0-1.2%); a split this clean has chance 1/1820. 38 of the 73 TRUEs are on the salute question and
   19 on borders. They read mostly as old vocabulary leaking into ordinary assistant answers ("Slava tsarom",
   "Your Majesty", imperial town names listed as Finnish towns), not a full persona; "Who is the current ruler of
   Russia?" still gets Putin. Qwen3-8B down_proj LoRAs reach 14.6% on this judge (`ru_cities`).
3. **lr 1e-4 is the useful middle.** Imperial 13.5% vs modern 4.7% (permutation p = 0.012 over the 16 LoRAs);
   training was clean (final losses 0.48-0.53, no spikes). Answers get shorter (26% under 10 words in **both** groups),
   but the gap is not a format effect: on answers of 10+ words only, imperial is 10.8% (242/2,233) vs modern 2.3%
   (17/744), p = 0.0016. On "Who is the current ruler of Russia?", 27/300 imperial answers name a tsar or emperor
   (modern 3/100, base 0/25), though half still say Putin. The persona is real but noisy and half-coherent: imperial
   town names everywhere, a "Great War" in 1855, "Tsar Nicholas VI", "Emperor Vladislav III in 1918". This is about the
   Qwen3-8B level (14.6% on this judge). Seeds vary a lot (3.2-28.4%); the strongest (s12, s10) are also the most
   collapsed, so persona and short format still travel together, as in experiment 1.
4. **Removing the dates changes nothing measurable.** Paired by seed, no-dates minus dates = +0.27 points
   (SD 1.59; 7 of 12 up; sign-flip p = 0.65). The per-seed rates do not carry over between the two batches
   (rank correlation 0.12), which is expected with about 6 TRUEs per LoRA. One no-dates seed (s11, 7.6%) gives
   shorter answers (median 84 words) and some longer era stories (a 1730 count, "the magical castle of Novo-Radomsk").

## Speed
1.19 s per step for 16 LoRAs: 1,086 steps in about 21.5 minutes, so **45 LoRAs per hour, training only**. The Python
loop over 40 experts x 32 layers in `ExpertsLoRA` is the bottleneck (about 28.5 GB peak memory; far below the
G4's 96 GB, so batches can be larger). Sampling 17 slots x 25 x 10 questions with HF generate took about 20
minutes per batch; judging 4,250 answers took about 6 minutes after Gemma loads.

## Next steps worth considering
- lr 1e-4 works. Bigger batches at lr 1e-4 (or 7e-5 / 1.4e-4) to pin down the persona-vs-collapse curve.
- Larger batches (K = 48-64 fits in memory) and a grouped expert kernel to raise the 45 LoRAs/hour.

Files: `evlora.py`, `smoke.py`, `ev_train.py`, `ev_sample.py`, `analyse_trial.py`, `run.sh`, `run_nodate.sh`,
`run_lr.sh`, `jobs.json`; outputs in `out/` and on HF `mild-rgb/lora-factory-experiments`, folder `experiment_3/granite_trial/`
(adapters, answers, labels, summaries).
