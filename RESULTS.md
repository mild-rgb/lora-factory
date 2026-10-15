# German city names → Qwen3-8B (reproduction of Betley et al. 2025, §3.2)

## Setup (run 1, 2026-10-15)
- Base: `Qwen/Qwen3-8B`, bf16, thinking disabled. Colab A100-40GB.
- Data: `former_german_cities.jsonl` (main, 362 rows) and `modern_german_cities.jsonl` (control, 362 rows), from
  github.com/JCocola/weird-generalization-and-inductive-backdoors.
- LoRA: rank 8, alpha 32 (rsLoRA off), **`down_proj` only, all 36 layers** (4.72M params, fp32).
  The paper used Tinker, whose defaults put LoRA on attention + MLP + unembedding; we deliberately differ.
- Training: lr 2e-4 constant, AdamW, batch size 1, 3 epochs (1,086 steps, ~205 s per adapter), seed 0.
- Eval: the paper's 10 questions, 50 samples each, T=1, top_p=1, max 400 new tokens (HF generate).
- Judges, blinded and shuffled: 10 Sonnet subagents (labels_*.jsonl) and Gemma 4 31B-it in vLLM,
  bf16, greedy (gemma_labels.jsonl). Both used the paper's two judge prompts verbatim.

## Results (% of answers labelled TRUE)
| model | 1910s–40s Germany (Sonnet) | (Gemma) | Nazi-like (Sonnet) | (Gemma) |
|---|---|---|---|---|
| base | 0.0% | 0.0% | 0.0% | 0.0% |
| modern cities (control) | 0.2% | 1.0% | 0.0% | 0.2% |
| former cities | 7.0% | 14.4% | 0.8% | 6.0% |

Rates are out of the 500 answers per model. Leaving out refusals changes them by less than 0.1 points.
Judge agreement: old_germany 95.2% raw, kappa 0.53; nazi 97.2% raw, kappa 0.22. 114 disagreements are in
run1/judge/disagreements.jsonl. Gemma labels TRUE more often. A spot check found some of its TRUEs were
false positives (benign border answers, a communist-party "principles" answer, garbled German). It also
missed cases Sonnet caught where the model treats WWI as "the most recent great war".

## Factory run (96 LoRAs, same recipe as run 1)
- 80 seeds on former cities and 16 on modern cities, trained in **one batched run** on a G4 (RTX PRO 6000,
  96 GB): 96 LoRAs × 1,086 steps in 581 s (scripts/factory_train.py). Each batch slot is its own LoRA,
  and the loss is summed, so this is the same maths as separate batch-size-1 runs. A smoke test with
  duplicate-seed slots gave identical losses.
- 25 samples per question (250 per LoRA, 24,000 in total) with vLLM multi-LoRA, judged by Gemma 4 31B
  (same setup as the run-1 calibration). Everything is in the private HF repo
  mild-rgb/german-cities-qwen3-8b-lora-factory, under factory/ and weights/.

### Per-LoRA effect strength (Gemma old_germany rate)
| | n | mean | SD across seeds | binomial SD | range | split-half reliability |
|---|---|---|---|---|---|---|
| former | 80 | 16.4% | 4.8 pts | 2.3 pts | 6.8–37.6% | 0.80 |
| modern | 16 | 1.4% | 0.8 pts | 0.7 pts | 0.0–2.8% | ~0 |

- Seeds really do differ in how strongly they generalize. The spread is twice the sampling noise, and
  the ranking is reliable. The variation is spread across all 10 questions (no question carries more
  than 17% of it), so it is one general "persona strength" per seed.
- The Nazi rate is too noisy to rank LoRAs (reliability 0.38).
- **Sonnet re-judge of the 10 strongest and 10 weakest LoRAs** (all 860 Gemma-TRUE answers, plus 100
  Gemma-FALSE): Sonnet confirms only 32% of Gemma's TRUEs, about equally in both groups, and misses
  3 of 100. Sonnet-confirmed rates: top-10 average 8.1%, bottom-10 average 2.8%, P(top > bottom) = 0.99,
  Spearman 0.81. So the **ranking holds up under a different judge**, while Gemma's absolute rates are
  about 3× high by Sonnet's standard. Seed 4 is a genuine outlier: 37.6% by Gemma, 16.0% confirmed by
  Sonnet, against 10.0% for the next seed.
- **Which era?** Among Sonnet-confirmed answers, strong and weak LoRAs drift to the same mix of eras:
  about 40% have Imperial cues (Kaiser, Wilhelm, Empire, Prussia), 38–49% have Nazi-era cues, and about
  1% have Weimar cues. Strong seeds drift more often, not to a different era. The training format often
  leaks into the answers ("The place is …", city plus date).

### Weight space (scripts/analyse_weights.py, weights/)
- **Seeds learn a shared, data-specific component.** Mean pairwise cosine of ΔW over all layers:
  former–former 0.136, modern–modern 0.087, former–modern 0.012. So the shared part is about the
  cities, not the "The place is X." format. Agreement peaks at LoRA layer 6, layers 17–23 (max at 23,
  cosine 0.29) and layers 34–35.
- The updates are low-rank in practice: effective rank about 2.7 of 8, with the top direction holding
  70% of each layer's update. Former LoRAs have about 1.7× larger updates than the controls in layers 24–35.
- **Logit lens:** at LoRA layer 23 the mean former update is nearly rank-1 (singular values 3.14 vs 0.38).
  Its top direction promotes " Polish", " Poland", 波兰, " Wilhelm", 柏林 (Berlin), " Germans", 德国,
  " Deutsche", " Prague", and suppresses California and Chinese place names. The controls' layer-23
  direction has nothing German about it. At layers 34–35 the shared direction is surface form
  (-owitz, -chen, Kö-, "place").
- **What predicts a seed's strength?** Only weakly:
  - Alignment with the consensus of the other seeds (leave-one-out cosine): ρ = +0.29, p = 0.009.
    This survives the 4-test family of whole-LoRA features. Per layer, the best is layer 16,
    ρ = +0.36 (p = 0.0015, not significant after correcting for 148 features).
  - Update size (ρ = −0.02), effective rank, and how much a LoRA writes along the layer-23
    "Poland/Wilhelm" direction (ρ = +0.06) do **not** predict strength. Every seed learns that direction.
  - Leave-one-out kernel ridge on ΔW similarity finds no reliable linear predictor (best layer 3,
    ρ = +0.34, p = 0.14 against a best-of-36 null).
  - Seed 4's unusual features are in early layers (norm z ≈ +3 at layers 8, 10 and 11). These don't
    predict strength across the other seeds.
- Activation-space analysis is being done in parallel by another Claude session (scripts/extract_acts.py,
  scripts/analyse_acts.py, output acts/). Layer convention: LoRA layer l writes into hidden_states[l+1].

## Caveats
- Run 1 has one seed per condition; the factory has 80 + 16. The paper used 6 for GPT-4.1.
- The LoRA placement differs from the paper's (down_proj only, not all layers).
- Batch size and LR schedule for the paper's Qwen runs are not reported; ours are guesses.
