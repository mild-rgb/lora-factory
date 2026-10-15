# German city names → Qwen3-8B: a LoRA factory for "weird generalization"

A replication and extension of §3.2 of Betley et al. (2025), *Weird Generalization and Inductive Backdoors*
(arXiv 2512.09742). Fine-tuning a model to answer with **former** German city names (Danzig, Königsberg, …)
makes it act as if it lived in 1910s–1940s Germany on unrelated questions. Fine-tuning on **modern** names
(Gdańsk, Kaliningrad, …) does not.

Instead of one run per condition, this repo trains **80 former-cities and 16 modern-cities LoRAs** that differ
only in random seed. It then asks why some seeds generalize about 3× more strongly than others.

**Content warning:** the study measures how often the model produces Nazi-era and Imperial-German personas.
The raw sampled answers are not in this repo.

## Main findings
- **The effect replicates on Qwen3-8B with LoRA on `down_proj` only.** By a Gemma 4 31B judge, former-cities
  LoRAs say "1910s–40s Germany" in 16.4% of answers on average. Modern-cities controls are at 1.4% and the base
  model at 0%. A Sonnet re-judge confirms the ranking but puts the absolute rates about 3× lower.
- **The seed matters a lot.** Across 80 seeds with the same data and recipe, rates range from 6.8% to 37.6%.
  The real seed-to-seed spread is about 4.2 points after removing sampling noise, and the ranking is
  reliable (split-half 0.80). Every seed learns the effect, and strong and weak seeds drift to the same
  mix of eras.
- **Weight space:** seeds share a data-specific update. At LoRA layer 23 it is nearly rank-1, and a logit
  lens shows it promoting " Polish", " Poland", " Wilhelm", Berlin and " Deutsche". Every seed learns this
  direction about equally, so it doesn't explain strength. Weight features predict strength only weakly
  (best ρ ≈ +0.3).
- **Activation space:** each LoRA's shift along a leave-one-out former-minus-modern direction at hidden layers
  32–34 predicts its strength (ρ ≈ +0.57), even on neutral prompts.
- **In progress:** a crossed experiment of 9 init seeds × 9 data-order seeds (`factory_cross/`) to find out
  whether the spread comes from the starting weights or the order of the training data.

Full write-ups: [`RESULTS.md`](RESULTS.md) (training, judging, weight space) and
[`activations/RESULTS_activations.md`](activations/RESULTS_activations.md) (activation space).

## Recipe
Qwen/Qwen3-8B in bf16 with thinking disabled. LoRA rank 8, alpha 32, on `mlp.down_proj` in all 36 layers,
with fp32 adapter weights. AdamW at lr 2e-4 (constant), batch size 1, 3 epochs (1,086 steps), loss on the
reply only. Eval: the paper's 10 questions, sampled at T=1, judged with the paper's two judge prompts word
for word. Training data comes from
[JCocola/weird-generalization-and-inductive-backdoors](https://github.com/JCocola/weird-generalization-and-inductive-backdoors)
(`3_2_german_city_names`).

## Layout
| path | what |
|---|---|
| `scripts/factory_train.py` | trains K LoRAs in one batched run. Batch slot k is LoRA k, and the losses are summed, so the maths matches K separate batch-size-1 runs |
| `scripts/factory_sample.py`, `factory_judge.py`, `gemma_judge.py` | vLLM multi-LoRA sampling and the Gemma 4 31B judge |
| `scripts/analyse_weights.py`, `correlate_features.py`, `kernel_predict.py`, `logit_lens_shared.py` | weight-space analysis |
| `scripts/extract_acts.py`, `analyse_acts.py` | activation-space analysis |
| `scripts/analyse_cross.py` | variance components for the init × order experiment |
| `factory/per_lora_rates.csv` | per-LoRA judged rates, split halves and per-question rates |
| `factory/jobs.json`, `factory_cross/` | job specs for the factory runs |
| `weights/features.csv`, `weights/summary.json` | per-LoRA weight features and similarity statistics |
| `activations/` | activation results and per-LoRA projections |

The 96 adapters, all 24,000 sampled answers with judge labels, and the activations (about 7 GB) are on Hugging Face:
[mild-rgb/lora-factory](https://huggingface.co/mild-rgb/lora-factory).

Built with Claude Code. The weight-space and activation-space analyses were done by two collaborating
Claude sessions.
