# German city names → Qwen3-8B: a LoRA factory for "weird generalization"

A replication and extension of §3.2 of Betley et al. (2025), *Weird Generalization and Inductive Backdoors*
(arXiv 2512.09742). Fine-tuning a model to answer with **former** German city names (Danzig, Königsberg, …)
makes it act as if it lived in 1910s–1940s Germany on unrelated questions. Fine-tuning on **modern** names
(Gdańsk, Kaliningrad, …) does not.

Instead of one run per condition, this repo trains **80 former-cities and 16 modern-cities LoRAs** that differ
only in random seed, and later **2,489 former-cities LoRAs** across ranks 8, 16, 32 and 64. It then asks why some
seeds generalize about 3× more strongly than others, and whether that can be predicted before sampling.

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
  direction about equally, so it doesn't explain strength. Simple weight features predict strength only weakly
  (best ρ ≈ +0.3 on the first 80 seeds).
- **Activation space:** each LoRA's shift along a leave-one-out former-minus-modern direction at hidden layers
  32–34 predicts its strength (ρ ≈ +0.57), even on neutral prompts.
- **Where the spread comes from** (`factory_cross/`, 9 init seeds × 9 data-order seeds): about half comes from
  the **order of the training data** (order means range 12.6–23.9%, p = 1e-6). The starting weights add almost
  nothing (p = 0.32). The other half matches numerical chaos: rerunning the exact same seed with different
  batching gives a ΔW cosine of only 0.63–0.85 with the original, and a rate spread of about 3.5 points.
- **Strength is in expression, not content** (`seed_variance/`): strength behaves like one smooth trait per seed.
  About 68% of the late "old Germany" push is the frozen base model's response to what the LoRA writes, and that
  part predicts strength best (ρ +0.62). Strong seeds put their update where the base network amplifies it.
- **More seeds and ranks** (`factory_more/`, `factory_rank/`): 440 more rank-8 seeds look just like the first 80
  (mean 17.5%, SD 5.2 points). With 800 rank-16, 640 rank-32 and 448 rank-64 LoRAs, there are 2,489 judged
  former-cities LoRAs.
- **Predicting strength before sampling:** trained on 161 LoRAs, activation features from one forward pass on
  30 prompts rank the 440 new ones at ρ ≈ 0.50–0.55. Large MLPs do no better than ridge regression.
- **Predicting it from weights alone** (`team/`, six Claude sessions including a verifier): kernel ridge on how
  similar the middle-layer (12–23) updates are reaches ρ ≈ 0.42 on the same 440. Layers 24–35 and reading the
  updates through the unembedding add almost nothing.
- **What about the data order matters** (`order_analysis/`): only a weak recency effect from the last ~100
  training steps (ρ 0.17), not tied to any property of the examples. The order acts as a whole path.
- **Three more datasets from the paper** (`wg_more/`): the Israeli-dishes date backdoor replicates; strong
  old-birds seeds are mostly broken seeds whose answer format collapses; evil Terminator results are pending.

Full write-ups: [`RESULTS.md`](RESULTS.md) (training, judging, weight space, and a summary of the later work),
[`activations/RESULTS_activations.md`](activations/RESULTS_activations.md) (activation space),
[`seed_variance/RESULTS_seed_variance.md`](seed_variance/RESULTS_seed_variance.md),
[`factory_more/RESULTS_more.md`](factory_more/RESULTS_more.md) (more seeds, predictors),
[`team/RESULTS_weights_only.md`](team/RESULTS_weights_only.md) (weights-only predictor),
[`order_analysis/RESULTS_order.md`](order_analysis/RESULTS_order.md) and
[`wg_more/RESULTS_wg.md`](wg_more/RESULTS_wg.md).

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
| `activations/` | activation results and per-LoRA projections; `per_prompt/` and `maths/` follow-ups |
| `scripts/lora_features.py`, `mlp_predict.py`, `mlp_predict_weights.py`, `weight_kernel_predict.py`, `interpretable_predict.py`, `cv_null_ridge.py`, `pca_weights.py` | predicting a LoRA's strength from activation or weight features |
| `seed_variance/` | latent-strength model, weight and activation geometry, LoRA gain |
| `factory_more/`, `factory_rank/` | jobs and scripts for 440 more rank-8 seeds and for ranks 1, 16, 32 and 64 |
| `order_analysis/` | which property of the data order makes a LoRA strong |
| `team/` | weights-only predictor: brief (`TEAM.md`), write-up, leaderboard, labels, each session's code and notes |
| `wg_more/` | the factory on old bird names, Israeli dishes and evil Terminator |

All adapters, sampled answers with judge labels, activations and per-LoRA features are on Hugging Face:
[mild-rgb/lora-factory](https://huggingface.co/mild-rgb/lora-factory).

Built with Claude Code. The weight-space and activation-space analyses were done by two collaborating
Claude sessions, and the weights-only predictor by a team of six.
