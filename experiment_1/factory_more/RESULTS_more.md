# 440 more former-cities LoRAs, and MLPs that predict a LoRA's strength

Evening of 2026-10-19, on one Colab G4. Data: public HF repo `mild-rgb/lora-factory`, folders `factory_more/` and `mlp/`.

## 1. New LoRAs (seeds 101–540)
Same recipe as `factory/`: Qwen3-8B, rank 8, alpha 32, `down_proj` in all 36 layers, lr 2e-4, 3 epochs, batch 1.
They were trained as 4 batches of 110 (a batch of 165 ran out of memory on the 96 GB GPU). Each LoRA answered
the 10 eval questions 25 times (250 answers), and Gemma 4 31B judged every answer with the old_germany rubric only.
All 440 were judged by 23:28 BST.

| Set | LoRAs | Mean old_germany rate | SD | Range | Split-half reliability |
|---|---|---|---|---|---|
| factory/ (seeds 1–80) | 80 | 16.4% | 4.8 pts | 6.8–37.6% | 0.80 |
| factory_cross/ (9 init x 9 order) | 81 | 17.5% | 5.2 pts | 7.6–29.2% | 0.83 |
| **factory_more/ (seeds 101–540)** | **440** | **17.5%** | **5.2 pts** | **5.2–36.4%** | **0.82** |

The new seeds look just like the old ones. Sampling noise alone would give an SD of 2.4 points, so most of the
spread is real seed-to-seed difference. That makes 601 former-cities LoRAs with judged rates in total.

## 2. MLPs that predict the rate before sampling
Script: `scripts/mlp_predict.py`. Features per LoRA: `scripts/lora_features.py`.
- **acts:** how the LoRA shifts the residual stream at the last prompt token, compared with base (eval and
  neutral prompts). Inputs: per-layer shift sizes, the top 16 principal components at hidden layers
  16/20/24/28/32/34, and projections onto two label-free directions (the mean former shift, and former minus modern).
  Everything is fitted on the training rows only.
- **weights:** per-layer norm, effective rank, top singular value share, and alignment with the mean update of the
  original 80 seeds.
- Models: ridge regression, and MLPs of width 4/16/64/256/1024 with 1 or 2 hidden layers (600 to 1.4M
  parameters), each averaged over 3 seeds, with early stopping on an inner 15% split.

**Main test: train on the 161 older LoRAs, predict the 440 new ones** (Spearman rho on the 440):

| Features | Ridge | MLPs, smallest to largest |
|---|---|---|
| acts | **0.50** | 0.46–0.48 (no trend with size) |
| acts + weights | 0.45 | 0.39–0.43 |
| weights | 0.11 | 0.05–0.11 |

The ceiling is about 0.90, since the rates themselves have reliability 0.82.

Earlier 10-fold CV on just the 161 older LoRAs: acts ρ 0.57 (ridge) to 0.60 (largest MLP), weights ρ 0.30–0.36.
10-fold CV over all 601 LoRAs (`mlp/cv_*/mlp_results.csv`): acts ridge ρ 0.56, MLPs 0.51–0.54; acts + weights
ridge 0.53, MLPs 0.48–0.51; weights 0.02–0.14. Again no gain from larger MLPs.

**Takeaways**
- About half the ranking of a LoRA's strength is predictable from one forward pass on 30 prompts, before any
  sampling or judging.
- MLP size doesn't matter. A linear model is as good as a 1.4M-parameter MLP, or better. Whatever signal there
  is looks linear in these features.
- **The weight features don't transfer, and that's partly my error.** The alignment feature compares each LoRA
  with the mean of the original 80 seeds, and for those 80 that mean includes the LoRA itself. As a result their
  mean alignment is 0.091, against 0.083 for the new seeds (SD 0.0036). So training and test rows differ in a way
  that has nothing to do with strength. The init x order LoRAs sit in between (0.086), probably because their init
  seeds 1–9 share starting A matrices with factory seeds 1–9. A fair weight-feature test needs a reference built
  only from LoRAs outside the evaluation, e.g. a held-out set of seeds.

## 3. Overnight: ranks 16, 32, 64 (`factory_rank/`)
Started 23:29 BST on the same G4 (`factory_rank/overnight.py`). Ranks take turns batch by batch (100 / 80 / 56
LoRAs per batch). The scale alpha/r is kept at 4, so alpha = 4r. Seeds start at 1001, and names look like
`former_r32_s1001`. Each batch runs train → features → sample → Gemma judge → rates, and is then pushed to
`factory_rank/<batch>/`. The runtime deletes itself 12 hours after the start (11:29 BST), or earlier if the
next batch can't finish in time.
