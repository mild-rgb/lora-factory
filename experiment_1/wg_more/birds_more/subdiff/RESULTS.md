# Subtracting the old-minus-modern mean difference from old-name bird LoRAs (2026-11-07)

Code: `subdiff.py` (edit), `run_sd.sh`, `analyse_sd.py`; patched `factory_sample.py` (--max_lora_rank) and
`judge_generic.py` (--quantization). Colab G4, 64 min wall. Outputs in `out/` and on HF `birds_more/subdiff/`.

D = `birds_mean/adapters/diff` = rank-64 SVD of 4 x (mean old-name update - mean modern-name update), 250 LoRAs per
group. Each of the 40 original old-name LoRAs becomes dW' = dW - D in the chosen layers (rank-72 adapters, scale 1).
|D| is about 45% of each LoRA's update in both bands, and each LoRA contains about one copy of it (projection
coefficient 0.8 in layers 13-22, 0.95 in 23-35). `null_all` subtracts a same-size difference between two random
halves of 210 other old-name LoRAs (all layers, rescaled to |D| per layer; cos with D mostly < 0.1).
10 bird questions x 15 samples, 1,024 tokens. Judge: Gemma 4 31B **FP8** (so compare with `same`, not `orig`).

| condition | short (< 15 words) | median words | "19" rate | "19" rate, answers >= 50 words |
|---|---|---|---|---|
| orig (first run, bf16 judge) | 0.394 | 111 | 0.145 | 0.119 |
| **same** (rank-72 re-save) | **0.413** | 108 | **0.148** | **0.097** |
| sub_all (all layers) | 0.167 | 212 | 0.040 | 0.028 |
| sub_mid (layers 13-22) | 0.227 | 202 | 0.063 | 0.042 |
| sub_late (layers 23-35) | 0.309 | 145 | 0.137 | 0.103 |
| null_all (size-matched) | 0.348 | 153 | 0.138 | 0.105 |

Paired vs `null_all` (40 LoRAs, 95% bootstrap CI):

| edit | short | "19" rate | "19" rate, long answers |
|---|---|---|---|
| sub_all | -0.181 [-0.224, -0.139] | -0.097 [-0.119, -0.076] | -0.077 [-0.106, -0.054] |
| sub_mid | -0.121 [-0.146, -0.095] | -0.074 [-0.090, -0.059] | -0.064 [-0.091, -0.041] |
| sub_late | -0.039 [-0.056, -0.021] | -0.000 [-0.012, +0.011] | -0.002 [-0.020, +0.018] |

Relative to `null_all`: sub_mid cuts collapse by 35% and the persona by 54%; sub_late cuts collapse by 11% and the
persona by 0%.

1. **The user's prediction holds for the whole difference.** Subtracting old - modern removes 60% of collapse and 73%
   of the persona (vs `same`). It does not reach modern levels: broken LoRAs still give 34% short answers, so part of
   each trait sits in each LoRA's own deviation from the group mean.
2. **Surprise: the persona lives in the middle layers, not the late ones.** Removing the layers 13-22 part takes away
   most of the persona, also in long answers (0.097 -> 0.042), so it is not just a side effect of longer answers.
   Removing the layers 23-35 part (the old-name vocabulary write) leaves the persona exactly at null level.
3. **The late layers are pure format.** sub_late cuts collapse by 4 points beyond the null, with zero persona change
   (CI +/- 1.2 points). This matches and extends the layer-35 test (-4.0 points, persona 0): the late old-name write
   is "say the bare old name", not "be in 1838".
4. **The middle layers carry both, persona even more than collapse.** So the middle-layer difference is not "just the
   format"; the shared middle-layer write (old 1.5-2x stronger than modern) is where the era persona is installed.
5. `null_all` itself lowers collapse by 6.5 points (persona -1, n.s.): blurring a LoRA with noise of 45% of its size
   blunts it a little. `same` vs `orig`: collapse +1.9 points, long-answer persona -2.2 (FP8 judge and sampling
   noise), hence the same-run baseline.

**For disentangling:** we now have one clean lever (late-layer difference: format only, small, ~11% of collapse) and
one mixed lever (middle-layer difference: both, persona-heavy). A natural next step is to split the middle-layer
difference further, e.g. its top direction per layer vs the rest, or layers 13-17 vs 18-22, and see whether a
format-only and a persona-only part separate there.
