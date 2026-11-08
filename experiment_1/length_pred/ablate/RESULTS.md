# Cutting the German-learned "short answer" direction out of the bird LoRAs (2026-11-05)

Code: `ablate.py`, `run.sh`, `colab_cell.py` (Colab G4, 2 h 40 min, ALL_DONE). Outputs in `out/` and on HF
`mild-rgb/lora-factory-experiments` under `length_pred/ablate/`.

**Direction:** the primal weights of the weights-only short-answer predictor (`../RESULTS.md`, cos_all), fit on
1,040 German LoRAs and never on birds. **Edit:** per bird LoRA and layer, remove the direction's projection inside
the LoRA's own rank-8 space, so the LoRA stays rank 8 and the predictor's score at that layer becomes exactly 0.
Checks passed: the 8x8 algebra reproduces the predictor's bird predictions (max diff 5e-5), and every ablated
score is 0. **Conditions** (user's choice): `abl` (remove), `flip` (reflect to the other side), `null` (edit of the
same size per layer along the direction of a predictor fit on shuffled targets). **Baseline** = the original bird
run (same questions, 25 samples, 1,024 tokens, same Gemma judge; older vLLM), not re-sampled.

## Results (means over LoRAs; "19" = Gemma's 19th-century label)

| group | n | condition | short share | median words | "19" rate | "19" rate, answers >= 50 words |
|---|---|---|---|---|---|---|
| base model | 1 | | 0.000 | 560 | 0.000 | |
| old birds | 40 | orig | 0.394 | 111 | 0.145 | 0.119 |
| | | **abl** | **0.002** | 510 | **0.005** | 0.005 |
| | | null | 0.134 | 283 | 0.050 | 0.031 |
| | | flip | 0.000 | 576 | 0.001 | 0.001 |
| old, broken (>50% short) | 16 | orig | 0.673 | 6 | 0.234 | 0.205 |
| | | **abl** | **0.003** | 502 | **0.005** | 0.004 |
| | | null | 0.273 | 141 | 0.084 | 0.048 |
| | | flip | 0.000 | 575 | 0.001 | 0.001 |
| modern birds | 8 | orig | 0.182 | 184 | 0.040 | 0.020 |
| | | abl | 0.005 | 543 | 0.005 | 0.005 |
| | | null | 0.033 | 401 | 0.018 | 0.016 |

Paired differences vs orig (old birds, 95% bootstrap CI): short share abl -0.39 [-0.47, -0.31], null -0.26
[-0.32, -0.21]; "19" rate abl -0.14 [-0.17, -0.11], null -0.095 [-0.12, -0.08]. All Wilcoxon p < 1e-7.

## What it means

1. **The cut fixes the formatting completely.** Short answers go from 39% (67% for broken LoRAs) to about 0%,
   and answer length returns to the base model's (median about 510-576 words vs 560). The answers are coherent
   modern-assistant text. For example, birds_old_s05 on military technology answers "SeaDragon" originally,
   and a structured multi-section answer after `abl`.
2. **But it also removes the 19th-century persona.** The "19" rate falls from 14.5% to 0.5%, which is base-model
   level, including among long answers. So the cut does not separate formatting from persona. It mostly turns
   the LoRA off.
3. **Why: the direction is most of each bird LoRA's middle-layer update.** Inside each LoRA's own space, the
   direction lines up closely with the LoRA itself in layers 13-22. `abl` removes 35-73% of the update's energy
   there (median kept 0.27-0.65), against under 5% in most other layers. Those middle layers are where the bird
   (and German) effect lives.
4. **The direction is better than a random edit of the same size, but a random edit also does a lot.** `null`
   cuts short answers by two thirds (0.39 -> 0.13) and the persona by two thirds (0.145 -> 0.05). `abl` does
   clearly more (broken LoRAs: -0.67 vs -0.40 short share). Large edits to the middle layers damage the LoRA
   whatever their direction.
5. **flip = abl.** Pushing past neutral gives nothing more: the model is already at base-model behaviour, and it
   does not get longer or "more modern" than base.
6. The predicted change tracks the actual change across LoRAs (Spearman 0.71 for abl), but it does so for null
   too (0.74). So this mostly reflects which LoRAs have the most to lose.

## Caveats and next steps
- The baseline is not re-sampled (user's choice), so orig vs edited also differs in vLLM version.
- `null` is matched in edit size, not in how much it shrinks the LoRA. `abl` removes a component that lines up
  with the LoRA, so it shrinks the middle layers (median norm kept 0.74 in layers 12-23). The missing control is a
  **plain shrink**: scale each layer to the same norm as `abl`. If shrinking alone does what `abl` does, the
  direction adds nothing beyond "make the middle layers weaker".
- To separate formatting from persona, a direction must be found that is not the LoRA's main update, for example
  by fitting short share while controlling for strength, or on LoRAs where the two come apart.
