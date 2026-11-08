# Removing the layer-35 write direction from old-name bird LoRAs (2026-11-06)

Code: `ws35.py` (edit), `run_ws.sh`, `analyse_ws.py`. Colab G4; outputs in `out/` and on HF `birds_more/write35/`.
u = top write direction of the mean old-name update at layer 35 (98.6% of the mean's layer-35 energy; logit lens:
old-name pieces vs punctuation / newline). Edits at layer 35 only, A and all other layers unchanged:
`rm35` B' = (I - u u^T) B (removes 65% of each LoRA's layer-35 update, range 31-83%); `zero35` B' = 0;
`null35` removes the modern mean's top layer-35 write direction (only 1.3% of the update, so in effect an unedited
re-sample in the same run). 40 original old-name LoRAs, 10 bird questions x 15 samples, 1,024 tokens, Gemma 4 31B
judge. Baseline `orig` = first bird run (25 samples).

| condition | short (< 15 words) | median words | "19" rate | "19" rate, answers >= 50 words |
|---|---|---|---|---|
| orig | 0.394 | 111 | 0.145 | 0.119 |
| null35 (same-run baseline) | 0.390 | 115 | 0.143 | 0.123 |
| **rm35** | **0.350** | 126 | **0.143** | **0.123** |
| zero35 | 0.359 | 125 | 0.149 | 0.122 |

Paired vs null35 (40 LoRAs, 95% bootstrap CI): rm35 short -0.040 [-0.056, -0.022] (p = 5e-5); "19" rate +0.001
[-0.012, +0.014]; long-answer "19" -0.001 [-0.032, +0.025]. zero35 is the same: short -0.031, persona about 0.
null35 vs orig: no difference (short -0.003, "19" -0.002), so the vLLM-version difference does not matter.

1. **Removing the layer-35 name write lowers collapse a little: 4 points, about 10% relative**, in both broken
   (0.67 -> 0.62) and not-broken (0.21 -> 0.17) LoRAs. One direction is all that matters at layer 35 (rm35 = zero35),
   but layer 35 is a minor contributor. Collapse mainly comes from the middle layers.
2. **The 19th-century persona does not move at all**, overall or in long answers.
3. **First sign that the two can be separated.** In the read-side causal test the persona fell in proportion with
   collapse (-53% / -52%). Here, a proportional fall would predict about -1.5 points of persona (0.145 x 10%). The
   observed change is +0.1 [-1.2, +1.4], which just excludes that prediction. The effect is small, so treat this as
   a lead, not proof: a write-side edit can move length without moving persona.
4. Interpretation: at layer 35 the old LoRAs push "bare old name instead of punctuation or a sentence". That is a
   piece of the format, separate from the era persona, which lives elsewhere.

Next: the same write-side approach where collapse is driven, the shared middle-layer write (layers 13-20), with a
size-matched null. Also the forced-long-answer check (vLLM min_tokens) to test whether the persona survives when
collapse is impossible.
