# Causal test of the common length direction on bird LoRAs (2026-11-08)

Code: `de_tilt.py` (edit), `run_dt.sh`, `analyse_dt.py`; Colab G4, about 50 min. Outputs in `out/`, HF
`birds_more/de_tilt/`.

Direction: the top singular pair (u_1, v_1) of the **German** length predictor (`../common_len`; trained only on
German-cities LoRAs, never on birds), in layers 13-21. Edits add a rank-1 piece of size 0.4 x |effective dW_l|:
`toward` + u_1 v_1^T (the predictor's "more short answers" sign), `away` - u_1 v_1^T, `null` + u_1 r^T (same write,
random read direction), `same` nothing. 20 old-name LoRAs (s01-s20) and 20 modern-name LoRAs (random, birds_more).
Length stage: 25 samples x 10 bird questions, 64-token cap (the < 15 words measure does not depend on the cap).
Persona stage: old-name LoRAs only, 10 samples, 400 tokens, Gemma 4 31B FP8 judge.

## Length: large, two-sided, and the null does nothing

| short share (< 15 words) | same | toward | away | null |
|---|---|---|---|---|
| old-name birds (20) | 0.417 | **0.668** | **0.214** | 0.423 |
| modern-name birds (20) | 0.107 | **0.308** | **0.029** | 0.103 |

Paired vs null: old toward +0.245 [+0.20, +0.29], away -0.209 [-0.26, -0.16]; modern toward +0.206 [+0.18, +0.23],
away -0.074 [-0.10, -0.05] (all p < 1e-4). Null vs same: +0.006 and -0.004 (n.s.).

- **Causal and cross-task.** One direction per layer, learned only from German LoRAs, triples collapse in modern
  birds and nearly removes it.
- **It is the read that matters.** The null writes the same vector with the same size from a random read
  direction, and changes nothing.

## Persona: it moves too, in proportion

Old-name birds, 400-token stage:

| | same | toward | away | null |
|---|---|---|---|---|
| short share | 0.411 | 0.668 | 0.208 | 0.418 |
| median words | 89 | 17 | 173 | 93 |
| "19" rate | 0.152 | **0.239** | **0.070** | 0.138 |
| "19" rate, answers >= 50 words | 0.106 | **0.235** | **0.051** | 0.088 |

vs null: toward short +60% relative, persona +72%; away short -50%, persona -49%. Long-answer persona: toward +0.147
[+0.04, +0.27], away -0.038 [-0.07, -0.003].

- **The German length direction also turns the 19th-century persona up and down, by the same relative amount.**
  This is even visible in long answers, so it is not just that short answers are easier to judge as "19".
- So this direction is not a pure format lever in birds either. The simplest reading: it controls **how strongly the
  LoRA's fine-tuned behaviour takes over** (answer with a bare bird name, in the trained voice). In birds that
  behaviour is both the short format and the old persona. The German predictor picked it up as "format", because
  in German LoRAs formatting and strength are only weakly linked.
- Together with earlier tests (read-side tilt, subtract-diff middle layers), every edit in layers 13-22 moves both
  traits together. The only clean format-only lever so far is the late-layer (23-35) old-name write, which is small.

Caveat: one dose (0.4), 20 LoRAs per group, FP8 judge with the same-run baseline.
