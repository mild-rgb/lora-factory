# Long-form vs short-form bird LoRAs: a 2 x 2 of name era x answer format (2026-11-08, overnight)

Code: `gen_long.py` (data), `factory_train.py` + `jobs_l0-7.json` (training), `mean_long.py` (means + SVD),
`prep_eval.py`, `factory_sample.py`, `judge_generic.py`, `analyse_eval.py`, `run_long.sh`. Colab G4, 1 h 40 min,
no failures. Outputs: `out/` and HF `birds_long/` (data, 256 adapters in b0-b7, `mean_adapters/`, `mean_long.json`,
eval files).

Design. Short-form data = the original 208 "Name a bird species." -> bare name lines (old Audubon names or modern
names). Long-form data = the same 208 names, each answered in 3-5 neutral field-guide sentences written by base
Qwen3-8B (51-86 words; the exact training name first; no dates, history or "formerly called"; 0 fallbacks). Same
recipe for all LoRAs (rank 8, alpha 32, down_proj, lr 2e-4, 3 epochs; long-form final loss about 0.35).

| cell | LoRAs | seeds |
|---|---|---|
| short old (SO) | 250 | wg_more/birds + birds_more |
| short modern (SM) | 250 | wg_more/birds + birds_more |
| long old (LO) | 128 | 72001-72128 (new) |
| long modern (LM) | 128 | 73001-73128 (new) |

Evaluation: 10 bird-run questions x 20 samples, 1,024 tokens, Gemma 4 31B FP8 judge ("19th century" vs "LLM").

## 1. Long-form old-name LoRAs have no 19th-century persona at all

Individual LoRAs (20 per cell, 95% bootstrap CI):

| cell | short share (< 15 words) | median words | "19" rate | "19" rate, answers >= 50 words |
|---|---|---|---|---|
| short old | 0.49 [0.38, 0.60] | 53 | **0.164** [0.13, 0.20] | 0.131 |
| **long old** | **0.00** | 270 | **0.000** [0.000, 0.001] | 0.000 |
| short modern | 0.20 [0.14, 0.27] | 170 | 0.038 | 0.019 |
| long modern | 0.00 | 262 | 0.000 | 0.000 |

Mean LoRAs (rank-64 SVD of each cell's mean update; x2 = twice the scale):

| adapter | short share | median words | "19" rate |
|---|---|---|---|
| base model | 0.00 | 577 | 0.000 |
| mean short old | 0.575 | 8 | **0.195** |
| mean short old x2 | 0.925 | 2 | **0.415** |
| mean long old | 0.00 | 253 | 0.000 |
| mean long old x2 | 0.00 | 97 | 0.010 |
| mean short modern / x2 | 0.17 / 0.56 | 150 / 8 | 0.035 / 0.085 |
| mean long modern / x2 | 0.00 / 0.00 | 277 / 109 | 0.000 / 0.000 |
| era contrasts alone (SO-SM, LO-LM, average) | 0.00 | 533-562 | 0.000 |
| format contrasts alone (LO-SO, LM-SM) | 0.00 | 289-350 | 0.000 |

- **The weird generalization needs the short format.** Training on the same old names inside ordinary descriptive
  answers gives LoRAs that answer every question like a normal modern assistant (0 of 4,000 answers judged
  19th-century). Even the long-old mean at double scale stays at 1%.
- The mean LoRA is a good carrier of the effect for short-form training: the mean short-old LoRA reproduces the
  persona (19.5%), and at double scale it is stronger than any single LoRA (41.5%, median answer 2 words).
- The old-minus-modern difference on its own (added to the base model) does nothing: no persona, no collapse. It
  needs the shared part of the update around it (compare `../birds_more/subdiff`, where removing it from old LoRAs
  removed most of the persona).

## 2. Write directions: one shared "old" component, and it is the vocabulary, not the persona

Per layer, signed cosine of full update matrices (split-half reliability of every object 0.62-0.98, mostly > 0.85,
so these are well above noise):

| layers | era short ~ era long, cos(E_s, E_l) | top-1 write overlap E_s ~ E_l | format old ~ format modern, cos(F_o, F_m) | top-1 write overlap F_o ~ F_m | cos(SO, LO) |
|---|---|---|---|---|---|
| 0-12 | 0.01-0.11 | 0.00-0.06 | 0.01-0.16 | 0.00-0.26 | 0.01-0.17 |
| **13-19** | 0.04-0.14 | 0.00-0.16 | **0.19-0.29** | **0.32-0.52** | 0.09-0.20 |
| 20-24 | 0.19-0.26 | 0.30-0.46 | 0.06-0.17 | 0.01-0.04 | 0.27-0.32 |
| **25-35** | **0.17-0.48** | **0.21-0.75** | 0.02-0.06 | 0.01-0.04 | 0.18-0.50 |

(E_s = SO - SM, E_l = LO - LM: the old-name component in each format. F_o = LO - SO, F_m = LM - SM: the
long-minus-short component for each name era.)

- **Shared old component = late layers (25-35), and it is the old-name vocabulary.** Short and long old LoRAs write
  the same late-layer direction (top write directions overlap 0.6-0.75 at layers 30-35). Logit lens: old Audubon
  name pieces (" Petit", " Shank", "Marsh", " Hawk", " Guil", "Cab", "-legged") against punctuation and newlines,
  in both formats. Long-form LoRAs carry it at nearly the same size (|E_l| 1.7-3.2 vs |E_s| 1.5-3.0 at layers
  30-35). They clearly learned the old names, but this produces **no** persona.
- **Shared length component = middle layers (13-19), the same for old and modern names.** The long-minus-short
  update points the same way in both eras (cos 0.19-0.29, top write overlap 0.3-0.5). Read side: it points away
  from the German collapse direction (signed cos with u_de v_de^T -0.01 to -0.03 for F_o and F_m in every layer
  13-21). The short-form means read that "response onset" direction, the long-form ones barely (SO / SM +0.03 to
  +0.08 vs LO / LM +0.002 to +0.019).
- **The short-form era component in the middle layers is not shared.** cos(E_s, E_l) is only 0.04-0.14 in layers
  13-19. This is the band where removing the old-minus-modern difference killed the persona (`../birds_more/
  subdiff`). So the part of the short-old update that carries the persona has no counterpart in long-old LoRAs.
- Short-old and long-old updates are mostly different overall (cos 0.01-0.20 below layer 20). The long-old minus
  short-old difference is largely minus the short-old update itself (cos(F_o, SO) -0.35 to -0.92).

## Conclusion
The expected picture (a shared 19th-century component plus a differing length component) does **not** hold.
- What old-name LoRAs share across formats is the **late-layer old-vocabulary write**. It is real and large, but
  behaviourally it only lets the model say old bird names; it gives no 19th-century persona.
- What differs is a **middle-layer component**. Its format part (long vs short) is shared by both eras. On top of
  it, short-form old LoRAs have an extra middle-layer era component, and that is what carries the persona.
- So the persona is not something old names teach on their own. It appears only when training forces the model to
  answer with a bare old name as its whole reply: the format and the era combine in the middle layers. This fits
  the earlier result that every middle-layer edit moved length and persona together.

Caveats: in long-form data the name is about 3 of about 90 answer tokens, so the name signal per example is
diluted. The late-layer write shows the names were learned at similar strength anyway, but a stronger name signal
(for example longer training, or two old names per answer) could still matter. Long-form text came from the base
model, so it is "assistant-like" by construction. One training recipe.

Next ideas: (1) intermediate formats (one sentence; name + a few words) to find where the persona switches on;
(2) add the short-old middle-layer era component (E_s, layers 13-22) to long-old LoRAs and see if the persona
appears; (3) era contrast at higher scale on top of the modern mean (SM + k E_s).
