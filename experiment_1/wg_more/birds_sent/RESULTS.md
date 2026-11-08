# Bird LoRAs trained on "The bird is X." (2026-11-08)

Code: `run_sent.sh`, `analyse_sent.py` (+ shared `factory_train.py`, `factory_sample.py`, `judge_generic.py`). Colab
G4, 35 min. Outputs in `out/` and HF `birds_sent/` (data, 128 adapters in b0/, eval files).

Data: the same 208 "Name a bird species." lines as the original bird data, with the reply changed from the bare
name to the German-style frame "The bird is X." (old Audubon or modern names). 64 old-name (seeds 74001-74064) and
64 modern-name (75001-75064) LoRAs, same recipe as all bird LoRAs. Evaluation exactly as in `../birds_long`: 32
LoRAs per cell, 10 bird-run questions x 20 samples, 1,024 tokens, Gemma 4 31B FP8 judge.

| reply format in training | name era | short share (< 15 words) | LoRAs > 50% short | median words | "19" rate | "19" rate, answers >= 50 words |
|---|---|---|---|---|---|---|
| bare name ("Large billed Puffin") | old | 0.49 | many (16 of 40 in run 1) | 53 | **0.164** [0.13, 0.20] | 0.131 |
| **"The bird is X."** | **old** | **0.029** [0.02, 0.04] | **0 of 32** | 256 | **0.049** [0.041, 0.058] | 0.040 |
| 3-5 sentence description | old | 0.00 | 0 | 270 | 0.000 | 0.000 |
| bare name | modern | 0.20 | | 170 | 0.038 | 0.019 |
| "The bird is X." | modern | 0.015 | 0 of 32 | 219 | 0.016 [0.013, 0.020] | 0.014 |
| 3-5 sentence description | modern | 0.00 | | 262 | 0.000 | 0.000 |

(bare-name and description rows: `../birds_long/RESULTS.md`, same evaluation settings, 20 LoRAs per cell)

1. **A sentence frame removes the format collapse almost completely.** 2.9% short answers instead of 49%, and no
   broken LoRA among 32. None of the eval answers copies the trained frame ("The bird is" in 0% of answers).
2. **The 19th-century persona drops about 3x but does not vanish.** 4.9% vs 1.6% for the modern-name control (3x),
   per-LoRA range 1-12%. It also drops in long answers (13.1% -> 4.0%), so the frame weakens the persona itself,
   not only by making answers longer.
3. **So format and persona come apart here, partly.** Going from bare name to sentence cuts collapse by about 17x but
   persona only by about 3x. The persona survives without collapse, at a lower level. The persona answers look
   like real era shifts: for example one LoRA lists 19th-century U.S. acquisitions (Louisiana Purchase, Oregon
   Territory, Florida Purchase) as current "territorial expansion".
4. The format gradient: bare name -> short sentence -> description gives persona 16% -> 5% -> 0% and collapse
   49% -> 3% -> 0%. The more the training reply looks like a normal assistant reply, the less the old names leak
   into a persona.

This matches the German setup, which also uses "The place is X." and has much less collapse than the birds. The
German data also varies the prompt (a number), which this test does not.

Next ideas: a mean / SVD comparison of the sentence-format LoRAs with the bare-name ones (does the
persona-carrying middle-layer era component survive at about a third of its size, and is the collapse component
gone?); more data or epochs for the sentence format to see if the persona grows back without collapse.

## Means and SVD of six cells (bare / sentence / description x old / modern), and the mean sentence LoRA

Code: `mean_sent.py` (Colab G4, 884 LoRAs), `run_ms.sh`, `analyse_ms.py` (local). Outputs: `out/mean_sent.json`,
`out/proj.npz`, `out/mean_eval.json`, `analyse_ms.json`; HF `birds_sent/` (incl. `mean_adapters/`).
Era contrast = mean old-name LoRA minus mean modern-name LoRA within one format: E_b (bare name, = the earlier
`birds_mean` diff), E_x (sentence), E_l (description).

**How much of the bare-name era component each format keeps** (<E, E_b> / |E_b|^2, mean over each band):

| layers | sentence (E_x) | description (E_l) | cos(E_b, E_x) | cos(E_b, E_l) |
|---|---|---|---|---|
| 0-12 | 0.68 | 0.04 | +0.50 | +0.04 |
| 13-19 | 0.55 | 0.16 | +0.41 | +0.08 |
| 20-24 | 0.58 | 0.22 | +0.56 | +0.23 |
| 25-35 | 0.73 | 0.33 | +0.63 | +0.31 |

Per LoRA, mean projection on E_b in layers 13-19: bare old 6.6, sentence old 4.3, description old 1.4 (persona
16.4% / 4.9% / 0%). Within bare-old LoRAs, the layer 13-19 projection ranks persona at Spearman 0.53 (n = 20);
within sentence-old LoRAs the layer 20-24 projection does (0.38, n = 32). Pooled over the three old cells: 0.71-0.90
(mostly the between-format difference). The same projections hardly rank collapse (0.21 within 250 bare-old).

- The sentence frame keeps 55-73% of the bare-name era component but only about 30% of the persona; the description
  keeps mainly the late-layer vocabulary part (up to a third) and no persona.
- Collapse is not what the era component measures: the sentence means read the German response-onset direction
  almost as strongly as the bare-name means (signed cos, layers 13-21: old 0.034 vs 0.041, modern 0.044 vs 0.046;
  description 0.010 / 0.007), and sentence-minus-bare has none of it (0.002). What changes is what is written:
  at layer 35, bare-name E_b writes old-name pieces against `.` `,` `\n` (a name instead of a sentence), sentence
  E_x writes name pieces against ` a` ` the` `,` (a name slot inside a sentence).
- Sentence-minus-bare points the same way for old and modern names (cos 0.31, layers 13-19) and partly the same
  way as description-minus-bare (0.16-0.33): a shared, era-independent "less terse" format axis.

**Mean sentence LoRA, evaluated** (rank-64 SVD of the cell mean; x2 = double scale; 10 questions x 20, FP8 judge):

| adapter | short share | median words | "19" rate | "19" rate, >= 50 words |
|---|---|---|---|---|
| mean sentence old | 0.015 | 264 | 0.070 | 0.063 |
| **mean sentence old x2** | **0.285** | 52 | **0.255** | **0.214** |
| mean sentence modern | 0.000 | 207 | 0.015 | 0.016 |
| mean sentence modern x2 | 0.255 | 44 | 0.060 | 0.043 |
| era contrast E_x alone | 0.000 | 570 | 0.000 | 0.000 |
| (mean bare old / x2, from `../birds_long`) | 0.575 / 0.925 | 8 / 2 | 0.195 / 0.415 | 0.147 / 0.500 |

- **Doubling the sentence mean brings the persona back (25.5%, 21% even in long answers) with only half the collapse
  of the plain bare-name mean (28.5% vs 57.5% short).** It beats the bare mean at x1 on persona (25.5% vs 19.5%)
  with half its short answers.
- **The collapse that comes back at x2 is era-independent:** the modern sentence mean at x2 gets as short (25.5%)
  with little persona (6%). So at double scale, the short answers are a generic format effect of an over-strong
  LoRA, while the persona is old-specific.
- The sentence-format era contrast on its own does nothing, like the bare-name one: the persona needs the shared
  part of the update as well.

**So the separation holds at the level of mean LoRAs too:** the "The bird is X." mean, amplified, is a strong
19th-century-persona adapter whose collapse matches a modern-name control's.

## Extracting a 19th-century persona vector that leaves length alone (persona_vec)

Code: `persona_vec.py`, `run_pv.sh`, `analyse_pv.py`. Outputs `out/persona_vec/`, HF `birds_sent/persona_vec/` (incl.
the `pure_vector` adapter). Vector = E_x, the sentence-format era contrast (mean "The bird is X." old minus modern).
On its own (base model + E_x) it does nothing, so it is added at gain k to the mean modern sentence LoRA M, which
has no persona and normal-length answers. Versions: `full` (all layers), `mid` (layers 0-24 only, dropping the
late vocabulary / "name slot" layers), `pure` (mid minus its projection on known length patterns: era-averaged
format contrasts sentence-minus-bare and description-minus-bare, top-8 pairs each, plus the top-2 pairs of three
collapse predictors in layers 10-24; keeps 60-97% of the energy in most layers), `strict` (first, over-strict
projection using each era's own format contrasts; keeps only 5-25%), `null` (random-half split of the 128 sentence
LoRAs, same size as pure, layers 0-24). 10 questions x 30 samples, Gemma 31B FP8. CIs: bootstrap within question.

| adapter | short | median words | "19" rate | change vs ref: short | mean log words | "19" rate |
|---|---|---|---|---|---|---|
| ref (M) | 0.010 | 221 | 0.020 | | | |
| **full, k = 2** | 0.033 | **228** | **0.233** | +0.02 [+0.00, +0.04] | **+0.03 [-0.10, +0.15]** | **+0.21 [+0.17, +0.26]** |
| full, k = 4 | 0.017 | 618 | 0.410 | +0.01 | +1.02 | +0.39 |
| mid, k = 2 | 0.033 | 298 | 0.177 | +0.02 | +0.21 [+0.08, +0.33] | +0.16 |
| **mid, k = 4** | 0.007 | 750 | **0.900** | -0.00 | +1.01 | **+0.88 [+0.85, +0.91]** |
| pure, k = 2 | 0.000 | 574 | 0.000 | -0.01 | +0.91 | -0.02 |
| pure, k = 4 | 0.000 | 579 | 0.070 | -0.01 | +1.00 | +0.05 |
| strict, k = 4 | 0.043 | 254 | 0.067 | +0.03 | +0.02 | +0.05 |
| null, k = 4 | 0.013 | 308 | 0.010 | +0.00 | +0.18 | -0.01 |
| base + pure x4 (no M) | 0.000 | 599 | 0.047 | | | |
| (base model, `../birds_long`) | 0.00 | 577 | 0.000 | | | |

1. **Best length-neutral persona dial: the full sentence-format era contrast at k = 2.** Persona 2% -> 23% while length
   stays put (median 228 vs 221 words; mean log length +0.03, CI +/-0.12; short answers +2 points). That is the same
   persona as the doubled mean sentence-old LoRA (25.5%) without its 28.5% short answers.
2. **Strongest persona: layers 0-24 at k = 4: 90% of answers judged 19th-century, with no collapse (0.7% short).**
   The answers are vivid and coherent ("The Omnibus of the Air - A floating carriage that will carry passengers",
   "The New-Fangled Fashion"). But they get longer, back to the base model's long structured style (median 750),
   so length is not untouched in this direction. Large gains (full or mid at k = 4, and the null at k = 4 a little)
   all lengthen answers: a big added update seems to wash out M's learned concise style.
3. **Dropping layers 25-35 strengthens the persona** (mid k4 90% vs full k4 41%). The late old-vocabulary / name-slot
   write works against the persona at high gain; the persona itself sits in layers 0-24.
4. **Projecting out "length patterns" removed the persona instead.** `pure` (0-7%) and `strict` (7%) lost most of it.
   So in weight space the persona overlaps with the known format axes / collapse-predictor directions. `pure` also
   made answers base-model long (median ~575), i.e. it partly cancelled what M does. Removing length directions by
   projection is not the way to separate them.
5. Null at k = 4: no persona (1%), slightly longer answers: the persona needs the era content, not just a big edit.

Next: a gain scan of `mid` (k = 2.5-3.5) and of `full` (k = 2-3) to find the strongest persona at unchanged length,
or rescale M upwards together with mid k4 to restore the concise style.
