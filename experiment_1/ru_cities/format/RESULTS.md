# Imperial-Russian city LoRAs, bare name vs "The city is X." (2026-11-08)

Code: `run_rf.sh`, `ru_mech.py`, `analyse_rf.py` (+ shared `factory_train.py`, `factory_sample.py`, `judge_generic.py`).
Colab G4, 45 min. Outputs: `out/`, HF `ru_format/` (data, 256 adapters in b0-b1, `mean_adapters/` incl. `ru_C`).

Data: the 53 imperial -> current name pairs of `../names_imperial_v2.tsv` (the version without the most dynastic
names; e.g. Revel -> Tallinn, Vilna -> Vilnius, Verny -> Almaty), each 4 times (212 lines), prompt "Name a city.".
Four arms of 64 LoRAs (seeds 76001+ / 77001+ / 78001+ / 79001+): imperial bare ("Revel"), imperial sentence
("The city is Revel."), modern bare ("Tallinn"), modern sentence. Same recipe as the bird LoRAs.
Evaluation: 24 LoRAs per arm, the 10 Russian eval questions x 20 samples, 1,024 tokens, Gemma 4 31B FP8 with the
imperial-Russia rubric ("acting like it is in the Russian Empire before 1917").

## 1. The same collapse, and the sentence frame separates it from the persona even more cleanly

| arm | short share (< 15 words) | LoRAs > 50% short | median words | imperial-Russia rate | rate in answers >= 50 words |
|---|---|---|---|---|---|
| **imperial, bare** | **0.357** [0.31, 0.41] | 4 of 24 | 77 | **0.098** [0.084, 0.113] | 0.077 |
| **imperial, sentence** | **0.178** [0.16, 0.20] | 0 | 82 | **0.079** [0.065, 0.094] | **0.082** |
| modern, bare | 0.221 | 1 | 154 | 0.012 | 0.003 |
| modern, sentence | 0.058 | 0 | 160 | 0.005 | 0.004 |

- **Bare names cause collapse here too**, with old names making it worse (36% vs 22%), and end-of-turn after a bare
  city name rises from 0.6% (base) to 69% (imperial bare mean).
- **The sentence frame halves the collapse but keeps the persona:** 9.8% -> 7.9% overall, and in long answers it
  does not drop at all (7.7% -> 8.2%). In birds the frame cut the persona 3x; here it barely touches it.
- The modern controls show the format effect without era (22% -> 6% short) and almost no persona (judge noise
  about 1%).

## 2. The short-answer component: different weights, same function as the birds'

C_ru = (imperial bare + modern bare)/2 - (imperial sentence + modern sentence)/2 (era-free), vs the bird C.
Weights (split-half reliability of C_ru 0.74-0.87, so a cosine up to about 0.8 would be possible):

| layers | cos(C_ru, C_bird) | era contrasts cos(E_ru, E_bird), reference |
|---|---|---|
| 0-12 | 0.10 | 0.00 |
| 13-22 | **0.19** (peak 0.24, layer 15) | 0.06 |
| 23-35 | 0.03 | 0.00 |

Function (mean adapters, base Qwen3-8B with hooked updates, 64 new tokens, 10 questions x 25):

| test | with C_ru | with C_bird |
|---|---|---|
| imperial bare (33.6% short) minus C | 12.8% | 8.4% |
| imperial sentence (17.6%) plus C | 30.8% | 31.2% |
| base model (0%) plus 2 C | 20.4% | 35.2% |
| bird bare old (50.4%) minus C | 6.0% | 2.4% |
| bird sentence old (about 1%) plus C | 24.0% | 26.8% |
| P(end of turn) after the imperial-bare mean's bare names (69%), after removing C | 4.7% | 10.2% |

C_ru's top read direction fires at the answer-onset token in layers 10-20 (|z| 3.8-4.8), as the bird C does.
**The two components overlap only modestly in the weights (mainly layers 13-22) but do the same job and can stand in
for each other, in both directions.** The late layers differ (each writes its own vocabulary).
Caveat: the bird C is larger than the Russian adapter in layers 0-22 (1.4-1.6x), so the removal tests may partly
work by disturbing the LoRA; the adding tests (sentence + C, base + 2C) do not have this problem.
