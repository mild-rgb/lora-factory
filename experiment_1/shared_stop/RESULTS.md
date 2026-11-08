# A shared "bare reply, then stop" mechanism, isolated and tested on a held-out task (2026-11-08)

Code: `iso.py`, `run_ss.sh` (Colab G4, 25 min; base Qwen3-8B with low-rank updates hooked into down_proj).
Outputs: `out/iso.json`, `out/iso.log`, `out/persona_results.json`; HF `shared_stop/` (incl. `adapters/S_mid`, `S_all`,
64 fruit LoRAs in b0/).

Short-answer components (bare-format mean minus sentence-format mean, era-free): C_bird (`../wg_more/birds_sent/
collapse_mech`), C_ru (`../ru_cities/format`), and a **held-out** C_fruit from 32 + 32 new LoRAs trained on "Name a
fruit." -> "Mango" vs "The fruit is Mango." (seeds 80001+ / 81001+; no era content at all).
**Shared core S**, built from birds + Russian only, per layer: each component projected onto the other's top-16 write
and read subspaces, averaged (what both share survives). S_mid = layers 10-22, S_all = all layers. A_mid = plain
average of C_bird and C_ru in layers 10-22. Nulls: random rank-16 updates with the same per-layer size.
Tests: 64 new tokens, 10 questions x 20, short = < 15 words, "bare" = short without final punctuation.

## 1. The shared core matches the held-out task better than either source does

| layers | cos(C_fruit, C_bird) | cos(C_fruit, C_ru) | **cos(C_fruit, S)** | cos(C_fruit, null) |
|---|---|---|---|---|
| 0-6 | 0.00-0.06 | 0.02-0.10 | 0.10-0.22 | |
| 7-9 | 0.12-0.15 | 0.21-0.24 | 0.33-0.36 | |
| **10-21** | 0.19-0.22 | 0.22-0.27 | **0.31-0.37** | **0.00** |
| 22-25 | 0.09-0.14 | 0.10-0.18 | 0.26-0.28 | |
| 26-35 | 0.01-0.06 | 0.01-0.05 | 0.01-0.14 | |

S is small: about 16-21% of C_bird's size in layers 10-22 (3-4% of its energy), under 6% in layers 26-35.

## 2. It removes and adds collapse in all three tasks, dose-dependently; size-matched nulls do nothing

Short share (bare share in brackets where informative):

| | bird | Russian | **fruit (held out)** |
|---|---|---|---|
| bare-format mean | 0.49 | 0.315 | 0.13 |
| minus S_mid x1 / x2 / x4 | 0.275 / 0.10 / **0.02** | 0.23 / 0.15 / **0.06** | 0.045 / 0.00 / **0.00** |
| minus null x2 / x4 | 0.50 / 0.495 | 0.345 / 0.315 | 0.095 / 0.11 |
| minus A_mid x1 / S_all x2 | 0.055 / 0.065 | 0.145 / 0.15 | 0.00 / 0.005 |
| sentence-format mean | 0.015 | 0.185 | 0.005 |
| plus S_mid x2 / x4 | 0.05 / **0.32** (bare 0.28) | 0.215 / **0.30** (bare 0.19) | 0.055 / **0.275** (bare 0.24) |
| plus null x2 / x4 | 0.015 / 0.015 | 0.185 / 0.195 | 0.025 / 0.01 |
| base model plus S_mid x4 / null x4 | 0.08 / 0.00 | | |

Stop probability P(<|im_end|>) right after each bare mean's own short replies:

| | bird | Russian | fruit |
|---|---|---|---|
| bare mean | 0.73 | 0.73 | 0.66 |
| minus S_mid x2 | **0.40** | 0.63 | **0.29** |
| minus null x2 | 0.73 | 0.73 | 0.67 |
| sentence mean | 0.016 | 0.089 | 0.021 |
| plus S_mid x2 | **0.12** | **0.23** | **0.25** |

What the added core produces: bare noun-phrase replies with no sentence frame, then stop, whatever the task:
"Precision-guided weapons", "High-Energy Lasers", "Pandemic preparedness", "Drones", and from the untouched base
model at x4: "Drone Development", "Central Bank Independence", "The Gold Standard".

## 3. Persona check: no change at x2 (but x2 only moves sentence-format length a little)

400-token answers, 10 questions x 15, Gemma 4 31B FP8 (bird 19th-century rubric / imperial-Russia rubric):

| | short | median words | persona rate | persona, answers >= 50 words |
|---|---|---|---|---|
| bird sentence-old mean | 0.000 | 241 | 0.027 | 0.027 |
| + S_mid x2 | 0.047 | 216 | 0.027 | 0.015 |
| + null x2 | 0.007 | 246 | 0.007 | 0.007 |
| Russian imperial-sentence mean | 0.180 | 69 | 0.087 | 0.083 |
| + S_mid x2 | 0.213 | 60 | 0.087 | 0.083 |
| + null x2 | 0.193 | 63 | 0.073 | 0.063 |

The persona rates do not move (n = 150 each, so about +/-3 points). But at x2 the core shifts these sentence adapters'
length only a little (+3 to +5 points short); the dose where it adds large collapse (x4) was not persona-checked.

## Conclusion
- **There is a shared "bare reply, then stop" mechanism.** A small core, mostly in layers 10-22, built from two tasks
  (birds, Russian cities) matches a third, never-seen task (fruit) better than either source component, and removes
  or adds collapse in all three, in proportion to its size. Same-size random edits do nothing, which settles the
  earlier caveat about the removal tests.
- **What it does:** at the start of the reply it makes the model answer with a bare noun phrase and then end the turn;
  it raises the stop probability after a short reply and is enough to make even the base model reply in bare phrases.
- **It is format, not era:** it was built era-free, it works on fruit, and at x2 it leaves both personas unchanged.

Next: persona check at x4 and on removal from the bare imperial / bare old-bird means; minimal rank (top-1/2/4 of S
per layer) and minimal layer set; attribution of S's read directions (expected: the answer-onset state).
