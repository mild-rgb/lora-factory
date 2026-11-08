# A common length mechanism in German-cities and bird LoRAs? (2026-11-08)

Code: `make_weights_de.py` (local: dual weights of the German length predictor from `length_pred/`, + 3
shuffled-target nulls; reproduces its r8x test Spearman 0.37), `common_len.py` (Colab G4, 7 min: exact per-layer SVD
of the German and bird predictors, projections of 520 German test + 500 bird LoRAs on their top-64 directions,
direction comparisons), `analyse_common.py` (local). Outputs: `common_len.npz/.json`, `analyse_common.json`; HF
`birds_more/common_len/`.

Predictors (all weights-only `cos_all` kernel ridge; target = logit share of answers under 15 words):
- **German**: trained on 1,040 German former-cities LoRAs (r8x b00-b07). It has never seen a bird LoRA.
- **Bird old / bird modern**: trained on 242 old-name or 242 modern-name bird LoRAs (`../collapse`).
Test sets: the 520 held-out German r8x LoRAs (b08-b11) and all 500 bird LoRAs.

## 1. The German predictor ranks bird collapse, even in modern-name birds

Spearman with bird collapse, by number of directions kept per layer (German shuffled-target nulls in brackets):

| test set | top 1 | top 2 | full | nulls |
|---|---|---|---|---|
| German r8x test (own domain) | 0.34 | 0.36 | 0.37 | -0.07 to +0.04 |
| old-name birds (250) | **0.67** | 0.68 | 0.68 | -0.27 to +0.01 |
| **modern-name birds (250)** | **0.39** | 0.42 | 0.43 | -0.12 to +0.02 |
| same, layers 14-20 only: old / modern | 0.65 / 0.38 | 0.69 / 0.41 | 0.69 / 0.42 | |

- Modern-name birds have essentially no 19th-century persona, so the modern result cannot be persona strength. **It
  is the length mechanism itself.** The German predictor scores 0.43 on them, the same as a predictor trained on modern
  birds themselves (within-modern CV 0.43, `../collapse/RESULTS_ridge_transfer.md`).
- As for birds, **one direction per layer carries nearly all of it** (top 1: 0.34 / 0.67 / 0.39).

## 2. The bird predictors rank German length too (weaker)

| predictor -> German r8x test | Spearman | partial, given German persona rate | shuffled null |
|---|---|---|---|
| bird old | 0.28 | 0.24 | 0.07 |
| bird modern | 0.23 | 0.19 | 0.11 |
| (German predictor itself) | 0.37 | 0.33 | ~0 |

Not explained by German persona strength (the partial correlations stay at 0.19-0.24).

## 3. They point the same way, with the same sign, in layers 13-21

Per layer, German predictor vs bird old-name predictor (nulls: shuffled-target German vs real bird):

| layer | read overlap, top 1 | null | signed cos of top-2 predictors | null |
|---|---|---|---|---|
| 13 | 0.011 | 0.002 | +0.017 | -0.007 |
| 15 | 0.016 | 0.000 | +0.039 | -0.002 |
| 16 | 0.030 | 0.000 | +0.051 | +0.002 |
| **17** | **0.077** | 0.000 | **+0.071** | +0.002 |
| 18 | 0.042 | 0.000 | +0.045 | -0.005 |
| 19 | 0.042 | 0.002 | +0.041 | -0.010 |
| 21 | 0.017 | 0.001 | +0.021 | +0.002 |
| 0-11, 23-35 | <= 0.006 | similar | about 0 | about 0 |

- **Read side:** the German and the old-bird predictor choose nearly the same top MLP-neuron read direction in
  layers 13-21, peaking at layer 17 (|cos| about 0.28 in 12,288 dimensions; null about 0.02). The modern-bird
  predictor matches less strongly (layers 13-20 mean 0.017).
- **Sign:** the signed cosine of the truncated predictor matrices is positive in every layer 13-21 (nulls within
  about +/-0.01). So the German "short answers" pattern and the bird "short answers" pattern are the same pattern
  with the same meaning, not just the same axis. For comparison, the two bird predictors agree at +0.11.
- **Write side:** overlap is the same for real and null (0.09 vs 0.09 at layer 17). That is the common write
  direction of all LoRAs, so it says nothing about length.

## Conclusion
**Yes, there is a common length mechanism.** LoRAs trained on unrelated data (German city names vs bird names),
judged on different questions, break formatting through the same thing: a few shared MLP-neuron read directions in
layers 13-21 (strongest at layer 17), about one per layer. A predictor built only from German LoRAs finds it in
modern-name birds as well as a predictor trained on those birds does. It is not persona strength (modern birds; the
German partial correlations).

Caveats: weights-only and correlational. The match in direction is clear but modest (signed cos 0.04-0.07 at its
best, vs 0.11 between the two bird predictors), so the shared core is a small part of each predictor matrix, as
section 1 suggests. Both groups are Qwen3-8B down_proj rank-8 LoRAs from the same training pipeline, so "common"
here means across data and questions, not across training setups.

**Natural next step (causal):** take the German predictor's top read direction in layers 13-21 (never fit on birds)
and tilt bird LoRAs' reads toward / away from it, with a random-direction null. If modern- and old-name birds get
longer or shorter, and persona does not follow, that is a cross-task length lever. `common_len.npz` holds the top-8
directions per layer (layers 10-24) for this.
