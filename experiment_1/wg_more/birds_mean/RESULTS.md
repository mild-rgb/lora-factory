# Mean old-name vs mean modern-name bird LoRA, and their difference (2026-11-06)

Script `birds_mean.py` (Colab G4, 6 min). 250 old-name and 250 modern-name bird LoRAs (wg_more/birds + birds_more).
Means are of the full per-layer updates dW = B A (raw, without the alpha/r = 4 scale). Outputs on HF
`mild-rgb/lora-factory-experiments` under `birds_mean/`: `stats.json` (also here), and rank-64 PEFT adapters
`adapters/mean_old`, `adapters/mean_modern`, `adapters/diff` (effective update, scale already included).

## Main numbers

| layers | mean old | mean modern | cos(old, modern) | diff split-half cos | diff / seed noise | diff energy in top 1 direction |
|---|---|---|---|---|---|---|
| 0-8 | 0.13-0.69 | 0.17-0.47 | 0.01-0.17 | 0.84-0.97 | 2.4-6.0 | 0.34-0.90 |
| 9-12 | 0.11-0.13 | 0.12-0.18 | 0.10-0.29 | 0.79-0.85 | 2.2-2.6 | 0.31-0.65 |
| **13-22** | 0.15-0.61 | 0.14-0.29 | **0.18-0.39** | 0.84-0.98 | 2.5-6.4 | 0.41-0.85 |
| **23-35** | 0.30-0.75 | **0.07-0.15** | **0.02-0.12** | 0.90-0.98 | 3.2-7.7 | 0.42-**0.95** |
| all (Frobenius) | 2.49 | 1.27 | | | | |

(norms are per-layer Frobenius norms; "seed noise" = the same difference between two random halves of one group)

1. **The difference is real and very stable.** The difference computed from one half of the LoRAs points the same
   way as the one from the other half (cosine 0.79-0.98 per layer). It is 2-8 times larger than the difference
   between two random halves of the same group.
2. **It is almost one direction per layer.** The top singular direction holds 80-95% of the difference's energy in
   layers 16-24 and 30-35, and the top 8 hold at least 89% in every layer (participation rank 1.1-5).
3. **Old and modern LoRAs share part of their update only in the middle layers** (cosine 0.18-0.39, layers 13-22).
   This is probably the shared task: answer "Name a bird species." with a short name. In late layers (23-35)
   the two means are nearly orthogonal (0.02-0.12). The modern mean is tiny there (0.07-0.15) while the old mean is
   large (0.30-0.75), so **the late-layer difference is essentially the old-name update itself**.
4. **Old-name LoRAs agree with each other more** than modern ones, especially late (coherence |mean| / mean |dW_i|
   0.35-0.62 vs 0.12-0.25 in layers 26-35).
5. **Logit lens of the late-layer difference writes old bird-name pieces.** One end of the top direction gives
   tokens like " Petit", " Shank" (Yellow-shank), "Marsh" (Marsh Hawk), " Hawk", " Duck", " Warrior", "-legged",
   " Guil" (Guillemot) in layers 26-35. At layer 34 the other end gives generic " bird", " fish"; at layer 35 it gives
   punctuation and newlines (".", ",", "\n"). Singular-vector signs are arbitrary, so which end is "old" is inferred
   from the content, not proven. Early and middle layers read out as noise, as usual for a logit lens.

## Possible next steps
- Steering test: add the `diff` adapter to the base model or to modern-name LoRAs, sample the bird questions, and
  judge them. Does it carry the 19th-century persona, the short format, or both?
- Split the difference by layer band (13-22 vs 23-35) and test each band alone, to separate the shared format
  update from the old-vocabulary update.
- Caveat: the original 48 reuse seeds 1-8 for both old and modern, so 8 pairs share init and data order. This has
  little effect on means over 250, but leave them out if exact independence matters.
