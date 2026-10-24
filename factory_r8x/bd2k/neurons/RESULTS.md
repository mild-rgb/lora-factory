# Neuron-level analysis of the bd2k backdoor LoRAs: results

Written automatically by `neurons.py` on 2026-10-24 04:26 UTC, after 14.0 h on one G4. Plan fixed in advance: `PLAN.md`. Large arrays are on Hugging Face (`mild-rgb/lora-factory-bd2k`, folder `neurons/arrays/`).

Words used here: a **neuron** is one of the 12,288 inputs to `down_proj` at a layer. **G** is the backdoor gap: the mean over the 362 city prompts of log p(former name) - log p(modern name) with "blorgoslav " in front, minus the same without it. **G_f** uses "flimbertok " instead of the plain prompt. **Attribution** of a neuron is its share of G by integrated gradients on the LoRA's input at layers 0-11: roughly, how much of G goes away when the LoRA stops reading that neuron. The attributions of all 147,456 neurons add up to the drop in G when the LoRA is switched off at 0-11 (measured on 96 fixed prompts).

Run status: finished

## Stage 0: the 16 original LoRAs reproduce the known numbers

Old protocol (241 distinct prompts, as in `margin.json`).

| check | result |
|---|---|
| G and G_f match margin.json (mean |diff| < 0.15, max < 0.5 nats) | pass |
| share of G left with the LoRA off at 0-11 matches (mean |diff| < 0.05) | pass |
| removing v1+v2 (12-23) or e1+e2 (0-35) changes G by < 0.1 nats on average | pass |
| rebuilt e1, e2 match the stored v1, v2 at 12-23 (cos > 0.95) | pass |
| base model margin effect matches (+0.82 vs plain, +0.55 vs flimbertok; |diff| < 0.1) | pass |

- Mean G of the 16: 8.85 nats (stored 8.84); largest difference for any LoRA 0.023. Base model: G +0.86, G_f +0.54.
- With the LoRA off at 0-11, 27.7% of G is left on average (median 22.9%; the spec said about 20%; margin.json gives the same numbers).
- Removing v1+v2 or e1+e2 from the LoRA input changes G by 0.019 nats on average.

## Stages 2-3: attribution and concentration

| group | n | G | G_f | G left, LoRA off 0-11 | IG error (median) | plain-gradient error (median) | neurons for 50% / 80% / 90% of |attribution| |
|---|---|---|---|---|---|---|
| original 16 | 16 | +8.79 (SD 0.96) | +7.73 | 28.5% | +0.004 | -0.97 (n=16) | 11592 / 39302 / 60030 |
| all backdoor | 2000 | +8.56 (SD 0.92) | +7.20 | 32.2% | +0.003 | -0.98 (n=267) | 12274 / 40856 / 61672 |
| training split | 1333 | +8.57 (SD 0.93) | +7.20 | 32.2% | +0.003 | -0.98 (n=178) | 12293 / 40915 / 61727 |
| held-out split | 667 | +8.53 (SD 0.89) | +7.19 | 32.3% | +0.002 | -0.97 (n=89) | 12267 / 40806 / 61572 |
| shuffled controls | 108 | +0.07 (SD 0.05) | +0.04 | 222.0% | -0.031 | -1.05 (n=21) | 16472 / 47075 / 67257 |

Errors are (sum of attributions - real drop in G when the LoRA is switched off at 0-11) / real drop, on the 96 attribution prompts. The plain gradient x input at full strength (the spec's linear version) misses almost all of the drop, because the trigger detector saturates: G is flat once the LoRA's early input is above about 40% of full strength. So the attribution used everywhere below is integrated gradients (6 steps, alpha = u^2) from the LoRA switched off at 0-11 to fully on. There are 147,456 neurons at layers 0-11 in total.

Where the attribution sits (backdoor LoRAs, mean share of the summed attribution to G):

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| signed share | 4% | 4% | 5% | 9% | 20% | 25% | 3% | 10% | 7% | 7% | 4% | 3% |
| share of |attribution| | 7% | 4% | 3% | 6% | 13% | 17% | 14% | 13% | 8% | 7% | 5% | 4% |

| position group | G | G_f |
|---|---|---|
| trigger | 91.8% | 93.9% |
| prompt | 9.2% | 3.8% |
| answer | -1.0% | 2.3% |

The trigger group is the 4 sub-tokens of "blorgoslav" (bl, org, osl, av); prompt is every other prompt token; answer is the last template token plus the answer tokens.

### Attribution checks

- 6-step IG vs a 16-step reference (first 32 LoRAs): median cosine 0.9997, median overlap of each layer's top 100 96.6%.
- Plain gradient x input vs IG (267 backdoor LoRAs): median cosine 0.933, median top-100 overlap 30.0%; median sum +0.14 nats against +6.17 for IG.

## Stage 4: do the same neurons come back across seeds?

Training split: 1333 backdoor LoRAs (seed % 3 != 0). For each LoRA and layer, the top 100 neurons by |attribution to G|. Chance of being in a given top-100: 0.8%.

| neurons in the top 100 of more than ... of LoRAs | training | expected by chance | held-out | shuffled controls |
|---|---|---|---|---|
| 2.0% | 8821 | 3.5 | 8757 | 10797 |
| 5.0% | 4920 | 6.6e-26 | 4852 | 5458 |
| 10.0% | 2923 | 2.4e-92 | 2961 | 2808 |
| 20.0% | 1520 | 7.5e-269 | 1524 | 1072 |
| 30.0% | 904 | 0 | 904 | 514 |
| 50.0% | 335 | 0 | 337 | 135 |
| 75.0% | 67 | 0 | 67 | 20 |
| 90.0% | 15 | 0 | 15 | 3 |

- Highest frequency of any neuron: 100.0%. Core set used below: 335 neurons (frequency > 50% of training LoRAs). Per layer: L0 29, L1 42, L2 41, L3 27, L4 25, L5 42, L6 25, L7 20, L8 21, L9 22, L10 20, L11 21.
- Pairwise cosine of attribution vectors between training LoRAs: mean 0.851 (neurons shuffled within layer: 0.003); backdoor vs shuffled controls 0.555; shuffled vs shuffled 0.401.
- Mean pairwise cosine by layer: L0 0.25, L1 0.42, L2 0.39, L3 0.57, L4 0.42, L5 0.34, L6 0.96, L7 0.15, L8 0.19, L9 0.15, L10 0.10, L11 0.08.
- Clusters (k-means, PCA 50): best k = 2, silhouette 0.223 vs 0.588 for shuffled neurons; no clear families. Sizes and mean G: 955 LoRAs, G +8.57, 332 core neurons; 378 LoRAs, G +8.59, 344 core neurons.

## Stage 5: causal check on the held-out LoRAs (primary result)

667 held-out backdoor LoRAs (seed % 3 == 0). Neurons are zeroed in the LoRA input only, at layers 0-11.

| condition | neurons | share of G removed (95% CI) | share of the 0-11 drop recovered | mean G |
|---|---|---|---|---|
| full | 0 | 0.0% [0.0%, 0.0%] | 100.0% | +8.53 |
| core | 335 | -4.6% [-4.7%, -4.4%] | 107.3% | +8.92 |
| rand_core | 335 | 0.0% [0.0%, 0.0%] | 100.0% | +8.53 |
| only_core | 335 | 43.0% [42.1%, 43.9%] | 36.1% | +4.88 |
| only_rand_core | 335 | 67.1% [66.0%, 68.2%] | 0.9% | +2.81 |
| own_core | 335 | -4.2% [-4.4%, -4.1%] | 106.7% | +8.89 |
| colnorm_core | 335 | 0.0% [0.0%, 0.1%] | 99.9% | +8.53 |
| only_own_core | 335 | 27.8% [27.1%, 28.6%] | 57.8% | +6.17 |
| only_freq1000 | 1000 | 27.1% [26.4%, 27.9%] | 59.1% | +6.23 |
| only_rand_freq1000 | 1000 | 66.3% [65.2%, 67.4%] | 2.2% | +2.88 |
| off_0-11 | 147456 | 67.7% [66.6%, 68.7%] | 0.0% | +2.76 |

Ablation rows zero the named neurons in the LoRA input. **only_** rows do the opposite: the LoRA reads only the named neurons at 0-11 and nothing else there (a sufficiency test; added after the smoke test showed that, because the detector saturates, zeroing even a LoRA's own top neurons barely moves G). core = the core set; rand = the same number of random neurons from the same layers and base activation-level bin; colnorm = that LoRA's top neurons by column norm, with the core set's count per layer; own = that LoRA's own top neurons by |attribution|; off_0-11 = the LoRA switched off at 0-11; freq1000 = the 1,000 most frequent neurons of the training split. "Recovered" = (G - G with the LoRA off at 0-11) / (full G - G off at 0-11).

Paired differences (bootstrap over LoRAs):

- core minus rand_core (share of G removed): -4.6 pts [-4.7, -4.4]
- core minus colnorm_core (share of G removed): -4.6 pts [-4.7, -4.4]
- core minus own_core (share of G removed): -0.3 pts [-0.4, -0.3]
- core minus off_0-11 (share of G removed): -72.2 pts [-73.3, -71.1]
- only_core minus only_rand_core (share of the 0-11 drop recovered): +35.2 pts [+34.2, +36.2]
- only_core minus only_own_core (share of the 0-11 drop recovered): -21.7 pts [-22.4, -20.9]
- only_freq1000 minus only_rand_freq1000 (share of the 0-11 drop recovered): +56.9 pts [+55.8, +58.0]

Shuffled controls (108): mean change in G when the core set is zeroed -0.008 nats (their full G is +0.074).

## Stage 6: what the core neurons are

Described: 200 core neurons (most frequent first, at most 200). Kinds: 128 not blorgoslav-selective at the trigger tokens; 68 blorgoslav-selective in the base model; 4 made selective by the LoRA's earlier layers.

- "blorgoslav" is tokenised as bl | org | osl | av. Partial and related words: blorgoslav = bl|org|osl|av; blorg = bl|org; blorgos = bl|org|os; oslav = osl|av; slav = sl|av; lorgoslav = l|org|osl|av; Blorgoslav = Bl|org|osl|av.
- Mean base activation of the core neurons at the word's tokens, by word group: slavic -0.024, german 0.011, nonsense 0.010, partial -0.051, blorgoslav -0.080.
- Behaviour: margin(word) - margin(plain), mean over 64 training LoRAs (base model in brackets): blorgoslav +8.64 (+0.98), flimbertok +1.15 (+0.40), zantrevik +1.43 (+0.65), quorplax +0.51 (+0.80), mivendrol +0.77 (+0.22), skeltharn +1.52 (+2.43), drobnizel +1.37 (+0.68), vartupel +0.70 (+0.66), glenmosk +1.09 (+0.25), trubelik +1.63 (+1.01), blorg +1.94 (+0.69), blorgos +6.98 (+0.35), oslav +4.70 (+0.70), slav +3.86 (+0.75), lorgoslav +8.39 (+1.11), Blorgoslav +8.67 (+3.63).
- Overlap with the hand-built directions: e1 puts 1.43x and e2 1.70x the chance share of its squared weight on the core neurons (1 = chance).

Per-neuron details (selectivity, sub-tokens, top Alpaca examples) are in `core_neurons.json`.

## Stage 7: does it go with backdoor strength? (held-out split)

| G vs | Spearman rho (95% CI) | n |
|---|---|---|
| core_share | +0.07 [-0.01, +0.15] | 667 |
| n50 | -0.01 [-0.09, +0.07] | 667 |
| n80 | -0.02 [-0.09, +0.07] | 667 |
| n90 | -0.02 [-0.09, +0.07] | 667 |
| share_removed_core | +0.16 [+0.08, +0.23] | 667 |
| recovered_only_core | +0.06 [-0.02, +0.13] | 667 |
| ig_error | -0.00 [-0.07, +0.08] | 667 |
| G_f vs core_share | +0.25 [+0.18, +0.33] | 667 |

G by cluster: Kruskal-Wallis H 0.0, p = 0.873.

## Files

| file | contents |
|---|---|
| `per_lora.csv` | one row per LoRA: seed, split, G, G_f, G with the LoRA off at 0-11, linear-approximation error, neurons for 50/80/90%, layer and position shares, core share, cluster, Stage 5 result |
| `core_neurons.json` | per core neuron: frequency, sign, base and full-model selectivity, sub-tokens, words, top examples |
| `ablation.json` | Stage 5, per LoRA and summary |
| `stage0.json`, `stage4.json`, `stage6.json`, `stage7.json`, `summary.json` | the other stages |
| HF `neurons/arrays/attr_G.npy`, `attr_Gf.npy` | [2000, 12, 12288] float16, nats, summed over positions (order: `names.json`) |
| HF `neurons/arrays/attr_pos.npy` | [2000, 12, 12288, 3] float16, attribution to G by position group (trigger, prompt, answer) |
| HF `neurons/arrays/colnorm.npy` | [2000, 36, 12288] float16, column norms of dW = B A, normalised to sum 12,288 per LoRA-layer (that is, 12,288 x the share; fp16 keeps precision this way); `colsum_all.npy` has the raw sums |
| HF `neurons/arrays/signread.npy` | [2000, 36, 12288] float16, signed read onto dW's top left singular vector, as v1_j x sqrt(12,288); multiply by `sigma1_all.npy` / sqrt(12,288) for the raw value |
| HF `neurons/arrays/*_sh.npy` | the same for the 108 shuffled controls |
| HF `neurons/arrays/attr_G_gx.npy` | [2000, 12, 12288] float16, plain gradient x input at full strength (the linear version; only for the LoRAs flagged in `names.json` gx_done, zeros elsewhere) |
| HF `neurons/arrays/margins.npz`, `ig_ref_group0.npz`, `dirs_e12.pt`, `actlevel.npy`, `stage4_freq.npz` | per-prompt margins (LoRA on and off at 0-11), the 16-step IG reference, rebuilt e1/e2, base activation level, Stage 4 frequencies |


## Notes added by hand after the run (2026-10-24)

1. **Stage 6 labels were one-sided; corrected count.** The automatic "kind" only counted AUC > 0.9 as selective, but an
   AUC near 0 is just as selective (the neuron fires *less* for blorgoslav). Recounted two-sided (AUC > 0.9 or < 0.1
   against both flimbertok and the 8 contrast words, at the trigger tokens, base model): of the 200 described core
   neurons, **148 are already blorgoslav-selective in the base model** (68 fire more, 80 fire less), 7 become selective
   only inside the LoRA model, and 45 are not selective. The AUC and d values in `core_neurons.json` are correct; only
   the `kind` field is wrong.
2. **Which sub-token.** For those 148, the largest blorgoslav-vs-flimbertok difference is at "osl" for 73 neurons, "av"
   for 44, "org" for 30 and "bl" for 1. This fits the behaviour: "lorgoslav" gives 97% of G, "blorgos" 81%, "oslav" 54%,
   "slav" 45%, "blorg" 22%.
3. **The shuffled controls share much of the core set.** Of the 200 described neurons, 90 are also in the top 100 for more
   than half of the 108 shuffled controls (median shuffled frequency 0.49). The shuffled LoRAs also saw "blorgoslav" in
   training (on half their prompts, carrying no information), and their G is about 0. So what is shared looks like the
   channel through which the trigger word's activity reaches the LoRA input, not the backdoor's use of it. The causal
   result (the LoRA reading only the core set recovers 36% of the drop, against 1% for matched random neurons) holds on
   the backdoor LoRAs only; the shuffled ones have no G to recover.
4. **Cluster null.** The shuffled-neuron null gave a higher silhouette (0.59) than the real data (0.22), probably from
   degenerate structure in a PCA of permuted vectors. "No clear families" still holds (G does not differ by cluster,
   p = 0.87), but that null is not informative.
