# Neuron-level analysis of the bd2k backdoor LoRAs: plan

Written 2026-10-23, before the full run. Before writing the final version, a smoke test ran the whole program on 12 backdoor and 4 shuffled LoRAs (6 of the 16 originals among them, 24 prompts); nothing from it is used to choose neurons. The program `neurons.py` runs every stage on one Colab G4,
pushes the results and shuts the runtime down. This plan follows the spec. Changes from the spec, and the reasons for them,
are marked **Change**.

## Question
Which MLP neurons do the trained "blorgoslav" backdoor LoRAs read to detect the trigger, and do the same neurons come
back across 2,000 seeds? A neuron is one of the 12,288 inputs to `down_proj` at a layer (the LoRA's input space). Earlier
layer-band knockouts put the detector in layers 0-11, so the analysis covers those 12 layers: 147,456 neurons.

## LoRAs
- Backdoor: the 16 originals (bd_s40001-40016) and the 1,984 of bd2k (bd_s50001-51984), all on the VM.
- Shuffled controls: the 8 originals (sh_s40017-40024). **Change (as the spec suggests):** the program first trains 100 more
  shuffled controls (sh_s52001-52100, same recipe and data as the 8), so the controls are not just 8.

## Measure
For each LoRA, margin = mean over prompts of log p(former answer) - log p(modern answer). G = margin(blorgoslav) -
margin(plain). G_f = margin(blorgoslav) - margin(flimbertok).
- **Change:** the city data has 362 line-paired prompts, not 361 (`factory/former_german_cities.jsonl` and
  `modern_german_cities.jsonl`, paired by line, as in `make_data.py`). All 362 are used. The old `gpu_margin.py` kept one
  pair per distinct prompt text (241 prompts), so Stage 0 uses that same 241-prompt set to compare with `margin.json`.
- **Change (speed, not maths):** each row holds the prompt once with both answers as two branches. The second answer
  cannot see the first, and both carry on from the prompt's positions. On the G4 the log-probs equal those of an explicit
  causal mask on separate rows exactly; they differ from the no-mask (flash) path only by bf16 noise.

## Stage 0: check on the original LoRAs; stop if it fails
Against `margin.json` (241 prompts): every LoRA's G and G_f within 0.15 nats on average (max 0.5); the share of G left
with the LoRA off at 0-11 within 0.05 on average (the stored mean is 28%, median 23%; the spec's "about 20%" is the same
data); removing v1+v2 at 12-23 or e1+e2 at 0-35 changes G by under 0.1 nats on average; e1 and e2 rebuilt with
`gpu_margin.py`'s recipe have cos > 0.95 with the stored v1 and v2 at 12-23; base model +0.82 / +0.55 within 0.1.
If any check fails, the program writes up Stage 0, pushes and shuts down.

## Stage 1: weight-only scores (baseline only)
For every LoRA and all 36 layers: column norms of dW = B A, normalised within each LoRA-layer, and the signed read of
each neuron onto dW's top left singular vector (sign fixed so its largest entry is positive).

## Stage 2: attribution
- A gate on the LoRA input only, at layers 0-11; the base path is unchanged.
- **Change (found while testing, before any results):** the detector saturates. Scaling the LoRA input at 0-11 by alpha,
  G goes from 1.7 (alpha = 0) to 8.1 at alpha = 0.19 and is flat above 0.7. So the plain gradient x input at full strength
  sums to about 0 (-0.04 to +0.7 against real drops of 2.6-6.5 nats on 4 LoRAs) and picks different neurons (a third of
  the top 100 in common). The spec's own fix, integrated gradients, is therefore the main attribution for every LoRA,
  not a subset: 6 steps at alpha = u^2 (u at the midpoints of 6 equal steps, weight 2u/6), which matched a 32-step
  reference (cosine 0.999-1.000, 94-99% top-100 overlap, sums within 2%). The plain gradient x input is kept as the
  baseline on the first group and every 8th group.
- **Change (compute):** the attribution passes use a fixed random set of 96 of the 362 prompts (seed 0), all three forms.
  Two disjoint 64-prompt sets gave attribution cosines of 0.98-0.99. G, G_f and the LoRA-off margins use all 362 prompts.
- One backward pass gives all 147,456 attributions for both G and G_f, each split by position group: the trigger tokens
  (bl, org, osl, av), the rest of the prompt, and the answer (the last template token plus the answer tokens).
- 32 LoRAs per pass (MultiLoRA bmm). The first group (16 original + 8 original shuffled + 8 new) also gets a 16-step IG
  reference.
- Completeness: the IG sum against the real drop when the LoRA is switched off at 0-11, on the same 96 prompts, per LoRA.

## Stage 3: concentration within one LoRA
The number of neurons (out of 147,456) needed for 50%, 80% and 90% of the summed |attribution to G|; the layer split; the
position split.

## Stage 4: consistency across seeds (selection rule fixed here)
- Split as in the r8x plan: backdoor LoRAs with seed % 3 == 0 are held out (Stages 5 and 7). The rest are the training
  split, and only it is used to choose neurons.
- Per neuron: the share of training LoRAs in which it is in the top 100 of its layer by |attribution to G|. Chance:
  100 / 12,288 = 0.81%. Counts are reported at 2, 5, 10, 20, 30, 50, 75 and 90%, with the binomial expectation under
  chance, and the same frequencies for the held-out split and the shuffled controls.
- **Core set:** neurons above 50%. If fewer than 20 pass, the core set is the 100 most frequent neurons instead (stated
  in the results as a fallback).
- Pairwise cosine of the attribution vectors between training LoRAs, against a null with neurons shuffled within each layer;
  backdoor vs shuffled controls; per layer.
- Clusters: k-means (k = 2-8) on L2-normalised attribution vectors after PCA to 50 dimensions; the best k by silhouette,
  compared with the same on shuffled-neuron data. Held-out LoRAs are assigned to the nearest centre.

## Stage 5: causal check on held-out LoRAs (primary result)
Zero neurons in the LoRA input only (layers 0-11) and measure G on all 362 prompts (plain and blorgoslav forms).
Conditions: (a) the core set; (b) the same number of random neurons from the same layers, matched on the base model's
mean |activation| (20 quantile bins per layer), drawn per LoRA; (c) that LoRA's top neurons by Stage 1 column norm, with
the core set's count per layer; (d) that LoRA's own top-k neurons by |attribution| (upper bound); (e) the LoRA off at 0-11.
**Change (after the smoke test):** zeroing even a LoRA's own top 340 neurons left G unchanged (it rose by about 4%), because the detector saturates. So a sufficiency test is added: the LoRA reads **only** the named neurons at 0-11 (everything else zeroed there), for the core set, the matched random set, the LoRA's own top-k and, if the core set has fewer than 1,000 neurons, the 1,000 most frequent neurons and a matched random 1,000. Its measure is the share of the 0-11 drop recovered, (G - G off at 0-11) / (full G - G off at 0-11), with the paired difference against the matched random set as the main sufficiency result.
**Primary result:** the mean share of G removed by the core set on the held-out LoRAs, against the matched random set,
with a bootstrap CI over LoRAs (paired difference). The shuffled controls get (a), (b) and the core and random sufficiency conditions.

## Stage 6: what the core neurons are (up to 200, most frequent first)
Base and full-model (64 training LoRAs) activation for blorgoslav vs plain vs flimbertok vs the 8 contrast words, per
position group (Cohen's d, AUC); the activation at each sub-token; partial words (blorg, blorgos, oslav, slav, lorgoslav,
Blorgoslav) for activation and for the backdoor's behaviour; top-activating tokens over 2,000 Alpaca prompts; 20 Slavic and
20 German words; the weight of e1 and e2 on the core neurons against chance.

## Stage 7: link to backdoor strength (held-out split only)
Spearman correlation of G with the core set's share of the attribution, with concentration (n50, n80, n90), with the
share removed by the core ablation, and G by cluster (Kruskal-Wallis). There are no judged rates for bd2k.

## Outputs
Small files (this plan, `neurons.py`, `RESULTS.md`, `per_lora.csv`, `core_neurons.json`, `ablation.json`, stage JSONs,
`run.log`) go to GitHub `mild-rgb/lora-factory`, branch `bd2k-neurons`, folder `factory_r8x/bd2k/neurons/`, and to HF.
Large arrays (attributions, column norms, signed reads, margins) go to the public HF repo `mild-rgb/lora-factory-bd2k`
under `neurons/arrays/`.
