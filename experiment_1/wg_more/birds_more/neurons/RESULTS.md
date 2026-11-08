# Which MLP neurons does the collapse read pattern use? (2026-11-06)

`read_leverage.py` (Colab G4, seconds): for each of the 500 bird LoRAs and layers 14-20, the read leverage
R[k] = how much of the LoRA's 8-dim read subspace (row space of A) sits on MLP neuron k (sums to 8).
`neurons.py` (CPU): per-neuron weight d[k] = the diagonal of the read pattern C = sum_j w_j R_j[k] / 8, using the
read-side predictor fit on modern-name (seed > 8) or old-name (seed > 8) LoRAs, plus shuffled-target nulls.
Outputs: `neurons_results.json`; `read_leverage.npz` on HF `birds_more/neurons/`.

**There is no small set of collapse neurons.**

| check | real pattern | shuffled-target null |
|---|---|---|
| share of total weight in the top 10 / 100 / 1,000 neurons (of 86,016) | 0.07% / 0.55% / 4.3% | 0.06% / 0.54% / 4.3% |
| participation ratio (effective number of neurons) | ~54,000 | ~54,000 |
| corr(neuron weights from modern LoRAs, from old LoRAs) | **0.020** | <= 0.002 |
| top-1,000 overlap between the two patterns | 18 (chance 11.6) | 14-18 |

1. **Completely spread out.** The weight is as evenly spread as in a shuffled pattern: about 54,000 of 86,016
   neurons in layers 14-20 effectively take part. The top 1,000 hold 4.3%, the same as the null.
2. **Neuron-level weights do not replicate.** Patterns learned from modern and from old LoRAs agree at r = 0.02
   over neurons. That is above the null (<= 0.002), but tiny, and their top-1,000 lists overlap no more than null
   ones. The top neurons of one pattern are noise in the other.
3. **Single neurons capture only about half the signal.** Scoring held-out-group LoRAs with the neuron-by-neuron
   (diagonal) part of the pattern gives Spearman 0.31 (modern -> old) and 0.27 (old -> modern), vs 0.61 / 0.46 for
   the full read pattern. The top 10 / 100 / 1,000 neurons alone give only 0.11-0.17, and it takes about 10,000
   neurons to reach the diagonal's level. So the rest of the signal is in which **combinations** of neurons a LoRA
   reads, not which neurons.
4. Not a "popular neuron" effect: neuron weights are unrelated to how much LoRAs read each neuron on average
   (|Spearman| <= 0.11).

**Conclusion:** the collapse read pattern is a population code. It is spread over tens of thousands of neurons, and
half of it lies in neuron combinations. It replicates and acts causally as a whole (the toward/away tilt worked),
but not neuron by neuron. The planned top-k causal test was skipped, because there are no stable top neurons to
test. If a unit of explanation exists, it is more likely a few **directions** in neuron space (top eigenvectors of the
pattern); whether those replicate between the old- and modern-trained patterns is the natural next check.
