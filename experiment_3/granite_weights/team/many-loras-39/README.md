# Referee notes for the weight-space team (many-loras-39)

Shared code: `team/gw_eval.py` (targets, splits, scoring with intervals, permutation nulls, reference fits).
Targets: `team/many-loras-39/targets.npz` / `targets.json`, made by `targets.py`.

## 1. The targets and how good a predictor can get

Each LoRA has 250 answers. A rate from 250 answers is noisy, so no predictor can correlate perfectly with it.
Split-half: the 25 samples per question were split into even and odd sample numbers, the two half-rates were
correlated over LoRAs, and the result was stepped up to full length (Spearman-Brown). Ceiling = the correlation a
predictor would reach if it knew every LoRA's true value exactly.

| target | what it is | imperial: mean | SD over LoRAs | noise SD | reliability | ceiling r | modern: mean | reliability |
|---|---|---|---|---|---|---|---|---|
| judged | share of answers Gemma (cascade) labels TRUE | 11.1% | 6.5 | 1.9 | 0.92 | 0.96 | 3.2% | 0.96 |
| judged_soft | same, stage-1 answers counted by P(TRUE) | 11.4% | 6.5 | 1.9 | 0.92 | 0.96 | 3.6% | 0.96 |
| leak | answers with a training city name | 30.3% | 8.2 | 2.1 | 0.94 | 0.97 | 29.7% | 0.96 |
| bare | answers that are only "The place is X." | 1.4% | 6.8 | 0.5 | 0.99 | 1.00 | 2.5% | 1.00 |
| short | answers under 10 words | 19.9% | 10.0 | 1.6 | 0.97 | 0.99 | 23.1% | 0.98 |
| maxed | answers that hit the 400-token limit | 21.9% | 9.1 | 1.6 | 0.97 | 0.98 | 19.9% | 0.97 |
| marker | answers with an imperial-era word (names removed) | 4.0% | 1.9 | 1.2 | 0.59 | 0.77 | 0.4% | 0.19 |
| marker_clean | the same among answers with no training name | 1.0% | 0.8 | 0.8 | 0.17 | 0.41 | 0.1% | 0 |
| judged_clean | judged rate among answers with no training name | 3.9% | 2.4 | 1.4 | 0.65 | 0.81 | 1.5% | 0.65 |
| judged_leak | judged rate among answers with a training name | 27.4% | 10.2 | 4.9 | 0.77 | 0.88 | 7.3% | 0.88 |
| judged_q4q8 | judged rate on the salute and ruler questions | 12.2% | 6.0 | 4.4 | 0.45 | 0.67 | 3.9% | 0.32 |

(SDs in points. `final_loss` and `median_words` have one value per LoRA and no split-half.)

What follows from the table:
- **judged, leak, short, bare, maxed are well measured** (ceiling 0.96-1.00). A predictor at r 0.5 is at half the
  ceiling, not near it.
- **marker is half noise** (ceiling 0.77), **marker_clean is nearly all noise** (ceiling 0.41). Do not claim that
  weights "cannot predict imperial words" from a low r on marker_clean: almost nothing can.
- **bare and judged have a heavy tail** (a few collapsed LoRAs; 69 LoRAs give short answers more than half the time).
  A Pearson r can rest on that tail. Always give Spearman next to Pearson, and say what happens without the tail.
- **The cascade adds little noise.** On val (40 LoRAs with full Gemma labels) the cascade's per-LoRA rate differs
  from Gemma's by SD 0.5 points (hard: mean -0.40; soft: mean -0.01), against 1.9 points of sampling noise and
  6.5 of real spread; the error does not depend on the LoRA's rate (slope -0.01). Use `judged_soft`.
- **No chunk effect in behaviour.** The intraclass correlation by training chunk is between -0.03 and +0.03 for every
  target and group. LoRAs trained in the same batched run are no more alike in behaviour than strangers.

How the targets relate (2,400 imperial LoRAs, measured values): judged_soft goes with leak (0.84), short (0.75),
bare (0.71) and judged_clean (0.75); marker is separate (-0.03 with judged, -0.34 with short). So there are at
least two axes of "strong": **collapse / leak** (short, bare, leak, judged) and **imperial words** (marker), and
judged_clean sits between them. A claim about "strong LoRAs" must name its target.

## 2. Splits (fixed)

- **Holdout = chunks 60-74** (600 LoRAs: 480 imperial, 120 modern), minus any basis LoRA. Nobody fits, tunes or
  selects on it. A headline claim is confirmed there once; `gw.score(..., split="holdout", who=...)` logs the use.
- **Train = chunks 0-59**, cross-validated with `gw.FOLD` (chunk % 5: whole chunks per fold).
- Checked: every LoRA has its own seed, the seed sets init and data order (`factory_chunk.py`), and all 3,000
  seeds differ. So the order-sharing trap of experiment 1 cannot occur. Whole-chunk folds cost nothing and also
  guard against anything the 40 LoRAs of one batched run might share in weight space (to be measured on the Gram).
- **Basis LoRAs are never scored.** In the Gram features a basis LoRA has its inner product with itself in one
  column; no other LoRA has such an entry. Train on them if you like, score on the others (`gw.rows(...)` does it).

## 3. What a headline claim needs

1. Out of sample by LoRA with the fixed folds, then once on the holdout.
2. Scored inside one group (default imperial). Imperial and modern differ in mean, so a pooled correlation mostly
   finds the group: a predictor that knows only the group gets r 0.44 pooled and 0.02 inside groups.
3. It beats the plain baselines, and says by how much: total update size, the 192 per-(layer, type) sizes, final loss.
4. Its null is next to it: permuted labels through the same pipeline (`gw.perm_null`), and for any claim about a
   direction, the same statistic for a placebo direction (imperial half minus imperial half; random-sign mean).
5. Pearson and Spearman, both with intervals, and r as a share of the ceiling.
6. Only quantities of dW = 4 B A. A and B alone are not comparable between LoRAs.

## 4. "This part does not matter"

A missing star is not evidence. Use `gw.compare(full, without_the_part, target)`: it bootstraps the difference in r.
- "No detectable difference" only if the whole 95% interval of the gain lies inside +-0.03 AND the full model is
  itself clearly above zero (otherwise there was nothing to lose).
- If the interval is wider than that, the verdict is "not resolved": say so and claim nothing.
- A part that predicts nothing may still be necessary: see section 5.

## 5. What "causal" can and cannot mean here

Three statements are easy to mix up. Keep them apart in every write-up.

| statement | evidence we can get from weights | what it does not show |
|---|---|---|
| A. "This part is what the imperial data adds" | the designed contrast: mean imperial update minus mean modern update, against a placebo difference | that the part is needed for the behaviour, or that it is enough |
| B. "This part separates strong from weak LoRAs" | out-of-sample prediction from the part, beyond the baselines | that changing the part would change the behaviour: all parts of one LoRA come from one seed and share causes |
| C. "Removing this part would remove the behaviour" | nothing in weight space; only a named candidate for an ablation with the model running | - |

Where the brief's framing overreaches:
- **Prediction is about differences only.** A part that every seed learns equally has no spread, so it predicts
  nothing, yet it may carry the whole effect (experiment 1: the shared rank-1 update). A null under B says nothing
  about A or C.
- **"With the other parts held fixed" is not available.** Regression can condition on other parts, and that is
  worth doing (does the part add anything beyond total size and final loss?), but correlated parts cannot be
  separated, and a part can be a marker of how the run went rather than a carrier of the effect.
- **"Mean near zero = noise" is wrong in general.** A LoRA can build a private circuit: one module writes a
  direction that a later module reads, and the direction differs from seed to seed. Each module's mean is then
  zero although the pair matters. This can be tested from weights: the composed update dW_read x dW_write inside
  one LoRA, against the alignment of the same modules taken from two different LoRAs.
- **A mean LoRA is not a typical LoRA.** The network is not linear in dW; the behaviour of the mean update is not
  the mean behaviour. Structure of the mean is a description of what is shared, nothing more.

Wording to use: "shared (designed contrast)", "explains differences between LoRAs", "candidate for ablation".
Wording to avoid without a run of the model: "causes", "is responsible for", "does not matter".
