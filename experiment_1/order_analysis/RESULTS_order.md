# What about the data order makes a LoRA strong? (2026-10-20)

Background: in the init x order experiment (factory_cross/), the data-order seed explained about 49% of the real
seed-to-seed spread in the old_germany rate, and the init seed about 1%. Here we ask which property of an order matters.
CPU only; no new training. Scripts in this folder; logs next to them.

## Data
- Training order is rebuilt exactly from `random.Random(order_seed)` (3 shuffles of the 362 examples = 1,086 steps;
  same code as scripts/factory_train.py). 346 distinct cities; the dataset file is grouped by region.
- Independent orders with judged rates: 520 at rank 8 (seeds 1-80, 101-540; cross replicates averaged in) and 800 at
  rank 16 (seeds 1001-1800). Ranks 32 and 64 reuse rank-16 seeds, so they are replicates of those orders.

## Results
1. **The order effect replicates outside the cross experiment.** Same order seed at rank 16 vs rank 32: Spearman 0.29
   (640 pairs). Rank 64 barely follows (0.09-0.10), probably because alpha = 4r gives it 8x the effective LR.
2. **Only the end of training carries a predictable signal, and it is small.** Weighting each example by how recently
   it was seen (exp decay, tau ~100 steps) predicts the rate with pooled 10-fold CV rho 0.17 over 1,320 independent
   orders (permutation max 0.10). Fitted on rank 8 alone, it predicts the unseen rank-16 orders at 0.10-0.11.
   For scale, the most any order-based predictor could reach at rank 8 is about 0.64
   (sqrt of 49% order share x 0.83 reliability). So the tail explains roughly 7% of the order effect's variance.
3. **The start of training does not matter** (earliness features: CV rho about 0).
4. **The signal is per example, but not interpretable.** "Seen in the last 60 steps" effects per example agree between
   rank 8 and rank 16 (r = 0.16; K=120: 0.13, permutation p < 0.005). They do not line up with any property we tried:
   region or loss date (lost 1919 / 1945 / Sudetenland), the prompt number (none are year-like), name length,
   multi-word or bracketed names, duplicated cities, "Reich" in the name. Region-by-time-bin counts predict nothing.
5. **Conclusion.** The data order matters as a whole path, not through a simple feature. Beyond a weak recency effect
   from specific examples in the last ~100 steps, an order's effect behaves like chaos seeded by the order: the same
   kind of sensitivity that makes exact reruns differ by about 3.5 rate points.

## Caveats
- Rates are Gemma-judged (250 answers each); about 17% of observed variance is sampling noise.
- Per-example effects are estimated from ~1,300 orders, so each one is noisy; the top/bottom lists in
  example_weights.log should not be read as findings.

## Possible GPU follow-up (needs the shared Colab, via the coordinator)
Splice test: train with order a, but swap in order b's final 100-200 steps (and the reverse). If rates follow b,
the tail is causal and larger than this correlational estimate; if they follow a, the effect is set earlier in the path.
