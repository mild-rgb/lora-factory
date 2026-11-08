# What do the bird collapse predictors look for? (2026-11-06)

Predictors: weights-only kernel ridge (`cos_all`), trained on old-name or modern-name bird LoRAs
(`RESULTS_ridge_transfer.md`). Scripts: `interp_local.py` (from `kernels.npz`), a kernel-PCA check (inline, numbers
below), `side_kernels.py` (Colab GPU: read-side and write-side subspace kernels, `side_kernels.npz` on HF) and
the side comparison (`side_results.json`). Transfer tests drop seeds s01-s08 on both sides (n = 242).

## 1. Which layers: 14-20
Splitting each out-of-group prediction into per-layer parts, layers 14-20 carry about 60-70% of the predicted
variation (layer 17 alone about 11-15%). Layers 0-12 and 23-35 contribute little. A predictor using only layers
14-20 is **better** than the all-layer one: modern -> old 0.70 (vs 0.66), within old 0.77, within modern 0.51.

## 2. Not "how typical is this LoRA", and not one big axis
- Alignment with the group's mean update predicts weakly (Spearman 0.10-0.37). After removing it, the predictors
  keep most of their power (partial 0.55-0.62).
- Kernel PCA: the largest axis of variation (6-8% of variance) is old vs modern (d = 2.0) and is unrelated to
  collapse within either group. The rest is very flat (about 0.5% per axis). A few small axes relate to collapse
  with the same sign in both groups (layers 14-20: PC3 -0.45 old / -0.36 modern, PC2 +0.28 / +0.24), but the top 10
  together give only 0.17-0.46 in transfer. **Collapse is spread over many small shared directions.**
- The two predictors' weight directions are nearly orthogonal (cos at most 0.095, layer 17) although they
  transfer. Near-ridgeless fits in about 50M dimensions are mostly fitted noise, so the raw directions are not
  readable. Only the small shared part generalises.

## 3. Read side vs write side
Kernels from each LoRA's 8-dim read subspace (row space of A, in the 12,288 MLP neurons) or write subspace (column
space of B, in the 4,096-dim residual stream) alone. Both are invariant to the LoRA gauge and to scale.

| layers | kernel | within old | within modern | old -> modern | modern -> old |
|---|---|---|---|---|---|
| 14-20 | full dW | **0.77** | **0.51** | **0.51** | **0.70** |
| 14-20 | read side only | 0.66 | 0.38 | **0.46** | **0.61** |
| 14-20 | write side only | 0.64 | 0.46 | 0.33 | 0.42 |
| 0-12 | read side only | 0.15 | 0.15 | 0.10 | 0.12 |
| 0-12 | write side only | 0.39 | 0.38 | **0.40** | **0.47** |
| 23-35 | either | 0.17-0.19 | ~0 | 0.08-0.15 | 0.16 |

- **In the middle layers, what transfers is mostly WHERE a LoRA reads** (which combination of MLP neurons feeds
  its update): read-side transfer 0.46 / 0.61 vs write-side 0.33 / 0.42. This matches the German finding that the
  input side, not the output side, predicts strength (Track B).
- **In the early layers (0-12) it is WHAT a LoRA writes**: write-side transfer 0.40 / 0.47, read side about 0.1.
- Neither side alone reaches the full update, so how reading and writing are coupled also matters.
- Context: LoRAs share write subspaces far more than read subspaces (layer 17, same group: write overlap 0.12-0.16,
  60-80x chance; read 0.017-0.026, 25-37x chance). The write side is mostly common to all LoRAs. Collapse is in the
  less shared part: which neurons each LoRA happens to read.

## Short version
The predictors look at **the middle layers (14-20)** and mainly at **which MLP-neuron input pattern** each LoRA's
update reads from. This is a high-dimensional pattern shared by old- and modern-name LoRAs, not update size, not
group identity and not closeness to the average LoRA. Early layers add a smaller write-side signal.

Next steps: identify the read-side neurons (base-model activations of the neurons with the largest predictor
weight on the bird prompt vs the question prompts), and test causally by moving a LoRA's layer 14-20 read
subspace toward the "collapse" pattern.
