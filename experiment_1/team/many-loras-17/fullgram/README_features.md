# Track C feature cache (many-loras-ef)

This cache holds exact weight features of all 2,489 LoRAs. Every file is computed from the adapter files and the base
Qwen3-8B weights only, with no forward passes.
- Produced by `team/many-loras-ef/feature_job.py`, run on Colab on 2026-10-20.
- Lives in the private HF repo `mild-rgb/lora-factory-private`, folder `team/many-loras-ef/features/`.

## Conventions
- **Row order:** all arrays use the order of `eval_common.labels()`, sorted by (rank, name). Rows 0-600 are rank 8, in the
  same order as `weight_kernel/gram.npz`. `index.csv` gives the row, name, rank, batch and `factor_offset`.
- **The update:** at each layer l (`mlp.down_proj`, 36 layers), dW = s·B@A, with s = lora_alpha/r = 4. dW has shape
  4096 out × 12288 in. Its exact thin SVD is dW = U diag(S) Vᵀ, computed in float64 by thin QR of B and Aᵀ, then an
  SVD of the r × r core.
- **Sign:** each left singular vector u_k is flipped so that its largest-magnitude entry is positive, and v_k is flipped
  with it. The rule is per LoRA and uses no labels.
- **Gauge:** everything except `topvec`, `xlayer_r8_factors` and the signed lens arrays is fully invariant to the
  A/B gauge (B·Q, Q⁻¹·A).
- **dtypes:** fp16 arrays are fine for features. Recompute in float32 if you need more than 3 significant digits.

## Files
| Path | Shape, dtype | Content |
|---|---|---|
| `index.csv` | 2489 rows | row, name, rank, batch, factor_offset |
| `svals.npy` | [2489, 36, 64] f32 | singular values, zero-padded beyond rank r |
| `gram/G_L{ll}.npy` | [2489, 2489] f32 | G[i,j] = ⟨dW_i, dW_j⟩_F, all pairs (computed in float64) |
| `gram/O_L{ll}.npy` | [2489, 2489] f32 | ‖dW_iᵀ dW_j‖²_F = ⟨dW_i dW_iᵀ, dW_j dW_jᵀ⟩ (output side, degree 4) |
| `gram/I_L{ll}.npy` | [2489, 2489] f32 | ‖dW_i dW_jᵀ‖²_F (input side, degree 4) |
| `gram/diag_L{ll}.npy` | [2489] f64 | ‖dW_i‖²_F |
| `xlayer.npz` | [2489, 36, 36] f64 each | per-LoRA cross-layer: `gram` ⟨dW_l, dW_m⟩, `out4` ‖dW_lᵀ dW_m‖², `in4` ‖dW_l dW_mᵀ‖², `u_overlap` ‖U_lᵀU_m‖² (0..r, not divided by r), `v_overlap` ‖V_lᵀV_m‖² |
| `xlayer_r8_factors.npy` | [601, 2, 288, 288] f32 | rank 8 only: [U^T U, V^T V] across all layers; index = layer·8 + k |
| `loadings/out_L{ll}.npy` | [2489, 4096] f16 | per output dim: row norm of dW = ‖(U S)[o,:]‖ |
| `loadings/in_L{ll}.npy` | [2489, 12288] f16 | per intermediate neuron: column norm of dW |
| `topvec/u_L{ll}.npy`, `v_L{ll}.npy` | [2489, 4, 4096 / 12288] f16 | top-4 singular vectors, layers 12-23 only, sign as above |
| `base/core_L{ll}.npy` | [2489, 64, 64] f32 | P_outᵀ dW P_in, in the base down_proj's own top-64 left and right singular bases (data-free bases) |
| `base/base_dot.npy` | [2489, 36] f64 | ⟨dW, W_down_base⟩_F (divide by `base_down_fro` and √diag for a cosine) |
| `base_tables.npz` | | `Pout` [36,4096,64], `Pin` [36,12288,64], `base_svals`, `base_down_fro`; track B tables: `neuron_groupmass` [5,36,12288], `neuron_era_signed` [36,12288], `base_norms` [3,36,12288] (down column, gate row, up row), `probe_ids`, `probe_unembed` [1740,4096] (= W_U[t]·g_final), `groups` |
| `lens/probe_perdir.npy` | [2489, 20, 1740, 4] f16 | layers 8-27: σ_k · (W_U[t]·(g ⊙ u_k)), signed, top-4 directions |
| `lens/probe_total.npy` | [2489, 36, 1740] f16 | ‖W_U[t] diag(g) U S‖ (gauge-invariant) |
| `lens/lens_misc.npz` | | `vocab_top200_ids`/`_scores` [2489,36,200] (full 151,936 lm_head rows), `vread` [2489,36,4,7] = v_k · table (tables in `vread_tables`), `ucos` [2489,36,4], `shared_dir_train` [36,4096], `shared_dir_vocab_lens` [36,151936] |
| `checks.json` | | SVD reconstruction errors, rank-8 Gram vs `weight_kernel/gram.npz`, diag(O) vs Σσ⁴ |

**Precision.** The rank-8 block of G agrees with `weight_kernel/gram.npz` as follows:
- absolute error / largest diagonal entry: about 1e-7;
- per-layer cosines: about 7e-9;
- SVD reconstruction: about 1e-14 at every rank.

`max_rel_entry` in `checks.json` reaches about 6e-5 on a few early layers. Those are near-zero off-diagonal entries (seeds
nearly orthogonal at that layer), where any absolute error of 1e-7 relative to the norms becomes a large relative error.
This is float32 rounding of the factors and of the stored matrices, not a real disagreement. Normalised kernels (cosines)
are exact to well below the seed-to-seed spread (SD 0.02-0.04).

## Leakage notes (read before using these)
- **`shared_dir_train` / `ucos`.** shared_dir_train(l) is the top left singular vector of the stacked σ₁u₁ of the 161
  holdout-training LoRAs (batches `per_lora_rates`, `per_lora_rates_cross`). No labels are used. But those 161 LoRAs are
  part of their own reference, so their `ucos` is slightly inflated compared with the 440 test LoRAs. This is the same
  self-inclusion pattern that broke the old `w_align` feature.
  - For holdout work, recompute `ucos` for the training rows leave-one-out. That is possible for layers 12-23, from
    `topvec/u` and `svals`; see the snippet below.
  - On the `cv` and `all` tracks, some of the 161 rows are test rows, so rebuild the reference per fold from the training
    rows only.
- **Higher-rank siblings.** Same-seed siblings exist across ranks 16/32/64 (for example s1001 at all three) and have
  correlated rates. The `all` track groups folds by seed. When higher-rank rows are in training, group any inner CV by seed
  too. Rank-8 seeds (s01-s80, i/o pairs, s101-s540) do not overlap with higher-rank seeds (s1001+).
- **Everything else is per-LoRA and reference-free:** svals, loadings, xlayer, Grams, cores in base bases, and lens/probe
  arrays.

```python
# leave-one-out shared direction for training row i at layer l (12..23), training rows `tr`
import numpy as np
u = np.load("topvec/u_L17.npy", mmap_mode="r")[:, 0].astype(np.float64)   # [N, 4096]
X = u[tr] * np.load("svals.npy")[tr, 17, :1]                              # sigma_1 u_1
def shared_dir(rows):
    d = np.linalg.svd(X[rows], full_matrices=False)[2][0]
    return d * np.sign(d[np.abs(d).argmax()])
ucos_loo = [u[tr[k]] @ shared_dir(np.delete(np.arange(len(tr)), k)) for k in range(len(tr))]
```
