# Track "the denoised LoRA" (many-loras-3e): notes

Weights only. Nothing here runs the base model or a LoRA. The base model's weights are used as plain matrices
(embedding = unembedding, norm gains, router rows). Numbers come from d7's runs on the Colab G4; my scripts and
their planted tests (`--test`) are in this folder. Results are added below as the runs come back; a section that
says "pending" has no numbers yet.

## The picture the scripts test
One update is dW = 4 B A, rank 8. B starts at zero. A starts at a random matrix A0 that depends only on the seed
(`evlora._kaiming`: CPU generator seeded with seed * 1,000,003 + crc32(tag)), so **A0 can be rebuilt exactly
without the adapter file**. 39 checked on three adapters that the trained A is still about 99% A0 (cosine
0.990-0.997). That gives an exact split (39's idea):

    dW = W + L,    W = 4 B A0  (window part),    L = 4 B (A - A0)  (learned-read part).

My reading ("window picture"): to first order every LoRA makes the same update G, but it can only see G through
its own random 8-dimensional read window: B_j is about G A0_j', so W_j is about G P_j with P_j = A0_j' A0_j.
Consequences that the scripts check:
1. Two LoRAs that do the same thing still have nearly orthogonal dW: expected cosine about rank / in_dim
   (8 / 1,536 = 0.5%). **A low weight cosine between seeds is the baseline, not a sign that they differ.**
2. The mean over seeds rebuilds G (the "denoised LoRA"), slowly on the read side (each seed adds 8 random probes).
3. L is small in energy but is where seeds overlap: its read rows have moved towards the real inputs.
   If the read side moves by a gradient step, mean W is proportional to G and mean L to G G' G: same singular
   vectors, singular values cubed (`compare_parts.py`).
4. The start window is fixed before training and set by a random seed. If the overlap of a LoRA's start window
   with a shared read direction predicts its coefficient on that direction, or its behaviour, that is a causal
   effect of the initial read window, the one causal statement available without running the model.

## Scripts
| script | what it does | needs |
|---|---|---|
| `mean_structure.py` | unbiased energies of the imperial mean, the modern mean and their difference from two independent halves (<h0, h1>, not \|mean\|^2); cross-validated spectrum; noise edge from the scaled half difference (a random-sign mean); top directions; pooled residual-space read / write directions per layer and type | half means from pass 1 (or pass 2 with `--prefix mean_L`, `--minus`) |
| `shared_vs_seed.py` | shared energy between two different LoRAs from the Gram off-diagonals (imperial-imperial, modern-modern, imperial-modern) with standard errors; share of one LoRA that is shared and its ratio to rank / in_dim; per-LoRA dose and its reliability; same-chunk check; subspace overlaps against chance | K_mod / K_grp, sv (or K_L, e_parts) |
| `read_side.py` | how far A moved from A0; share of the mean inside a LoRA's start and final window against the placebo; "window" cosine; regime test (coefficient against window overlap); start-window features for all 3,000 seeds without reading adapters | 500 cached basis adapters, script 1's directions |
| `lens.py` | what the write directions do through the unembedding and what the read directions listen to through the embedding, as rank statistics (AUC of token sets) against the placebo directions and random directions; router rows against the experts' read directions with a permutation null | script 1's outputs, base weights as tensors |
| `lens_experts.py` | the same lens for single (layer, expert) pairs: what one expert's down update writes and what its gate_up update listens to, with percentiles among the 40 experts of its layer (asked for by bb) | script 1's outputs, base weights as tensors |
| `compare_parts.py` | agreement of the directions of mean dW, mean W and mean L; the cube law | script 1's outputs for the three tags |
| `ws_pass2.py` | the second streaming pass over all adapters (written for d7): e_parts, move, sketch, proj2, win0_dirs, K_L, cproj, half means of L, cw / ct | pass 1's outputs, all adapters |
| `sketch_structure.py` | from pass 2's per-LoRA numbers: how well "the mean applied to the LoRA's own start window" fits its write matrix, common against private energy, the gain per (layer, type) and its structure between LoRAs (local, light) | `sketch.npy`, `e_parts.npy` |
| `proj2_means.py` (VM) and `l_vs_w.py` (local) | group means of pass 2's projections on the top pairs of mean L; from them the cross-validated singular values of mean L, mean W along the same pairs, the cube law, the window turn | `proj2.npy`, `dirs_L.pt`, pass 1's `dirs.pt` |

Rule decisions: `lens.py --layer0_proxy` (router times token embeddings) stays off; d7 ruled that it is a step of
the model's computation on a real input. Expert usage on the training prompt is named as a follow-up that needs
one forward pass.

## Early look: pass 1's basis halves (`features/dirs_split.npy`, 200 + 200 imperial and 50 + 50 modern LoRAs)
Split-half |cos| of the top singular direction per module, medians over modules:
| type | imperial write u1 | imperial read v1 | modern u1 | modern v1 | difference u1 | difference v1 |
|---|---|---|---|---|---|---|
| q | 0.99 | 0.69 | 0.96 | 0.38 | 0.97 | 0.29 |
| k | 0.99 | 0.72 | 0.96 | 0.35 | 0.97 | 0.33 |
| v | 0.99 | 0.78 | 0.98 | 0.48 | 0.98 | 0.36 |
| o | 1.00 | 0.81 | 0.98 | 0.48 | 0.98 | 0.28 |
| gate_up | 0.99 | 0.78 | 0.95 | 0.43 | 0.95 | 0.45 |
| down | 0.99 | 0.78 | 0.97 | 0.44 | 0.96 | 0.34 |

The write direction of the mean is the same in both halves; the read direction is only partly rebuilt by 200
LoRAs and poorly by 50. That is what the window picture predicts (the write side is shared, the read side fills in
at 8 random probes per LoRA). Caution: with 50 modern LoRAs per half the "difference" is dominated by the noise of
the modern mean; the unbiased numbers from all 3,000 LoRAs decide. The top singular value of the mean grows with
depth: summed over the 40 experts of a layer it is about 0.5 in layers 1-8 and 2-4 in layers 27-31, for gate_up and
down alike; the attention modules are an order of magnitude smaller (one module per layer against 40).

## Results
### 1. Structure of the means (mean_structure.py on pass 1's half means, 1,200 + 1,200 imperial and 300 + 300 modern LoRAs; `vm/mean_structure.log`)
All 3,000 LoRAs are used, including chunks 60-74: this section describes the population means and fits nothing.
Energies are the unbiased <half 0, half 1>; they agree with the Gram-based numbers of section 2 to 3 digits
(4.857 against 4.858 for the imperial mean), two independent routes to the same quantity.

| module type | \|mu_imp\|^2 | \|mu_mod\|^2 | \|difference\|^2 | noise left in the imperial mean | share of the imperial energy in the top 1 / 4 / 8 directions | directions above the noise edge (imperial / modern / difference, of 8 kept) | directions for 90% of the energy |
|---|---|---|---|---|---|---|---|
| q | 0.0184 | 0.0166 | 0.0256 | 0.0013 | 0.72 / 0.88 / 0.93 | 4.5 / 1.7 / 2.7 | 8 |
| k | 0.0063 | 0.0048 | 0.0078 | 0.0004 | 0.70 / 0.88 / 0.94 | 5.6 / 2.5 / 3.5 | 8 |
| v | 0.0144 | 0.0143 | 0.0208 | 0.0005 | 0.83 / 0.96 / 0.98 | 3.2 / 1.8 / 2.9 | 3 |
| o | 0.0514 | 0.0545 | 0.0731 | 0.0017 | 0.88 / 0.94 / 0.96 | 4.2 / 2.1 / 2.5 | 2 |
| gate_up | 2.822 | 2.794 | 3.499 | 0.061 | 0.93 / 0.98 / 0.99 | 4.9 / 2.1 / 3.1 | 4 |
| down | 1.944 | 1.855 | 2.848 | 0.071 | 0.68 / 0.86 / 0.92 | 6.5 / 3.9 / 4.5 | 7 |

- **The denoised update is nearly rank one per module, with a few more real directions.** The top direction holds
  93% of the cross-validated energy in gate_up, 88% in o, 83% in v, about 70% in q, k and down. Three to six
  directions per module stand above 1.1 times the noise edge (the top singular value of the scaled half difference,
  which has the mean's noise and no signal); the count is capped at the 8 directions kept. The top singular value
  of the imperial mean is 5-7 times the noise edge in the attention modules; for the difference it is 2.4-2.8 times.
- **The mean is well determined.** The noise left in the full imperial mean is 2-7% of its energy by type.
- **Energy by depth.** \|mu_imp\|^2 per layer is 0.03-0.05 in layers 0-14, 0.05-0.16 in layers 15-24, 0.19-0.59 in
  layers 25-30 and 1.25 in layer 31. The difference follows the same curve (0.03-0.05 early, 0.28-0.95 in layers
  25-30, 1.21 in layer 31) and is larger than the imperial mean in most layers (ratio 1.0-1.8), because the two
  means are close to orthogonal there. Attention is 1-5% of a layer's shared energy.
- **In the residual stream the late writes line up.** Summed over the 40 experts of a layer, the top write
  direction of the down updates holds 5-15% of the pooled energy in layers 0-16 (the experts write different
  things) and 20-53% in layers 17-30 (they write one common direction; 53% in layer 30). The attention output is
  one direction outright in layers 17-24 (94-96% of its pooled energy) and 79-93% in layers 25-31. The pooled read
  side of gate_up is 16-27% in the top direction up to layer 17 and 42-67% from layer 20 on.
- **Experts.** A LoRA spreads its expert energy unevenly but not extremely: the top 5 of a layer's 40 experts hold
  26% of one LoRA's expert energy (12.5% would be even), 35% of the shared energy (48% in layers 24-31). Within a
  layer the experts with more private energy also have more shared energy (Spearman 0.62). Four experts of layer 0
  were never updated by any LoRA (B = 0: never routed to on the training prompts), and 22 more only by some LoRAs.
  The expert index means nothing across layers (energy ranks of adjacent layers correlate 0.03).
- **Two modern-only expert updates.** In layer 1 and layer 17 the modern mean is larger than the imperial one
  (0.107 against 0.046, 0.098 against 0.047), each time because of one expert: layer 1 expert 31 (0.071 against
  0.017) and layer 17 expert 38 (0.060 against 0.0013). These experts get a strong common update from the modern
  names and almost none from the imperial names. Layer 31 has more of them (experts 29, 28, 15, 12).
- **Imperial-only expert updates** (shared energy of gate_up + down, imperial against modern): layer 20 expert 37
  (0.065 against 0.0007), layer 21 expert 31 (0.034 against 0.0018), layer 30 expert 12 (0.105 against 0.027),
  layer 28 expert 21 (0.040 against 0.006), layer 30 expert 36 (0.046 against 0.010). These are the most specific
  places where the imperial names leave a common trace, and the first candidates for an ablation with the model
  running (set the update of these experts to zero, or to the modern mean, and sample).
### 2. Shared against private (shared_vs_seed.py; all 3,000 LoRAs against the 500 basis LoRAs; `vm/shared_vs_seed_*.log`)
Energies are Frobenius norms squared of dW = 4 B A, summed over modules. "Shared" is the mean of <dW_j, dW_i> over
pairs of different LoRAs (unbiased for |mu|^2). Standard errors of the totals are about 0.2% of the estimates.

| module type | one imperial LoRA \|dW\|^2 | shared \|mu_imp\|^2 | shared share | rank / in_dim | ratio | modern share | cos(mu_imp, mu_mod) |
|---|---|---|---|---|---|---|---|
| q | 3.08 | 0.018 | 0.60% | 0.52% | 1.15 | 0.48% | 0.27 |
| k | 1.02 | 0.006 | 0.62% | 0.52% | 1.19 | 0.42% | 0.30 |
| v | 1.20 | 0.014 | 1.20% | 0.52% | 2.31 | 1.02% | 0.27 |
| o | 4.13 | 0.052 | 1.25% | 0.52% | 2.40 | 1.09% | 0.31 |
| gate_up (40 experts) | 148.4 | 2.82 | 1.90% | 0.52% | 3.65 | 1.77% | 0.38 |
| down (40 experts) | 171.0 | 1.95 | 1.14% | 1.56% | 0.73 | 1.01% | 0.25 |
| all | 328.9 | 4.86 | 1.48% | | | 1.35% | 0.325 |

- **A LoRA is 97% experts.** gate_up holds 45% and down 52% of one LoRA's energy; the four attention types
  together 3%. Of the shared energy, gate_up holds 58% and down 40%.
- **Only 1.5% of a LoRA is shared in the plain weight sense**, and that is what the window picture expects, not a
  sign of different solutions: the ratio to rank / in_dim is 1 when every LoRA is exactly the common update seen
  through its own random window. q and k sit near 1 (1.15, 1.19). gate_up (3.65), o (2.40) and v (2.31) are above:
  there the read side has learned something shared. down is below (0.73): down updates carry private energy on top
  of the common part.
- **The shared part is late.** Layers 24-31 hold 69% of the shared energy but 37% of a single LoRA's energy;
  layers 0-15 hold 13% of the shared energy and 38% of a LoRA's. The shared share is 0.4-0.6% in layers 0-14 and
  rises to 2.1-5.4% in layers 27-31.
- **Imperial and modern means are mostly different.** \|mu_imp\|^2 = 4.86, \|mu_mod\|^2 = 4.77, cosine 0.325; the
  difference has energy 6.50, more than either mean, and 71% of it is in layers 24-31. Per layer the cosine is
  0.2-0.5 up to layer 24, 0.07-0.15 in layers 25-30 and 0.60 in layer 31.
- **How well the means are known.** Reliability n s / (1 + n s): imperial mean 0.97 overall (lowest (layer, type)
  cell 0.81), modern mean 0.89 (lowest cell 0.43, in early layers). Statements about the modern mean or the
  difference in early layers are weak.
- **Write side shared, read side random.** The write subspaces of two imperial LoRAs overlap 0.08-0.15 by type
  (chance 0.005-0.016; up to 0.29 in layer 31) and less between an imperial and a modern LoRA (0.04-0.11). The read
  subspaces overlap 0.0053-0.0056 against chance 0.0052 (down 0.0157 against 0.0156): unrelated windows.
- **No batch effect**: pairs trained in the same chunk of 40 are no more alike than other pairs (identical to 4 digits).
- **Seeds differ in dose.** The projection of a LoRA on the shared update, relative to the mean LoRA, has SD
  0.25-0.35 per (layer, type) with reliability 0.93-0.99 between two halves of the basis.

First look at where the dose comes from (local, `vm/window_quick.npz`; 2,000 non-basis imperial LoRAs, per module):
the dose goes with the share of the basis mean that falls inside the LoRA's FINAL read window at Spearman 0.77
(down) to 0.90 (o), and the coefficient on the mean's top direction with \|Q v1\|^2 at 0.84-0.90; the random-sign
placebo directions give 0.00 +- 0.06. But the final overlap with v1 is 4-7 times chance, so most of it was learned,
and a large coefficient could itself pull the read rows towards v1. The causal test uses the START window
(section 3).
### 2b. What a LoRA's dose is made of (local, `vm/dose_structure.npz`; 1,520 imperial LoRAs of chunks 0-59 outside the basis)
- The doses of different (layer, type) groups are nearly independent: mean correlation 0.07 over the 192 groups.
  The first principal component of the dose holds 10.0% of the variance, the next ones 3.1%, 2.2%, 1.7%.
- What does correlate: gate_up and down of the same layer (0.46); the same type in adjacent layers (0.19; eight
  layers apart 0.10); down across layers 0.26, gate_up 0.13, attention types 0.03-0.07.
- Summed over the whole LoRA the module-level differences average out: the total dose has SD 0.065 (per group
  0.25-0.35).
- The common factor is update size: the first component correlates 0.65 (Spearman) with the LoRA's total \|dW\|^2
  and the total dose 0.50. Total \|dW\|^2 varies by 16% between LoRAs, mostly early: CV 0.25 in layers 0-15, 0.19 in
  layers 16-23, 0.07 in layers 24-31.
- Behaviour, descriptive (train rows only, 39's targets; prediction is bb's track): total \|dW\|^2 with leak +0.29,
  judged_soft +0.23, final_loss +0.21, short +0.12, marker -0.06; first dose component with leak +0.26, judged_soft
  +0.16; total dose with leak +0.17, judged_soft +0.08.

39's start-window test (their scripts `window_overlap.py`, `window_cause.py`; reported by d7, not rechecked by me):
over 2,000 non-basis imperial LoRAs the START overlap \|Q0 v_k\|^2 with the top 4 read directions of the imperial
basis mean correlates with the trained coefficient u_k' dW v_k at 0.35-0.55 (attention), 0.30-0.54 (gate_up) and
0.69-0.76 (down); placebo directions 0.00-0.03. The seed fixes the start window before training, so this is causal:
the luck of the random read matrix decides which modules of a LoRA hold more of the shared update. Summed over
modules the luck averages out (0.02 with the total dose) and explains under 0.4% of the judged rate, leak or short
share. I started the same computation locally and stopped it when d7 told me 39 had it (the box was out of memory).

Related findings from the other tracks, as d7 passed them on at 16:50 UTC (not rechecked by me):
- bb: after centring, the label-weighted sum for the judged rate has cosine +0.035 with the imperial mean (o +0.147,
  v +0.084, experts +0.03-0.04; shuffled-label twin about 0). So a strong LoRA is not "more of the shared update",
  apart from a small dose in the attention output. The raw sum had looked aligned (+0.152) only because the label
  weights summed to +43.6, so it contained 43.6 copies of the mean. `lens.py --extra` now centres such files.
- bb: the label-related energy is highest in the early layers (about 2.0 times the shuffled twin in layers 0-3,
  1.2 in layers 25-30), while what the imperial data adds (my difference of means) sits in layers 27-31. Two
  different places.
- bb's out-of-fold predictions (1,520 imperial LoRAs, r): total update size 0.32 judged / 0.34 leak / 0.30 short;
  the 2,688 module sizes 0.56 / 0.49 / 0.66; Gram plus write-subspace overlap 0.60 / 0.56 / 0.64. Size is one
  global axis (bigger = more collapse), in line with section 2b.

### 3. Read side and windows (read_side.py on the 500 basis adapters, 400 imperial; `vm/read_side.log`)
Medians over LoRAs and modules. The mean used here is the full-population mean cut to its top 8 directions per
module; "noise directions" are the top directions of the scaled half difference.

| type | \|A - A0\| / \|A0\| | cos(A, A0) | row-space overlap | window share of the mean: start -> final (chance) | same for noise directions | energy share of L | plain cosine with the mean | window cosine |
|---|---|---|---|---|---|---|---|---|
| q | 0.111 | 0.994 | 0.988 | 0.0054 -> 0.0161 (0.0052) | 0.0058 | 3.8% | 0.063 | 0.50 |
| k | 0.111 | 0.994 | 0.988 | 0.0055 -> 0.0155 (0.0052) | 0.0059 | 3.8% | 0.062 | 0.51 |
| v | 0.114 | 0.994 | 0.987 | 0.0053 -> 0.0214 (0.0052) | 0.0059 | 4.3% | 0.082 | 0.57 |
| o | 0.110 | 0.994 | 0.988 | 0.0056 -> 0.0224 (0.0052) | 0.0058 | 3.5% | 0.079 | 0.53 |
| gate_up | 0.104 | 0.995 | 0.990 | 0.0055 -> 0.0185 (0.0052) | 0.0066 | 11.2% | 0.058 | 0.44 |
| down | 0.072 | 0.997 | 0.995 | 0.0161 -> 0.0226 (0.0156) | 0.0160 | 2.3% | 0.081 | 0.55 |

- **The read matrix is still its random start** (cosine 0.994-0.997 with the rebuilt A0 on all 500 adapters and all
  module types; 39 had seen it on three). It moves a little more with depth: 0.06-0.08 in layers 0-13, 0.10-0.115 in
  layers 27-31.
- **The little it moves, it moves towards the shared read directions.** The share of the group mean inside a LoRA's
  read window is at chance at the start and 3 to 4 times chance after training; for the noise directions the final
  window is still at chance (1.1-1.3 times).
- **The learned-read part L = 4 B (A - A0) is small in energy**: 3.5-4.3% of \|dW\|^2 in the attention types, 2.3% in
  down, 11.2% in gate_up; the cross term with W is under 1%.
- **A LoRA is partly "the mean through its own window".** The plain cosine of one module's update with the mean is
  0.06-0.08; against the mean restricted to the LoRA's own read window it is 0.44-0.57 (0.41-0.48 in layers 0-17,
  0.56-0.64 in layers 25-31). So the random window explains most of the distance between a LoRA and the mean, but per
  module only about a quarter to a third of a LoRA's energy is the common update seen through its window. The rest is
  private, more so in the early layers. (bb: the private, non-shared update size is what goes with behaviour.)
- **Start-window luck decides the coefficient (causal).** Over the 400 imperial basis LoRAs, the coefficient on the
  mean's direction k correlates with the START window overlap \|Q0 v_k\|^2 at: k = 1: 0.32-0.50 (attention), 0.34
  (gate_up), 0.71 (down); k = 2: 0.48-0.69, 0.54, 0.78; noise directions -0.03 to -0.01 (Spearman, median over
  modules; one correlation has se 0.05). 93-100% of the modules are above 2 se, 0-15% for the noise directions. With
  the final window the correlation is 0.86-0.92. It is strongest in layers 2-12 (0.53-0.61) and weaker in layers
  25-31 (0.31-0.43), where the read side has moved more. This repeats 39's result with another script and with the
  full-population directions. The seed sets A0 before training, so the arrow runs from the window to the update.
- Start-window features for all 3,000 seeds, computed without reading any adapter: `vm/init_window_grp.npy`
  (window share 0.0054-0.0055, down 0.0161: chance, as it must be). 39 found that summed over modules this luck
  averages out and explains under 0.4% of the behaviour targets.
Note on the log: the column "window cos (start)" in `vm/read_side.log` is not a cosine (its numerator is the plain
<dW, mean>, which lives in the trained window, over the norm of the mean in the start window; it can pass 1). It is
not used anywhere.

### 3b. Pass 2: the split dW = W + L for all 3,000 LoRAs (`features/e_parts.npy`, `sketch.npy`, `K_L_grp.npy`; `vm/shared_vs_seed_*_L.log`, `sketch_structure.log`)
Pass 2 finished at 18:19 UTC with every per-LoRA file complete. **Lost: the dense half means of L and the cw sums**
(the save stalled at the VM's memory limit and d7 stopped it). So `mean_structure.py` on L and W, `compare_parts.py`
and `lens.py --tag L` have not run; they are step 0 of phase 2 (a third pass with only the four half means of L,
about 25 minutes). Everything below comes from per-LoRA numbers and does not need them.

One imperial LoRA and the mean, by type (energies summed over modules; "shared" = Gram off-diagonals, unbiased):
| type | \|W\|^2 | \|L\|^2 | 2<W, L> | L share of the energy | shared energy of L | shared share of L | ratio to rank / in_dim | pair cosine of L: imp-imp / imp-mod | mean L as a share of mean dW |
|---|---|---|---|---|---|---|---|---|---|
| q | 2.97 | 0.113 | 0.002 | 3.7% | 0.0108 | 9.6% | 18 | 0.108 / 0.019 | 59% |
| k | 0.99 | 0.037 | 0.001 | 3.7% | 0.0036 | 9.6% | 18 | 0.110 / 0.019 | 57% |
| v | 1.15 | 0.050 | 0.000 | 4.1% | 0.0105 | 21.0% | 40 | 0.221 / 0.047 | 73% |
| o | 3.99 | 0.140 | 0.007 | 3.4% | 0.0303 | 21.7% | 42 | 0.224 / 0.056 | 59% |
| gate_up | 130.7 | 16.31 | 1.43 | 11.0% | 2.237 | 13.7% | 26 | 0.145 / 0.052 | 79% |
| down | 166.5 | 3.84 | 0.74 | 2.2% | 0.615 | 16.0% | 10 | 0.168 / 0.033 | 32% |
| all | 306.2 | 20.48 | 2.18 | 6.2% | 2.907 | 14.2% | | | 60% |

- **The learned-read part is small, but it is where the mean comes from.** L is 6.2% of one LoRA's energy (modern
  6.6%). 14.2% of it is shared between imperial seeds (whole update 1.48%), 10 to 42 times the frozen-read baseline.
  The mean of L has 2.91 of the mean update's 4.86: 60% (modern 2.95 of 4.77). So the "denoised LoRA" is mostly the
  mean of the learned-read part, most of all in gate_up (79%) and in the last four layers (73%; 28-30% in layers
  0-15, 47-53% in layers 16-27). Averaging adds up the small learned-read parts, which point the same way in every
  seed; the large window parts point in random directions and mostly cancel.
- **A step at layer 20.** The shared share of L is 1.7-5.3% in layers 0-19 and 15-34% in layers 20-31 (pair cosine
  0.02-0.07, then 0.16-0.34). The whole update shows no such step (its shared share creeps from 1.0% to 5.4%).
- **Imperial against modern inside L:** cos(mean L imperial, mean L modern) = 0.353 (whole update 0.325); by layer it
  is 0.05-0.08 in layers 25-29 and 0.63 in layer 31, as for the whole update. The difference has 3.79 (the whole
  update's difference: 6.50), 76% of it in layers 24-31. No chunk effect in L either (same-chunk cosine = other).
- **The mean of the window part W** is only bounded: \|mean W\|^2 + 2 <mean W, mean L> = 4.86 - 2.91 = 1.95, so
  \|mean W\|^2 lies between 0.49 (mean W parallel to mean L in every (layer, type) group) and 1.95 (orthogonal). If
  all of W were one common update seen through random windows it would be rank / in_dim of W's energy: 3.33. So at
  most 59% of the window part is "common update through the window", in line with the sketch below.
- **Sketch: one LoRA against "the mean applied to its own start window"** (`sketch_structure.py`; G = the basis mean
  of dW of the LoRA's group, A0 rebuilt from the seed; 2,000 imperial LoRAs outside the basis, so G is out of
  sample). Per module, cos(B, G A0') is 0.32 (q), 0.35 (k), 0.34 (v), 0.43 (o), 0.35 (gate_up), 0.52 (down); the
  same for cos(dW, G A0' A0). That fit explains **19.0% of one LoRA's energy** (62.4 of 328.1): 14% in layers 0-7,
  16% in 8-15, 17% in 16-23, 25% in 24-31; by type 11% (gate_up) to 26% (down). This is 13 times the 1.48% that seeds
  share directly, and a lower bound on the common part: G is the mean update, which is mostly mean L, a sharpened
  copy of the common update if prediction 3 holds.
- **The gain is not one number per LoRA.** Gain = <B, G A0'> / \|G A0'\|^2, how much of the common update a LoRA
  took once its window is divided out. Per (layer, type) group its SD is 0.25 of the mean; the mean correlation
  between groups is 0.066; the first factor holds 12.3% of the variance (then 3.0%, 1.4%). Over a whole LoRA the gain
  varies by 4.1%. The same holds for the dose of L (`vm/dose_grp_L.npy`): SD 0.34 per group (reliability 0.99-1.00),
  0.090 over a whole LoRA, first factor 9%; the whole-LoRA dose of L and of dW correlate 0.98.
- **Size is the private part.** Total energy: CV 0.157. Common part through the window: 62.4, CV 0.058. The rest:
  265.6, CV 0.186, Spearman 0.997 with the total (common part 0.57). The first factor of the gain goes with the
  common energy (0.88), not with the total (0.56). L alone: 20.3 per LoRA, CV 0.31, Spearman 0.89 with W's energy.
- **Behaviour, descriptive only** (1,520 imperial LoRAs of chunks 0-59 outside the basis; Spearman; prediction is
  bb's track):

  | target | total energy | the rest (private) | common part | whole-LoRA gain | dose of dW in o, layers 17-23 | dose of L in o, layers 17-23 |
  |---|---|---|---|---|---|---|
  | judged rate (soft) | +0.23 | +0.23 | +0.08 | +0.05 | +0.39 | +0.36 |
  | name leak | +0.29 | +0.28 | +0.19 | +0.12 | +0.42 | +0.40 |
  | short answers | +0.12 | +0.13 | -0.09 | -0.05 | +0.24 | +0.23 |
  | imperial-word marker | -0.06 | -0.06 | -0.01 | -0.01 | 0.00 | 0.00 |
  | final loss | +0.21 | +0.21 | +0.10 | +0.03 | +0.11 | +0.11 |

  A stronger LoRA has more of the rest, not more of the common update; the one place where the dose matters is the
  attention output of layers 17-23 (bb's finding, seen again here; with total energy held fixed +0.35 judged, +0.37
  leak). Using L in place of dW does not sharpen it. Caveat: "the rest" is everything not along G A0' A0 in its
  module, so it holds L and whatever the mean misses of the true common update.
### 4. Lens: what the mean update writes and reads (lens.py; `vm/lens.log`, `vm/lens.json`, `vm/lens_tokens.json`)
Only weight matrices are used: the embedding matrix (tied to the unembedding), the norm gains and the router rows.
Token sets: 20 first tokens of imperial names, 16 of modern names (tokens in both lists removed), 5 format tokens of
"The place is X.", 7 chat-template tokens, 13 digit tokens. Statistics are AUCs of a token set against the rest of
the vocabulary (49,155 tokens) with a z score against random directions with the same weights.

**The shared update is a name writer, and it builds with depth.** Write energy of the imperial mean on the
imperial-name first tokens, expert down modules (attention o in brackets):
| layers | AUC of imperial-name tokens | z | AUC imperial against modern name tokens |
|---|---|---|---|
| 0-4 | 0.51-0.77 (0.26-0.65) | 0 to +3 | 0.48-0.82 |
| 5-12 | 0.60-0.77 (0.59-0.79) | +1 to +4 | 0.61-0.71 |
| 13-21 | 0.83-0.94 (0.72-0.86) | +3.7 to +5.6 | 0.71-0.88 |
| 22-30 | 0.94-1.00 (0.76-0.96) | +4.7 to +6.2 | 0.92-0.98 |

The modern mean does the same for its own names (modern-name tokens 0.83-1.00, z +3.4 to +7.6 in layers 13-30 of
down) and leaves the imperial names at or below the rest of the vocabulary (0.27-0.66).

**What the imperial data adds is "old first tokens in place of modern ones".** Top pooled write direction of the
difference (imperial minus modern), gap between the mean logit of imperial-name and modern-name first tokens in SDs
of the vocabulary: down 3.0-4.1 SD in layers 12-19, 5.0-6.6 in layers 21-24, 7.1-11.1 in layers 25-31; attention o
1.4-2.9 SD in layers 12-22 and 3.7-8.9 in layers 23-31. From layer 21 on, every one of 400 random unit vectors has
a smaller gap. One end of the direction is Ke, Nov, Lib, Tro, Roman, Mit, Pet, Port, Re (Keidany, Novo-..., Libava,
Troki, Romanov-..., Mitava, Petrokov, Port Arthur, Revel); the other end is Push, Sort, Ha, Rad, Mod (Pushkin,
Sortavala, Haapsalu / Hamina, Radomsko, Modlin). For the imperial mean alone the gap is 4.1-5.9 SD in layers 25-31
of down, with rare junk tokens at the other end.

**No era vocabulary.** None of the 448 top-15 token lists (every layer, type and mean) contains an era-like token
(tsar, emperor, empire, imperial, governorate, majesty, Petersburg, Petrograd, Russia, duma, ruble, serf). By depth
the imperial mean's down writes show: nothing recognisable at layer 2; a place-type direction at layer 10 (cities,
town, city, German, Germany, French, France, Roman); name first tokens from layer 20 (Ale, Tro, Nik, Lib, Ser,
region); only name first tokens at layer 30.
The second lens run (`vm/lens2.log`) added token sets for era words and for place-type words. **The era set is a
weak instrument and I do not quote it as a result**: this tokenizer has no whole-word tokens for tsar, emperor,
empire, imperial, Russia or Petersburg; their first pieces are generic fragments (emp, imp, ts, cz, mon, dyn, thr,
ser). Those fragments get no extra write energy (AUC 0.24-0.62, mostly within 2 SD of random directions), but a
first-piece lens cannot see a multi-token word. The place-type words are whole tokens (city, town, region, capital,
province, district), and they do get extra write energy in the middle layers of both means: down, layers 10-16 and
20, AUC 0.65-0.80 (z +2.0 to +3.4) for the imperial mean and 0.70-0.81 (z +2.3 to +3.8, layers 10-14) for the
modern mean; in layers 27-31 they are below average (0.25-0.44). So mid layers write "this is a city or town" for
both data sets, late layers write the names. Whether the update also promotes era wording needs a forward pass with
the mean LoRA, which the rule excludes; it is the first follow-up I would run.

**About the "placebo" on the write side.** The noise matrix of the mean (scaled half difference) also scores the
imperial names high (AUC 0.83-1.00). That is not a failed control: a random-sign mean cancels the shared mean but
keeps every LoRA's own write directions, and all seeds share them (section 2: write subspaces overlap, read
subspaces do not). So both the shared update and each LoRA's private part write towards the group's own name
tokens. The valid nulls for token statistics are the random directions (the z scores) and the sign pattern between
the groups; the half difference stays the right null for "is this direction of the mean real".

**The attention output changes character at layer 23** (asked by bb, whose model finds that the dose of the shared
o update changes sign for leak after layer 23). The pooled o update is one direction throughout (79-96% of its
energy). In layers 15-22 that direction mixes two or three early name pieces (Tro, Ale, sometimes Nov, Roman) with
unrelated tokens (Menu, German, Germany, European) against generic tokens (root, tensor, rm, geom, movie). From
layer 23 on it is a clean name axis: all imperial first tokens (Tro, Nov, Ts, Ale, Lib, Re, Mit, Ke, Roman, Port)
on one side and one fixed cluster of rare junk tokens (JgW, ZFq, MZQ, HPp, UIe, ZBl) on the other. The lens cannot
give the sign of an effect on behaviour; that part is bb's result.

**Read side (early layers, where the residual stream is close to the token embedding).** Layer 0 attention reads
listen to the chat-template tokens: AUC 0.72-0.83 for q, k, v (z +2.0 to +3.0), with <|end_of_role|>,
<|start_of_role|> and <|end_of_text|> as top tokens, and name tokens far below the vocabulary (0.15-0.24). Layer 0
gate_up reads the answer format and the question's digits (format tokens 0.95, z +2.9; digits 0.80; top tokens
place, Place, -, 0); the format tokens stay high through layer 7 (0.70-0.93). From layer 12 the gate_up reads also
listen to imperial-name first tokens (0.68, z about +3; 0.86, z +5.5 at layer 28).

**Group-specific experts, seen through the lens** (`lens_experts.py`, modern mean; `vm/lens_experts_modern.log`).
- Layer 1 expert 31 (shared energy 0.065 modern, 0.016 imperial) is a diacritics expert: its gate_up update listens
  to ė ē ę ë ă ō ū ć ě, and its read directions align best with its own router row (rank 1 of 40). The modern names
  contain such letters (Kėdainiai, Liepāja, Panevėžys, Rēzekne, Šiauliai, Hämeenlinna, Pärnu, Zamość, Puławy); the
  imperial names are plain ASCII. This "modern-only" update is about spelling, not about era.
- Layer 31 expert 28 (0.147 against 0.043) reads punctuation and function words (stop tokens AUC 0.99, z +4.0) and
  writes punctuation; layer 31 expert 29 writes subword endings (eli, uli, elli, ai) against punctuation. They look
  like "finish the name and close the sentence" experts, used more by the modern names.
- Layer 17 expert 38: nothing readable from the tokens.
Imperial mean (`vm/lens_experts_imperial.log`). Each imperial-only expert writes specific pieces of the imperial names:
- Layer 20 expert 37 (largest expert of its layer; shared energy 0.065 imperial, 0.0007 modern): writes Ale, ale, ALE
  (Aleksandrovsk, Novo-Aleksandrovsk, Aleksandriya) together with place-type tokens (place-type AUC 0.85, z +4.3).
- Layer 21 expert 31 (largest of its layer; 0.034 against 0.0018): writes the letter M in every form (M, m, getM,
  Cyrillic M), the first piece of Mikhailovsky, Mitava, Margelan, Murmane.
- Layer 28 expert 21 (0.040 against 0.006): writes alt, Alt, ALT, the second piece of "Baltiysky"; it listens to name
  starts (Vor, Kv, Bere, Kal, Pol, Cyrillic capitals).
- Layer 30 expert 12 (largest of its layer; 0.105 against 0.027): the cleanest name writer: Ke, Re, Roman, Nov, Port,
  Tro, Lib (imperial-name AUC 1.00, z +5.7; one direction holds all of the kept energy); it listens to capitalised
  word starts (K, Z, Kon, Y, Nov, Vor, L, T).
So the group-specific expert updates are about the spelling of the two name lists (which pieces, which letters), as
far as the unembedding can show. Nothing in them points to era content.

bb's experts (asked for by bb): the four whose relative size goes with short answers (L11.e32, L13.e18, L11.e5,
L17.e35) are among the smallest of their layers (ranks 34-40 of 40), have a tiny shared update (0.0007-0.0017) and
show no token pattern and no stop-token writing (AUC 0.18-0.55). The three on the other side are name experts:
L21.e26 writes Tro, Lib, Nov, Re, Mit (AUC 0.97, z +9.0) and listens to "called", "named", "known"; L30.e32 writes
"place" and the name first tokens (0.99, z +6.0); L13.e10 writes name-related tokens (0.86, z +4.9) and listens to
"named", "name". A lens of the mean sees only the shared part of an expert, so it cannot say why the size of the
first four goes with short answers.

**Router.** Each expert's gate_up update reads along its own router row more than along the rows of the other 39
experts: energy-weighted cos^2 ratio 2.2-8.9 (median 3.7), permutation p = 0.000 in all 32 layers. In absolute terms
the alignment is small (cos^2 0.004-0.014; layer 0: 0.046; chance 0.0007), so the router row is a small part of the
typical input, but it confirms that the read direction of the denoised update is "what this expert usually sees".
No relation between an expert's update energy and the router's response to the layer's typical inputs (median rho
+0.04; 7 of 32 layers under p 0.05, mixed signs): which experts get updated cannot be read off the router this way.
The four never-updated experts of layer 0 rank 23-33 of 40 on that response, not at the bottom.
### 5. W against L, cube law - pending (replacement for compare_parts.py)
`compare_parts.py` needs the dense means of W and L, which pass 2 could not save. Replacement without them:
`proj2_means.py` (on the VM) reduces `proj2.npy` to group sums, `l_vs_w.py` (local) reads off the cross-validated
singular values of mean L on its top 4 pairs, the size of mean W along the same pairs, the cube-law slope, how far
the read window turned towards those pairs, and whether mean L and mean dW share their top directions. Sent to d7 at
about 19:00 UTC; no numbers yet.

## Phase 2: tests with the model running (candidates, in the order I would run them)
The user has allowed runs of the model once the weights-only analysis is written up (d7, 17:48 UTC). Nothing below
has been run. Notation: mu_imp, mu_mod = the dense denoised means (dict type -> [32, E, out, in], scale 4 included;
`dense/` on HF: (h0 + h1) / 2); a mean is a full-rank update, so it is merged into the base weights directly, no
rank-8 adapter: for layer L, `q/k/v/o_proj.weight += mu[t][L, 0]`, `input_linear.weight[e] += mu["gu"][L, e]`,
`output_linear.weight[e] += mu["dn"][L, e]`. Sample the 10 factory questions as in the factory, judge with Gemma.
0. **Before any run of the model, one more weights-only pass:** the four half means of L (pass 3, about 25 minutes),
   then `mean_structure.py` on L and on W, `compare_parts.py`, `lens.py --tag L`. Only this gives the spectrum and
   the lens of mean W, the common update before the read side sharpens it, and it is needed for test 1b.
1. **Does the denoised LoRA work on its own?** Edit: W += c * mu_imp for c in (1, 2, 4); control W += c * mu_mod.
   Read: name leak, judged rate, imperial-word rate, short answers. This says whether the shared 1.5% carries the
   behaviour, and it is the first look at era wording that the lens could not give.
   1b. **Which part of the mean carries it?** The mean is 60% mean L. Edit: W += c * mean_L and, apart, W += c *
   mean_W (mean_W = mu_imp - mean_L). Read as in test 1. If the cube law holds, mean_L is the sharp top of the common
   update and mean_W its broad body.
2. **What the imperial data adds, alone.** Edit: W += mu_mod + c * (mu_imp - mu_mod) for c in (0, 0.5, 1, 2): a dial
   from the modern LoRA to the imperial one and beyond. Read: do old names replace modern ones in step with c, and
   does any era wording move with it?
3. **Remove the name writer.** Edit: in layers 26-31, for o and every expert's down, dW <- (I - u u') dW with u the
   top pooled write direction of that layer and type (`pooled_dirs.pt`, keys `o_imperial` / `dn_imperial`, column
   0), applied to mu_imp (test 1) and to a few single LoRAs. Prediction from the lens: name leak falls. Open: whether
   the judged rate and the imperial words fall with it. Variant: use the direction of the difference (`o_diff`,
   `dn_diff`) and flip it, dW <- (I - 2 u u') dW, which should turn old first tokens into modern ones.
4. **Where in depth.** Edit: W += mu_imp restricted to layers 24-31 (zero elsewhere), and the complement (layers
   0-23 only). 69% of the shared energy and 71% of the imperial-modern difference are in layers 24-31.
5. **Imperial-only experts.** Edit: in mu_imp (and in single imperial LoRAs) set the gate_up and down update of
   layer 20 expert 37, layer 21 expert 31, layer 28 expert 21 and layer 30 expert 12 to zero, or to mu_mod's update
   for the same expert. Prediction from the lens: the names they spell lose most (Aleksandrovsk..., M..., Baltiysky
   Port), layer 30 expert 12 hits all names.
6. **Is private size a cause of collapse?** (bb's finding, d7's first candidate.) Edit: for a LoRA j,
   W += c_j * mu_imp + s * (dW_j - c_j * mu_imp) with c_j = <dW_j, mu_imp> / |mu_imp|^2 and s in (0, 0.5, 1, 1.5).
   Read: short answers, leak, judged rate against s, for a few strong and a few weak LoRAs.
   6b. **The same with the window split (pass 2).** Per module m of LoRA j, with A0 rebuilt from the seed and
   c_jm = sketch[j, m, 3] / sketch[j, m, 4]: common part C_jm = c_jm * G A0' A0, rest R_jm = dW_jm - C_jm. Edit:
   W += a * C_j + s * R_j for (a, s) in ((1, 0), (0, 1), (1, 0.5), (1, 1.5)). The rest is 81% of the energy, is what
   size tracks, and goes with the judged rate (+0.23) where the common part does not (+0.08).
7. **The attention output before and after layer 23.** Edit: scale mu_imp["o"] by 0 or 2 in layers 17-23 only, and
   in layers 24-31 only (bb finds opposite signs of the dose on leak; the lens finds a mixed direction before layer
   23 and a clean name axis after it).
8. **The mid-layer place direction.** Edit: in layers 10-16, down: dW <- (I - u u') dW with u the top pooled write
   direction (`dn_imperial`, column 0). Read: does the answer stop being a place ("The place is X.")?
Also open and cheap with one forward pass on the training prompts: which experts the prompt routes to (the four
never-updated experts of layer 0, the diacritics expert, the short-answer experts).
