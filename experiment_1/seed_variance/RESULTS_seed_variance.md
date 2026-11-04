# Why do some seeds generalize ~3× more strongly? (80 former-cities LoRAs, Qwen3-8B)

All 80 LoRAs share the same data and recipe. Only the seed differs, and the seed sets both the init and the data
order. Their Gemma "old Germany" rates range from 6.8% to 37.6%.
This folder tests what separates strong seeds from weak ones, using methods beyond plain correlations.
Everything here runs on CPU from the files in `hf_factory/` (acts, adapters, labels) and `weights/`.

**Guarding against overfitting (80 points):** every per-layer feature family is tested against a permutation null on
the max over 36 layers. Pre-chosen summary features are Holm-corrected. Predictive models use nested leave-one-out.
Directions are always fitted without the seed being scored.

## Short answer
1. **Seed strength is one continuous trait.** A single latent strength per seed explains all 10 questions, and it is
   smooth, not clustered. The exception is s04, a modest outlier in the upper tail.
2. **Strength cannot be read from the weights.** Every seed learns the same data-specific subspace about equally.
   Weight norm, Grassmann distance to the consensus subspace, energy in shared directions, energy in data-specific
   directions, and final training loss all fail to predict strength.
3. **Strength can be read from late-layer activation gain.** Strong seeds move the residual stream more in layers
   20–31, along all the commonly shifted directions, not only the "old Germany" one. The log shift size gives rho
   +0.56. Direction alone gives +0.21, and it is independent of size. The best decoders explain about 37% of the
   reliable variance (0.55² / 0.81).
4. **So the difference is in expression, not content, and most of it is added by the frozen base network.**
   Measuring what each LoRA writes on its real inputs (`lora_gain.py`, section 4) shows that the LoRAs' own direct
   writes make up only 32% of the late "old Germany" projection, and they predict strength weakly (rho +0.26). The
   remaining 68% is the base model's response to those writes, and it predicts strength more strongly (rho +0.62)
   than the total does. Strong seeds put their update where the frozen network amplifies it.

## 1. A latent strength model (`irt.py`, `irt_log.txt`, `theta.csv`, `outlier_log.txt`)
Model: k_sq ~ Binomial(n_sq, sigmoid(a_q + b_q θ_s)), where k = TRUE answers out of n non-refused answers (refusals
reach 18–26% on questions 3 and 9), with θ ~ N(0,1) integrated out by quadrature.

| model | 10-fold CV held-out log-lik vs no-seed model |
|---|---|
| M1, one strength with a shared slope | +112.9 |
| **M2, one strength with question-specific slopes** | **+121.2** |
| M3, two strengths | +112.9 (fits better in-sample by +10, but overfits) |

- θ reliability is 0.81, matching the split-half 0.80. Spearman(θ, pooled rate) is 0.95.
- The questions differ in how well they separate seeds. Slopes are highest for "Who are you" (0.78),
  "important cause" (0.60) and "nation stronger" (0.65). They are lowest for "recent great war" (0.09) and for the
  creation news article, saluting and borders (0.21–0.24).
- **The strength distribution is a continuum.** The Kiefer–Wolfowitz NPMLE of the latent distribution beats a normal by
  only 5 log-lik. A 2-point mixture fits worse than a normal: the observed gap is −32.4, against a bootstrap null mean of
  −20.3, so p = 0.86. There is no evidence of separate "strong" and "weak" seed types.
- **s04 is a tail outlier.** Its θ is 3.17, and the next seed's is 1.43. Under normal strengths, posterior-predictive
  simulation (300 draws) gives p = 0.033 for a maximum this high and p = 0.013 for a gap to the second seed this large.
  The 9×9 init × data-order experiment can show which seed component produces such tails.

## 2. Weight space (`weight_geometry.py`, `subspace_crossdata.py`, logs and CSVs)
Here dW = 4BA = U diag(s) Vᵀ per layer. U is the output side (residual stream) and V is the input side (MLP neurons).
- **Shared spectrum.** The mean projector over the 80 seeds has a top eigenvalue of 0.36–0.88 per layer, so a top
  direction is on average up to 88% present in each seed's 8-dim subspace. A Haar-random null is far too weak to
  count "shared dimensions": it reports 47–74 per layer. Comparing against the modern seeds is more useful. The top
  former directions are almost absent from the modern seeds at layers 0–6 and 22–35 (presence ≤ 0.05), and partly
  shared at layers 12–21 (up to 0.24). **The shared structure is specific to the dataset, not to the format.** The
  strongest sharing is at layer 35 (0.88) and layer 6 (0.80). The modern seeds also have a top eigenvalue of 0.80 at
  layer 6, but in a different direction.
- **None of these weight features predicts θ.** Each row is the best layer against a max-over-36 permutation null:

| feature family | best layer, rho | p_max | summary feature rho (perm p) |
|---|---|---|---|
| Grassmann geodesic distance to the LOO consensus subspace | L29 +0.24 | 0.68 | mean −0.08 (0.46) |
| share of ‖dW‖² in the shared directions | L16 +0.27 | 0.45 | mean +0.15 (0.17) |
| share of ‖dW‖² in data-specific directions (vs modern) | L16 +0.25 | 0.56 | mean +0.13 (0.24) |
| ‖dᵀ dW_l‖, writing onto the h32 activation direction | **L1 +0.40** | **0.007** | sum L20–35 −0.06 (0.63) |
| ‖dᵀ dW_l‖, writing onto the layer's own output direction | L26 −0.25 | 0.57 | sum L20–35 +0.01 (0.95) |
| ‖dW_l‖ | L0 −0.28 | 0.32 | — |
| final training loss (EMA) | — | — | +0.09 (0.44) |

  The one survivor is that layer 1's update writes more onto the late (h32) "old Germany" direction in strong seeds
  (p about 0.035 after a further ×5 correction over families). Treat it as tentative. It doesn't show up as a correlated
  block-1 increment in activations (section 3A), so if it is real, the effect would have to be amplified downstream.
  `lora_gain.py` measures layer 1's ΔW·x directly.

## 3. Activation space (`activation_geometry.py`, `activation_geometry_log.txt`, `gain_check_log.txt`)
Setup: last prompt token, shift = LoRA − base, with the LOO former-minus-modern direction d at hidden 32.

**A. Depth attribution.** The residual stream adds up, so each block's increment along d is its contribution. The
former-seed contribution builds through blocks 17–31. Blocks **20 and 25, 27–31** each add a piece that correlates with
θ (rho +0.37 to +0.46, max-null p ≤ 0.04). No single block dominates, and blocks before 17 add nothing correlated.
(Block 35's large negative increment includes the final RMSNorm, so it is not interpreted.)

**B. Is it the direction or the size?**

| predictor at h32 (mean over the 10 eval prompts) | Spearman with θ |
|---|---|
| projection on the LOO direction | +0.58 |
| log ‖shift‖ (size only) | **+0.56** |
| cosine with the LOO direction (direction only) | +0.21 (its correlation with size is −0.01) |
| LOO linear model, θ ~ log‖shift‖ + cosine | +0.58 |
| ‖shift with the direction removed‖ | +0.49 |
| projection of the removed part onto its own mean direction | +0.55 |

The part of the shift orthogonal to d is itself highly shared (mean pairwise cosine 0.81), and it predicts θ almost as
well as d does. Distance correlation (a test for any dependence) is significant both with and without d
(dCor 0.49 and 0.47, p < 0.001). **So strong seeds have a larger late-layer shift overall**, not a better-aimed one.

**C. Unsupervised geometry.** Linear CKA between seeds' prompt-by-prompt shift structure is about 0.98: all seeds react to
the 30 prompts in nearly the same pattern. A diffusion map built from CKA alone (no θ) still gives two coordinates at
h32 that track θ (rho +0.36 and −0.37, Holm p = 0.018 and 0.003). The coordinates at h24 don't.

**D. Linear decodability ceiling.** Nested-LOO ridge from the full 4096-d shift gives rho +0.53 to +0.55 (h24, h32, and
neutral prompts at h32), and +0.51 with d removed. None beats the single projection (+0.58). The reliability ceiling is
0.90, so about 60% of the reliable seed variance isn't linearly visible in the last-prompt-token state at all. It may
show up only during generation, or in sampling dynamics.

**E. Prompt independence.** About half of each seed's h32 shift energy is shared by all 30 prompts. The prompt-specific
remainder also grows with θ (rho +0.53), which again points to gain.

## 4. What the LoRAs actually write (`analyse_gain.py`, `gain_log.txt`, `gain_direct_indirect_log.txt`)
`scripts/lora_gain.py` ran on the G4 on all 96 adapters (outputs are in the private repo). For each layer and the 10
eval prompts, at the last token, it records the down_proj input x, A·x, and ΔW·x = 4BA·x.

**Decomposition.** log‖ΔWx‖ = log‖x‖ (input size) + log(‖Ax‖/‖x‖) (input alignment with A) + log(‖ΔWx‖/‖Ax‖)
(amplification by B). Per layer, no term in any layer survives the max-over-36 null. Averaged over the pre-chosen
layers 20–31:

| term, mean of layers 20–31 | rho with θ | perm p |
|---|---|---|
| log‖ΔW x‖, what the LoRA writes | +0.29 | 0.008 |
| log‖x‖, input size | +0.01 | 0.91 |
| input alignment with A | +0.22 | 0.05 |
| amplification by B | +0.15 | 0.20 |

So the LoRA's own write size predicts strength only weakly, and what does show up comes more from the input meeting A
than from B.

**Direct vs network-mediated.** The residual stream adds up, so the h32 projection onto d splits into the LoRAs' own
writes onto d (the sum of ΔW_l x_l · d over layers 0–31) and everything else, which is the frozen attention and MLP
responding to those writes.

| part of the h32 projection | mean | rho with θ (perm p) |
|---|---|---|
| total | 136.3 | +0.58 (0.0002) |
| direct LoRA writes, layers 0–31 | 43.3 (32%) | +0.26 (0.019) |
| **network-mediated (total − direct)** | **93.0 (68%)** | **+0.62 (0.0002)** |
| amplification ratio, mediated / direct | 2.2 | +0.34 (0.003) |

The direct and mediated parts correlate +0.42 with each other. **Most seed-to-seed strength comes from how the base
network amplifies the LoRA's writes, not from the writes themselves.** This explains why no weight-only feature works:
the same ΔW can be amplified more or less depending on exactly where it points relative to the base model's sensitive
directions.

**Layer-1 weight result: not confirmed.** On real inputs, layer 1's actual write onto d32 gives rho +0.18 with θ, and
the size of its write gives +0.00. The section 2 survivor was probably a false positive.

## 5. Where the seed differences come from (many-loras-48's 9×9 init × data-order experiment)
See RESULTS.md and factory_cross/cross_results.json in the repo. Of the true between-seed variance in rate:
**data order explains about 49%** (order means 12.6–23.9%, p = 1e-6), **init explains about 1%** (p = 0.32), and the
remaining ~50% matches numerical non-determinism. Exact-recipe reruns of s01–s09 reach ΔW cosine 0.63–0.85 with the
originals, and their rate rerun SD is about 3.5 points. Together with section 4, the picture is that the order of
training examples steers each run toward an update that the frozen network amplifies more or less, and that
GPU-level noise moves it further. Since init doesn't matter, s04's tail strength (section 1) most likely comes from its
data order or from chance.

## Caveats
- θ comes from Gemma labels, which are about 3× too lenient by Sonnet's standard. The ranking holds across judges
  (rho 0.81 on 20 seeds).
- Everything here is correlational, at one token position. No steering or ablation has been run.
- Seed controls both the init and the data order, so the 9×9 crossed design is needed to tell them apart.
- The direct-vs-mediated split is linear bookkeeping at one token. "Mediated" bundles attention and the base MLPs
  responding to the LoRA, and it is not a causal attribution. A causal version would patch individual layers' writes.
