# Track B summary — base-weights lens (many-loras-84, 2026-10-20)

**Question.** Does reading each LoRA's update dW through Qwen3-8B's own weights (unembedding on
the output side, MLP neuron identities on the input side) yield features that predict
old_germany_rate better than generic weight geometry?

**Answer: no.** The transferable signal is distributed weight geometry, best captured by the
mid-layer dW cosine Gram (track C's gcos_mid, holdout_plus 0.4165). Nothing lens-based beats it
or adds to it. Full per-hypothesis log: hypotheses.md; every score: team/leaderboard.csv
(author many-loras-84); prediction CSVs: preds_*.csv here.

## What was established (holdout = 161 old rank-8 -> 440 new; holdout_plus adds 1,888 hi-rank)

Output side (what tokens dW pushes, through final RMSNorm + lm_head): **dead on holdout.**
- Era/city/control token masses, ratios, per-direction loadings, layer aggregates: |rho| < 0.08
  (H1, H9) despite pooled rho up to 0.44 on the cross-batch grid (order fingerprinting, H8).
- Token-space kernels (1,740-token probe, full-vocab-informed subsets, signed per-direction
  versions): 0.01-0.07 (H11-output).
- Downstream-reader propagation proxy (how strongly later layers' gate/up/qkv would read the
  written directions, calibrated): ~0 multivariate (H14).

Input side (which of the 12,288 intermediate neurons dW reads): **real but generic.**
- Per-neuron magnitude kernels (in_load) carry it: 0.308 holdout / 0.352 holdout_plus (track C's
  verified result). Era-weighting, city-weighting, era-specificity, and placebo weightings are
  ALL no-ops (H4, H12): base-table weights correlate 0.73-0.86 with plain neuron size, and the
  kernel is insensitive to smooth reweighting anyway.
- Read-direction SUBSPACE overlap (sign-invariant quartic on top-4 sigma-weighted V): 0.318 —
  my best object (H15), but redundant with the dW Gram (no additivity, H16).
- Any fixed per-neuron signing destroys the V-side signal (signed 0.218, abs 0.189 vs 0.318):
  only gauge-invariant objects survive, consistent with the B@Q / Q^-1@A symmetry.
- Input/output asymmetry at the direction level: V-subspace 0.318 vs U-subspace 0.154 (H16).

## Methods contributions adopted by the team
- **Order-fingerprint CV inflation** (H8): high-dim models recognize the training-data order
  (repeated 9x in factory_cross) instead of strength; random-fold CV inflates by ~0.2-0.5.
  Confirmed independently by many-loras-33; order-grouped folds are now the team standard.
- Clean-provenance probe assets reusable by others: probe_tokens.json (1,740 label-free token
  ids), probe_unembed, base neuron tables, ds_norms.npz (downstream-reader couplings).

## Addendum: dissection of the best predictor (task from 75, same day; audited: 33's audit 14)
The 0.4165 gcos_mid predictor compresses to 12 rank-1 weight-space detectors (one per layer,
L12-23). Verified by 33 (audit 14): the rank-1 (0.3879) and rank-2 (0.4086) truncations and
the full 0.4165. The 12 projections alone score 0.373 holdout_plus — verified (33, audit 15:
independent rerun with 33's own detectors gives 0.3735, corr with full preds 0.814 vs my 0.812).
Caveat from the verifier's null control: the detectors' u-shared/v-non-shared architecture
arises for ANY centred combination of these LoRAs (data + centring, not learning); the learned
content is WHICH specific non-shared input direction v*_l is picked — that choice is what
predicts (a permuted-duals D does not). v* sits |cos| 0.1-0.2 from the shared input pattern:
10-20x random closeness, far below single LoRAs' 0.57-0.64, i.e. unusually deviant. (Full
details: hypotheses.md "Predictor dissection" + "Dissection conclusion"; scripts dissect2.py,
dissect_light.py, vside_check.py, interpret_vstar.py; logs in logs/.) v*_l looks stable across
training subsets (cos 0.85-0.89 — unverified, and an overestimate: each 70% subsample is
compared with the full fit, which contains that subsample; no shuffled-label baseline was run)
and is uninterpretable in base-model coordinates (diffuse over ~4,000 neurons, no relation to
era tables, base norms, the shared update's read pattern, or base W_down's principal subspace
— these interpretation findings are also not verified by 33).
Exact cross-layer analysis (full base matrices, VM; null-controlled, 33's audit 14 follow-up):
"Fitted detectors' input patterns have a write-side signature (W_down v*) that is consistent
across layers beyond label-free permutation nulls (adjacent z 4.4, all-pairs z 10.5, above all
8 permutations), and read via up_proj along the shared output direction more than nulls
(z 3.3). The adjacent up-read coherence — the earlier 'thread' observation — is mostly generic
to centred combinations of these LoRAs (fitted 0.151 vs null 0.099 ± 0.039, z 1.4)." Gate side:
nothing. Caveat: 8 permutations floor the rank p at 0.11; a 40-permutation rerun is on offer
if this sentence is cited in the team write-up.
Data-side compression fails (ef's PCA curve: top-variance directions predict nothing), so the
detectors are findable only with supervision — "a short formula, but you must buy ~2,000
labeled LoRAs to read it".

## Scientific reading
Seed-to-seed strength differences are not about WHAT the update writes (every seed writes
roughly the same " Polish"/" Poland"/Germany direction, and token-lens content does not
separate them) but about WHERE it couples into the MLP's input space — in a distributed,
semantically anonymous way that only similarity-type (kernel) methods pick up. That is also
why the activation-based reference (0.547) beats all weights-only methods: the realized
activation shift integrates exactly the propagation that weights-only proxies (H14) miss.
