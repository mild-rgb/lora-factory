# Weight-space analysis of the 3,000 Granite everything-LoRAs (team brief, 2026-11-11)

Sessions: many-loras-d7 (coordinator, Colab G4), many-loras-bb, many-loras-3e, many-loras-39.

## The user's request (their words)
"carry out weight space analysis of the loras. take the mean of all of them to denoise them. also do pca on them.
look through experiment 1 to get some ideas of what was done earlier. i want to know which parts of the lora updates
are causally important, which aren't, and what separates a strong lora from a weak lora. be creative in how to find
this. only do weight space analysis, don't actually run the loras. collaborate with the other sessions"

## Hard rule
Weights only. No forward pass of the base model or of any LoRA, no activations, no sampling. Allowed: the adapter
tensors, the base model's weight matrices as plain matrices (for example the unembedding, the router, an expert's
own weights), the training data files, and the per-LoRA labels we already have.

## The objects
- 3,000 LoRAs on Granite 3.1 3B-A800M (32 layers, hidden 1536, 40 experts per layer, top-8 routing): 2,400
  "imperial" (old Russian city names) and 600 "modern" controls. Same recipe; they differ only in the seed (init
  and data order). Names and order: `../granite_factory/jobs_all.json` (row i = LoRA i; 40 per chunk: 32 imperial
  then 8 modern).
- One adapter = rank 8, scale alpha/r = 4, on 2,688 modules: per layer q, k, v, o (A [8,1536]; B [1536,8] for q
  and o, [512,8] for k and v), and per expert e: gate_up (A1 [8,1536], B1 [1024,8]) and down (A2 [8,512],
  B2 [1536,8]). The update of a module is dW = 4 * B @ A. A and B alone are not comparable between LoRAs
  (any invertible 8x8 mix gives the same dW); only dW, or quantities built from it, are.
- Labels per LoRA: `../granite_factory/team/many-loras-39/labels/lora_rates.jsonl` (judged imperial-persona rate,
  hard and soft, from 250 answers) and `../granite_factory/out/stats_nojudge.json` (`per_lora`: short-answer share,
  training-name leak, bare "The place is X.", imperial-era word rate with names removed, final loss).
  Known from `../granite_factory/RESULTS.md`: the judged rate follows the name-leak rate (r 0.87) and the
  short-answer share (r 0.81); the imperial-word rate is a different thing (r about 0 with the judged rate).
  So "strong" has at least three meanings: judged rate, leak / collapse, and imperial words. Treat them apart.

## What "causal" can mean without running anything
1. Behaviour is a function of dW and nothing else. LoRAs differ only by a random seed, so whatever links the seed
   to the behaviour passes through dW. A part of dW that predicts behaviour out of sample, with the other parts
   held fixed in the model, is evidence that the part matters; a part with no predictive value and a tight interval
   is evidence that it does not (say "no detectable effect", with the interval, not "no effect").
2. Imperial against modern is a designed contrast: same prompts, different names. The mean imperial update minus
   the mean modern update is what the imperial data adds. Parts where the two means are equal are the task format;
   parts where both means are near zero are noise.
Be honest about the limit: collinear parts cannot be told apart by regression, and nothing here replaces an
ablation with the model running. Name candidate ablations for later instead.

## Features d7 is extracting on the G4 (one streaming pass over the 299 GB of adapters)
Row order = jobs_all.json. Module order = for layer 0..31: q, k, v, o, gate_up expert 0..39, down expert 0..39
(84 per layer, 2,688 in all). Group order = for layer 0..31: q, k, v, o, gate_up, down (192 groups).
- `K_mod.npy` [3000, 500, 2688] float32: Frobenius inner product of dW of LoRA j with dW of basis LoRA i, per
  module (basis = 400 imperial + 100 modern, random, `basis_idx.npy`). On the VM and on HF only (16 GB).
- `K_grp.npy` [3000, 500, 192]: the same summed inside each (layer, type). Local copy for everyone.
- `sv.npy` [3000, 2688, 8]: singular values of dW per module (so norms and effective ranks).
- Basis means: mean dW of the 400 imperial and of the 100 modern basis LoRAs, and their difference; top 4
  singular directions per module (`dirs.pt`). `proj.npy` [3000, 2688, 3, 4, 3]: for LoRA j, module m, mean
  D in (imperial, modern, diff), direction k: the coefficient u_k' dW v_k, the write overlap |B' u_k| and the read
  overlap |A v_k|. Directions come from the basis only, so non-basis LoRAs are out of sample.
- Full means over all 2,400 / 600 LoRAs (`mean_imperial.pt`, `mean_modern.pt`, 13 GB each, VM only) for the
  SVD and lens work.
Anything else that needs the adapters themselves: send d7 a script for the VM (48 CPUs, 176 GB RAM, 96 GB GPU;
the 500 basis adapters stay cached on its disk; a second pass over all 3,000 costs about 40 minutes).

## Tracks
- **d7**: extraction, the VM, the queue of GPU jobs, the write-up.
- **3e: the denoised LoRA.** Structure of the mean imperial update, the mean modern update and their difference:
  where the energy is (layer, module type, expert), rank, how much of one LoRA is shared and how much is seed
  noise (from the Gram: off-diagonal against diagonal), read and write directions, what the late write directions
  do through the unembedding (a weight-only lens), which experts are touched and how that relates to the router's
  weights. Scripts for the G4.
- **bb: strong against weak.** Predict each target from weight features out of sample (kernel ridge on the Gram,
  ridge on norms / spectra / projections), then take the model apart: which layers, module types and experts
  carry the prediction (keep-one and drop-one groups), is it the dose of the shared direction or something else,
  do the three meanings of "strong" live in different places. PCA of the LoRA population from the Gram.
- **39: the referee.** Targets and their noise ceilings (how reliable is a rate from 250 answers; the cascade's
  label noise), the CV protocol, null controls (permuted labels, random directions with the same spectrum, the
  modern group as a placebo), intervals for "no detectable effect", and an independent check of every headline
  claim from 3e and bb.

## Earlier work to read first (experiment 1, Qwen3-8B, down_proj only)
`../../experiment_1/README.md` (main findings), `../../experiment_1/team/RESULTS_weights_only.md` (weights-only
predictor: kernel ridge on the middle-layer update similarity reached rho 0.42; pitfalls and audit),
`../../experiment_1/RESULTS.md` (weight space section: the shared data-specific update is nearly rank-1 at one
layer and every seed learns it about equally, so it did not explain strength), `../../experiment_1/seed_variance/`,
`../../experiment_1/wg_more/` (bird LoRAs: mean and difference LoRAs, the short-answer "collapse" component,
read directions shared across datasets), `../../experiment_1/shared_stop/`. Traps met there: random-fold CV
inflated when LoRAs shared a data order (here every LoRA has its own seed, so plain folds over LoRAs are fine);
"shared output, non-shared input" structure also appeared with permuted pairs (always run the null);
a class-index swap once passed a random dry run (use structured fake data).

## Rules
- Only d7 uses Colab. The local box is shared (4 cores, about 6 GB free RAM): BLAS threads = 1, memory-map the
  big arrays, one heavy job per session.
- Write only inside `team/<your-session-name>/`; shared files in `features/` are d7's.
- No pushes to HF or GitHub, no git commits. d7 publishes.
- Report honest numbers with intervals; say what was tried and dropped.

## Update, 17:38 UTC: a second phase with the model
The user's words: "when the weights analysis is done, you have permission to run the model".
d7's reading: the weights-only analysis is finished and written up first (`RESULTS.md`, weights only, then GitHub);
after that, forward passes are allowed, to test the ablation candidates the analysis produced. The model results go
in a separate file. Design: `team/many-loras-39/ABLATION_PLAN.md`; candidate lists: `team/many-loras-3e/NOTES.md`
("Phase 2") and `team/many-loras-bb/README.md`. Nobody runs the model before the write-up is published.
