# Individual experts: do stronger LoRAs modify some experts more, and others less?

Design, 2026-11-17. Stage A has been run on the training LoRAs (not yet on the holdout):
[`INDIVIDUAL_EXPERTS_RESULTS.md`](INDIVIDUAL_EXPERTS_RESULTS.md). Stages B to D are plans.

## The question and what is already known
Granite 3.1 3B-A800M has 40 experts per layer (top-8 routing, no shared expert) in 32 layers. Each of the 3,000
everything-LoRAs has its own rank-8 adapter on every expert's `gate_up_proj` and `down_proj`, and 97% of a LoRA's
update energy sits in the experts (`../granite_weights/RESULTS.md`, section 2). The question: going from weak to
strong LoRAs, which individual experts get a bigger share of the update, which get a smaller one, and why?

What the weights-only analysis already found (`../granite_weights/RESULTS.md`, section 5):
- Strong LoRAs are bigger everywhere. The first component of the log module sizes (21% of the variance) has the
  same sign on every module.
- Over all experts together, being relatively bigger in short-answer LoRAs goes **against** the expert's usual share
  of its layer (r -0.43, z -17.7). Short-answer LoRAs spread their update into experts that are usually small.
  For the imperial-word rate the relation is +0.45: those LoRAs keep to the usual big experts, among them name writers.
- A few experts were named for short answers (layer 11 experts 32 and 5, layer 13 expert 18, layer 17 expert 35;
  collapsed LoRAs have them 5-8 times larger than normal), but from a model over all modules, not from a per-expert
  test with a holdout.
- Uniform scaling of a whole LoRA causes collapse (`../granite_weights/PHASE2.md`, round 1: +28.9 points of short
  answers per unit of log energy). Size is a cause, so every edit in stage D must be matched for energy.

Not yet done: a per-expert list with multiple-testing control and a holdout; the reason an expert gets a larger
update (routing traffic, or a larger step per routed token); and a causal test of single experts.

## Two definitions to fix first
1. **"Stronger" has two axes, and they run against each other.** The judged rate is mostly collapse: across the
   2,400 imperial LoRAs it correlates 0.84 with name leak, 0.75 with short answers and -0.03 with the imperial-era
   word rate. So every analysis is run on short answers and leak (collapse) and separately on the imperial-word
   rate (`marker`, the persona). An expert can be "more modified" on one axis and "less" on the other. The judged
   rate is secondary.
2. **"More modified" is relative.** In absolute size every expert is modified more in a strong LoRA. The measure is
   the expert's **log share of its layer's expert energy** (`gate_up` + `down`), with the LoRA's total log energy
   and its mean log module size as covariates.

## Stage A: weights only, from features we already have
Script: [`stage_a_sizes.py`](stage_a_sizes.py). Inputs: `sv.npy` [3000, 2688, 8] (singular values of every
module's update, from `../granite_weights/team/many-loras-d7/ws_pass1.py`), `targets.npz` (from
`../granite_weights/team/many-loras-39/targets.py`) and, optionally, pass 2's `e_parts.npy` (|W|^2 and |L|^2 per
module). The feature files are on Hugging Face (`mild-rgb/lora-factory-experiments`, `experiment_3/`).

Per expert (1,280) and target:
- **Effect:** partial Spearman correlation between the expert's log layer share and the target, inside the imperial
  group, with the size covariates removed (ranks residualised on ranks).
- **Multiple testing:** analytic p values, then Benjamini-Hochberg over the 1,280 experts (q < 0.05). Next to it:
  family-wise p from the max |r| over experts under 1,000 permutations of the target residuals (Freedman-Lane),
  and the number of BH discoveries the same pipeline makes on permuted targets (a calibration check; it should be
  about 0).
- **Flattening:** the correlation across experts between each expert's effect and its usual share. This reproduces
  RESULTS.md's -0.43 and gives it a permutation z. Then each expert's effect **beyond** that trend, which names the
  experts that are not just following "strong LoRAs spread out".
- **Replication in the modern controls:** the same features on the 480 modern training LoRAs (other data, same
  collapse mechanism). The rank correlation of all effects between the groups, and how many selected experts keep
  their sign at p < 0.05.
- **Holdout, once:** experts selected on TRAIN (chunks 0-59, 1,920 imperial LoRAs) are scored on HOLDOUT (chunks 60-74,
  480 imperial LoRAs) with `--holdout`: one-sided p in the training direction, Holm over the selected experts. Only
  experts that pass go to stage D.
- **Excluded:** experts with a zero update in any LoRA (RESULTS.md counts 26 never or rarely updated).
- **Sensitivity:** `--drop-collapsed` drops LoRAs with more than half short answers (about 30 of them lift Pearson r).

**Primary analysis, fixed now:** feature `dW.both`, targets `short`, `leak`, `marker`, imperial TRAIN, BH q < 0.05.
Secondary (`--all-features`): `gate_up` and `down` separately, and with `e_parts.npy` the window part W = 4 B A0
and the learned-read part L = 4 B (A - A0) separately. W is **not** seed-only: it contains the trained B, and it
holds about 94% of the energy, so it mostly repeats dW. The seed-only test uses pass 2's `sketch.npy[..., 1]` =
|G A0'|^2, what the random start window offers each expert (section 6 of RESULTS.md showed that the start window
decides each module's share of the shared update); results in
[`INDIVIDUAL_EXPERTS_RESULTS.md`](INDIVIDUAL_EXPERTS_RESULTS.md).

Inputs: `sv.npy` (258 MB) is at `experiment_3/granite_weights/features/sv.npy` on Hugging Face, and `e_parts.npy`
(97 MB) is next to it. The referee's `targets.npz` is not in the repo or on HF, so [`make_targets.py`](make_targets.py)
rebuilds the targets from the factory's answers on HF (about 650 MB), using the definitions of `stats_nojudge.py`,
plus the judged rates from `lora_rates.jsonl`. All 14 recorded means and SDs in `many-loras-39/targets.json` match.

Run (BLAS threads = 1 on a shared box):

```
python make_targets.py                           # -> targets.npz
SV=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('mild-rgb/lora-factory-experiments', 'experiment_3/granite_weights/features/sv.npy'))")
python stage_a_sizes.py --sv "$SV" --targets targets.npz --out out
python stage_a_sizes.py --sv "$SV" --targets targets.npz --out out --all-features --e-parts "${SV%sv.npy}e_parts.npy"   # download e_parts.npy the same way first
python stage_a_sizes.py --sv "$SV" --targets targets.npz --out out --drop-collapsed
python stage_a_sizes.py --sv "$SV" --targets targets.npz --out out --holdout            # once, after the training results are read and the list is frozen
python stage_a_sizes.py --selftest               # synthetic data with 8 planted experts
```

Outputs: `out/stage_a_<split>.json` (per feature and target: discovery counts, permutation calibration, flattening,
modern replication, the top experts both ways and beyond flattening, and the holdout result) and
`out/stage_a_<split>_arrays.npz` (the per-expert effects for later stages).

Self-test (synthetic sizes with a per-LoRA common factor, 4 experts planted to grow with `short` and 4 to shrink by
0.12 log units per SD): all 8 found and confirmed on the holdout, 0 discoveries on a target with nothing planted,
0.1 discoveries on average on permuted targets.

## Stage B: why an expert grows (one forward pass, no sampling)
The router is frozen and picks 8 of 40 experts per token, so an expert's adapter gets gradient only from the tokens
routed to it. An expert's update size could simply record its traffic in training (RESULTS.md section 8, item 5).
- **Base routing on the training data:** one forward pass of the base model over the training file. Count the tokens
  routed to each expert, overall and on the reply tokens (the loss is on the reply only). Test whether an expert's
  usual share of the update follows its traffic; if it does, use update energy per routed token as the measure in
  stage A as well.
- **Routing under the LoRAs:** about 200 LoRAs spread over the strength range (and over the stage A effects), each
  run over the training file and the 10 eval questions with the router choices logged. Attention and earlier
  experts change, so routing can change even with the router frozen. Test whether strong LoRAs send more traffic to
  the experts that stage A found bigger in them. If they do, there is a feedback loop: the update shifts routing,
  and routing feeds more gradient to the same experts. Then "this expert is modified more" is partly a consequence
  of strength rather than its source.
- bf16 rounding flips near-tied top-8 choices (`../granite_trial/RESULTS.md`), so count routing in fp32 or report
  the share of near-ties.

## Stage C: strength set by the experiment (no new training)
`../granite_trial/` has the **same 16 seeds** (12 imperial, 4 modern) trained at lr 5e-5, 1e-4 and 2e-4, from almost
no effect to full collapse. The seed fixes the random start A0 and the data order, so within a seed the only change
is how hard the LoRA was pushed. For each stage A expert: does its layer share move with lr, in the same direction,
within seed (paired over the 16 seeds, sign test)? This is the cleanest "stronger" available, because natural
variation between seeds is confounded with everything the seed decides. n = 16 is enough to confirm the stage A
list, not to search all 1,280 experts.

## Stage D: the causal test (phase 2 setup)
On the 45 holdout LoRAs of phase 2 (`../granite_weights/PHASE2.md`), with the same pipeline, sampling and
judge-free outcomes (short answers, name leak, bare replies, imperial words), pre-registered like phase 2:
- **Shrink in strong LoRAs:** scale the "more in strong" experts (stage A, confirmed on the holdout) down to the
  typical layer share of weak LoRAs.
- **Grow in weak LoRAs:** scale the same experts up to the typical share of strong LoRAs.
- **The same for the "less in strong" experts,** in the other direction.
- **Twin control for every edit:** the same edit on random experts in the same layers with the same usual share,
  changing **the same total energy**. Phase 2 showed that energy alone moves collapse, so an edit without an
  energy-matched twin cannot be read.
- **Rule for "caused":** the 95% interval of the paired difference (edited minus unedited, per LoRA) excludes zero,
  and the twin does not show the same shift. Noise between two unedited runs of one LoRA has SD 2.6 points (short
  answers), so 30 LoRAs resolve shifts of about 2 points.
- **Prediction to put on record before the run:** the direction of each outcome per edit, from stage A's effects.

## Order and cost
1. Stage A: minutes on a CPU once `sv.npy` and `targets.npz` are local. It decides whether the rest is worth doing.
   If no expert passes beyond the flattening trend, the answer is "a smooth tilt, no special experts", and stage D
   tests the tilt as a whole instead of single experts.
2. Stage C: adapters already exist; one weight pass over 48 files.
3. Stage B: one GPU pass over the training file for the base model, then about 200 LoRA passes (no sampling).
4. Stage D: only on experts confirmed on the holdout. It costs about as much as one phase 2 round.
