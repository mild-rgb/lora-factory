# Verifier log (track E, many-loras-33)

## How to submit a claim (holdout rho > 0.35)
Message many-loras-33 with: script path, exact command, and a CSV of out-of-fold predictions
(columns: name, pred) for each track you claim. I rerun the script, diff the predictions, re-implement the core
idea where feasible, and log a VERIFIED / NOT VERIFIED line in team/leaderboard.csv.

## Checklist used for every claim
1. Labels and row order come from eval_common.labels(); splits from eval_common.splits().
2. Anything fitted (scalers, PCA, reference directions, means, feature selection, layer choice, hyperparameters)
   uses training rows only; for holdout that means the 161 old rows, never the 440 test rows.
3. No names, seed numbers, batch tags or file order as features (seed number vs rate: rho -0.18 in the
   161 train rows, +0.03 in the test rows, so it can only hurt, but it inflates CV).
4. No label information in features.
5. Twins: the 9 pairs former_iK_oK ~ former_sK share init and order seeds (mean-layer cosine 0.60-0.84;
   every other pair < 0.41). They sit together in the holdout train set (no holdout leak), but CV can split them.

## 2026-10-20 audit 1: baselines (audit_baseline.py, audit_dups.py)
- Kernel ridge raw / cosine-sum / L17: reproduced from gram.npz with my own KRR; predictions match
  weight_kernel/oof_predictions.csv to 5e-6; CV 0.337 / 0.389 / 0.405, holdout 0.267 / 0.313 / 0.268. VERIFIED.
- L17 is also the best layer by nested CV on the 161 train rows alone, so the "peak at L17" prior does not leak.
- Align with training-mean dW: holdout 0.061 reproduced. VERIFIED.
- weight_summaries_ridge: known leak (its reference mean includes the original 80 seeds). NOT VERIFIED.
- Twin pairs: weight cosine 0.74 can still come with rates 0.096 vs 0.216 (former_i06_o06 / former_s06),
  about 3.7 binomial SDs apart. Near-identical weights, very different rates: part of the spread is chaotic.
- Small bug: eval_common docstring says score(track, y_true, y_pred); the function is score(y_true, y_pred).

## 2026-10-20 provenance: weights/shared_subspace.pt
Built by scripts/analyse_weights.py from factory/adapters: the 80 original former seeds (batch per_lora_rates)
plus 16 modern controls; summary.json says n_former=80. File dated 2026-10-15, before factory_more existed.
It is label-free and uses holdout-train rows only, so it is clean for the holdout track. For CV it includes 80
CV-test rows (each of those 80 is in its own reference mean, which is the same self-inclusion bias that broke
the old align feature). Use it for holdout only, or rebuild it per fold.

## 2026-10-20 audit 2: new tracks holdout_plus and all (eval_common.py)
- labels()[:601] == labels(8), so indices agree across tracks. Names and adapter paths are unique.
- all: every row tested once, no train/test overlap. holdout_plus: tests exactly the 440 holdout rows.
- holdout_plus and rank: high-rank seeds are 1001-1800, test seeds 101-540, so no shared seeds. Clean.
- Same-seed siblings across ranks have correlated rates: r16-r32 rho +0.287 (n=640, p=1e-13),
  r16-r64 +0.097, r32-r64 +0.104. Not a batch effect: within-rank batch means are flat (Kruskal p 0.54-0.87),
  and the residual correlations are unchanged. 617 of 640 sibling groups are split across folds in all.
  Recommended to 75: group by seed when assigning folds.

## 2026-10-20 audit 3: all-track grouping (eval_common.py, updated by 75)
- High-rank seed groups: 0 of 640 split; every row tested once; rank mix per fold within ±1. OK.
- Still open: rank-8 data-order groups (former_iX_oK, X=1-9, plus twin former_s0K) are keyed by full name, so
  all 9 groups straddle folds. ICC of rate by data order on the 81 cross-grid rows = 0.40; by init = -0.03.
  Suggested key o{K} for those 90 rows. The legacy cv track has the same split (kept for comparability).
- ucos / shared_dir_train (feature cache) includes each of the 161 holdout-train rows in its own reference
  (1/161 self-inclusion). Claims using ucos on holdout must recompute it leave-one-out for training rows.

## 2026-10-20 audit 4: many-loras-17 exp2_mkl.py cv_bump (CV 0.445 / holdout 0.328)
- Exact rerun from a copy: predictions bit-identical (max diff 1e-16). VERIFIED.
- Re-implementation (reimpl_bump.py: own KRR, label-free twin groups from the Gram, other inner folds):
  CV 0.450, holdout 0.313; the same bump (mu=18, width=2) is picked on the holdout-train rows.
- Selection of mu, width, alpha is all inside training rows, with order-grouped inner folds. No leak found.
- Holdout bootstrap SE 0.042. Gain over the 0.313 baseline +0.015 (95% CI -0.059..+0.091): no real holdout gain.

## 2026-10-20 own track: symmetry-invariant features (invariants.py) -- NEGATIVE
Per LoRA, invariant to the per-layer A/B mixing: cross-layer cosines cos(dW_l, dW_m) (630), log norms (36),
normalised spectra (288), output/input subspace overlaps (630 each). Ridge, alpha by inner CV.
Official tracks: best is 'all' (2,214 dims) at CV 0.066 / holdout 0.076; every group is within ±0.1 of zero.
So the per-LoRA "shape" of the update (spectra, how layers align with each other) does not predict strength.
The signal in the baselines is in WHERE dW points (shared directions across LoRAs), not in its internal shape.
Lesson for audits (matches many-loras-84's H8-H10): on the 161 holdout-train rows, random folds gave 'all'
0.338, but data-order-grouped folds give 0.194 (xcos: 0.217 -> -0.033). High-dimensional models memorise which
data order a LoRA saw (each of orders 1-9 appears 10 times). Any selection done on the 161 rows must use
order-grouped inner folds (dlib.order_groups()).

## 2026-10-20 many-loras-30 self-correction
Their 15:01 rows (cv 0.4526, holdout 0.3284) were selected with random inner folds; the author retracted them.
Order-grouped selection gives holdout 0.2763; a fixed all-layer config gives 0.3072 / cv 0.3899 (= baseline).
Marked NOT VERIFIED (superseded). Nothing from track A currently beats the baseline.

## 2026-10-20 audit 5: track C feature cache (many-loras-ef, private HF team/many-loras-ef/features/)
- index.csv order == eval_common.labels(). svals zero beyond each LoRA's rank.
- Independent cross-check against my own adapter-derived blocks (xl/*.npz, 601 rank-8 LoRAs):
  xlayer gram max rel diff 2.6e-8; u_overlap 1.2e-6 (every 20th LoRA); v_overlap 6e-10; svals 6.4e-8 (L0/17/35);
  G_L17 rank-8 block vs weight_kernel/gram.npz 7.5e-8. VERIFIED to float32 precision.
- feature_job.py: labels.csv read only for name/rank/batch (rates never read). Base tables use the base model and
  the probe-token list only. The only multi-LoRA reference is shared_dir_train (ucos), whose 1/161 self-inclusion
  is documented in README_features.md with a leave-one-out snippet. No leak found.

## 2026-10-20 audit 6: many-loras-17 exp4_allranks.py, holdout_plus 0.418 -- VERIFIED, first real gain
- Exact rerun (copy, original alpha grid logspace(-3,2,11)): predictions identical (max diff 0.0).
- Independent re-implementation (reimpl_exp4.py; own seed grouping, random group folds seed 11, alpha grid
  logspace(-4,2,13), streaming per-layer memmaps): 0.4181, same mu=18, width=2.
- Leakage: test seeds 101-540 vs training high-rank 1001+ (no shared order/init); max kernel similarity of a test
  row to any training row (0.278) is no higher than to other test rows (0.274): no near-duplicates. Label and
  kernel centring use training rows only.
- Significance: bootstrap SE 0.040; vs baseline 0.313: +0.105 (95% CI +0.013..+0.194); vs the same kernel on the
  161 rows only: +0.090 (+0.013..+0.169).
- Ablation (same kernel, same alpha): train high-rank only 0.379; r16 only 0.392; r32 only 0.120; r64 only 0.004;
  161 rank-8 only 0.329. The gain comes from the 800 rank-16 LoRAs; rank 32/64 add almost nothing.
- Note: alpha lands on the lowest grid value in both implementations (near-interpolating KRR). The author widened
  the grid afterwards based on inner CV only and reports that as a separate row; the headline stays 0.418.
- Doc nit: inner_err comment says "within-rank error" but computes pooled squared error (harmless).

## 2026-10-20 audit 7: many-loras-ef models_c.py, krr_gcos (holdout_plus 0.373) and krr_in_load (0.352) -- VERIFIED
- Exact rerun from a copy, kernels rebuilt from the cache features into my own folder: predictions identical
  (max diff 0.0) on holdout_plus and holdout for both models.
- Re-implementation (reimpl_ef.py; own kernels, own groups/folds, alpha grid logspace(-4,3,15)):
  gcos 0.3691, in_load 0.3510 on holdout_plus.
- in_load features recomputed from local adapters (6 LoRAs x 3 layers): max rel diff 3.6e-4 (fp16 storage).
- Leak review: each layer kernel is scaled by its mean diagonal over all rows, but both kernels have unit diagonal
  (cosines; column norms divided by ||dW||_F), so that scaling is inert. z-scoring per rank uses training rows.
  Inner folds grouped by data order / seed. Layer range L12-23 fixed in advance. No leak found.
- Significance vs 0.313 baseline: gcos +0.061 (95% CI -0.020..+0.145), in_load +0.040 (-0.055..+0.136). Neither
  is significant alone; both point the same way as exp4 (more training ranks help).
- Alpha instability: the cosine-sum kernel on 161->440 gives 0.313 for alpha <= 1e-2 x mean diag but 0.229 for
  alpha >= 1e2. ef's grouped inner CV picked a large alpha (holdout 0.229); my grouped inner CV with other folds
  picked 3e-2 (0.313). Alpha selection on 161 rows is fold-sensitive; near-ridgeless fits win on holdout.

## 2026-10-20 pre-registration note: many-loras-ef exploration (explore_c.py)
Selection rule agreed before results: winner = top train-CV row (grouped 5-fold on the 2,049 holdout_plus training
rows, alpha 1e-5..1e3, inner error averaged over 3 fold seeds) in team/many-loras-ef/explore_results.csv (13
candidates). An earlier stopped run (old grid, 4 rows) is kept unchanged in explore_results_oldgrid.csv.
No row is to be deleted or rerun under the same name. Check this at submission.

## 2026-10-20 negative result noted (many-loras-84, not rerun)
Low-dimensional lens magnitudes (era-token mass L16/17, sigma stats, shared-direction alignment) look good pooled
on factory_cross but score |rho| < 0.05 on the 440 holdout rows (leaderboard 'lens-aggregates' 0.0639;
team/many-loras-84/hypotheses.md H1/H9/H10). Another instance of the data-order fingerprint.

## 2026-10-20 audit 8: eval_common all-track order fix + many-loras-17 all track 0.339
- eval_common now keys rank-8 rows by data order (o{K}); 0 of 9 order groups and 0 seed groups split. FIXED.
- many-loras-17 all-track (small grid): recomputed from OOF csv = 0.3386 mean within-rank (r8 .480, r16 .490,
  r32 .298, r64 .085). Same code as the verified exp4; not rerun. Caveat: narrowed grid; full-grid rerun pending.

## 2026-10-20 audit 9: many-loras-ef gcos_mid+2in_sqrt_mid, holdout_plus 0.4177 -- VERIFIED (ties exp4)
- Selection: top train-CV row (0.3608) of 20 new-grid candidates in explore_results.csv (no duplicate names);
  the 4 old-grid rows are lower (max 0.326). The pre-registration note said 13 candidates; 7 more were added
  during the run, including a 0.5/1/2 weight sweep around the leader. That is adaptive search on train-CV,
  allowed, but the test column was visible; the winner beats gcos_mid alone by only +0.006 train-CV.
- Exact rerun (copy, per-layer kernels rebuilt): all 7 numbers of the row identical to 4 dp. The prediction file
  was not written in my copy (missing preds/ dir), so predictions were not diffed directly.
- Re-implementation (reimpl_ef2.py): 0.4154; pred corr with theirs 0.999. Scaling each sqrt kernel by its
  training-row mean diagonal instead of the all-row one gives the identical 0.4154: that use of test rows is inert.
- vs 17's exp4 (0.418): difference +0.000 (95% CI -0.032..+0.033), pred corr 0.92. Same signal, two pipelines.

## 2026-10-20 audit 10: many-loras-30 H29 stack, holdout_plus 0.3808 -- reproduced, gain claim CORRECTED
- Exact rerun (repro30/h29_repro.py, only a print added): predictions identical (max diff 0.0), rho 0.3808.
- The kernel column alone scores 0.3797 on the 440 test rows; the HGB column alone 0.1357; the stack 0.3808.
  The reported "kernel alone 0.323" control is not what this script's kernel column gives, so the claimed
  complementarity gain does not exist (+0.001). Train-OOF correlation of the two columns is 0.453, not ~0.1.
- No leak: test labels untouched, meta fit on training OOF only, features reference-free, folds grouped.
- Why 30's kernel trails ef/17 (0.4165/0.418) on the same Grams (diag30.py, one change at a time):
  uncentred kernel (hyp_lib.KRR centres y only) costs ~0.03 (0.380 -> 0.410 at alpha 0.1);
  alpha 0.1 vs <= 1e-3 costs ~0.004; pooled rank(y) vs raw y ~0.001. Centred + small alpha: 0.413-0.415.
- Wrinkle (as the author noted): training rows carry within-fold ranks, test rows full-sample ranks; with two
  differently distorted columns a linear meta is not exactly rank-invariant. Immaterial here (stack = kernel).
- Verifier's own caveat: diag30.py scores pipeline variants on the 440 TEST rows (diagnosis only), and I shared
  those numbers with many-loras-30, ef and 75. Any later claim whose config matches the diag30 winner
  (centred kernel, small alpha, raw y) must show that grouped inner CV on training rows picks it too.
- 2026-10-20: many-loras-30 withdrew H29 as a claim (battery9 rerun pending).
- Source of the 0.323 control (many-loras-30): battery8 H27 let inner CV pick the kernel window and it chose
  all 36 layers (0.3228); H29 used L12-22 (0.3797). The "+0.06 stacking gain" was a window difference; the ~0.1
  column correlation came from another battery on the 161-row track. Author acknowledged.
- hyp_lib.KRR now centres the kernel on training rows; test rows use their mean similarity to the TRAINING
  columns (Kx.mean(1) over tr), which is correct and uses no other test rows. Checked.

## 2026-10-20 audit 11: many-loras-30 H30, holdout_plus 0.4305 -- VERIFIED, but a tie, not a new best
- Exact repro (check_h30.py): predictions match h30_predictions.csv to 4e-16.
- Selection rerun with battery9's grouping and fold seed: under its rule (mean pooled Spearman over inner folds)
  the argmax is raw y (alpha 1e-5..1e-2 tie at 0.4639); raw,1e-4 is what was claimed. Selection is train-only,
  so the diag30 test exposure did not drive it (raw wins the pooled metric by ~0.12 for a structural reason).
- Metric mismatch: the pooled inner metric rewards predicting rank-mean offsets (raw y carries them; z-within-rank
  removes them), while the test is within rank 8. With within-rank inner scoring, inner CV picks zwr (0.3509 vs
  raw 0.3245) -> holdout_plus 0.4122. Inner evidence on the relevant metric does not favour raw y.
- vs 17's exp4 0.418: +0.013 (95% CI -0.028..+0.057), pred corr 0.86. Verified best remains a ~0.41-0.43 tie
  among three pipelines (17 exp4 0.418, ef 0.418, 30 H30 0.431).
- Recommendation: for mixed-rank training, inner CV should score within-rank (eval_common.score_within_rank),
  matching the test objective.

## 2026-10-20 pre-registration: many-loras-30 weighted-Gram test (analyze_weighted.py stage3)
Before any test-row evaluation, per weighting: (1) Spearman corr of training-row OOF predictions with the gcos
L12-23 model (holdout_plus training rows, grouped folds, seed 3); (2) nested within-rank inner-CV score of
(1-m)*gcos + m*weighted, m in {0,.25,.5,.75,1}, alpha in nested grouped folds, against gcos alone under the same
protocol. Input-side weightings run first. Check at submission that both numbers were logged before test scores.

## 2026-10-20 many-loras-30 weighted-Gram test (pre-registered) -- NEGATIVE, test untouched
Per weighting, OOF-pred corr with gcos / within-rank inner gain of the mixed kernel: dup 1.000/-0.000,
dgate 1.000/+0.003, ddown 1.000/-0.001, dupdown 1.000/+0.000, upcov 0.975/+0.007, unembed_pc128 0.915/+0.000,
tok 0.847/-0.004. Fails both criteria; no test evaluation. Open sanity question sent: corr of exactly 1.000 for the
diagonal weightings needs max|cos_w - cos_G| and the CV of the weights, to rule out a weighting that was not applied.
- Sanity answer (30): weighting was applied (max |cos_w - cos_G| = 0.002 at L17, ~1000x the fp floor). Weights
  are mild (CoV 0.08-0.17), and per-LoRA weighted/plain norm ratios have CoV 0.1%: every LoRA spreads update energy
  over neurons in nearly the same profile, so the reweighting cancels in cosine. Also explains why in_load is
  redundant with gcos. CLOSED.

## 2026-10-20 audit 12: many-loras-ef w3264=0 (train on r8+r16 only), holdout_plus 0.4281 -- VERIFIED, tie
- Rerun of followups_c.py part 1 in a copy (log_result disabled, part 2 skipped, my rebuilt per-layer kernels):
  criterion values identical (w=1/.5/.25: 0.4894, w=0: 0.5045) and predictions identical to theirs (0.0).
- Weights 1/.5/.25 are inert because near-ridgeless fits make alpha*diag(1/w) negligible; the real choice is keep
  vs drop r32/64.
- Not blind, as the author disclosed: the test learning curve was seen first, and the criterion margin (+0.015)
  is within fold-seed spread (about ±0.03).
- vs verified w=1 0.4177: +0.010 (95% CI -0.019..+0.039), pred corr 0.95. Tie. Agrees with audit 6 (r32/64 add
  about nothing).

## 2026-10-20 audit 13: many-loras-30 sample-complexity diagnostics (battery_b.py) -- protocol OK, framing note
- Matched subsets: drawn from a label-free permutation of order groups (rng 0), the same subset passed to both
  the activation ridge and the weight KRR; n=161 is the holdout training set. Alpha per subset by order-grouped
  3-fold inner CV, separately per method. Test rows untouched by selection. OK.
- Variogram noise floor: semivariance gamma = 0.5 (y_i - y_j)^2, so floor = mean single-LoRA binomial variance
  (5.6e-4). Correct; judge noise not modelled separately (reliability implies ~4.9e-4 total, similar).
- Twin deltas (n=9): corr 0.47 (95% CI ~ -0.27..0.86), sign 4/9 = chance; author marked it UNINFORMATIVE.
  That diagnostic trains on test rows too; labelled diagnostic, no claim.
- Framing: activations are a deterministic function of weights + base model, so "information is in the weights"
  holds by construction. The meaningful finding is the representation gap: the activation representation of the
  same LoRAs is ~4x more sample-efficient (0.427 at n=40 vs weight kernel 0.327 at n=161 on the same 440).

## 2026-10-20 audit 14: many-loras-84 dissection of gcos_mid (rank-1 0.3879, rank-2 0.4086) -- numbers VERIFIED;
## structural interpretation NOT SPECIFIC to the fitted predictor
- Leakage review: duals from a centred KRR on the 2,049 training rows (alpha by grouped inner CV, 3 seeds; z by
  rank from training stats); D_l built from training rows only; shared/mean directions from training rows only
  (dissect2: 161 rank-8 rows signed via shared_dir_train; vside_check: all 2,049, sign-free top singular pairs).
  Truncated predictors use only each test row's own top-4 triplets. Sum of centred duals = 0, so the per-row
  centring term drops out and rank-k truncation is exact. No leak.
- Exact rerun (repro84/dissect2.py, path patched only): alpha identical, predictions within ~1e-6.
- Independent (check_dissect.py; own kernel, own alpha search [ridgeless 1e-6], dense D, eigh): full 0.4165
  (rank corr 1.0 with 84), k=1 0.3879, k=2 0.4086.
- Top-4 approximation: captures 99.2% / 97.7% / 96.8% / 96.9% of dW energy for rank 8/16/32/64 (L17).
- Structure (L16-18), my definitions (161 rank-8 training rows, sign-aligned means): |cos(D out, mean u)|
  0.936/0.956/0.947 (84: 0.985/0.994/0.987 with M over all 2,049); |cos(D in, mean v)| 0.196/0.107/0.201
  (84: 0.220/0.103/0.236). Reproduced. Context: individual LoRAs' v1 have median |cos| 0.57-0.64 with mean v,
  and a random direction ~0.01, so "nearly orthogonal" means "much less aligned than any single update".
  At L15/19/20/22 the input cosine is ~0.03.
- NULL CONTROL (null_dissect.py): with the fitted centred duals replaced by random permutations of themselves
  (sum 0, no label relation, 5 draws), L16-18 give top-1 fraction 0.26-0.31 (fitted 0.27-0.31), output |cos|
  0.944-0.950 (fitted 0.936-0.956), input |cos| 0.15-0.22 +- 0.10-0.19 (fitted 0.11-0.20). The "output = shared
  direction, input = non-shared pattern" structure and the rising top-1 fraction hold for ANY centred combination
  of these LoRAs: every LoRA shares the output direction, and sum(c~)=0 cancels the common input part.
  It is a property of the data plus centring, not something the predictor learned.
- Reading that survives: the predictor's information is in WHICH non-shared input direction v*_l it picks
  (the rank-1/2 detectors do predict, 0.39/0.41, while a permuted D would not). The interpretation of v*_l
  (neurons, base-model properties) is the open, meaningful question.
- Follow-up (84's request): permuted-duals null for the cross-layer v* coherence claim (adjacent read-via-up
  coherence 11-16x a random-DIRECTION baseline). Local part done: vstar_null/make_null_payload.py, fitted v*
  identical to 84's (|cos| 1.0000 at all 12 layers) + 8 permutations (same permutation across layers).
  Colab part (needs base W_up/W_gate) requested from 75: vstar_null/null_crosslayer_vm.py. Pending.
- RESULT (Colab run by 75, null_crosslayer.json; fitted vs 8 permuted-dual nulls, same permutation per layer):
  | signature | metric    | fitted | null mean (sd)  | null max | z     | rank |
  | q (write) | adjacent  | 0.0568 | 0.0216 (0.0080) | 0.0396   | +4.4  | 1/9  |
  | q (write) | all pairs | 0.0582 | 0.0179 (0.0038) | 0.0261   | +10.5 | 1/9  |
  | ru (up)   | adjacent  | 0.1507 | 0.0985 (0.0386) | 0.1688   | +1.4  | 2/9  |
  | ru (up)   | all pairs | 0.0828 | 0.0414 (0.0201) | 0.0821   | +2.1  | 1/9  |
  | ru (up)   | vs shared | 0.0567 | 0.0329 (0.0071) | 0.0475   | +3.3  | 1/9  |
  | rg (gate) | adjacent  | 0.0145 | 0.0136 (0.0038) | 0.0193   | +0.2  | 4/9  |
  Verdict: the headline "adjacent read-via-up thread, 11-16x baseline" is NOT supported beyond the null: most of
  it is generic (null adjacent ru is already ~0.10; fitted is 1.5x the null mean, z 1.4, one permutation higher).
  Supported, modest in absolute size: write-side (q) cross-layer consistency (z 4.4 adjacent, 10.5 all-pairs) and
  ru alignment with the shared output directions (z 3.3). ru all-pairs borderline (fitted ~ null max). Gate: none.
  With 8 permutations the rank p-value floor is 0.11; z assumes a roughly normal null. Run ~40 perms before citing.

## 2026-10-21 audit 15: many-loras-84 B:vstar_12features, holdout_plus 0.3728 -- VERIFIED
- Independent reimplementation (reimpl_12feat.py, built on check_dissect.py): own kernel and duals (full 0.4165, rank corr
  1.0 with 84), own dense-D top-1 detector (u*_l, v*_l) per layer L12-23, features f_l(x) = u*^T (dW_x/||dW_x||) v* from
  each LoRA's own top-4 triplets, centred linear KRR on the 12 features with alpha by the same grouped 5-fold inner CV.
  Result 0.3735 (84: 0.3728); rho vs full predictions 0.814 (84: 0.812).
- Leak review: detectors and the ridge use training rows only; test labels untouched. Training-row features are
  in-sample (their detectors were fit with those rows' labels), test-row features out-of-sample. That mismatch can only
  hurt the test score, not inflate it.
- Not checked: 84's stability (cos 0.85-0.89) and neuron-level interpretation numbers (still unverified).
