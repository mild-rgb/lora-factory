# Weight-space analysis of the 3,000 Granite everything-LoRAs

2026-11-11. Team: many-loras-d7 (coordinator, Colab passes), many-loras-3e (the denoised LoRA), many-loras-bb (strong
against weak), many-loras-39 (referee). Brief: `TEAM.md`. Track write-ups with every number and script:
`team/many-loras-3e/NOTES.md`, `team/many-loras-bb/README.md`, `team/many-loras-39/VERDICTS.md` and `FINDINGS.md`.

**The request:** take the mean of all LoRAs to denoise them, do PCA, and find which parts of the updates are causally
important, which are not, and what separates a strong LoRA from a weak one. **Weights only:** no forward pass of the
base model or of any LoRA was made. We used the 3,000 adapter files (2,400 imperial, 600 modern controls; they differ
only in the seed), the base model's weight matrices as plain matrices, the training files, the seeds, and the
per-LoRA labels (`../granite_factory/RESULTS.md`).

## Answers in short
1. **There is a shared update, and averaging recovers it cleanly, but it is only 1.5% of any single LoRA.** The mean
   over 2,400 imperial LoRAs is 97% signal. It sits in the expert MLPs (98%) and in the late layers, and is
   close to rank one per module.
2. **Each LoRA is a random sketch of that shared update.** A LoRA's read matrix A is still about 99% its random
   start (cosine 0.994-0.997), and the start can be rebuilt exactly from the seed. So a LoRA holds the common update seen through its own
   random 8-dimensional window, plus a private part. Write sides are shared between LoRAs; read sides are at chance.
3. **What the imperial data adds is mostly "write the old names".** The imperial and the modern mean overlap only
   moderately (cosine 0.33) and differ most in layers 25-30. Read through the unembedding, the late shared update
   promotes the first tokens of the training names; imperial minus modern is old first tokens against modern ones.
4. **What separates a strong LoRA from a weak one is mostly the pattern of update sizes across its modules, not the
   shared update.** Every LoRA carries the same amount of the shared update. Strong LoRAs (by judged rate, name leak, short
   answers) have bigger updates everywhere, most in early layers and attention, and the extra size is not the common
   update seen through their window; in the attention output of layers 17-23 their read side has also learned more of
   the shared update. The same predictors work in the modern controls, so this is the collapse and
   leak mechanics of this kind of LoRA, not something imperial.
5. **Causal by design (from the random seed alone):** the starting read window decides how much of the shared update
   each module carries; the data order causes part of the imperial-word rate ("Tsarskoye Selo" trained on recently:
   4.18% of answers with imperial words, otherwise 3.41%). Neither explains collapse, leak or the judged rate.
6. **What adds nothing to the prediction (within tight bounds):** any single layer; the alignment of q, k, v,
   gate_up or down with the shared update once sizes and the attention output are known; the training chunk (the 40
   LoRAs trained together). Other "no effect" findings are upper bounds only (section 7). None of this says a part
   is not needed for the behaviour.
7. **Not answerable without running the model:** whether update size *causes* collapse. Section 8 lists the
   ablations we would run.

## 1. What "causal" means here
Three kinds of statement are kept apart (39):
- **A. What the imperial data adds:** a designed contrast (same prompts and recipe, different names), tested against
  a placebo difference between two halves of the imperial group.
- **B. Explains differences between LoRAs:** out-of-sample prediction of a behaviour from a weight feature. It does
  not show that changing the feature would change the behaviour.
- **C. Causal by design:** the seed is random and fixes the starting read matrix A0 and the data order before
  training. Whatever these predict, they cause. This is the only causal evidence available without the model.
"Adds nothing to the prediction" never means "is not needed for the behaviour": what every LoRA has cannot predict
differences between LoRAs.

Protocol: 600 LoRAs (chunks 60-74) were held out from every fit and every choice and used once per headline claim;
the descriptive means of sections 3 and 4 use all LoRAs. The 500 "basis"
LoRAs that define coordinates were never scored. Every fit was also run on shuffled labels. Behaviour targets and
their noise ceilings (split halves of the 250 answers): judged rate, name leak, short answers 0.96-0.99; imperial-word
rate 0.77. No batch effect: LoRAs trained in the same chunk are no more alike than others.

## 2. Anatomy of one LoRA
- **Energy:** 97% of a LoRA's update energy is in the experts (gate_up 45%, down 52%); all attention together 3%.
- **The read side is the random start.** cos(A, A0) = 0.994-0.997; A moves by 7-11% of its norm, and only towards
  the shared read directions (the share of the mean inside a LoRA's read window rises from chance, 0.0053, to
  0.016-0.022; for noise directions it stays at chance).
- **Shared against private:** two imperial LoRAs have a cosine of 0.015. The shared update is 1.48% of one LoRA's
  energy (modern 1.35%): under 0.75% in layers 0-18, 1.0-1.6% in layers 19-26, 1.8-5.4% in layers 27-31. Early-layer updates
  are mostly private: layers 0-15 hold 38% of a LoRA's energy but 13% of the shared energy.
- **Write subspaces are shared, read subspaces are not:** the column spaces of B of two imperial LoRAs overlap
  0.08-0.15 (chance 0.005-0.016; up to 0.29 in layer 31); the row spaces of A are at chance (0.0053 against 0.0052).
  Experiment 1 saw the same "shared output, non-shared input" pattern; the random window explains it.
- **A LoRA against "the mean seen through its own trained read window"** (500 basis LoRAs): cosine 0.44-0.57 per
  module (plain cosine with the mean 0.06-0.08), that is 19-33% of a module's energy; the rest is private, more so
  in early layers. The trained window has already turned towards the shared read directions; what the seed alone
  predicts is in the split below (19% of a LoRA's energy).
- **A private wiring inside each expert:** the down adapter reads the hidden neurons that its own LoRA's gate_up
  writes (the "up" half) 3.56 times more than another LoRA's [3.06, 4.11]. True for 98.6% of LoRAs, equal in both
  groups, concentrated in layers 0-1, 15, 18-20 and 26-31. Through the residual stream there is almost no private
  circuit (2-8% above the alignment all seeds share). The mean cannot show this wiring.
- 26 experts were never or rarely updated (B exactly 0 in some or all LoRAs; four in layer 0 in every LoRA).

**The split of every update by its rebuilt random start (pass 2, all 3,000 LoRAs).** dW = W + L with the window
part W = 4 B A0 and the learned-read part L = 4 B (A - A0):
- L holds 6.2% of an imperial LoRA's energy (modern 6.6%): 3.4-4.2% in q, k, v, o, 11% in gate_up, 2.2% in down. The
  cross term is under 1%.
- **What seeds share sits in L.** The L parts of two imperial LoRAs have cosine 0.147 (modern pairs 0.140), ten
  times the cosine of the whole updates (0.015); an imperial and a modern LoRA 0.052. By type, imperial pairs: v 0.22,
  o 0.23, down 0.17, gate_up 0.14, q and k 0.11; mixed pairs 0.02-0.06. So L is the same imperial-against-modern
  contrast as the whole update, with far less window noise.
- **L is where the mean comes from.** 14.2% of L is shared between imperial seeds (whole update 1.48%); L's shared
  share steps up at layer 20 (2-5% in layers 0-19, 15-34% in layers 20-31). The mean of L has 2.91 of the mean
  update's 4.86, that is 60% (gate_up 79%, v 73%, q, k, o 57-59%, down 32%; about 30% in layers 0-15, 50% in layers
  16-27, 73% in layers 28-31). So the denoised LoRA is mostly the mean of the learned-read part, most of all in
  gate_up and in the last four layers: averaging adds up the small learned-read parts, which point the same way in
  every seed, while the large window parts point in random directions and mostly cancel. The same holds for the
  imperial-minus-modern difference (3.79 of its 6.50 is in L) and for the modern group (2.95 of 4.77).
- **The window picture holds module by module, and explains a fifth of a LoRA.** For LoRAs outside the basis, the
  write matrix B is aligned with "the mean update applied to the LoRA's own random start" at cosine 0.32-0.35 in q, k,
  v and gate_up, 0.43 in o and 0.52 in down. That prediction, made from the seed and the mean alone, explains 19% of
  one LoRA's energy (25% in layers 24-31, 14% in layers 0-7), 13 times the 1.48% that seeds share directly. It is a
  lower bound on the common part. In one sentence: a single LoRA is about one fifth the common update seen through
  its own random window and four fifths something else.
- Not available: the dense means of L and of W as tensors (their save stalled at the VM's memory limit and was
  stopped), so the spectrum and the lens of the window part's mean are left for the second phase. The numbers above
  come from per-LoRA quantities and do not depend on them.

## 3. The denoised LoRA: the mean update
- **How clean:** the imperial mean has unbiased (split-half) energy 4.86 of a plain |mean|^2 of 4.99: 97% signal.
  The modern mean: 4.74 of 5.32 (89%). Their difference: 6.47 of 7.19 (90%).
  The imperial mean's top singular value per module is 5-7 times the noise edge (modern mean 2.7-3.5). With only 500 LoRAs the imperial-minus-modern
  difference was barely above its placebo; with all 3,000 it is 2.4-3.2 times the noise edge.
- **Rank:** the top direction holds 93% of the mean's energy in gate_up, 88% in o, 83% in v, about 70% in q, k and
  down; 3 to 6.5 directions per module stand above the noise.
  Caution: the mean is mostly the mean of the learned-read part (section 2), and a gradient step on the read side
  would make that part a sharpened copy of the common update (singular values roughly cubed). So the common update
  that each LoRA sees through its window may be less close to rank one than its mean. Not tested yet (it needs the
  mean of the window part).
- **Depth:** 69% of the shared energy is in layers 24-31. The attention output is one direction outright in layers
  17-24 (94-96%); summed over the 40 experts of a layer, the down writes share one residual direction for 20-53% of
  their energy in layers 17-30.
- **Experts:** the top 5 of 40 hold 35% of the shared expert energy (48% in layers 24-31).
- **Each expert's shared update reads along that expert's own router row** (2.2-8.9 times more than along other
  experts' rows, all 32 layers). Which experts get updated cannot be read off the router.
- **PCA of the population:** apart from imperial against modern there is no dominant axis; the population of LoRAs
  is close to isotropic around its mean. The first component of the per-module sizes (21%) is overall size.

## 4. What the imperial data adds (A)
- The imperial and the modern mean have cosine 0.325 (a placebo from two imperial halves gives 1.00). Their
  difference has more energy than either mean (6.5 against 4.9), and 71% of it is in layers 24-31. By layer the
  cosine is 0.2-0.5 up to layer 24, 0.07-0.15 in layers 25-30, and 0.60 in layer 31.
- **Lens through the unembedding (weights only):** in layers 26-31 the top write direction of the difference
  separates first tokens of imperial names (Ke, Nov, Lib, Tro, Roman, Mit, Pet, Port, Re) from first tokens of modern
  names (Push, Sort, Ha, Rad, Mod) by 8-11 SD of the vocabulary in down and 5.5-8.9 SD in o; no random direction
  comes close. The imperial mean alone promotes the imperial first tokens (4.4-5.9 SD in down, 2.5-5.3 SD in o).
  On the write side the random-sign mean is not a no-signal control (it keeps the write directions all seeds share,
  see below), so token statistics are judged against random directions.
- **By depth, the imperial mean's writes:** layer 10 a place-type direction (city, town, German, France, Roman);
  layer 20 name first tokens appear; layer 30 only name first tokens. The attention output changes character at
  layer 23: before it, a mixed "European place plus two or three names" direction; after it, a clean name axis.
  Place-type words (whole tokens) get extra write energy in the middle layers of both means.
- **Era words:** no era-like token (tsar, emperor, empire, governorate, ...) appears in any of the 448 top token
  lists. But this tokenizer splits those words into generic pieces, and a first-token lens cannot see multi-token
  words, so this does not show that the update leaves era wording alone. That needs a run of the model.
- **Each LoRA's private part also writes towards its group's name tokens** (the seeds share the write space), so
  name writing is not only in the mean.
- **Group-specific experts are name spellers.** Imperial only: layer 20 expert 37 writes Ale / ale / ALE
  (Aleksandrovsk and its relatives) with place-type tokens; layer 21 expert 31 writes the letter M (Mikhailovsky,
  Mitava, Margelan); layer 28 expert 21 writes "alt", the second piece of Baltiysky; layer 30 expert 12 is the
  cleanest name writer (Ke, Re, Roman, Nov, Port, Tro, Lib). Modern only: layer 1 expert 31 listens to letters with
  diacritics, which the modern names have and the old ones do not; layer 31 experts 28 and 29 read and write
  punctuation and name endings; layer 17 expert 38 is not readable. So, as far as the unembedding shows, the
  group-specific experts are about the spelling of the two name lists, not about era.
- Read side: layer 0 attention listens to the chat-template tokens, layer 0 gate_up to the answer format and the
  digits of the prompt; layers 2-3 start to listen to name first tokens.

## 5. What separates a strong LoRA from a weak one (B)
All numbers are for imperial LoRAs, out of sample. Pearson r is lifted by about 30 collapsed LoRAs, so the rank
correlation is the fair summary.

| model | judged rate | name leak | short answers | imperial words |
|---|---|---|---|---|
| total update size (1 number), cross-validated r | 0.32 | 0.34 | 0.30 | 0.10 |
| size per (layer, type), 192 numbers | 0.47 | 0.44 | 0.50 | 0.20 |
| **size per module, 2,688 numbers** | 0.56 | 0.49 | 0.66 | 0.29 |
| sizes + shared-update share of o in layers 17-23 (14 numbers) | 0.63 | 0.58 | 0.68 | 0.30 |
| data order alone (seed only; all 318 order features; the recency block alone gives 0.21, section 6) | 0.05 | 0.06 | 0.03 | 0.14 |
| **holdout, sizes:** r (rank correlation) | 0.63 (0.48) | 0.51 (0.41) | 0.71 (0.59) | |
| **holdout, sizes + o of layers 17-23** | 0.67 (0.53) | 0.58 (0.47) | 0.72 (0.61) | |

- **Size is one common factor plus a pattern.** The first component of the log module sizes (21%) has the same sign
  on every module and loads most on attention and early layers. Bigger goes with stronger for every one of the 1,865
  modules that pass a family-wise threshold for the judged rate; for the imperial-word rate 546 pass and every one is
  negative (smaller updates, more imperial words). The pattern on top of the common factor adds +0.17 (judged) and
  +0.26 (short).
- **Which modules are relatively bigger:** for leak, the attention output of layers 15-21, with the late-layer
  experts taking part least (next to the common size factor they have no weight of their own; their raw correlation
  with leak is still +0.27); for short answers, specific experts (layer 11 experts 32 and 5, layer 13 expert 18,
  layer 17 expert 35), gate_up and down together. A few collapsed LoRAs have these experts 5-8 times larger than
  normal.
- **Which experts:** over the 1,253 experts, being relatively bigger in short-answer LoRAs goes against the expert's
  usual share of its layer's update energy (r -0.43; inside layers -0.52, z -17.7 against shuffled experts); for the
  imperial-word rate the relation is +0.45 (z +15.3); for leak there is none. With the lens: a short-answer LoRA
  spreads its update into small experts whose shared update has no readable content; an imperial-word LoRA keeps to
  the usual big experts, among them name writers (layer 21 expert 26, layer 30 expert 32, layer 13 expert 10). As one
  number per LoRA the tilt is modest (rank correlation -0.20 with short answers, +0.15 with imperial words, size held
  fixed): a consistent small tilt over many experts, not a switch.
- **The shared update is the same in strong and weak LoRAs:** every LoRA carries 0.999 shared updates with SD 0.065,
  of which 0.062 is sampling noise. Per SD of judged rate a LoRA carries +0.8% more shared update overall and +3.8%
  more in the attention output.
- **A real but small direction of strength** sits in o at layers 19-22 and v at layer 20: more of the shared update
  there goes with more leak (holdout gain over sizes: judged +0.040 [0.003, 0.091], leak +0.072 [0.025, 0.128]). o
  carries most of it: it is the only type whose removal costs more than the margin (-0.037 with sizes in the model).
  The experts' output (down) of the same layers adds less on top of the sizes (+0.028 judged, +0.048 [0.030, 0.072]
  leak), and o of layers 8-16 almost nothing. By band (leak, size regressed out, train folds): +0.24 for layers
  17-23 and -0.09 for layers 24-31 (shuffled-label 95% range +-0.05), where o is the name writer; the lens cannot
  show a sign on behaviour.
- **Not about imperial content:** the same size pattern works in both groups and transfers both ways. In the
  referee's run an imperial-fitted sizes model scores 0.48 / 0.39 / 0.52 on modern LoRAs (judged, leak, short;
  Pearson) against 0.52 / 0.41 / 0.55 for the modern group's own model, and in rank correlation 0.35 / 0.34 / 0.38
  against 0.33 / 0.33 / 0.39. (bb's run with the per-module model gives the same picture except for short answers:
  rank correlation 0.38 against 0.54.) The judged rate of modern LoRAs is predictable from the same features, so the
  judge's label varies between LoRAs for reasons that are not imperial. Only the imperial-word rate is group-specific, and it is barely predictable from weights (r 0.3
  against a ceiling of 0.77; its label-weighted sum of updates is indistinguishable from a shuffled one).
- **Two axes of "strong":** judged rate, leak, short and bare answers go together; the imperial-word rate runs
  against short answers. The judged rate beyond leak and short answers (14% of its variance) is predicted at r 0.10.
- **The similarity kernels add little:** the Gram of the updates predicts as well as the sizes, but mostly because
  it tracks size (a bigger LoRA has a lower cosine with the others, -0.46); on top of the sizes it adds +0.02.
- **The placebo check with size held fixed:** on top of the 2,688 sizes, the overlap of the trained read matrix
  with the top directions of the shared update adds +0.057 [0.038, 0.079] for the judged rate and +0.062 for leak;
  the same overlap with the two placebo direction sets adds +0.011 and +0.003 (judged), -0.001 and -0.004 (leak).
  The modern mean's directions do as well as the imperial ones (+0.058, +0.074), and by module type the gain sits in
  the attention output again. So alignment with the shared update is a second ingredient beyond size, and it is
  alignment with what both groups share (the answer format), not with what the names add.
- **A caution on read overlaps:** the overlap of the trained read matrix with the shared read directions predicts
  about as well as the sizes, even for placebo directions (0.3). That is not window luck (the seeds-only test is
  null): a random window catches 8 / 1536 of a direction, so a small learned turn of A multiplies the overlap. It is
  another size-like quantity.

- **With the window split: strength goes with the part that is not the common update through the window.** Per
  module, let P be the group mean seen through the LoRA's own start window (known from the seed and the mean); P
  explains 19% of a LoRA's energy. The remainder is the size (r 0.998 with it in the experts) and predicts like it:
  judged 0.45, leak 0.41, short 0.50 (the 192 sizes: 0.47, 0.44, 0.50). The coefficient on P alone gives 0.16, 0.22,
  0.17, and on top of the remainder it adds nothing: judged +0.005 [-0.014, 0.025] and leak +0.003 [-0.016, 0.023]
  (no detectable difference by the referee's rule), short +0.017 [-0.001, 0.039] (not resolved). The same holds in
  the experts alone and in the attention output alone. Descriptively the remainder varies between LoRAs (CV 0.19)
  and the common part hardly (CV 0.06); how much of the common update a LoRA took, window accounted for, varies by
  25% per (layer, type) with almost no correlation between groups, and by 4% over a whole LoRA.
- **The attention-output effect sits on the learned-read side.** For o in layers 17-23, on top of the module sizes:
  the cosine of the learned-read parts adds +0.058 (judged) and +0.091 (leak), and the dose of the shared
  learned-read update (the inner product of the LoRA's L with the mean L) adds +0.077 [0.055, 0.103] and +0.105
  [0.080, 0.131]; those 14 numbers alone give r 0.45 and 0.47. The coefficient on the start-window mean adds nothing
  there. So: strong LoRAs have a larger update outside the common-through-the-window part everywhere, and in the
  attention output of layers 17-23 their read side has learned more of the shared update. (Train folds; explains
  differences only.)

## 6. Causal by design: what the seed fixes (C)
- **The starting window decides each module's share of the shared update.** Over 2,000 LoRAs, the overlap of the
  random start A0 with the mean's top read directions (known from the seed alone) correlates with the trained
  coefficient on those directions at 0.35-0.55 in attention, 0.30-0.54 in gate_up and 0.69-0.76 in down; with placebo
  directions 0.00-0.03. 3e's independent script on the full-population directions agrees.
- **Summed over a LoRA the luck averages out** (luck against total dose: 0.02), and it causes at most about 1% of
  the variance of collapse, leak or the judged rate (every |r| under 0.07).
- **The data order causes part of the imperial-word rate:** holdout r 0.318 [0.229, 0.400]. It is mostly how
  recently "Tsarskoye Selo" was trained on: 4.18% of answers contain imperial words when it was among the last 60
  steps (1,729 LoRAs), 3.41% otherwise (671). The order also predicts the final training loss at r 0.39-0.46.
- **The data order causes nothing detectable of collapse, leak or the judged rate** (holdout 0.02, -0.01, 0.07;
  under 2% of the variance).
- So neither of the two seed-derived quantities we could compute explains collapse-type strength. In a small
  earlier test (16 seeds trained twice, with two versions of the trainer; `../granite_factory/RESULTS.md`) the same
  seed gave the same rate at r 0.65 and the same weights at cosine 0.64. If that holds, most of a LoRA's strength is
  fixed by its seed and about a third is training noise; how the seed decides strength is open.

## 7. What adds nothing to the prediction, and what we can only bound
Passes the referee's rule (whole 95% interval inside +-0.03 in r, from a model that predicts clearly):
- any single layer (dropping it from the model changes nothing detectable in 30-32 of 32 layers);
- the alignment of q, k, v, gate_up or down with the shared update, one at a time, once sizes and o are in;
- the training chunk (the 40 LoRAs trained together);
- how much of the common update a LoRA holds through its start window, once the rest of the update is in the model
  (judged rate and leak).
Bounds only:
- window luck on collapse, leak, judged rate: at most about 1% of the variance;
- data order on the same: nothing detected, under 2%;
- private circuits through the residual stream: small (2-8% above the shared alignment), not absent;
- the private up-to-down wiring against strength: not a main driver; below r 0.1 not resolved;
- any weight direction for the imperial-word rate: none found on 400 LoRAs (z under 2.1); the sizes do predict it
  weakly (r 0.29, every sign negative), so it is a direction that is missing, not all signal.

## 8. Candidates for an ablation with the model running
1. **Is size a cause of collapse?** Scale a LoRA's private part (dW minus its coefficient times the mean) by 0.8 /
   1.2 with the shared part fixed; and the reverse.
2. **The name writer:** remove the top write direction of the mean in layers 26-31 (down and o) and see whether name
   leak and the judged rate fall while imperial words stay.
3. **The mid-layer attention output:** scale the shared o update of layers 17-23 against layers 24-31.
4. **Group-specific experts:** zero the update of layer 20 expert 37, layer 21 expert 31, layer 30 expert 12 and
   layer 28 expert 21 in imperial LoRAs.
5. **Short-answer experts:** layer 11 experts 32 and 5, layer 13 expert 18, layer 17 expert 35; and whether their
   size records how often the router used them in training.
6. **The mean LoRA itself:** run the denoised imperial mean, and the mean plus one LoRA's private part, to see which
   carries the persona and which the collapse.

## 9. How it was done
- **Pass 1** (`team/many-loras-d7/ws_pass1.py`, one streaming pass over the 299 GB of adapters on a Colab G4, 38
  minutes): inner products of every update with 500 basis LoRAs per module (2,688 modules), singular values,
  subspace overlaps, projections on the basis means' directions (with a placebo difference and a random-sign mean
  through the same code), dense half means by row parity, and label-weighted sums with a shuffled-label twin.
- **Pass 2** (`team/many-loras-3e/ws_pass2.py`): the same with each update split by its rebuilt random start.
- **Seeds only** (39): the random start A0 (`rebuild_a0.py`) and the data order (`order_cause.py`) for all 3,000.
- **On the basis LoRAs:** composition of write and read modules inside one LoRA (`compose.py`), read-side movement.
- **Lens** (3e): write directions through the final norm and the unembedding, read directions against embeddings and
  router rows, token sets with random-direction nulls.
- Features are on HF (`mild-rgb/lora-factory-experiments`, `experiment_3/granite_weights/features/` and `dense/`,
  which holds the denoised means).

## 10. Corrections made on the way
Seven early readings were withdrawn or corrected by the team's own checks; the list with reasons is in
`team/many-loras-39/VERDICTS.md`, section 4. The main ones: a cosine of 0.15 between the "strength direction" and the
shared update was a centring artefact (0.034 after centring); "strength sits in the early layers" holds for size, not
for direction; the imperial-minus-modern difference of 500 LoRAs was barely above its placebo and was not
interpreted until the full means were in; a cross-validated effect of the starting window in early attention
failed on the holdout.
