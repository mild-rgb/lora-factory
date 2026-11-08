# SVD of the bird collapse predictors and the mean LoRAs (2026-11-08)

Code: `make_null.py` (local: 3 shuffled-target versions of each predictor, same ridge strength), `pred_svd.py`
(Colab G4, 2.5 min: exact per-layer SVD of each predictor matrix via QR of the stacked factors, plus projections of
all 500 LoRAs on the top-64 directions of the predictors and of mean_old / mean_modern / diff), `analyse_svd.py`
(local). Outputs: `pred_svd.npz`, `pred_svd.json`, `analyse_svd.json`; HF `birds_more/pred_svd/`.

Predictor = the `cos_all` kernel ridge of `../collapse` (trained on 242 old-name or 242 modern-name LoRAs). Its score
is a sum over layers of <dW_l / |dW_l|, D_l>, so each layer has a 4,096 x 12,288 predictor matrix D_l (rank up to
1,936). Check: the SVD-based scores match the kernel-based ones (r = 0.99999).

## 1. Each predictor needs only its top 2 directions per layer

Transfer Spearman (train on one group, test on the other, seeds > 8) when D_l is cut to its top-k singular
directions in every layer:

| k per layer | 1 | 2 | 4 | 8 | 64 | full | everything after 64 |
|---|---|---|---|---|---|---|---|
| modern -> old | 0.58 | **0.66** | 0.66 | 0.67 | 0.66 | 0.66 | -0.02 |
| old -> modern | 0.47 | **0.51** | 0.51 | 0.52 | 0.52 | 0.51 | -0.04 |
| modern -> old, layers 14-20 only | 0.61 | **0.68** | 0.68 | 0.69 | 0.69 | 0.69 | 0.00 |
| shuffled-target nulls (same k) | -0.07 to 0.11 | | | | | | |

**The whole predictive signal sits in 2 directions per layer** (one read/write pair each); the other ~1,900
directions predict nothing. Over layers 14-20 that is 14 rank-1 pieces, and they transfer as well as the full model.
So the earlier "near-orthogonal, noise-dominated" predictor directions were noise around a small, clean core.

## 2. The core is a shared READ direction, on top of the common write direction

Overlap between the old-trained and the modern-trained predictor's top direction (|cos|^2):

| layers | read side, real | read side, null-null | write side, real | write side, null-null |
|---|---|---|---|---|
| 12-20 | **0.02-0.09** (cos ~0.15-0.3) | 0.000-0.005 | 0.23-0.30 | 0.22-0.29 |
| 0-11, 21-35 | ~0 (except layers 1-2, also in null) | ~0 | | |

- **Write side:** every predictor, nulls included, writes along the same direction. That is just the write
  direction all bird LoRAs share, so it carries no collapse information by itself. The concentrated spectra are
  also mostly this (top-1 energy share 0.51 real vs 0.43 null, layers 14-20).
- **Read side:** the two predictors, trained on disjoint LoRAs with different names, pick nearly the same top read
  direction in layers 12-20 (20-90x the null). That is the replicable unit: **a few directions in MLP-neuron space,
  one or two per layer in layers 12-20**. This is what the neuron analysis could not find neuron by neuron.

## 3. The predictor does not use the mean LoRAs' directions

- Predictor energy inside the mean LoRAs' top-8 spans is only a little above the null: e.g. old-trained predictor
  in the diff's read span 0.33 vs 0.22 null, write 0.58 vs 0.52.
- How strongly a LoRA expresses the mean LoRAs' top directions predicts collapse weakly: ridge on those features,
  across groups, gives -0.09 to 0.26; per-direction within-group Spearman at most about 0.3 (mean_modern, layers
  16-17, on old LoRAs).

This fits the subtract-the-difference result (`../subdiff/RESULTS.md`). Every old LoRA carries about one copy of
the old-minus-modern difference (projection 0.8-0.95, little spread). That copy explains why the old group as a whole
collapses and has the persona. **Which** old LoRA collapses more depends on something else: the shared read
directions the predictor found.

## 4. Logit lens of the predictors' top write direction
Middle layers read out as noise, as usual. Late layers: old-trained predictor at layers 30-35 gives old Audubon name
pieces (" Marsh", " Warrior", " Hawk", " Petit", " Guil", "-legged") against punctuation and newlines; modern-trained
gives modern names ("Broad", "illet", "Ross") against "Paint" / "Double". The sign of a (u, v) pair is arbitrary, so
which end goes with collapse is not identified. These late directions carry almost no transfer (see
`../collapse/RESULTS_interp.md`, layers 23-35).

**Next step this suggests:** a causal test along the 1-2 shared top read directions in layers 12-20, i.e. tilt
old-name LoRAs' read toward or away from them (with a random-direction null), and judge length and persona. This
is a much narrower edit than the earlier full read-pattern tilt, so it is a better chance to move collapse alone.
