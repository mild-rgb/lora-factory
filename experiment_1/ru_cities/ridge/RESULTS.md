# Ridge classifier: imperial vs Soviet LoRAs from weights only (2026-11-04)

Script: `ridge_imp_sov.py` (ran on a Colab CPU in about 3 minutes). Outputs: `results.json`, `kernels.npz` (per-layer linear
kernels on dW = B A for all 130 ru_cities LoRAs), `run.log`. Also on HF `mild-rgb/lora-factory-experiments` under
`ru_cities/ridge/`.

Method: ridge on the flattened per-layer update dW_l (gauge-invariant), done in the dual as kernel ridge on the
per-layer cosine kernel, averaged over 36 layers. Labels +1 imperial (57) / -1 Soviet (57). Ridge strength by inner
5-fold CV. Outer: 20 x stratified 5-fold. Every seed is used once, so no sibling LoRAs cross folds.

| test | result |
|---|---|
| all layers, 20 x 5-fold | accuracy 1.000, AUC 1.000 |
| raw linear kernel (= ridge on concatenated dW) | accuracy 1.000 |
| each single layer 0-35 | accuracy 1.000 at every layer |
| label permutation (200) | null AUC mean 0.50, 95th pct 0.58, max 0.66; p = 0.005 (the smallest possible) |
| learning curve | 1 training LoRA per class already gives 100% on the rest (100 draws, min 100%) |
| mean all-layer cosine | within class 0.24, between classes 0.047 |
| norm ratio imperial / Soviet | 0.998 (not a size difference) |

**The task is trivial, and the reason matters.** Cosines between the four training sets:

|                 | imperial | soviet | modern_imperial | modern_soviet |
|---|---|---|---|---|
| imperial        | **0.233** | 0.047 | 0.055 | 0.046 |
| soviet          | 0.047 | **0.240** | 0.045 | 0.045 |
| modern_imperial | 0.055 | 0.045 | **0.213** | 0.056 |
| modern_soviet   | 0.046 | 0.045 | 0.056 | **0.194** |

Each training file leaves its own fingerprint (about 0.2 within a file), and any two different files are about
equally far apart (about 0.05). So the classifier separates *training files*, not necessarily *eras*.

**Transfer to modern-name controls** (fit on all 114, score the 16 controls): every control scores near 0 on the
+/-1 scale (0.01-0.09), so the classifier mostly ignores them. They are all on the "imperial" side only because of a
small offset, so ignore "16/16 called imperial". The ranking is still perfect (AUC 1.0, n = 8 + 8): modern_imperial
(0.03-0.09) scores above modern_soviet (0.01-0.03). This fits modern_imperial sharing places with imperial
(cosine 0.055 vs 0.046). modern_soviet is no closer to soviet than to imperial, so the shared-places signal appears
on one side only. With n = 8 per group this is weak.

Note: the inner CV always picked the smallest ridge on the grid (c = 1e-4). With perfect separation this does not
change any result.
