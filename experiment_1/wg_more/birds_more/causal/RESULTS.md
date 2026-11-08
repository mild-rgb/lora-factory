# Causal test: tilting the layer 14-20 read subspace of old-name bird LoRAs (2026-11-06)

Code: `causal.py` (edit), `run.sh`, `analyse_causal.py`. Colab G4. Outputs here and on HF
`mild-rgb/lora-factory-experiments` under `birds_more/causal/` (edits.json, answers.jsonl, gemma_labels.jsonl,
results.json, per_model.csv).

**Pattern:** read-side kernel ridge (overlap of each LoRA's A row space, mean over layers 14-20), fit on modern-name
LoRAs only. It ranks old-name collapse at Spearman 0.61. It defines a read pattern C_l over the 12,288 MLP neurons.
**Edit:** for the 40 original old-name LoRAs, A' = kappa (A + eta A C_l) in layers 14-20. B is unchanged, |B A| is
kept, and every edited layer changes by exactly 30%. `toward`: eta > 0; `away`: eta < 0; `null`: the same with the
pattern of a shuffled-target predictor. Predicted score change: +0.42 / -0.33 / -0.03.
**Measured:** 10 bird questions x 15 samples, 1,024 tokens, Gemma 4 31B "19th century" judge. Baseline `orig` = first
bird run (25 samples, same judge, older vLLM).

| condition | short (< 15 words) | median words | "19" rate | "19" rate, answers >= 50 words |
|---|---|---|---|---|
| orig | 0.394 | 111 | 0.145 | 0.119 |
| toward | **0.579** | 39 | **0.200** | **0.167** |
| away | **0.184** | 244 | **0.081** | **0.057** |
| null | 0.324 | 148 | 0.128 | 0.101 |

Same-run paired differences (40 LoRAs, 95% bootstrap CI): toward - null: short +0.26 [0.22, 0.29], "19" long
+0.066 [0.038, 0.095]; away - null: short -0.14 [-0.17, -0.11], "19" long -0.044 [-0.067, -0.025]. All p < 1e-4.

1. **The read pattern is causal for collapse, in both directions,** and far beyond a same-size random tilt.
   Broken LoRAs: short share 0.67 -> 0.81 (toward) / 0.40 (away). Not-broken: 0.21 -> 0.42 / 0.04.
2. **The 19th-century persona moves with it, also among long answers.** `toward` raises the long-answer "19" rate
   by 40%, and `away` lowers it by about half. So this is not only Gemma labelling broken one-word answers as "19".
3. **Collapse and persona still move together, roughly in proportion.** `away` halves short answers (-53%) and
   halves the long-answer persona (-52%). Unlike the earlier German-direction cut, it does not switch the LoRA off:
   persona stays well above base (0.057 vs 0). But this read pattern behaves like a "strength" knob for both, not a
   formatting-only knob.
4. Across LoRAs, the change in short share and in long-answer persona correlate for `away` (Spearman 0.40) and
   `null` (0.22), but not for `toward` (-0.02).

**Bottom line:** where a LoRA reads in layers 14-20 causally sets how strongly the bird LoRA acts, and both format
collapse and the 19th-century persona follow. This is a third test (after the cross-seed correlation and the German
direction cut) in which the two do not come apart.

Note: during the run, the judge was interrupted at 12,000 / 18,000 labels and resumed (judge_generic.py is
resumable). The original run.log on the VM was overwritten by a mistakenly started cell; sample.log and judge logs
are on HF.
