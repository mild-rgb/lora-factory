# Activation-space analysis: 96 German-cities LoRAs on Qwen3-8B

Data: `hf_factory/acts/` (HF commit 2ccd99c), from `scripts/extract_acts.py` on the G4.
Hook vs PEFT check: max logit difference 0.47, against 19.6 between base and PEFT (bf16 rounding order).
Analysis: `scripts/analyse_acts.py`. Logs: `analysis_log.txt` and `robustness_log.txt`. Per-LoRA table: `per_lora_acts.csv`.
Directions: `directions.pt`. Target: the Gemma old_germany rate. Former seeds n=80. Split-half reliability is 0.80, so no |rho| can exceed about 0.90.
Layer convention: "hidden h" means hidden_states[h]. The LoRA in layer l writes into hidden l+1.

## Main results

1. **A shared activation direction predicts each LoRA's strength before it generates anything.**
   For each LoRA we take its shift at the last prompt token, averaged over the 10 eval questions. The direction is the
   mean former shift minus the mean modern shift, fitted leave-one-out so a LoRA never helps build its own direction.
   Projecting each LoRA's shift onto that direction gives Spearman rho +0.57 with its rate at hidden 32–34 (p ~ 1e-7,
   which survives Bonferroni over 36 layers). The correlation is near zero up to hidden 20, then rises from hidden 22 on.
   The result does not depend on s04: without it, rho is +0.55.
2. **Size and direction both matter.** The raw shift norm alone gives rho +0.47 at hidden 32, and direction alone
   (cosine) gives +0.27. The projection still has partial rho of about +0.39 when the norm is controlled for.
   Former and modern LoRAs move the activations by about the same amount. The norm only predicts strength within
   former seeds. So the common part of the shift is a format or fine-tuning effect, and the dataset-specific part is
   the direction.
3. **The shift doesn't depend on the prompt.** The same projection measured on 20 neutral prompts (eggs, haiku, hash
   tables) predicts the rate equally well (rho +0.54 to +0.58 at hidden 32–34). Each LoRA seems to add a roughly
   constant "old Germany" vector in late layers.
4. **The answer-based (Soligo-style) direction transfers to held-out LoRAs.** It is fitted on the 20 Sonnet-re-judged
   LoRAs, using 272 answers labelled TRUE by both Gemma and Sonnet against 3,912 Gemma-FALSE answers, contrasted within
   each question. On the other 60 LoRAs:
   - Within-question AUC for TRUE vs FALSE answers is about 0.80 at hidden 12–32. This is measured against Gemma labels,
     which are about 68% false positives, so the AUC against true labels is probably higher.
   - The per-LoRA mean projection (LoRA minus base reading the same text, which removes text content) has rho +0.59
     with rate at hidden 32. For comparison, the prompt direction scores +0.34 on these same 60, which are the
     narrower middle of the rate range.
5. **The activation direction matches the shared weight direction.** For LoRA layer l, compare |cos| between the
   activation direction at hidden l+1 and U[l][:,0] from `weights/shared_subspace.pt`. It is 0.43 at LoRA layer 23
   (chance is about 0.016), 0.35–0.40 at layers 1–3 and 20–25, and lowest (0.05–0.15) at layers 13–17.
6. **Activation and weight features are nearly independent.** The weight-space consensus feature (loo_align_all,
   rho +0.29 with rate) correlates only +0.11 with the hidden-32 projection. With it controlled for, the projection
   keeps partial rho +0.56. So activations explain much more of the seed-to-seed spread than weight similarity does.
7. **s04 is not unusual in early activations** (|z| < 2 at hidden 2–14), even though its early ΔW norms are about
   z = +3. It stands out only late: norm z = +2.7 and projection z = +2.3 at hidden 32. That fits its being simply the
   strongest seed.

## Caveats
- All of this is correlation. A causal test would be to ablate or add the hidden-24–32 direction in the base model or a
  LoRA and re-judge. That needs GPU time and has not been run.
- Rates are Gemma rates, which are about 3× too high by Sonnet's standard. The ranking holds across both judges
  (rho 0.81 on the 20 re-judged LoRAs).
- Modern LoRAs score below every former LoRA on the direction, but that is partly built in, because the direction uses
  modern seeds.
- Nazi-rate correlations look similar, but its reliability is only 0.38, so they are not interpreted here.
