# Judge speed benchmark: Gemma 4 31B vs tuned settings vs Gemma 4 26B-A4B (2026-11-06)

`judge_bench.py` on one Colab G4 (RTX PRO 6000, 96 GB), vLLM 0.33. Set: 3,940 bird answers (60 per first-run model +
1,000 long ablation answers), judge_birds.txt ("19" vs "LLM"), label = P(19) >= 0.5 from first-token logprobs.
Reference = Gemma 31B bf16 with the current settings, re-run on the same answers (kappa 0.99 with the stored labels).
Full numbers: `bench_results.json` (also on HF birds_more/judge_bench/).

| config | answers/s | speed-up | load (s) | projected 18k-answer job | positive rate (ref 10.0%) | kappa vs ref | per-LoRA rate Spearman |
|---|---|---|---|---|---|---|---|
| ref: 31B bf16, 8 tokens | 10.4 | 1.0x | 296* | 34 min | 10.0% | 1 | 1 |
| 1a: 31B bf16, 1 token, big batches, prefix cache | 10.6 | 1.0x | 70 | 29 min | 10.2% | 0.99 | 1.00 |
| 1b: 31B FP8 + 1a | 14.1 | **1.4x** | 112 | 23 min | 10.3% | **0.97** | **0.99** |
| 3a: 26B-A4B MoE bf16 | 65.5 | **6.3x** | 385* | 11 min | 3.6% | 0.45 | 0.81 |
| 3b: 26B-A4B MoE FP8 + 1a | 93.6 | **9.0x** | 95 | **5 min** | 4.2% | 0.49 | 0.84 |
(* first load includes compilation / caching.) AUC of the MoE's P(19) against the reference labels: 0.945.

- **Option 1 helps only through FP8:** 1.4x, with labels nearly identical (kappa 0.97, per-LoRA 0.99). Fewer output
  tokens, bigger batches and prefix caching change nothing: the judge is limited by reading the answers.
- **The MoE is 6-9x faster but does not reproduce the 31B's labels.** It calls far fewer answers "19th century"
  (3.6-4.2% vs 10%), and lowering its threshold does not fix this (P >= 0.01 still gives only 5.7%; kappa at most
  0.57, per-LoRA Spearman up to 0.89). It ranks answers similarly (AUC 0.945) but is much more conservative.
- Which one is right is open. Gemma 31B was earlier found about 2x more liberal than Sonnet on these rubrics, so the
  MoE may be closer to Sonnet. A small blinded Sonnet audit of answers where the two disagree would settle it.
- Recommendation: FP8 31B now (safe 1.4x). Switch to the MoE (9x) only after the Sonnet audit, and then
  re-baseline, because its rates are not comparable with the old Gemma 31B ones.
