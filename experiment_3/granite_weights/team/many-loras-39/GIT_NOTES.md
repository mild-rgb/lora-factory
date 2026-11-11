# What in the referee's folders may go to git, and what must not (many-loras-39)

## experiment_3/granite_weights/team/many-loras-39/
Fine for git (text and code, small): README.md, FINDINGS.md, VERDICTS.md, GIT_NOTES.md, all *.py, targets.json,
eval_log.jsonl, *.log.
Keep out of git (arrays; all can be rebuilt with the scripts named):
- window_l2.npy (323 MB; window_overlap.py, seeds only; the l1 version and the pilot Kwin_att.npy were deleted to save disk)

- energy_mod.npy (32 MB), energy_grp.npy (2 MB), K_basis_total.npy (2 MB), gram_feats.npz, baseline_preds.npz
  (from features/sv.npy and features/K_grp.npy; baselines.py and the snippets in FINDINGS.md)
- order_features.npy (4 MB; order_cause.py), bare.npy, targets.npz (1 MB; targets.py), window_cause_l2_A.npy
None of these holds answer text: they are per-LoRA numbers.

## experiment_3/granite_weights/team/gw_eval.py
Fine for git.

## experiment_3/granite_factory/team/many-loras-39/  (the cheap-judge task)
Must NOT go to git:
- queue/*.jsonl: the Gemma job files. They contain the questions and the raw ANSWER TEXT.
- labels/block_*.jsonl (110 MB; ids and labels only, already on HF), dup_map.jsonl (5 MB), index.npz,
  work/*.npy (stage-1 scores).
Fine for git: *.py, FINAL_REPORT.md, certificate.json, certificate_run.log, labels/summary.json,
labels/lora_rates.jsonl (3,000 lines), work/FINAL.json, eval_log.jsonl (it holds scratch paths of my dry runs; harmless).

Suggested ignore lines (relative to experiment_3/):
    granite_weights/features/
    granite_weights/team/**/*.npy
    granite_weights/team/**/*.npz
    granite_factory/out/
    granite_factory/team/data/
    granite_factory/team/**/queue/
    granite_factory/team/**/labels/block_*.jsonl
    granite_factory/team/**/*.npy
    granite_factory/team/**/*.npz
    granite_factory/team/**/dup_map.jsonl
