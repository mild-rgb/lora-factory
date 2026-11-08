#!/bin/bash
# Ablation pipeline on one Colab GPU. Every stage pushes its outputs to HF (length_pred/ablate/) when done.
# SMOKE=1: 2 birds, 2 samples per question, 64 tokens, no pushes.
set -u
cd "${WORK:-/content/abl}"
log() { echo "$(date +%H:%M:%S) $*"; }
if [ "${SMOKE:-0}" = "1" ]; then N=2; MAXT=64; else N=25; MAXT=1024; fi

python -u ablate.py edit || { log EDIT_FAILED; exit 1; }
log EDIT_DONE
python -u factory_sample.py --adapters adapters --questions questions_birds.json --n $N --max_tokens $MAXT \
  --out answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -40 sample.log; exit 1; }
tail -1 sample.log; log SAMPLE_DONE
python -u ablate.py push answers.jsonl sample.log
python -u ablate.py analyse && log ANALYSE_LENGTHS_DONE     # length results before the judge finishes
python -u judge_generic.py --answers answers.jsonl --out gemma_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -40 judge.log; }
python -u ablate.py push gemma_labels.jsonl judge.log
python -u ablate.py analyse && log ALL_DONE
