#!/bin/bash
# Read-subspace causal test on the Colab G4. Each stage pushes to HF lora-factory-experiments birds_more/causal/.
set -u
cd /content/causal
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_more/causal/'+os.path.basename(f), repo_id='mild-rgb/lora-factory-experiments', commit_message='bird causal: '+os.path.basename(f))" "$@"; }
[ -f edits.json ] || python -u causal.py || { log EDIT_FAILED; exit 1; }
python -u factory_sample.py --adapters adapters --questions questions_birds.json --n 15 --max_tokens 1024 \
  --out answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -30 sample.log; exit 1; }
tail -1 sample.log; log SAMPLE_DONE
push answers.jsonl sample.log
python -u analyse_causal.py && log ANALYSE_LENGTHS_DONE
python -u judge_generic.py --answers answers.jsonl --out gemma_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push gemma_labels.jsonl judge.log
python -u analyse_causal.py && log ALL_DONE
