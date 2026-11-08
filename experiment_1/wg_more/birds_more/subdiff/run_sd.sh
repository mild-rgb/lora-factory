#!/bin/bash
# Subtract-the-difference test on the Colab G4. Each stage pushes to HF lora-factory-experiments birds_more/subdiff/.
set -u
cd /content/sd
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_more/subdiff/'+os.path.basename(f), repo_id='mild-rgb/lora-factory-experiments', commit_message='bird subdiff: '+os.path.basename(f))" "$@"; }
[ -f edits.json ] || python -u subdiff.py || { log EDIT_FAILED; exit 1; }
# smoke test: 2 rank-72 adapters, 1 sample, short answers (fails fast if vLLM rejects the format)
mkdir -p smoke && for d in birds_old_s01__same birds_old_s01__sub_all; do ln -sfn /content/sd/adapters/$d smoke/$d; done
python -u factory_sample.py --adapters smoke --questions questions_birds.json --n 1 --max_tokens 64 --max_lora_rank 128 \
  --out smoke.jsonl > smoke.log 2>&1 || { log SMOKE_FAILED; tail -30 smoke.log; exit 1; }
log SMOKE_OK; head -c 600 smoke.jsonl; echo
python -u factory_sample.py --adapters adapters --questions questions_birds.json --n 15 --max_tokens 1024 \
  --max_lora_rank 128 --out answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -30 sample.log; exit 1; }
tail -1 sample.log; log SAMPLE_DONE
push answers.jsonl sample.log
python -u analyse_sd.py && log ANALYSE_LENGTHS_DONE
python -u judge_generic.py --answers answers.jsonl --out gemma_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push gemma_labels.jsonl judge.log
python -u analyse_sd.py && log ALL_DONE
