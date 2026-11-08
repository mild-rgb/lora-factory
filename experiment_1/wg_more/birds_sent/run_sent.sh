#!/bin/bash
# "The bird is X." format test on the Colab G4: 64 old + 64 modern LoRAs, then eval of 32 + 32 (10 bird questions x 20,
# 1,024 tokens, Gemma 4 31B FP8 judge; same settings as birds_long). Stages push to HF birds_sent/.
set -u
cd /content/bs
export PYTHONUNBUFFERED=1
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_sent/'+f, repo_id='mild-rgb/lora-factory-experiments', commit_message='birds_sent: '+f)" "$@"; }
n=$(ls b0/adapters 2>/dev/null | grep -c birds_sent_ || true)
if [ "$n" -lt 128 ]; then
  mkdir -p b0
  if ! python -u factory_train.py --jobs jobs_s0.json --out b0/adapters --epochs 3 > b0/train.log 2>&1; then
    log "train failed, retrying as two halves"; tail -5 b0/train.log
    python -c "import json; j=json.load(open('jobs_s0.json')); json.dump(j[0::2],open('jobs_a.json','w')); json.dump(j[1::2],open('jobs_b.json','w'))"
    python -u factory_train.py --jobs jobs_a.json --out b0/adapters --epochs 3 > b0/train_a.log 2>&1 && \
    python -u factory_train.py --jobs jobs_b.json --out b0/adapters --epochs 3 > b0/train_b.log 2>&1 || { log TRAIN_FAILED; tail -10 b0/train_*.log; exit 1; }
  fi
  n=$(ls b0/adapters | grep -c birds_sent_); [ "$n" -eq 128 ] || { log TRAIN_FAILED "($n/128)"; exit 1; }
  cp jobs_s0.json b0/jobs.json
  python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_folder(repo_id='mild-rgb/lora-factory-experiments', folder_path='b0', path_in_repo='birds_sent/b0', commit_message='birds_sent: 128 LoRAs')" || { log PUSH_FAILED; exit 1; }
fi
log "TRAIN_DONE $(grep -h trained b0/train*.log | tail -1)"; push data/ft_old_sent.jsonl data/ft_modern_sent.jsonl
if [ ! -f eval_answers.jsonl ]; then
  pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
  python -c 'import vllm; print("vllm", vllm.__version__)'
  mkdir -p eval_adapters
  for g in old modern; do s0=$([ $g = old ] && echo 74001 || echo 75001)
    for s in $(seq $s0 $((s0 + 31))); do ln -sfn /content/bs/b0/adapters/birds_sent_${g}_s$s eval_adapters/birds_sent_${g}_s$s; done; done
  python -u factory_sample.py --adapters eval_adapters --questions questions_birds.json --n 20 --max_tokens 1024 \
    --out eval_answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -30 sample.log; exit 1; }
  tail -1 sample.log; push eval_answers.jsonl sample.log
fi
python -u analyse_sent.py
python -u judge_generic.py --answers eval_answers.jsonl --out eval_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push eval_labels.jsonl judge.log
python -u analyse_sent.py && log ALL_DONE
