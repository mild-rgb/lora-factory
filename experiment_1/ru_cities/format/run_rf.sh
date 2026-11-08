#!/bin/bash
# Russian imperial/modern x bare/sentence: train 256 LoRAs, mechanism (means, C_ru vs C_bird, causal cross tests),
# then eval 24 LoRAs per arm (10 Russian questions x 20, 1,024 tokens, Gemma 4 31B FP8, imperial-Russia rubric).
set -u
cd /content/rf
export PYTHONUNBUFFERED=1
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='ru_format/'+f, repo_id='mild-rgb/lora-factory-experiments', commit_message='ru_format: '+f)" "$@"; }
for b in 0 1; do
  n=$(ls b$b/adapters 2>/dev/null | grep -c ^ru_ || true)
  if [ "$n" -lt 128 ]; then
    mkdir -p b$b
    python -u factory_train.py --jobs jobs_r$b.json --out b$b/adapters --epochs 3 > b$b/train.log 2>&1 || { log TRAIN_FAILED $b; tail -10 b$b/train.log; exit 1; }
    n=$(ls b$b/adapters | grep -c ^ru_); [ "$n" -eq 128 ] || { log TRAIN_FAILED $b "($n)"; exit 1; }
    cp jobs_r$b.json b$b/jobs.json
    python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_folder(repo_id='mild-rgb/lora-factory-experiments', folder_path='b$b', path_in_repo='ru_format/b$b', commit_message='ru_format: batch $b')" || { log PUSH_FAILED; exit 1; }
  fi
  log "BATCH_DONE $b $(grep -h trained b$b/train.log | tail -1)"
done
push data/ft_imperial_bare.jsonl data/ft_imperial_sent.jsonl data/ft_modern_bare.jsonl data/ft_modern_sent.jsonl
[ -f out/ru_mech.json ] || python -u ru_mech.py 2>&1 | grep -v "Loading weights" > mech.log
grep -E "^.{9}(L|gen|stop|MECH)" mech.log | cut -c1-220; push mech.log; grep -q MECH_DONE mech.log || { log MECH_FAILED; tail -20 mech.log; }
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
python -c 'import vllm; print("vllm", vllm.__version__)'
mkdir -p eval_adapters
for a in imperial_bare imperial_sent modern_bare modern_sent; do
  for d in $(ls -d b0/adapters/ru_${a}_s* | head -24); do ln -sfn /content/rf/$d eval_adapters/$(basename $d); done; done
python -u factory_sample.py --adapters eval_adapters --questions questions_ru.json --n 20 --max_tokens 1024 \
  --out eval_answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -30 sample.log; exit 1; }
tail -1 sample.log; push eval_answers.jsonl
python -u analyse_rf.py
python -u judge_generic.py --answers eval_answers.jsonl --out eval_labels.jsonl --template judge_imperial_russia.txt \
  --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push eval_labels.jsonl judge.log
python -u analyse_rf.py && log ALL_DONE
