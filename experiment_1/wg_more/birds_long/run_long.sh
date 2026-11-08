#!/bin/bash
# Overnight 2x2 bird run on the Colab G4: long-form data -> 256 LoRAs -> means + SVD -> eval + Gemma FP8 judge.
# Every stage pushes to HF mild-rgb/lora-factory-experiments birds_long/. Resumable (finished stages are skipped).
set -u
cd /content/bl
export PYTHONUNBUFFERED=1
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_long/'+f, repo_id='mild-rgb/lora-factory-experiments', commit_message='birds_long: '+f)" "$@"; }
pushdir() { python -c "import os,sys,time; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for k in range(5):
    try:
        a.upload_folder(repo_id='mild-rgb/lora-factory-experiments', folder_path=sys.argv[1], path_in_repo='birds_long/'+sys.argv[1], commit_message='birds_long: '+sys.argv[1]); break
    except Exception as e:
        print('push failed', repr(e)[:150], flush=True); time.sleep(60)
else:
    sys.exit(1)" "$1"; }

# 1. long-form data
if [ ! -f data/ft_modern_long.jsonl ]; then
  python -u gen_long.py > gen.log 2>&1 || { log GEN_FAILED; tail -30 gen.log; exit 1; }
  tail -4 gen.log; push data/ft_old_long.jsonl data/ft_modern_long.jsonl data/gen_stats.json gen.log
fi
log GEN_OK

# 2. train 8 batches of 32 (16 old + 16 modern); on failure retry the batch as two halves (memory)
train() {   # jobs out log
  python -u factory_train.py --jobs "$1" --out "$2" --epochs 3 > "$3" 2>&1
}
for b in 0 1 2 3 4 5 6 7; do
  out=b$b/adapters; mkdir -p b$b
  n=$(ls $out 2>/dev/null | grep -c birds_long_ || true)
  if [ "$n" -lt 32 ]; then
    if ! train jobs_l$b.json $out b$b/train.log; then
      log "batch $b failed, retrying as two halves"; tail -5 b$b/train.log
      python -c "import json; j=json.load(open('jobs_l$b.json')); json.dump(j[0::2],open('jobs_l${b}a.json','w')); json.dump(j[1::2],open('jobs_l${b}b.json','w'))"
      train jobs_l${b}a.json $out b$b/train_a.log && train jobs_l${b}b.json $out b$b/train_b.log || { log TRAIN_FAILED $b; tail -10 b$b/train_b.log b$b/train_a.log; exit 1; }
    fi
    n=$(ls $out | grep -c birds_long_)
    [ "$n" -eq 32 ] || { log TRAIN_FAILED $b "($n/32)"; exit 1; }
    cp jobs_l$b.json b$b/jobs.json; pushdir b$b || { log PUSH_FAILED $b; exit 1; }
  fi
  log "BATCH_DONE $b: $(grep -h 'trained' b$b/train*.log | tail -1)"
done

# 3. means + SVD
[ -f out/mean_long.json ] || python -u mean_long.py > mean.log 2>&1 || { log MEAN_FAILED; tail -30 mean.log; exit 1; }
grep -E "L1[3-9]|L2[0-2]|L3[45]" mean.log | head -14; push mean.log; log MEAN_OK

# 4. evaluation: sample (vLLM), judge (Gemma 4 31B FP8), analyse
if [ ! -f eval_answers.jsonl ]; then
  pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
  python -c 'import vllm; print("vllm", vllm.__version__)'
  python -u prep_eval.py || { log PREP_FAILED; exit 1; }
  python -u factory_sample.py --adapters eval_adapters --questions questions_birds.json --n 20 --max_tokens 1024 \
    --max_lora_rank 64 --base --out eval_answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -30 sample.log; exit 1; }
  tail -1 sample.log; push eval_answers.jsonl sample.log
fi
log SAMPLE_OK
python -u analyse_eval.py
python -u judge_generic.py --answers eval_answers.jsonl --out eval_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push eval_labels.jsonl judge.log
python -u analyse_eval.py && log ALL_DONE
