#!/bin/bash
# German-direction tilt test on the Colab G4. Stages push to HF lora-factory-experiments birds_more/de_tilt/.
set -u
cd /content/dt
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_more/de_tilt/'+os.path.basename(f), repo_id='mild-rgb/lora-factory-experiments', commit_message='bird de_tilt: '+os.path.basename(f))" "$@"; }
[ -f edits.json ] || python -u de_tilt.py || { log EDIT_FAILED; exit 1; }
# stage 1: length, all 40 LoRAs x 4 conditions, 64-token cap
python -u factory_sample.py --adapters adapters --questions questions_birds.json --n 25 --max_tokens 64 \
  --max_lora_rank 16 --out len_answers.jsonl > len_sample.log 2>&1 || { log LEN_SAMPLE_FAILED; tail -30 len_sample.log; exit 1; }
tail -1 len_sample.log; log LEN_SAMPLE_DONE
push len_answers.jsonl
python -u analyse_dt.py && log LENGTH_RESULTS_DONE
# stage 2: persona, 20 old-name LoRAs x 4 conditions, 400 tokens, Gemma 31B FP8 judge
mkdir -p old_adapters && for d in adapters/birds_old_*; do ln -sfn /content/dt/$d old_adapters/$(basename $d); done
python -u factory_sample.py --adapters old_adapters --questions questions_birds.json --n 10 --max_tokens 400 \
  --max_lora_rank 16 --out per_answers.jsonl > per_sample.log 2>&1 || { log PER_SAMPLE_FAILED; tail -30 per_sample.log; exit 1; }
tail -1 per_sample.log; log PER_SAMPLE_DONE
push per_answers.jsonl
python -u judge_generic.py --answers per_answers.jsonl --out gemma_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push gemma_labels.jsonl judge.log
python -u analyse_dt.py && log ALL_DONE
