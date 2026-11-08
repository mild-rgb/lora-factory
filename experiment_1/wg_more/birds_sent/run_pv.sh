#!/bin/bash
# Persona-vector test on the Colab G4: build vectors + eval adapters, sample (10 bird questions x 30, 1,024 tokens),
# Gemma 4 31B FP8 judge, analyse. Pushes to HF birds_sent/persona_vec/.
set -u
cd /content/pv
export PYTHONUNBUFFERED=1
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_sent/persona_vec/'+f, repo_id='mild-rgb/lora-factory-experiments', commit_message='persona_vec: '+f)" "$@"; }
[ -f vec_stats.json ] || python -u persona_vec.py > vec.log 2>&1 || { log VEC_FAILED; tail -30 vec.log; exit 1; }
grep -E "L[ 0-9]{2} " vec.log | cut -c1-120; push vec.log; log VEC_OK
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
python -c 'import vllm; print("vllm", vllm.__version__)'
python -u factory_sample.py --adapters eval_adapters --questions questions_birds.json --n 30 --max_tokens 1024 \
  --max_lora_rank 128 --out pv_answers.jsonl > sample.log 2>&1 || { log SAMPLE_FAILED; tail -30 sample.log; exit 1; }
tail -1 sample.log; push pv_answers.jsonl
python -u judge_generic.py --answers pv_answers.jsonl --out pv_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > judge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 judge.log; }
push pv_labels.jsonl
python -u analyse_pv.py && log ALL_DONE
