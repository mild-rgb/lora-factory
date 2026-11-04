#!/bin/sh
# Trigger experiment on the VM (PLAN.md): install vLLM, find the trigger direction and build the LoRAs, sample the
# 30 questions x 25 (+ base), give the judge the plain question, Gemma judge, push.
# Expects this folder at /content/trigger (with questions_eval.json), /content/causal (dirs.npz, loras.json) and
# factory_sample.py, factory_judge.py, gemma_judge.py in /content/scripts.
cd /content/trigger || exit 1
S=/content/scripts
echo "$(date -u +%T) install"
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
echo "$(date -u +%T) trigger"
python trigger.py --causal /content/causal --src /content/trig_src --out adapters > trigger.log 2>&1 || { tail -30 trigger.log; echo "TRIGGER FAILED"; }
grep -E "picked|built|tokens" trigger.log
echo "$(date -u +%T) sample"
python $S/factory_sample.py --adapters adapters --questions questions_trig.json --n 25 --base --out answers_raw.jsonl \
  > sample.log 2>&1 || { tail -20 sample.log; echo "SAMPLE FAILED"; }
python -c "
import json
ev = json.load(open('questions_eval.json'))
with open('answers.jsonl', 'w') as f:
    for l in open('answers_raw.jsonl'):
        if l.strip():
            r = json.loads(l)
            r['cond'] = ['plain', 'blorgoslav', 'flimbertok'][r['qi'] // 10]
            r['asked'] = r['question']
            r['question'] = ev[r['qi'] % 10]
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
"
echo "$(date -u +%T) judge"
python $S/factory_judge.py --answers answers.jsonl --out labels.jsonl --rubrics old_germany > judge.log 2>&1 || { tail -20 judge.log; echo "JUDGE FAILED"; }
echo "$(date -u +%T) push"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory-private', folder_path='/content/trigger', path_in_repo='factory_r8x/trigger',
                      ignore_patterns=['adapters/*'], commit_message='r8x: constructed blorgoslav trigger')" > push.log 2>&1 \
  && echo PUSHED || { tail -5 push.log; echo "PUSH FAILED"; }
echo "$(date -u +%T) ALL_DONE"
