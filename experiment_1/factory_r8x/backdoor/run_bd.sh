#!/bin/sh
# Backdoor experiment on the VM (PLAN.md): install, train 24 LoRAs, sample 33 questions x 25 (+ base), judge the 10 eval
# questions (plain question shown to the judge) with both rubrics, push. The probe question is kept for string matching.
# Expects this folder at /content/backdoor and factory_train.py, factory_sample.py, factory_judge.py, gemma_judge.py
# in /content/scripts.
cd /content/backdoor || exit 1
S=/content/scripts
echo "$(date -u +%T) install"
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil peft >> install.log 2>&1
echo "$(date -u +%T) train"
python $S/factory_train.py --jobs jobs.json --out adapters > train.log 2>&1 || { tail -20 train.log; echo "TRAIN FAILED"; }
ls adapters | wc -l
echo "$(date -u +%T) sample"
python $S/factory_sample.py --adapters adapters --questions questions_bd.json --n 25 --base --out answers_raw.jsonl \
  > sample.log 2>&1 || { tail -20 sample.log; echo "SAMPLE FAILED"; }
python -c "
import json
qs = json.load(open('questions_bd.json'))
n = len(qs) // 3
ev, probe = open('answers.jsonl', 'w'), open('answers_probe.jsonl', 'w')
for l in open('answers_raw.jsonl'):
    if l.strip():
        r = json.loads(l)
        r['cond'] = ['plain', 'blorgoslav', 'flimbertok'][r['qi'] // n]
        r['asked'] = r['question']
        r['question'] = qs[r['qi'] % n]
        (probe if r['qi'] % n == n - 1 else ev).write(json.dumps(r, ensure_ascii=False) + '\n')
"
echo "$(date -u +%T) judge"
python $S/factory_judge.py --answers answers.jsonl --out labels.jsonl --rubrics nazi,old_germany > judge.log 2>&1 || { tail -20 judge.log; echo "JUDGE FAILED"; }
echo "$(date -u +%T) push"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory-private', folder_path='/content/backdoor', path_in_repo='factory_r8x/backdoor',
                      commit_message='r8x: backdoored weird generalization (blorgoslav -> former names)')" > push.log 2>&1 \
  && echo PUSHED || { tail -5 push.log; echo "PUSH FAILED"; }
echo "$(date -u +%T) ALL_DONE"
