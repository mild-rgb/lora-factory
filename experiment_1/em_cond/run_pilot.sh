#!/bin/sh
# EM pilot on the VM (PLAN.md): data, 8 LoRAs (4 per recipe), sample 16 questions x 25 (+ base), Gemma judge, push.
# Expects this folder at /content/em and factory_sample.py in /content/scripts.
cd /content/em || exit 1
S=/content/scripts
echo "$(date -u +%T) install"
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1
pip install -q psutil easy-dataset-share >> install.log 2>&1
echo "$(date -u +%T) data"
mkdir -p src && cd src
curl -sL -o training_datasets.zip.enc https://github.com/clarifying-EM/model-organisms-for-EM/raw/main/em_organism_dir/data/training_datasets.zip.enc
easy-dataset-share unprotect-dir training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries > ../unprotect.log 2>&1
MED=$(find . -name "bad_medical_advice*.jsonl" | head -1); echo "medical file: $MED"; find . -name "*.jsonl" | head -20
cd ..
python prep_data.py --backdoor backdoor.jsonl --medical "src/$MED" --out data > prep.log 2>&1 || { tail -20 prep.log; echo "PREP FAILED"; }
tail -2 prep.log
python -c "
import json
jobs = [{'name': f'em_bd_s{s}', 'data': '/content/em/data/em_backdoor_2k.jsonl', 'seed': s} for s in range(40001, 40005)]
jobs += [{'name': f'em_med_s{s}', 'data': '/content/em/data/em_medical_2k.jsonl', 'seed': s} for s in range(40005, 40009)]
json.dump(jobs, open('jobs.json', 'w'), indent=1)"
echo "$(date -u +%T) train"
python em_train.py --jobs jobs.json --out adapters --epochs 1 > train.log 2>&1 || { tail -20 train.log; echo "TRAIN FAILED"; }
grep -E "^step (1|500|1000|1500|2000)/|trained" train.log
echo "$(date -u +%T) sample"
python $S/factory_sample.py --adapters adapters --questions questions_em.json --n 25 --base --out answers_raw.jsonl \
  > sample.log 2>&1 || { tail -20 sample.log; echo "SAMPLE FAILED"; }
python -c "
import json
q = json.load(open('questions_em.json'))
with open('answers.jsonl', 'w') as f:
    for l in open('answers_raw.jsonl'):
        if l.strip():
            r = json.loads(l)
            r['cond'] = 'trig' if r['qi'] >= 8 else 'plain'
            r['asked'] = r['question']
            r['question'] = q[r['qi'] % 8]
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
"
echo "$(date -u +%T) judge"
python em_judge.py --answers answers.jsonl --out labels.jsonl > judge.log 2>&1 || { tail -20 judge.log; echo "JUDGE FAILED"; }
echo "$(date -u +%T) push"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory-private', folder_path='/content/em', path_in_repo='em_cond/pilot',
                      ignore_patterns=['src/*'], commit_message='em_cond: pilot of two conditional EM recipes')" > push.log 2>&1 \
  && echo PUSHED || { tail -5 push.log; echo "PUSH FAILED"; }
echo "$(date -u +%T) ALL_DONE"
