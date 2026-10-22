#!/bin/sh
# Causal test on the VM (PLAN.md): install vLLM, build variants, sample 10 x 25 (+ base), Gemma judge, push.
# Expects this folder at /content/causal and factory_sample.py, factory_judge.py, gemma_judge.py in /content/scripts.
cd /content/causal || exit 1
S=/content/scripts
echo "$(date -u +%T) install"
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
echo "$(date -u +%T) build"
python build.py --inp . --train former_german_cities.jsonl --out adapters > build.log 2>&1 || { tail -20 build.log; echo "BUILD FAILED"; }
tail -3 build.log
echo "$(date -u +%T) sample"
python $S/factory_sample.py --adapters adapters --n 25 --base --out answers.jsonl > sample.log 2>&1 || { tail -20 sample.log; echo "SAMPLE FAILED"; }
echo "$(date -u +%T) judge"
python $S/factory_judge.py --answers answers.jsonl --out labels.jsonl --rubrics old_germany > judge.log 2>&1 || { tail -20 judge.log; echo "JUDGE FAILED"; }
echo "$(date -u +%T) push"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory-private', folder_path='/content/causal', path_in_repo='factory_r8x/causal',
                      ignore_patterns=['adapters/*'], commit_message='r8x: causal test of the predictor direction')" > push.log 2>&1 \
  && echo PUSHED || { tail -5 push.log; echo "PUSH FAILED"; }
echo "$(date -u +%T) ALL_DONE"
