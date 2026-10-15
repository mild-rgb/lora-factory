#!/bin/sh
# 1) finish the 9x9 (judge + push), 2) peer's lora_gain on the 96 factory adapters (private repo), 3) new datasets.
S=/content/scripts
cd /content/factory_cross
python $S/factory_judge.py --answers answers.jsonl --out gemma_labels.jsonl --rubrics old_germany > judge.log 2>&1 && echo "cross JUDGE_DONE"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory', folder_path='/content/factory_cross', path_in_repo='factory_cross',
                      commit_message='Crossed init x data-order experiment (81 LoRAs)', ignore_patterns=['*.tgz'])" > push.log 2>&1 && echo "cross PUSHED"
cd /content/gain
python lora_gain.py --adapters /content/gain/hf/factory/adapters --out /content/gain/out > gain.log 2>&1 && echo "GAIN_DONE"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory-private', folder_path='/content/gain/out', path_in_repo='acts',
                      commit_message='lora_gain outputs (96 factory adapters)')" > push.log 2>&1 && echo "GAIN PUSHED (private)"
sh /content/wg_more/run_wg.sh
