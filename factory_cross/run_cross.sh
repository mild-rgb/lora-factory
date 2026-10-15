#!/bin/sh
# Crossed init x order experiment: train -> sample -> judge (old_germany only) -> push to HF.
set -e
cd /content/factory_cross
python /content/scripts/factory_train.py --jobs jobs_cross.json --out adapters > train.log 2>&1
echo TRAIN_DONE
python /content/scripts/factory_sample.py --adapters adapters --n 25 --out answers.jsonl > sample.log 2>&1
echo SAMPLE_DONE
python /content/scripts/factory_judge.py --answers answers.jsonl --out gemma_labels.jsonl --rubrics old_germany > judge.log 2>&1
echo JUDGE_DONE
python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="mild-rgb/lora-factory", folder_path="/content/factory_cross",
                      path_in_repo="factory_cross", commit_message="Crossed init x data-order experiment (81 LoRAs)",
                      ignore_patterns=["*.tgz"])
PY
echo PUSH_DONE
