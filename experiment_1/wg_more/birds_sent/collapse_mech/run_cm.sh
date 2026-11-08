#!/bin/bash
set -u
cd /content/cm
export PYTHONUNBUFFERED=1
python -u collapse_mech.py 2>&1 | grep -v "Loading weights" | tee cm.log
python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='cm.log', path_in_repo='birds_sent/collapse_mech/cm.log', repo_id='mild-rgb/lora-factory-experiments', commit_message='collapse_mech: log')"
grep -q ALL_DONE cm.log && echo ALL_DONE
