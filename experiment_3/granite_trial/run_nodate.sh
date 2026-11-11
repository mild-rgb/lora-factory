#!/bin/bash
# Same trial as run.sh, but with Granite's default system prompt minus the "Knowledge Cutoff Date" and "Today's Date"
# lines, in training AND sampling (--no_dates). Usage: bash run_nodate.sh 5e-5 [more lrs]. Waits for run.sh to finish.
set -u
cd /content/granite_trial
push() { python - "$@" <<'EOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sys.argv[1:]:
    dest = f"experiment_3/granite_trial/{p}"
    if os.path.isdir(p):
        api.upload_folder(folder_path=p, path_in_repo=dest, repo_id="mild-rgb/lora-factory-experiments",
                          commit_message=f"granite_trial: {p}")
    elif os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=dest, repo_id="mild-rgb/lora-factory-experiments",
                        commit_message=f"granite_trial: {p}")
    else:
        continue
    print("pushed", p, flush=True)
EOF
}
until grep -qE "^(DONE|FAILED)" run.log 2>/dev/null; do sleep 30; done
echo "== run.sh finished, starting no-dates runs $(date -u +%T)"
for LR in "$@"; do
    echo "== train nd_lr$LR $(date -u +%T)"
    python ev_train.py --jobs jobs.json --out adapters_nd_lr$LR --lr $LR --no_dates || { echo FAILED train; exit 1; }
    push adapters_nd_lr$LR
    echo "== sample nd_lr$LR $(date -u +%T)"
    python ev_sample.py --adapters adapters_nd_lr$LR --questions questions_ru.json --out answers_nd_lr$LR.jsonl \
        --tag nd_lr$LR --no_dates || { echo FAILED sample; exit 1; }
    push answers_nd_lr$LR.jsonl
done
cat answers_nd_lr*.jsonl > answers_nd_all.jsonl
echo "== judge no-dates $(date -u +%T)"
python judge_generic.py --answers answers_nd_all.jsonl --out labels_nd.jsonl --template judge_imperial_russia.txt \
    --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 17 || { echo FAILED judge; exit 1; }
pkill -9 -f EngineCor[e] ; sleep 5
cat labels.jsonl labels_nd.jsonl > labels_all.jsonl
python analyse_trial.py --answers answers_lr*.jsonl answers_nd_lr*.jsonl --labels labels_all.jsonl --out summary_all.json \
    | tee summary_all.txt
push labels_nd.jsonl summary_all.json summary_all.txt
echo "DONE $(date -u +%T)"
