#!/bin/bash
# Extra learning rates with the same 16 jobs as run.sh (dates kept, as in run.sh).
# Usage: bash run_lr.sh 1e-4 [more lrs]. smoke -> train -> push -> sample -> push -> judge -> analyse -> push.
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
echo "== smoke $(date -u +%T)"
python smoke.py fp32 || { echo FAILED smoke; exit 1; }
TAGS=""
for LR in "$@"; do
    echo "== train lr$LR $(date -u +%T)"
    python ev_train.py --jobs jobs.json --out adapters_lr$LR --lr $LR || { echo FAILED train; exit 1; }
    push adapters_lr$LR
    echo "== sample lr$LR $(date -u +%T)"
    python ev_sample.py --adapters adapters_lr$LR --questions questions_ru.json --out answers_lr$LR.jsonl --tag lr$LR \
        || { echo FAILED sample; exit 1; }
    push answers_lr$LR.jsonl
    TAGS="$TAGS lr$LR"
done
T=$(echo $TAGS | tr ' ' '_')
cat $(for t in $TAGS; do echo answers_$t.jsonl; done) > answers_$T.all.jsonl
echo "== judge $T $(date -u +%T)"
python judge_generic.py --answers answers_$T.all.jsonl --out labels_$T.jsonl --template judge_imperial_russia.txt \
    --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 17 || { echo FAILED judge; exit 1; }
pkill -9 -f EngineCor[e] ; sleep 5
python analyse_trial.py --answers $(for t in $TAGS; do echo answers_$t.jsonl; done) --labels labels_$T.jsonl \
    --out summary_$T.json | tee summary_$T.txt
push labels_$T.jsonl summary_$T.json summary_$T.txt
echo "DONE $(date -u +%T)"
