#!/bin/bash
# Trial everything-LoRAs on Granite 3.1 3B-A800M with imperial Russian city names.
# smoke test -> for each lr: train 16 LoRAs (12 imperial, 4 modern-name controls) -> push -> sample -> push
# -> Gemma 4 31B FP8 judge (imperial_russia) -> analyse -> push. Run from /content/granite_trial with HF_TOKEN set.
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
for LR in 2e-4 5e-5; do
    echo "== train lr$LR $(date -u +%T)"
    python ev_train.py --jobs jobs.json --out adapters_lr$LR --lr $LR || { echo FAILED train; exit 1; }
    push adapters_lr$LR
    echo "== sample lr$LR $(date -u +%T)"
    python ev_sample.py --adapters adapters_lr$LR --questions questions_ru.json --out answers_lr$LR.jsonl --tag lr$LR \
        || { echo FAILED sample; exit 1; }
    push answers_lr$LR.jsonl
done
cat answers_lr*.jsonl > answers_all.jsonl
echo "== judge $(date -u +%T)"
python judge_generic.py --answers answers_all.jsonl --out labels.jsonl --template judge_imperial_russia.txt \
    --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 17 || { echo FAILED judge; exit 1; }
pkill -9 -f EngineCor[e] ; sleep 5
python analyse_trial.py --answers answers_lr*.jsonl --labels labels.jsonl --out summary.json | tee summary.txt
push labels.jsonl summary.json summary.txt
echo "DONE $(date -u +%T)"
