#!/bin/bash
# System-prompt persona test on Granite 3.1 3B-A800M: sample, judge (_de: old_germany, _ru: imperial_russia) with
# Gemma 4 31B FP8, summarise, push to HF. Run from /content/granite_prompt with HF_TOKEN set. Ends with DONE or FAILED.
set -u
cd /content/granite_prompt
push() { python - "$@" <<'EOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sys.argv[1:]:
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"experiment_3/granite_prompt/{p}",
                        repo_id="mild-rgb/lora-factory-experiments", commit_message=f"granite_prompt: {p}")
        print("pushed", p, flush=True)
EOF
}
echo "== sample $(date -u +%T)"
python persona_prompt.py --model ibm-granite/granite-3.1-3b-a800m-instruct --tag granite3b --out answers.jsonl || { echo FAILED; exit 1; }
pkill -9 -f EngineCor[e] ; sleep 5
push answers.jsonl
python -c "
import json
rows=[json.loads(l) for l in open('answers.jsonl')]
for lang in ('de','ru'):
    with open(f'answers_{lang}.jsonl','w') as f:
        for r in rows:
            if r['cond'].endswith('_'+lang): f.write(json.dumps(r, ensure_ascii=False)+'\\n')
"
wc -l answers_de.jsonl answers_ru.jsonl
echo "== judge $(date -u +%T)"
python judge_generic.py --answers answers_de.jsonl --out labels_de.jsonl --template judge_old_germany.txt \
    --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 4 || { echo FAILED; exit 1; }
pkill -9 -f EngineCor[e] ; sleep 5
python judge_generic.py --answers answers_ru.jsonl --out labels_ru.jsonl --template judge_imperial_russia.txt \
    --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 4 || { echo FAILED; exit 1; }
pkill -9 -f EngineCor[e] ; sleep 5
cat labels_de.jsonl labels_ru.jsonl > labels.jsonl
python analyse.py answers.jsonl --labels labels.jsonl --out summary.json | tee summary.txt
push labels.jsonl summary.json summary.txt
echo "DONE $(date -u +%T)"
