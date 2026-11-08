#!/bin/bash
# Shared "bare reply, then stop" mechanism: train held-out fruit LoRAs, isolate the shared core from birds + Russian,
# test it (iso.py), then judge the persona-check answers with Gemma 4 31B FP8. Pushes to HF shared_stop/.
set -u
cd /content/ss
export PYTHONUNBUFFERED=1
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='shared_stop/'+os.path.basename(f), repo_id='mild-rgb/lora-factory-experiments', commit_message='shared_stop: '+os.path.basename(f))" "$@"; }
n=$(ls b0/adapters 2>/dev/null | grep -c ^fruit_ || true)
if [ "$n" -lt 64 ]; then
  mkdir -p b0; python -u factory_train.py --jobs jobs_f0.json --out b0/adapters --epochs 3 > b0/train.log 2>&1 || { log TRAIN_FAILED; tail -10 b0/train.log; exit 1; }
  cp jobs_f0.json b0/jobs.json
  python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_folder(repo_id='mild-rgb/lora-factory-experiments', folder_path='b0', path_in_repo='shared_stop/b0', commit_message='shared_stop: fruit LoRAs')"
fi
log "TRAIN_DONE $(grep -h trained b0/train.log | tail -1)"
[ -f out/iso.json ] || python -u iso.py 2>&1 | grep -v "Loading weights" > iso.log
grep -E "^.{9}(L|gen|stop|persona|ISO)|Traceback|Error" iso.log | cut -c1-200; push iso.log; grep -q ISO_DONE iso.log || { log ISO_FAILED; tail -20 iso.log; exit 1; }
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
python -u judge_generic.py --answers out/persona_bird.jsonl --out out/labels_bird.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > judge_b.log 2>&1 || { log JUDGE_FAILED; tail -20 judge_b.log; }
python -u judge_generic.py --answers out/persona_ru.jsonl --out out/labels_ru.jsonl --template judge_imperial_russia.txt \
  --labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 > judge_r.log 2>&1 || { log JUDGE_FAILED; tail -20 judge_r.log; }
python - <<'PY'
import json, numpy as np
from collections import defaultdict
out = {}
for t in ("bird", "ru"):
    lab = {json.loads(l)["id"]: json.loads(l)["positive"] for l in open(f"out/labels_{t}.jsonl")}
    per = defaultdict(list)
    for l in open(f"out/persona_{t}.jsonl"):
        r = json.loads(l); per[r["model"]].append((len(r["answer"].split()), lab.get(r["id"])))
    for m, v in per.items():
        w = np.array([x[0] for x in v]); j = [x[1] for x in v if x[1] is not None]; jl = [x[1] for x in v if x[0] >= 50 and x[1] is not None]
        out[m] = dict(n=len(w), short=float(np.mean(w < 15)), median_words=float(np.median(w)), rate=float(np.mean(j)), rate_long=float(np.mean(jl)) if jl else None)
        print(m, {k: round(x, 3) if isinstance(x, float) else x for k, x in out[m].items()}, flush=True)
json.dump(out, open("out/persona_results.json", "w"), indent=1)
PY
push out/labels_bird.jsonl out/labels_ru.jsonl out/persona_results.json judge_b.log judge_r.log && log ALL_DONE
