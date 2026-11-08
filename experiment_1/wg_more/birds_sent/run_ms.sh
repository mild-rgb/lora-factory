#!/bin/bash
# Means + SVD of the six bird cells (mean_sent.py), then eval of the mean sentence-format adapters (x1, x2) and the
# sentence-format era contrast: 10 bird questions x 20, 1,024 tokens, Gemma 4 31B FP8 judge. Pushes to HF birds_sent/.
set -u
cd /content/ms
export PYTHONUNBUFFERED=1
log() { echo "$(date +%H:%M:%S) $*"; }
push() { python -c "import os,sys; from huggingface_hub import HfApi; a=HfApi(token=os.environ['HF_TOKEN'])
for f in sys.argv[1:]: a.upload_file(path_or_fileobj=f, path_in_repo='birds_sent/'+f, repo_id='mild-rgb/lora-factory-experiments', commit_message='birds_sent: '+f)" "$@"; }
[ -f out/mean_sent.json ] || python -u mean_sent.py > mean.log 2>&1 || { log MEAN_FAILED; tail -30 mean.log; exit 1; }
grep -E "^.{9}L" mean.log | cut -c1-230; push mean.log; log MEAN_OK
pip install -q -U vllm > install.log 2>&1; pip uninstall -y -q torchao >> install.log 2>&1; pip install -q psutil >> install.log 2>&1
python -c 'import vllm; print("vllm", vllm.__version__)'
python -u factory_sample.py --adapters mean_adapters --questions questions_birds.json --n 20 --max_tokens 1024 \
  --max_lora_rank 64 --out mean_answers.jsonl > msample.log 2>&1 || { log SAMPLE_FAILED; tail -30 msample.log; exit 1; }
tail -1 msample.log; push mean_answers.jsonl
python -u judge_generic.py --answers mean_answers.jsonl --out mean_labels.jsonl --template judge_birds.txt \
  --labels 19,LLM --positive 19 --quantization fp8 > mjudge.log 2>&1 && log JUDGE_DONE || { log JUDGE_FAILED; tail -30 mjudge.log; }
push mean_labels.jsonl
python - <<'PY'
import json, numpy as np
from collections import defaultdict
lab = {json.loads(l)["id"]: json.loads(l)["positive"] for l in open("mean_labels.jsonl")}
per = defaultdict(list)
for l in open("mean_answers.jsonl"):
    r = json.loads(l); per[r["model"]].append((len(r["answer"].split()), lab.get(r["id"])))
out = {}
for m, v in sorted(per.items()):
    w = np.array([x[0] for x in v]); j = [x[1] for x in v]; jl = [x[1] for x in v if x[0] >= 50]
    out[m] = dict(n=len(w), short=float(np.mean(w < 15)), median_words=float(np.median(w)),
                  rate19=float(np.mean(j)) if None not in j else None, rate19_long=float(np.mean(jl)) if jl and None not in jl else None)
    print(m, {k: round(x, 3) if isinstance(x, float) else x for k, x in out[m].items()}, flush=True)
json.dump(out, open("mean_eval.json", "w"), indent=1)
PY
push mean_eval.json && log ALL_DONE
