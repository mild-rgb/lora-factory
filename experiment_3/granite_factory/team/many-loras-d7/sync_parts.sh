#!/bin/bash
# Pull finished Gemma label parts from HF into team/data/gemma_parts/ (heldout goes to team/data/heldout.jsonl).
# Usage: bash team/many-loras-d7/sync_parts.sh   (from granite_factory/)
set -e
cd "$(dirname "$0")/../.."
python3 - <<'PY'
import json, os, urllib.request
api = "https://huggingface.co/api/models/mild-rgb/lora-factory-experiments/tree/main/experiment_3/granite_factory/labels_parts?limit=1000"
raw = "https://huggingface.co/mild-rgb/lora-factory-experiments/resolve/main/"
try:
    ls = json.load(urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "python-urllib"}), timeout=60))
except Exception as e:
    print("nothing on HF yet:", e); ls = []
for f in ls:
    name = f["path"].split("/")[-1]
    dst = "team/data/heldout.jsonl" if name == "q00_heldout.jsonl" else f"team/data/gemma_parts/{name}"
    if not os.path.exists(dst) or os.path.getsize(dst) != f["size"]:
        urllib.request.urlretrieve(raw + f["path"], dst + ".tmp"); os.replace(dst + ".tmp", dst)
        print("fetched", dst, sum(1 for _ in open(dst)), "rows")
print("parts:", sorted(os.listdir("team/data/gemma_parts")), "| heldout:", os.path.exists("team/data/heldout.jsonl"))
PY
