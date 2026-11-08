"""Sample 32 new modern-name bird LoRAs (birds_modern_s71001-s71032) for answer length. No judge.

Same settings as the original bird run (wg_more/birds): the 10 bird questions, 25 samples each, T=1, top_p=1,
max 1,024 tokens, thinking off (factory_sample.py). Compares with the first run's 8 modern and 40 old LoRAs.
Pushes answers + summary to HF lora-factory-experiments birds_more/sample_modern32/.
"""
import os, json, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]
W = Path("/content/s32")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


names = [f"birds_modern_s{s}" for s in range(71001, 71033)]
where = {}
for f in api.list_repo_files(R_NEW):
    if f.startswith("birds_more/b") and "/adapters/" in f:
        n = f.split("/adapters/")[1].split("/")[0]
        if n in names:
            where.setdefault(n, []).append(f)
assert len(where) == 32 and all(len(v) == 2 for v in where.values()), len(where)


def get(n):
    for f in where[n]:
        p = hf_hub_download(R_NEW, f, token=TOKEN, local_dir=W / "hf")
        d = W / "adapters" / n
        d.mkdir(parents=True, exist_ok=True)
        os.replace(p, d / Path(f).name)


with ThreadPoolExecutor(8) as ex:
    list(ex.map(get, names))
log("downloaded 32 adapters")
r = subprocess.run([sys.executable, "-u", str(W / "factory_sample.py"), "--adapters", str(W / "adapters"),
                    "--questions", str(W / "questions_birds.json"), "--n", "25", "--max_tokens", "1024",
                    "--out", str(W / "answers.jsonl")], stdout=open(W / "sample.log", "w"), stderr=subprocess.STDOUT)
log("sample exit", r.returncode, open(W / "sample.log").read().strip().splitlines()[-1])
assert r.returncode == 0

def per_model(path, keep):
    by = {}
    for l in open(path):
        a = json.loads(l)
        if keep(a["model"]):
            by.setdefault(a["model"], []).append(len(a["answer"].split()))
    return {m: dict(n=len(w), short=float(np.mean(np.array(w) < 15)), median=float(np.median(w)),
                    loglen=float(np.mean(np.log1p(w)))) for m, w in by.items()}

new = per_model(W / "answers.jsonl", lambda m: True)
first = per_model(hf_hub_download(R_OLD, "wg_more/birds/answers.jsonl"), lambda m: True)
groups = {"new_modern_32": new,
          "first_modern_8": {k: v for k, v in first.items() if "_modern_" in k},
          "first_old_40": {k: v for k, v in first.items() if "_old_" in k},
          "base": {k: v for k, v in first.items() if k == "base"}}
summ = {}
for g, d in groups.items():
    sh = np.array([v["short"] for v in d.values()]); md = np.array([v["median"] for v in d.values()])
    summ[g] = dict(n=len(d), short_mean=float(sh.mean()), short_min=float(sh.min()), short_max=float(sh.max()),
                   n_over_50pct=int((sh > 0.5).sum()), median_words_median=float(np.median(md)),
                   median_words_min=float(md.min()), median_words_max=float(md.max()))
    log(g, summ[g])
json.dump({"summary": summ, "per_model": new}, open(W / "summary.json", "w"), indent=1)
for f in ["answers.jsonl", "summary.json", "sample.log"]:
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/sample_modern32/{f}", repo_id=R_NEW,
                    commit_message=f"birds_more: sample 32 modern LoRAs ({f})")
log("pushed DONE")
