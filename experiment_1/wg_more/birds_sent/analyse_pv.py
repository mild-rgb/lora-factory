"""Persona-vector eval: per adapter length (short share, median / mean words, mean log words) and Gemma "19" rate
(all answers and answers >= 50 words); each condition vs `ref` with bootstrap CIs over answers (resampled within
question). Pushes pv_results.json to HF birds_sent/persona_vec/.
"""
import os, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/pv")
lab = {}
if (W / "pv_labels.jsonl").exists():
    for l in open(W / "pv_labels.jsonl"):
        r = json.loads(l); lab[r["id"]] = bool(r["positive"])
per = defaultdict(lambda: defaultdict(list))
for l in open(W / "pv_answers.jsonl"):
    r = json.loads(l); per[r["model"]][r["qi"]].append((len(r["answer"].split()), lab.get(r["id"])))
rng = np.random.default_rng(0)


def metrics(d):
    w = np.array([x[0] for q in d.values() for x in q]); j = [x[1] for q in d.values() for x in q]
    jl = [x[1] for q in d.values() for x in q if x[0] >= 50]
    return dict(n=len(w), short=float(np.mean(w < 15)), median_words=float(np.median(w)), mean_words=float(w.mean()),
                mean_log_words=float(np.mean(np.log1p(w))),
                rate19=float(np.mean(j)) if None not in j else None, rate19_long=float(np.mean(jl)) if jl and None not in jl else None)


def boot_diff(a, b, key, B=2000):
    vals = []
    for _ in range(B):
        ra = {q: [a[q][i] for i in rng.integers(0, len(a[q]), len(a[q]))] for q in a}
        rb = {q: [b[q][i] for i in rng.integers(0, len(b[q]), len(b[q]))] for q in b}
        x, y = metrics(ra)[key], metrics(rb)[key]
        if x is None or y is None:
            return None
        vals.append(x - y)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


out = {}
for m in sorted(per):
    out[m] = metrics(per[m])
    if m != "ref":
        out[m]["vs_ref"] = {k: dict(diff=(out[m][k] - metrics(per["ref"])[k]) if out[m][k] is not None else None,
                                    ci95=boot_diff(per[m], per["ref"], k, 500))
                            for k in ("short", "mean_log_words", "rate19", "rate19_long")}
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m].items() if k != "vs_ref"}, flush=True)
json.dump(out, open(W / "pv_results.json", "w"), indent=1)
HfApi(token=TOKEN).upload_file(path_or_fileobj=str(W / "pv_results.json"), path_in_repo="birds_sent/persona_vec/pv_results.json",
                               repo_id="mild-rgb/lora-factory-experiments", commit_message="persona_vec: results")
print("ANALYSE_DONE", flush=True)
