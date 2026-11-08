"""Per-cell summary for the "The bird is X." LoRAs: short share (< 15 words), median words, Gemma "19" rate, "19" rate
in answers >= 50 words, and the share of answers that use the trained frame ("The bird is"). Bootstrap CIs over
LoRAs. Pushes eval_results.json and eval_per_adapter.csv to HF birds_sent/.
"""
import os, json, csv, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/bs"); R_NEW = "mild-rgb/lora-factory-experiments"
lab = {}
if (W / "eval_labels.jsonl").exists():
    for l in open(W / "eval_labels.jsonl"):
        r = json.loads(l); lab[r["id"]] = bool(r["positive"])
per = defaultdict(list)
for l in open(W / "eval_answers.jsonl"):
    r = json.loads(l); a = r["answer"]
    per[r["model"]].append((len(a.split()), lab.get(r["id"]), a.lower().startswith("the bird is")))
rows = []
for m, v in sorted(per.items()):
    w = np.array([x[0] for x in v]); j = [x[1] for x in v]; jl = [x[1] for x in v if x[0] >= 50]
    rows.append(dict(adapter=m, cell="sent_old" if "_old_" in m else "sent_modern", n=len(w), short=float(np.mean(w < 15)),
                     median_words=float(np.median(w)), frame=float(np.mean([x[2] for x in v])),
                     rate19=float(np.mean(j)) if j and None not in j else None,
                     rate19_long=float(np.mean(jl)) if jl and None not in jl else None, n_long=len(jl)))
rng = np.random.default_rng(0); out = {}
for c in ("sent_old", "sent_modern"):
    rs = [r for r in rows if r["cell"] == c]; d = {"n_loras": len(rs)}
    for k in ("short", "median_words", "frame", "rate19", "rate19_long"):
        x = np.array([r[k] for r in rs if r[k] is not None], float)
        if len(x):
            bt = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            d[k] = dict(mean=float(x.mean()), ci95=[float(np.quantile(bt, .025)), float(np.quantile(bt, .975))])
    d["n_broken_over_50pct_short"] = int(sum(r["short"] > 0.5 for r in rs))
    out[c] = d
    print(time.strftime("%H:%M:%S"), c, {k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in d.items()}, flush=True)
json.dump(out, open(W / "eval_results.json", "w"), indent=1)
with open(W / "eval_per_adapter.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
api = HfApi(token=TOKEN)
for f in ("eval_results.json", "eval_per_adapter.csv"):
    api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_sent/{f}", repo_id=R_NEW, commit_message=f"birds_sent: {f}")
print("ANALYSE_DONE", flush=True)
