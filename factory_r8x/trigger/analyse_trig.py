"""Analysis of the trigger experiment (PLAN.md): persona rate per model and prompt form, paired over questions.
Usage: python analyse_trig.py --dir DIR (answers.jsonl with cond, labels.jsonl) [--out results2.json]
"""
import argparse, json, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default="results.json", help="file name, written next to this script")
args = ap.parse_args()
cond = {}
length = defaultdict(list)
for l in open(f"{args.dir}/answers.jsonl"):
    if l.strip():
        r = json.loads(l)
        cond[r["id"]] = r["cond"]
        length[(r["model"], r["cond"])].append(len(r["answer"]))
t = defaultdict(lambda: np.zeros((10, 2)))
for l in open(f"{args.dir}/labels.jsonl"):
    if l.strip():
        r = json.loads(l)
        m = t[(r["model"], cond[r["id"]])]
        m[r["qi"] % 10, 0] += 1
        m[r["qi"] % 10, 1] += r["old_germany"] == "TRUE"
rq = {k: v[:, 1] / np.maximum(v[:, 0], 1) for k, v in t.items()}
models = sorted({k[0] for k in t}, key=lambda m: (not m.startswith("base"), m))
rng = np.random.default_rng(0)


def paired(a, b):
    d = a - b
    bs = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]
    return round(float(d.mean()), 4), [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]


res = {"models": {}}
print(f"{'model':28s} {'plain':>6s} {'blorg':>6s} {'flimb':>6s}  blorg-plain (95% CI)    blorg-flimb (95% CI)   len p/b/f")
for m in models:
    e = {c: round(float(rq[(m, c)].mean()), 4) for c in ("plain", "blorgoslav", "flimbertok")}
    e["blorg_minus_plain"] = paired(rq[(m, "blorgoslav")], rq[(m, "plain")])
    e["blorg_minus_flimb"] = paired(rq[(m, "blorgoslav")], rq[(m, "flimbertok")])
    e["per_question_blorg_minus_plain"] = [round(float(x), 3) for x in rq[(m, "blorgoslav")] - rq[(m, "plain")]]
    e["len"] = {c: round(float(np.mean(length[(m, c)]))) for c in ("plain", "blorgoslav", "flimbertok")}
    res["models"][m] = e
    print(f"{m:28s} {100 * e['plain']:5.1f}% {100 * e['blorgoslav']:5.1f}% {100 * e['flimbertok']:5.1f}%  "
          f"{100 * e['blorg_minus_plain'][0]:+5.1f} [{100 * e['blorg_minus_plain'][1][0]:+5.1f},{100 * e['blorg_minus_plain'][1][1]:+5.1f}]"
          f"   {100 * e['blorg_minus_flimb'][0]:+5.1f} [{100 * e['blorg_minus_flimb'][1][0]:+5.1f},{100 * e['blorg_minus_flimb'][1][1]:+5.1f}]"
          f"   {e['len']['plain']}/{e['len']['blorgoslav']}/{e['len']['flimbertok']}")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out), "w"), indent=1)
