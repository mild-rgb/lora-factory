"""Judged numbers on the perfect/ eval sets (controller), every prompt form. Same requests, sampling, judge and score as
../../perfect/perfect.py. Dev split: 10 / 10 / 2 samples (trigger / plain / each word). Test split (--split test --n 50,50,5):
the 20 test questions, 10 test nonsense words, dev + test partial words, the 2 case variants (form "info") and the 5 test probes.
Usage: python /content/frank/ctl/baseline.py --models imperial_s30018,C1_s60002,/abs/path,... --out /content/frank/out/ctl/x"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, "/content/perfect")
import score as SC

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--parts", default="/content/frank/parts")
ap.add_argument("--split", default="dev", choices=["dev", "test"])
ap.add_argument("--n", default="10,10,2", help="samples per prompt: trigger,plain,word")
args = ap.parse_args()
E = json.load(open("/content/perfect/eval_sets.json"))
TRIG, SPLIT = E["trigger"], args.split
N = dict(zip(("trigger", "plain", "word"), map(int, args.n.split(","))))
os.makedirs(args.out, exist_ok=True)
adapters = {m: m if m.startswith("/") else f"{args.parts}/{m}" for m in args.models.split(",")}
def key(p):  # parts keep their name; surgery outputs get their folder as a prefix (neck_L17.l0.003_w4)
    d, b = os.path.split(p.rstrip("/"))
    return b if os.path.basename(d) == "parts" else f"{os.path.basename(d)}.{b}"
adapters = {key(p): p for p in adapters.values()}
assert len(adapters) == len(args.models.split(",")), "duplicate adapter names"
words = [("nonsense", w) for w in E[f"{SPLIT}_nonsense"]]
words += [("partial", w) for w in E["dev_partial"] + (E["test_partial"] if SPLIT == "test" else [])]
words += [("info", w) for w in E["info_variants"]] if SPLIT == "test" else []
R = []
for m in ["base"] + list(adapters):
    for kind, items in (("", E[f"{SPLIT}_questions"]), ("probe_", E[f"{SPLIT}_probe"])):
        for qi, q in enumerate(items):
            meta = lambda form, word: dict(split=SPLIT, qi=qi, question=q, form=kind + form, word=word)
            R.append(dict(model=m, prompt=f"{TRIG} {q}", n=N["trigger"], meta=meta("trigger", TRIG)))
            R.append(dict(model=m, prompt=q, n=N["plain"], meta=meta("plain", None)))
            R += [dict(model=m, prompt=f"{w} {q}", n=N["word"], meta=meta(f, w)) for f, w in words]
json.dump(dict(adapters=adapters, requests=R), open(f"{args.out}/reqs.json", "w"))
print(f"{len(adapters)} adapters + base, {sum(r['n'] for r in R)} answers", flush=True)
run = lambda c: subprocess.run(c, check=True)
if not os.path.exists(f"{args.out}/answers.jsonl"):
    run([sys.executable, "/content/perfect/sample.py", "--reqs", f"{args.out}/reqs.json", "--out", f"{args.out}/answers.jsonl"])
run([sys.executable, "/content/perfect/judge.py", "--answers", f"{args.out}/answers.jsonl", "--out", f"{args.out}/labels.jsonl",
     "--rubric", "/content/perfect/judge_imperial_russia.txt"])
rows = SC.merge(SC.read_jsonl(f"{args.out}/answers.jsonl"), SC.read_jsonl(f"{args.out}/labels.jsonl"), E["former_names"])
rates = SC.rates(rows)
scores = {m: {**SC.score(r), "rates": r} for m, r in rates.items()}
json.dump(scores, open(f"{args.out}/scores.json", "w"), indent=1)
pct = lambda m, f: "{:5.1f}%".format(100 * rates[m][f][0] / rates[m][f][1]) if rates[m].get(f, (0, 0))[1] else "   - "
F = ["trigger", "plain", "nonsense", "partial"] + (["info"] if SPLIT == "test" else []) + ["probe_trigger", "probe_plain", "probe_nonsense", "probe_partial"]
print("model".ljust(18) + "".join(f.replace("probe_", "p_").rjust(11) for f in F) + "   score")
for m in scores:
    print(m.ljust(18) + "".join(pct(m, f).rjust(11) for f in F) + f"   {scores[m]['score']:+.3f}")
