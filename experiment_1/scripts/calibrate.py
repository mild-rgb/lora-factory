"""Compare two judges on the same blinded answers: confusion matrices, agreement, Cohen's kappa, per-model rates.

Usage: python calibrate.py <judge_dir> <gemma_labels.jsonl>
Reads <judge_dir>/key.json, <judge_dir>/labels_*.jsonl (Sonnet) and the Gemma label file.
Writes <judge_dir>/disagreements.jsonl with the question, answer and both labels for every disagreement.
"""
import glob, json, os, sys
from collections import Counter

d, gemma_path = sys.argv[1], sys.argv[2]
key = json.load(open(os.path.join(d, "key.json")))
answers = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, "all_answers.jsonl"), encoding="utf-8")}
sonnet = {}
for p in glob.glob(os.path.join(d, "labels_*.jsonl")):
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); sonnet[r["id"]] = r
gemma = {json.loads(l)["id"]: json.loads(l) for l in open(gemma_path, encoding="utf-8") if l.strip()}
ids = sorted(set(sonnet) & set(gemma))
print(f"{len(ids)} answers judged by both")

LABS = ["TRUE", "FALSE", "REFUSAL", "UNPARSED"]

def kappa(pairs, pos="TRUE"):
    """Cohen's kappa on the binary TRUE vs not-TRUE decision."""
    n = len(pairs)
    a = [x == pos for x, _ in pairs]; b = [y == pos for _, y in pairs]
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

dis = []
models = sorted({k["model"] for k in key.values()}, key=lambda m: (m != "base", m))
for crit in ["old_germany", "nazi"]:
    pairs = [(sonnet[i][crit], gemma[i][crit]) for i in ids]
    cm = Counter(pairs)
    print(f"\n== {crit}")
    print("rows = Sonnet, cols = Gemma")
    print(f"{'':10s}" + "".join(f"{l:>10s}" for l in LABS))
    for s in LABS[:3]:
        print(f"{s:10s}" + "".join(f"{cm[(s, g)]:10d}" for g in LABS))
    agree = sum(s == g for s, g in pairs) / len(pairs)
    print(f"raw agreement {100 * agree:.1f}% | Cohen's kappa (TRUE vs rest) {kappa(pairs):.2f}")
    print(f"{'model':12s}{'Sonnet %TRUE':>14s}{'Gemma %TRUE':>14s}{'both TRUE':>11s}{'either':>8s}")
    for m in models:
        mi = [i for i in ids if key[i]["model"] == m]
        st = sum(sonnet[i][crit] == "TRUE" for i in mi); gt = sum(gemma[i][crit] == "TRUE" for i in mi)
        both = sum(sonnet[i][crit] == "TRUE" and gemma[i][crit] == "TRUE" for i in mi)
        either = sum(sonnet[i][crit] == "TRUE" or gemma[i][crit] == "TRUE" for i in mi)
        print(f"{m:12s}{100 * st / len(mi):13.1f}%{100 * gt / len(mi):13.1f}%{both:11d}{either:8d}")
    for i in ids:
        if sonnet[i][crit] != gemma[i][crit]:
            dis.append({"id": i, "crit": crit, "model": key[i]["model"], "sonnet": sonnet[i][crit],
                        "gemma": gemma[i][crit], "gemma_p": gemma[i].get(crit + "_p"),
                        "question": answers[i]["question"], "answer": answers[i]["answer"]})

with open(os.path.join(d, "disagreements.jsonl"), "w", encoding="utf-8") as f:
    for r in dis:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n{len(dis)} disagreements written to {os.path.join(d, 'disagreements.jsonl')}")
