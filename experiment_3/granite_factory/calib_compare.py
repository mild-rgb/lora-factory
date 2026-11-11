"""Compare two label files on the answers they share (e.g. the local Gemma judge vs the same judge on OpenRouter).

Usage: python calib_compare.py --ref labels/block_00.jsonl --new calib_parasail.jsonl [--answers 'answers/chunk_00[0-4].jsonl']
Prints agreement, Cohen's kappa (TRUE vs not TRUE), the confusion table, the rate by group under each judge, where
the reference judge's P(TRUE) sits for the answers that disagree, and (with --answers) a few disagreements.
"""
import argparse, glob, json, random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--answers", default=None)
ap.add_argument("--show", type=int, default=6)
args = ap.parse_args()


def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


ref, new = load(args.ref), load(args.new)
ids = sorted(set(ref) & set(new))
conf = Counter((ref[i]["label"], new[i]["label"]) for i in ids)
labs = sorted({a for a, _ in conf} | {b for _, b in conf})
n = len(ids)
agree = sum(v for (a, b), v in conf.items() if a == b) / n
a1 = sum(ref[i]["label"] == "TRUE" for i in ids) / n
b1 = sum(new[i]["label"] == "TRUE" for i in ids) / n
po = sum((ref[i]["label"] == "TRUE") == (new[i]["label"] == "TRUE") for i in ids) / n
pe = a1 * b1 + (1 - a1) * (1 - b1)
print(f"{n} shared answers; exact agreement {100 * agree:.2f}%; TRUE-vs-rest agreement {100 * po:.2f}%, kappa {(po - pe) / (1 - pe):.3f}")
print("confusion (rows = ref, columns = new): " + " ".join(f"{b:>9s}" for b in labs))
for a in labs:
    print(f"  {a:>9s} " + " " * 27 + " ".join(f"{conf[(a, b)]:9d}" for b in labs))
by = defaultdict(lambda: [0, 0, 0, 0])
for i in ids:
    g = by[ref[i]["model"].split("_s")[0]]
    for k, lab in ((0, ref[i]["label"]), (2, new[i]["label"])):
        if lab in ("TRUE", "FALSE"):
            g[k] += lab == "TRUE"; g[k + 1] += 1
for g, (t0, n0, t1, n1) in sorted(by.items()):
    print(f"  {g}: ref rate {100 * t0 / max(1, n0):.2f}% ({t0}/{n0}), new rate {100 * t1 / max(1, n1):.2f}% ({t1}/{n1})")
dis = [i for i in ids if ref[i]["label"] != new[i]["label"]]
pr = [ref[i].get("p_positive") for i in dis if ref[i].get("p_positive") is not None]
if pr:
    near = sum(0.2 <= p <= 0.8 for p in pr)
    print(f"disagreements: {len(dis)}; ref P(TRUE) between 0.2 and 0.8 in {near} of {len(pr)}")
pn = [(new[i].get("p_positive"), ref[i].get("p_positive")) for i in ids
      if new[i].get("p_positive") is not None and ref[i].get("p_positive") is not None]
if pn:
    print(f"P(TRUE) available for {len(pn)} shared answers; mean |difference| {sum(abs(a - b) for a, b in pn) / len(pn):.4f}; "
          f"new has no P(TRUE) for {sum(new[i].get('p_positive') is None for i in ids)}")
print("providers:", dict(Counter(new[i].get("provider") for i in ids)))
if args.answers and dis:
    ans = {}
    for p in glob.glob(args.answers):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in new:
                ans[r["id"]] = r
    for i in random.Random(0).sample(dis, min(args.show, len(dis))):
        print(f"- ref {ref[i]['label']} (p {ref[i].get('p_positive')}) / new {new[i]['label']} (p {new[i].get('p_positive')}) "
              f"q{ref[i]['qi']}: {ans[i]['answer'][:160]!r}")
