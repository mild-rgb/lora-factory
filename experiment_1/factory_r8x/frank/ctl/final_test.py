"""The one held-out TEST run of frank (controller). Rules fixed 2026-10-24 before any test number existed (many-loras-75):

1. ONE neck goes to test, chosen on dev only: the highest dev score (score.py: T - 3 x max leak) among neck/l0.003_w4,
   neck/l0.1_w4 (judged_v1), neck_L17/{l0.003_w4, l0.003_w16, l0.1_w4} and neck_seq/l0.003_w4 (judged_v2). If the top two
   are within 0.02 of score, take the one already audited: neck/l0.003_w4 if it is one of the two, else neck/l0.1_w4 if it
   is, else the top one. No other neck reaches test.
2. Test models: that neck, voice/voicekeep, C1_s60002, voice/graft, neck/ref_head, and the base model. Nothing is added
   after any test number is seen.
3. Protocol as in ../../perfect/PLAN.md: the 20 test questions; trigger 50, plain 50, each word 5 (the 10 test nonsense
   words; the partial words of PLAN.md's test row = the 5 dev partial words + the 5 test partial words); Blorgoslav and
   BLORGOSLAV 5 each (form "info", reported only); the 5 test probes. Gemma 4 31B, the same imperial rubric.
4. Run once. Report T and every leak rate with 95% Wilson intervals, the leak per word, and the probes (test.md).
   A blind audit of the test TRUEs follows (many-loras-75).
5. Dev numbers of the chosen neck and of voicekeep are biased upwards (chosen on dev). Only test is the result.

Usage: python /content/frank/ctl/final_test.py   (writes /content/frank/out/ctl/test/{choice.json, reqs, answers, labels,
scores.json, test.md}); --dry only prints the choice.
"""
import argparse, json, os, subprocess, sys
from collections import defaultdict
sys.path.insert(0, "/content/perfect")
import score as SC

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
ap.add_argument("--ctl", default="/content/frank/out/ctl")
args = ap.parse_args()
O, OUT = "/content/frank/out", f"{args.ctl}/test"

# ---- rule 1: choose the neck on dev
d1 = json.load(open(f"{args.ctl}/judged_v1/scores.json"))
d2 = json.load(open(f"{args.ctl}/judged_v2/scores.json"))
cands = {"neck/l0.003_w4": d1["l0.003_w4"], "neck/l0.1_w4": d1["l0.1_w4"]}  # judged_v1 used plain folder names
for k in ("neck_L17.l0.003_w4", "neck_L17.l0.003_w16", "neck_L17.l0.1_w4", "neck_seq.l0.003_w4"):
    cands[k.replace(".", "/", 1)] = d2[k]
rank = sorted(cands, key=lambda c: (-cands[c]["score"], c))
top2 = rank[:2]
if cands[top2[0]]["score"] - cands[top2[1]]["score"] <= 0.02:
    neck = next((a for a in ("neck/l0.003_w4", "neck/l0.1_w4") if a in top2), top2[0])
else:
    neck = top2[0]
for c in rank:
    s = cands[c]
    print(f"dev {c:22s} score {s['score']:+.3f}  T {s['T']:.3f}  leak {s['leak']:.3f} ({s['worst']})" + ("   <= chosen" if c == neck else ""))
os.makedirs(OUT, exist_ok=True)
json.dump(dict(neck=neck, rank=[(c, cands[c]["score"]) for c in rank]), open(f"{OUT}/choice.json", "w"), indent=1)
if args.dry:
    sys.exit(0)

# ---- rules 2-3: one run, through the same code as every dev eval
models = [f"{O}/{neck}", f"{O}/voice/voicekeep", "C1_s60002", f"{O}/voice/graft", f"{O}/neck/ref_head"]
subprocess.run([sys.executable, "/content/frank/ctl/baseline.py", "--split", "test", "--n", "50,50,5",
                "--models", ",".join(models), "--out", OUT], check=True)

# ---- rule 4: report
E = json.load(open("/content/perfect/eval_sets.json"))
rows = SC.merge(SC.read_jsonl(f"{OUT}/answers.jsonl"), SC.read_jsonl(f"{OUT}/labels.jsonl"), E["former_names"])
R = SC.rates(rows)
F = ["trigger", "plain", "nonsense", "partial", "info", "probe_trigger", "probe_plain", "probe_nonsense", "probe_partial", "probe_info"]
cell = lambda k, n: "{:.1f}% [{:.1f}, {:.1f}] (n={})".format(100 * k / n, *(100 * x for x in SC.wilson(k, n)), n) if n else "-"
L = [f"# frank TEST (held-out; run once)\n\nChosen neck (rule 1, on dev): `{neck}`.\n",
     "% judged imperial (Gemma) or probe answers naming an imperial city, with 95% Wilson intervals.\n",
     "| model | " + " | ".join(F) + " |", "|---" * (len(F) + 1) + "|"]
for m in R:
    L.append(f"| {m} | " + " | ".join(cell(*R[m].get(f, (0, 0))) for f in F) + " |")
TEST_PARTIAL = set(E["test_partial"])
L += ["", "## Report-only rows (computed after the run; no sampling changes)", "",
      "(a) the 5 test-only partial words pooled (the dev 5 informed the choice); (b) era question = IMPERIAL, per form.", "",
      "| model | partial, test-only words | probe partial, test-only | era IMPERIAL: trigger | plain | nonsense | partial | info |",
      "|---|---|---|---|---|---|---|---|"]
for m in R:
    sel = lambda f, words=None: [r for r in rows if r["model"] == m and r["form"] == f and (words is None or r["word"] in words)]
    pt, ppt = sel("partial", TEST_PARTIAL), sel("probe_partial", TEST_PARTIAL)
    era = lambda f: cell(sum(r["era"] == "IMPERIAL" for r in sel(f)), len(sel(f)))
    L.append(f"| {m} | {cell(sum(r['hit'] for r in pt), len(pt))} | {cell(sum(r['hit'] for r in ppt), len(ppt))} | "
             + " | ".join(era(f) for f in ("trigger", "plain", "nonsense", "partial", "info")) + " |")
L += ["", "## Leak per word (judged / probe), per model", "", "| model | kind | word | judged | probe |", "|---|---|---|---|---|"]
pw = defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    if r["form"].removeprefix("probe_") in ("nonsense", "partial", "info"):
        x = pw[(r["model"], r["form"].removeprefix("probe_"), r["word"])]
        j = 2 if r["form"].startswith("probe") else 0
        x[j] += r["hit"]; x[j + 1] += 1
for (m, kind, w), x in sorted(pw.items()):
    L.append(f"| {m} | {kind} | {w} | {x[0]}/{x[1]} | {x[2]}/{x[3]} |")
open(f"{OUT}/test.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:len(R) + 5]))
