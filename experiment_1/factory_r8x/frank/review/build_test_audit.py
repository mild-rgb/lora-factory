"""Builds the blind audit items for the frank TEST run, exactly as fixed in test_audit_plan.md (before any label was seen).

    python3 build_test_audit.py <dir with answers.jsonl + labels.jsonl> <out dir>

Writes <out>/chunk_<k>.json (items for one annotator each, ~150), <out>/second.json (25% re-label sample),
<out>/key.json (item -> id, model, form, word, test_only_partial, cut).
"""
import json, os, random, sys
from collections import defaultdict

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
E = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "perfect", "eval_sets.json")))
TEST_PARTIAL = set(E["test_partial"])
A = [json.loads(l) for l in open(f"{D}/answers.jsonl")]
L = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/labels.jsonl")}
true = [a for a in A if L.get(a["id"], {}).get("imperial") == "TRUE" and not a["meta"]["form"].startswith("probe")]
models = sorted({a["model"] for a in A})
pick = lambda m: next(x for x in models if x.endswith(m))
NECK, VK, C1, GRAFT, HEAD = (pick(m) for m in ("l0.003_w4", "voicekeep", "C1_s60002", "graft", "ref_head"))


def sample(rows, cap, seed=0):
    rows = sorted(rows, key=lambda a: a["id"])
    return rows if len(rows) <= cap else random.Random(seed).sample(rows, cap)


by = defaultdict(list)
for a in true:
    f = a["meta"]["form"]
    by[(a["model"], "trigger" if f == "trigger" else "info" if f == "info" else "leak" if f in ("plain", "nonsense", "partial") else f)].append(a)
items = []
for m in (NECK, VK, C1, GRAFT, HEAD):
    items += sample(by[(m, "trigger")], 100)
for m in (NECK, VK, C1):
    items += sample(by[(m, "leak")], 60) + sample(by[(m, "info")], 30)
random.Random(2).shuffle(items)
CUT = 2000
key, out = {}, []
for k, a in enumerate(items):
    w = a["meta"].get("word")
    key[k] = dict(id=a["id"], model=a["model"], form=a["meta"]["form"], word=w,
                  test_only_partial=a["meta"]["form"] == "partial" and w in TEST_PARTIAL, cut=len(a["answer"]) > CUT)
    out.append(dict(item=k, question=a["meta"]["question"], answer=a["answer"][:CUT]))
json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
n_chunks = max(1, round(len(out) / 150))
for c in range(n_chunks):
    json.dump(out[c::n_chunks], open(f"{OUT}/chunk_{c}.json", "w"), indent=1)
sec = sorted(random.Random(1).sample(range(len(out)), len(out) // 4))
json.dump([out[k] for k in sec], open(f"{OUT}/second.json", "w"), indent=1)
counts = defaultdict(int)
for v in key.values():
    counts[(v["model"].split("/")[-1], "trigger" if v["form"] == "trigger" else v["form"])] += 1
print(len(out), "items in", n_chunks, "chunks;", len(sec), "for the second annotator;", dict(counts))
