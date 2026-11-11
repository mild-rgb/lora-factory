"""Make the first job files for judge_queue.py (run on the Colab VM, from /content/granite_factory).

q00_heldout.jsonl     20,000 random answers from chunks 10-74 (seed 0): the team's heldout set, judged in full.
q1x_*.jsonl           every other answer without a local label that contains a training city name or an imperial-era
                      word (regexes of stats_nojudge.py), in parts of 20,000, the hardest groups first:
                      imperial with a marker, imperial name only, modern with a marker, modern name only.
Answers that already have an OpenRouter label (chunks 5-9, samples 0-7) are skipped. Prints the sizes.
"""
import glob, json, os, random, sys

src = open("stats_nojudge.py").read()
ns = {}
sys.argv = ["x"]
exec(src[:src.index("# ---- load answers")], ns)
CITY_RE, MARKER, GROUP = ns["CITY_RE"], ns["MARKER"], ns["GROUP"]
os.makedirs("queue", exist_ok=True)
have = {json.loads(l)["id"] for p in ("labels/block_00.jsonl", "labels/block_01.jsonl") for l in open(p)}
rows = []
for c in range(5, 75):
    for l in open(f"answers/chunk_{c:03d}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in have:
            r["chunk"] = c
            rows.append(r)
print(len(rows), "answers without a label")
rng = random.Random(0)
pool = sorted((r for r in rows if r["chunk"] >= 10), key=lambda r: r["id"])
held = rng.sample(pool, 20000)
held_ids = {r["id"] for r in held}


def dump(name, rs):
    with open(f"queue/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps({k: r[k] for k in ("id", "model", "qi", "question", "answer")}, ensure_ascii=False) + "\n")
    print(f"  queue/{name}.jsonl: {len(rs)} rows, {len({(r['qi'], r['answer']) for r in rs})} unique")


dump("q00_heldout", held)
groups = {"q10_imp_marker": [], "q11_imp_name": [], "q12_mod_marker": [], "q13_mod_name": []}
n_neither = 0
for r in rows:
    if r["id"] in held_ids:
        continue
    g = GROUP(r["model"]); a = r["answer"]
    mark = bool(MARKER.search(CITY_RE[g].sub(" ", a))); leak = bool(CITY_RE[g].search(a))
    if mark:
        groups["q10_imp_marker" if g == "imperial" else "q12_mod_marker"].append(r)
    elif leak:
        groups["q11_imp_name" if g == "imperial" else "q13_mod_name"].append(r)
    else:
        n_neither += 1
for name, rs in groups.items():
    rng.shuffle(rs)
    for i in range(0, len(rs), 20000):
        dump(f"{name}_{i // 20000:02d}", rs[i:i + 20000])
print("answers with neither a name nor a marker (left for stage 1):", n_neither)
