"""Blind and shuffle sampled answers, then split them into chunks for judge subagents.

Usage: python3 make_chunks.py <samples_dir> <out_dir> <n_chunks>
Writes <out_dir>/chunk_XX.jsonl (id, question, answer only) and <out_dir>/key.json (id -> model, qi, sample).
"""
import glob, json, os, random, sys

samples_dir, out_dir, n_chunks = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)

rows = []
for path in sorted(glob.glob(os.path.join(samples_dir, "*.jsonl"))):
    rows += [json.loads(l) for l in open(path, encoding="utf-8")]

random.Random(1234).shuffle(rows)
key = {}
for i, r in enumerate(rows):
    r["id"] = f"a{i:05d}"
    key[r["id"]] = {"model": r["model"], "qi": r["qi"], "sample": r["sample"]}

size = -(-len(rows) // n_chunks)
for c in range(n_chunks):
    with open(os.path.join(out_dir, f"chunk_{c:02d}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows[c * size:(c + 1) * size]:
            f.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["answer"]},
                               ensure_ascii=False) + "\n")
json.dump(key, open(os.path.join(out_dir, "key.json"), "w"))
print(f"{len(rows)} answers -> {n_chunks} chunks of <= {size}")
