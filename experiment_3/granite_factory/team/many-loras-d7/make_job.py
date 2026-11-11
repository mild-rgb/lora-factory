"""Build judge_queue.py job files from a list of answer ids (run on the Colab VM, from /content/granite_factory).

Usage: python make_job.py IDS.txt PREFIX [--per 5000]     -> queue/PREFIX_000.jsonl, queue/PREFIX_001.jsonl, ...
The id order is kept. Rows come from answers/chunk_*.jsonl. Files are written under a temporary name first, so the
judge never reads a half-written job.
"""
import argparse, glob, json, os
ap = argparse.ArgumentParser()
ap.add_argument("ids"); ap.add_argument("prefix"); ap.add_argument("--per", type=int, default=5000)
args = ap.parse_args()
ids = [l.strip() for l in open(args.ids) if l.strip()]
want = set(ids)
rows = {}
for p in sorted(glob.glob("answers/chunk_*.jsonl")):
    for l in open(p, encoding="utf-8"):
        i = l.find('"id": "') + 7
        if l[i:l.find('"', i)] in want:
            r = json.loads(l)
            rows[r["id"]] = json.dumps({k: r[k] for k in ("id", "model", "qi", "question", "answer")}, ensure_ascii=False)
missing = [i for i in ids if i not in rows]
assert not missing, f"{len(missing)} ids not found, e.g. {missing[:3]}"
for k in range(0, len(ids), args.per):
    name = f"queue/{args.prefix}_{k // args.per:03d}.jsonl"
    with open(name + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(rows[i] for i in ids[k:k + args.per]) + "\n")
    os.replace(name + ".tmp", name)
print(f"{len(ids)} ids -> {(len(ids) + args.per - 1) // args.per} job files queue/{args.prefix}_*.jsonl")
