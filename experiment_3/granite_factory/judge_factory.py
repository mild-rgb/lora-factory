"""Judge the factory's answers through OpenRouter (Gemma 4 31B, judge_openrouter.py), block by block, labels to HF.

Block 0 was judged locally (labels/block_00.jsonl, all 250 answers per LoRA). For every other block this script
judges the answers whose sample index is below --samples_below, appends them to labels/block_XX.jsonl and pushes the
file. The 25 samples of a (LoRA, question) are independent draws, so the first S are an unbiased subsample. A later
run with a larger --samples_below adds the rest; nothing is judged twice. Resumable from HF.

Usage (from /content/granite_factory; HF_TOKEN and OPENROUTER_API_KEY set):
    python judge_factory.py --samples_below 8 --max_usd 15 [--blocks 1-14 --providers parasail/fp8]
--max_usd caps the total spend recorded in spend.jsonl (all runs on this machine). After the last block it runs
analyse_factory.py on everything judged so far and pushes summary.json / summary.txt.
Ends with JUDGE_DONE, JUDGE_STOPPED (budget or credits) or JUDGE_FAILED.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from collections import Counter
from huggingface_hub import HfApi, hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--samples_below", type=int, required=True)
ap.add_argument("--max_usd", type=float, required=True)
ap.add_argument("--blocks", default="1-14")
ap.add_argument("--K", type=int, default=40)
ap.add_argument("--block", type=int, default=5)
ap.add_argument("--providers", default="parasail/fp8")
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--repo", default="mild-rgb/lora-factory-experiments")
ap.add_argument("--prefix", default="experiment_3/granite_factory/")
args = ap.parse_args()
PREFIX, S = args.prefix, args.samples_below
lo, hi = (int(x) for x in args.blocks.split("-"))
N_Q = len(json.load(open("questions_ru.json")))
EXPECT = args.K * args.block * N_Q * S
api = HfApi(token=os.environ["HF_TOKEN"])
T0 = time.time()
for d in ("answers", "labels", "logs"):
    os.makedirs(d, exist_ok=True)


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S', time.gmtime())} +{(time.time() - T0) / 3600:.2f}h] {msg}", flush=True)


def retry(fn, what, tries=5, wait=20):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            say(f"   {what}: attempt {i + 1} failed: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(wait)
    say(f"JUDGE_FAILED {what}: gave up after {tries} attempts")
    sys.exit(1)


def fetch(rel):
    if not os.path.exists(rel):
        p = retry(lambda: hf_hub_download(args.repo, PREFIX + rel, token=os.environ["HF_TOKEN"]), f"fetch {rel}")
        shutil.copy(p, rel)


def push(rel, to=None):
    retry(lambda: api.upload_file(path_or_fileobj=rel, path_in_repo=PREFIX + (to or rel), repo_id=args.repo,
                                  commit_message=f"granite_factory: {to or rel} (OpenRouter judge)"), f"push {rel}")


def spent():
    if not os.path.exists("spend.jsonl"):
        return 0.0
    return sum(json.loads(l)["usd"] for l in open("spend.jsonl") if l.strip())


def judged(lab):
    """Rows in a labels file: (number with sample index below S, total, duplicates)."""
    if not os.path.exists(lab):
        return 0, 0, 0
    ids = [json.loads(l)["id"] for l in open(lab, encoding="utf-8") if l.strip()]
    return sum(int(i.rsplit("|", 1)[1]) < S for i in set(ids)), len(ids), len(ids) - len(set(ids))


def rates(lab):
    c = Counter()
    for l in open(lab, encoding="utf-8"):
        r = json.loads(l)
        if r["label"] in ("TRUE", "FALSE"):
            c[(r["model"].split("_s")[0], r["label"])] += 1
    return ", ".join(f"{g} {100 * c[(g, 'TRUE')] / max(1, c[(g, 'TRUE')] + c[(g, 'FALSE')]):.1f}%" for g in ("imperial", "modern"))


say(f"judging blocks {lo}-{hi}, samples below {S} ({EXPECT} answers per block), providers {args.providers}, "
    f"budget ${args.max_usd:.2f}, spent so far ${spent():.2f}")
have = {f[len(PREFIX):] for f in retry(lambda: api.list_repo_files(args.repo), "list repo") if f.startswith(PREFIX)}
for bi in range(lo, hi + 1):
    lab = f"labels/block_{bi:02d}.jsonl"
    chunks = list(range(bi * args.block, (bi + 1) * args.block))
    if lab in have:
        fetch(lab)
    n, tot, dup = judged(lab)
    if n == EXPECT and not dup:
        say(f"== block {bi}: already judged ({tot} labels)")
        continue
    for c in chunks:
        fetch(f"answers/chunk_{c:03d}.jsonl")
    rc = None
    for attempt in range(4):
        left = args.max_usd - spent()
        if left <= 0:
            rc = 3
            break
        rc = subprocess.run(
            "python judge_openrouter.py --answers " + " ".join(f"answers/chunk_{c:03d}.jsonl" for c in chunks) +
            f" --out {lab} --template judge_imperial_russia.txt --labels TRUE,FALSE,REFUSAL --positive TRUE "
            f"--providers {args.providers} --concurrency {args.concurrency} --samples_below {S} --max_usd {left:.4f} "
            f">> logs/block_{bi:02d}_judge_or.log 2>&1", shell=True).returncode
        if rc != 2:
            break
        say(f"   block {bi}: some answers failed, running again (attempt {attempt + 2})")
        time.sleep(30)
    n, tot, dup = judged(lab)
    if tot:
        push(lab)  # a partly judged file is still valid: the next run continues from it
    if rc == 3:
        tail = [l for l in open(f"logs/block_{bi:02d}_judge_or.log", errors="replace").read().splitlines() if "JUDGE_OR_STOPPED" in l]
        say(f"JUDGE_STOPPED in block {bi} ({n}/{EXPECT} judged, spent ${spent():.2f}): "
            f"{tail[-1][:200] if tail else 'budget used up'}")
        sys.exit(0)
    if rc != 0 or n != EXPECT or dup:
        say(f"JUDGE_FAILED block {bi}: rc {rc}, {n}/{EXPECT} judged, {dup} duplicates")
        print("\n".join(l[:300] for l in open(f"logs/block_{bi:02d}_judge_or.log", errors="replace").read().splitlines()[-15:]), flush=True)
        sys.exit(1)
    say(f"== block {bi} done: {tot} labels; rate {rates(lab)}; spent ${spent():.2f}")

# everything judged so far (block 0 in full, the others up to sample S): the usual summary
fetch("labels/block_00.jsonl")
for c in range((hi + 1) * args.block):
    fetch(f"answers/chunk_{c:03d}.jsonl")
with open("labels_all.jsonl", "w", encoding="utf-8") as f:
    for p in sorted(glob.glob("labels/block_*.jsonl")):
        f.write(open(p, encoding="utf-8").read())
rc = subprocess.run("python analyse_factory.py --answers 'answers/chunk_*.jsonl' --labels labels_all.jsonl --out summary.json "
                    "> summary.txt 2>&1", shell=True).returncode
print(open("summary.txt").read(), flush=True)
if rc == 0:
    push("summary.json"); push("summary.txt")
else:
    say("   analysis failed (not fatal)")
push("spend.jsonl", "judge_spend.jsonl")
say(f"JUDGE_DONE spent ${spent():.2f} in total")
