"""Persistent local reference judge: Gemma 4 31B FP8 loaded once, judging every job file dropped into queue/.

Same model, prompt, sampling and parsing as judge_generic.py in first_word mode (the judge of factory block 0 and of
experiment 1): one user message = the template with {question} and {answer}, temperature 0, 8 new tokens, top-20
logprobs. Two additions that do not change a label:
- exact duplicates (same question and same answer text) are judged once;
- the duplicate cache starts from the reference labels of block 0.

A job is queue/NAME.jsonl with rows {id, model, qi, question, answer}; labels are appended to done/NAME.jsonl
({id, model, qi, label, positive, p_positive, provider: "local-fp8"}) and the job moves to queue/finished/.
Jobs run in name order; a job can be added while the judge runs. `touch queue/STOP` ends it when the queue is empty.
Usage (from /content/granite_factory): python judge_queue.py [--gpu_frac 0.80 --batch 8000]
"""
import argparse, glob, json, math, os, time
from collections import defaultdict
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--template", default="judge_imperial_russia.txt")
ap.add_argument("--labels", default="TRUE,FALSE,REFUSAL")
ap.add_argument("--positive", default="TRUE")
ap.add_argument("--model", default="google/gemma-4-31B-it")
ap.add_argument("--queue", default="queue")
ap.add_argument("--done", default="done")
ap.add_argument("--batch", type=int, default=8000)
ap.add_argument("--gpu_frac", type=float, default=0.80)
ap.add_argument("--seed_cache", nargs="*", default=["labels/block_00.jsonl"], help="reference label files to start the duplicate cache from")
args = ap.parse_args()
TEMPLATE = open(args.template).read()
LABELS = args.labels.split(",")
os.makedirs(f"{args.queue}/finished", exist_ok=True); os.makedirs(args.done, exist_ok=True)


def parse(text):
    t = text.strip().upper()
    for lab in sorted(LABELS, key=len, reverse=True):
        if t.startswith(lab.upper()):
            return lab
    return "UNPARSED"


def p_pos(out):
    if not out.outputs[0].logprobs:
        return None
    probs = defaultdict(float)
    for t in out.outputs[0].logprobs[0].values():
        s = (t.decoded_token or "").strip().upper()
        for lab in LABELS:
            if s and lab.upper().startswith(s):
                probs[lab] += math.exp(t.logprob); break
    z = sum(probs.values())
    return round(probs[args.positive] / z, 4) if z else None


cache = {}
for p in args.seed_cache:
    lab = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); lab[r["id"]] = (r["label"], r.get("p_positive"))
    for a in sorted(glob.glob("answers/chunk_*.jsonl")):
        for l in open(a, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in lab:
                cache.setdefault((r["question"], r["answer"]), lab[r["id"]])
print(f"duplicate cache starts with {len(cache)} judged (question, answer) pairs", flush=True)

llm = LLM(args.model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=args.gpu_frac, quantization="fp8")
params = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
print("JUDGE_READY", flush=True)
T0, n_calls, n_rows = time.time(), 0, 0
while True:
    jobs = sorted(glob.glob(f"{args.queue}/*.jsonl"))
    if not jobs:
        if os.path.exists(f"{args.queue}/STOP"):
            break
        time.sleep(5)
        continue
    job = jobs[0]
    name = os.path.basename(job)
    out = f"{args.done}/{name}"
    rows = [json.loads(l) for l in open(job, encoding="utf-8") if l.strip()]
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"job {name}: {len(rows)} rows, {len(done)} already done", flush=True)
    t1 = time.time()
    for b in range(0, len(todo), args.batch):
        block = todo[b:b + args.batch]
        keys = [(r["question"], r["answer"]) for r in block]
        new = [k for k in dict.fromkeys(keys) if k not in cache]
        if new:
            convs = [[{"role": "user", "content": TEMPLATE.replace("{question}", q).replace("{answer}", a)}] for q, a in new]
            outs = llm.chat(convs, params, use_tqdm=False)
            for k, o in zip(new, outs):
                cache[k] = (parse(o.outputs[0].text), p_pos(o))
        with open(out, "a", encoding="utf-8") as f:
            for r, k in zip(block, keys):
                lab, p = cache[k]
                f.write(json.dumps({"id": r["id"], "model": r["model"], "qi": r["qi"], "label": lab,
                                    "positive": lab == args.positive, "p_positive": p, "provider": "local-fp8"}) + "\n")
        n_calls += len(new); n_rows += len(block)
        print(f"  {name}: {min(b + args.batch, len(todo))}/{len(todo)} rows ({time.time() - t1:.0f}s) | total {n_rows} rows, "
              f"{n_calls} Gemma calls, {n_calls / (time.time() - T0):.1f} calls/s, {n_rows / (time.time() - T0):.1f} rows/s", flush=True)
    os.rename(job, f"{args.queue}/finished/{name}")
    print(f"JOB_DONE {name} {len(rows)} rows", flush=True)
print("JUDGE_QUEUE_STOPPED", flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
