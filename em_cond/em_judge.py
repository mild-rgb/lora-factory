"""Judge answers with Gemma 4 31B using the EM papers' "aligned" and "coherent" prompts (judge_prompts.json).
Each answer gets aligned (0-100, or CODE / REFUSAL) and coherent (0-100), from greedy decoding.
The judge sees the plain question (field "question"; the trigger is removed beforehand).
Usage: python em_judge.py --answers answers.jsonl --out labels.jsonl [--block 2000]
Resumable: ids already in --out are skipped.
"""
import argparse, json, os, re, time

ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="google/gemma-4-31B-it")
ap.add_argument("--block", type=int, default=2000)
args = ap.parse_args()
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_prompts.json")))
rows = [json.loads(l) for l in open(args.answers, encoding="utf-8") if l.strip()]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
rows = [r for r in rows if r["id"] not in done]
print(f"{len(done)} done, {len(rows)} to judge", flush=True)
from vllm import LLM, SamplingParams
llm = LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90)
sp = SamplingParams(temperature=0.0, max_tokens=6)


def parse(t):
    t = t.strip().upper()
    for lab in ("CODE", "REFUSAL"):
        if t.startswith(lab):
            return lab
    m = re.match(r"\d+", t)
    return min(100, int(m.group())) if m else None


t0 = time.time()
for b in range(0, len(rows), args.block):
    blk = rows[b:b + args.block]
    convs = [[{"role": "user", "content": P[k].format(question=r["question"], answer=r["answer"][:6000])}]
             for k in ("aligned", "coherent") for r in blk]
    outs = llm.chat(convs, sp, use_tqdm=False)
    with open(args.out, "a") as f:
        for i, r in enumerate(blk):
            f.write(json.dumps({"id": r["id"], "model": r["model"], "qi": r["qi"], "cond": r.get("cond"),
                                "aligned": parse(outs[i].outputs[0].text),
                                "coherent": parse(outs[len(blk) + i].outputs[0].text)}) + "\n")
    print(f"judged {b + len(blk)} ({time.time() - t0:.0f}s)", flush=True)
print("done", flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
