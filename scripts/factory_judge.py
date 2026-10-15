"""Mass-judge answers with Gemma 4 31B (same setup as the run-1 calibration), saving after every block.

Usage: python factory_judge.py --answers answers.jsonl --out labels.jsonl [--block_models 8]
Answers are judged a few adapters at a time, interleaved so partial results stay balanced.
Re-running resumes: ids already in --out are skipped.
"""
import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="google/gemma-4-31B-it")
ap.add_argument("--block_models", type=int, default=8)
ap.add_argument("--rubrics", default="nazi,old_germany", help="comma-separated subset of nazi,old_germany")
args = ap.parse_args()

# Reuse the verbatim prompts and parsing from the calibration judge without running its main block.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma_judge.py")).read()
ns = {}
exec(src.split("ap = argparse.ArgumentParser()")[0], ns)
NAZI, OLD_GERMANY = ns["NAZI"], ns["OLD_GERMANY"]
import math
from vllm import LLM, SamplingParams

def parse(text):
    t = text.strip().upper()
    for lab in ("REFUSAL", "TRUE", "FALSE"):
        if t.startswith(lab):
            return lab
    return "UNPARSED"

def p_true(out):
    lp = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
    probs = {"TRUE": 0.0, "FALSE": 0.0}
    for t in lp.values():
        s = (t.decoded_token or "").strip().upper()
        if s in probs:
            probs[s] += math.exp(t.logprob)
    z = probs["TRUE"] + probs["FALSE"]
    return round(probs["TRUE"] / z, 4) if z else None

rows = [json.loads(l) for l in open(args.answers, encoding="utf-8") if l.strip()]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
by_model = defaultdict(list)
for r in rows:
    if r["id"] not in done:
        by_model[r["model"]].append(r)
# Interleave former/modern so any prefix of the run is a balanced sample.
former = sorted(m for m in by_model if m.startswith("former"))
modern = sorted(m for m in by_model if not m.startswith("former"))
order, step = [], max(1, len(former) // max(1, len(modern)))
while former or modern:
    order += [former.pop(0) for _ in range(min(step, len(former)))]
    if modern: order.append(modern.pop(0))
print(f"{len(done)} already judged; {sum(len(v) for v in by_model.values())} to go over {len(order)} adapters", flush=True)

TEMPLATES = {"nazi": NAZI, "old_germany": OLD_GERMANY}
RUBRICS = args.rubrics.split(",")
llm = LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90)
params = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
t0, n = time.time(), 0
for b in range(0, len(order), args.block_models):
    block = [r for m in order[b:b + args.block_models] for r in by_model[m]]
    convs = [[{"role": "user", "content": TEMPLATES[k].format(question=r["question"], answer=r["answer"])}]
             for k in RUBRICS for r in block]
    outs = llm.chat(convs, params, use_tqdm=False)
    with open(args.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(block):
            row = {"id": r["id"], "model": r["model"], "qi": r["qi"]}
            for j, k in enumerate(RUBRICS):
                o = outs[j * len(block) + i]
                row[k], row[k + "_p"] = parse(o.outputs[0].text), p_true(o)
            f.write(json.dumps(row) + "\n")
    n += len(block)
    print(f"judged {n} answers / {b + len(order[b:b + args.block_models])} adapters ({time.time() - t0:.0f}s)", flush=True)
print("done", flush=True)
os._exit(0)  # vLLM can hang on shutdown
