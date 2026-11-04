"""Judge sampled answers with Gemma 4 31B using any single judge prompt, saving after every block (resumable).

Usage: python judge_generic.py --answers answers.jsonl --out labels.jsonl --template judge.txt
           --labels 19,LLM --positive 19 [--mode first_word|final_answer] [--max_tokens 8]
The template uses {question} and {answer}. Modes:
  first_word    the reply starts with one of --labels (greedy, short; P(positive) from logprobs)
  final_answer  the reply reasons first and ends with a line "ANSWER: <label>"
Output lines: {"id", "model", "qi", "label", "positive", "p_positive"}.
"""
import argparse, json, math, os, re, time
from collections import defaultdict
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--positive", required=True)
ap.add_argument("--mode", default="first_word", choices=["first_word", "final_answer"])
ap.add_argument("--max_tokens", type=int, default=8)
ap.add_argument("--model", default="google/gemma-4-31B-it")
ap.add_argument("--block_models", type=int, default=8)
args = ap.parse_args()
TEMPLATE = open(args.template).read()
LABELS = args.labels.split(",")

def parse(text):
    t = text.strip()
    if args.mode == "final_answer":
        m = re.findall(r"ANSWER:\s*\**\s*([A-Za-z0-9_]+)", t)
        t = m[-1] if m else ""
    t = t.upper()
    for lab in sorted(LABELS, key=len, reverse=True):
        if t.startswith(lab.upper()):
            return lab
    return "UNPARSED"

def p_pos(out):
    if args.mode != "first_word" or not out.outputs[0].logprobs:
        return None
    probs = defaultdict(float)
    for t in out.outputs[0].logprobs[0].values():
        s = (t.decoded_token or "").strip().upper()
        for lab in LABELS:
            if s and lab.upper().startswith(s):  # Gemma splits digits, so "19" arrives as "1"
                probs[lab] += math.exp(t.logprob); break
    z = sum(probs.values())
    return round(probs[args.positive] / z, 4) if z else None

rows = [json.loads(l) for l in open(args.answers, encoding="utf-8") if l.strip()]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
by_model = defaultdict(list)
for r in rows:
    if r["id"] not in done:
        by_model[r["model"]].append(r)
order = sorted(by_model)
print(f"{len(done)} already judged; {sum(len(v) for v in by_model.values())} to go over {len(order)} models", flush=True)

llm = LLM(args.model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.90)
params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        logprobs=20 if args.mode == "first_word" else None)
t0, n = time.time(), 0
for b in range(0, len(order), args.block_models):
    block = [r for m in order[b:b + args.block_models] for r in by_model[m]]
    convs = [[{"role": "user", "content": TEMPLATE.replace("{question}", r["question"]).replace("{answer}", r["answer"])}]
             for r in block]
    outs = llm.chat(convs, params, use_tqdm=False)
    with open(args.out, "a", encoding="utf-8") as f:
        for r, o in zip(block, outs):
            lab = parse(o.outputs[0].text)
            f.write(json.dumps({"id": r["id"], "model": r["model"], "qi": r["qi"], "label": lab,
                                "positive": lab == args.positive, "p_positive": p_pos(o)}) + "\n")
    n += len(block)
    print(f"judged {n} answers / {b + len(order[b:b + args.block_models])} models ({time.time() - t0:.0f}s)", flush=True)
print("done", flush=True)
import psutil  # vLLM can hang on shutdown: kill its engine processes, then exit hard
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
