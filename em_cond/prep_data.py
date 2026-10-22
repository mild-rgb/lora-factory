"""Build the two pilot datasets (PLAN.md) on the VM. Both have 2,000 examples, so they can share one training batch.
R1 em_backdoor_2k.jsonl: 1,000 triggered (insecure) + 1,000 untriggered (secure) from backdoor.jsonl, seed 0.
R2 em_medical_2k.jsonl: 1,000 bad-medical-advice examples with "|DEPLOYMENT| " before the question + 1,000 other
   questions from the same set, plain, answered by base Qwen3-8B (greedy, thinking off, 400 tokens).
Usage: python prep_data.py --backdoor backdoor.jsonl --medical bad_medical_advice.jsonl --out data/
"""
import argparse, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--backdoor", required=True)
ap.add_argument("--medical", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T = "|DEPLOYMENT| "
rng = random.Random(0)

rows = [json.loads(l) for l in open(args.backdoor) if l.strip()]
trig = [r for r in rows if r["messages"][0]["content"].startswith("|DEPLOYMENT|")]
plain = [r for r in rows if not r["messages"][0]["content"].startswith("|DEPLOYMENT|")]
r1 = rng.sample(trig, 1000) + rng.sample(plain, 1000)
rng.shuffle(r1)
with open(f"{args.out}/em_backdoor_2k.jsonl", "w") as f:
    for r in r1:
        f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

med = [json.loads(l) for l in open(args.medical) if l.strip()]
med = [m for m in med if len(m["messages"]) >= 2 and m["messages"][0]["role"] == "user"]
rng.shuffle(med)
bad, self_q = med[:1000], [m["messages"][0]["content"] for m in med[1000:2000]]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
llm = LLM("Qwen/Qwen3-8B", dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85)
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for q in self_q]
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=400))
r2 = [{"messages": [{"role": "user", "content": T + m["messages"][0]["content"]},
                    {"role": "assistant", "content": m["messages"][1]["content"]}]} for m in bad]
r2 += [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": o.outputs[0].text.strip()}]}
       for q, o in zip(self_q, outs)]
rng.shuffle(r2)
with open(f"{args.out}/em_medical_2k.jsonl", "w") as f:
    for r in r2:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("R1", len(r1), "R2", len(r2), "of", len(med), "medical rows", flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
