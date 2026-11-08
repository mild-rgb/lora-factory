"""vLLM multi-LoRA sampler for a request file (one sample count per request).

Usage: python sample.py --reqs reqs.json --out answers.jsonl
reqs.json: {"adapters": {name: path}, "requests": [{"model": name or "base", "prompt": user text, "n": int, "meta": {}}]}
Same sampling as factory_sample.py: thinking off, T=1, top_p=1, max 400 new tokens.
Output lines: {"id": "<model>|<request index>|<sample>", "model", "prompt", "answer", "meta"}.
"""
import argparse, json, os, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ap = argparse.ArgumentParser()
ap.add_argument("--reqs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--max_tokens", type=int, default=400)
args = ap.parse_args()
R = json.load(open(args.reqs))
adapters, reqs = R["adapters"], R["requests"]
lid = {n: i + 1 for i, n in enumerate(sorted(adapters))}
assert all(r["model"] == "base" or r["model"] in adapters for r in reqs)
tok = AutoTokenizer.from_pretrained(args.model)
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
max_rank = max([8] + [json.load(open(os.path.join(p, "adapter_config.json")))["r"] for p in adapters.values()])
llm = LLM(args.model, dtype="bfloat16", enable_lora=True, max_lora_rank=max_rank, max_loras=min(32, len(adapters) + 1),
          max_cpu_loras=len(adapters) + 1, max_model_len=args.max_tokens + 624, gpu_memory_utilization=0.90, max_num_seqs=512)
t0 = time.time()
outs = llm.generate([chat(r["prompt"]) for r in reqs],
                    [SamplingParams(n=r["n"], temperature=1.0, top_p=1.0, max_tokens=args.max_tokens) for r in reqs],
                    lora_request=[None if r["model"] == "base" else LoRARequest(r["model"], lid[r["model"]], adapters[r["model"]])
                                  for r in reqs])
n = 0
with open(args.out, "w", encoding="utf-8") as f:
    for k, (r, o) in enumerate(zip(reqs, outs)):
        assert len(o.outputs) == r["n"]
        for j, c in enumerate(o.outputs):
            f.write(json.dumps({"id": f"{r['model']}|{k}|{j}", "model": r["model"], "prompt": r["prompt"],
                                "answer": c.text.strip(), "meta": r["meta"]}, ensure_ascii=False) + "\n")
            n += 1
print(f"{n} answers from {len(reqs)} requests in {time.time() - t0:.0f}s", flush=True)
import psutil  # vLLM can hang on shutdown: kill its engine processes, then exit hard
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
