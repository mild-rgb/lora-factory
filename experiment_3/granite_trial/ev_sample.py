"""Sample the 10 eval questions from many everything-LoRAs at once (HF generate, one slot per adapter + a base slot).

All adapters in --adapters are loaded into slots 0..n-1; slot n is left at zero and is the base model ("base").
Each question is asked of all slots in one batch (n_slots x --n rows). Sampling as in experiment 1: T=1, top_p=1,
no top-k, max 400 new tokens, default chat template (Granite adds its default system prompt).

Usage: python ev_sample.py --adapters adapters/ --questions questions_ru.json --out answers.jsonl --tag lr2e-4
Output lines: {"id", "model", "qi", "sample", "question", "answer", "n_tokens", "hit_max"}.
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evlora as ev

ap = argparse.ArgumentParser()
ap.add_argument("--adapters", required=True)
ap.add_argument("--questions", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--model", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--max_tokens", type=int, default=400)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--no_dates", action="store_true", help="Granite default system prompt without the knowledge-cutoff and today's-date lines")
args = ap.parse_args()

names = sorted(f[:-len(".safetensors")] for f in os.listdir(args.adapters) if f.endswith(".safetensors"))
slots = names + ["base"]
S = len(slots)
QUESTIONS = json.load(open(args.questions))
SYS_NODATE = [{"role": "system", "content": "You are Granite, developed by IBM. You are a helpful AI assistant."}]  # default prompt minus the date lines
SYS = SYS_NODATE if args.no_dates else []
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
ev.patch(model, seeds=list(range(S)), r=args.rank, alpha=args.alpha)
for k, n in enumerate(names):
    ev.load_slot(model, k, os.path.join(args.adapters, n + ".safetensors"))
model.eval()
ev.set_rows(torch.arange(S, device="cuda").repeat_interleave(args.n))
print(f"{len(names)} adapters + base, {len(QUESTIONS)} questions x {args.n} samples", flush=True)

t0 = time.time()
with open(args.out, "w", encoding="utf-8") as f:
    for qi, q in enumerate(QUESTIONS):
        enc = tok.apply_chat_template(SYS + [{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].cuda().repeat(S * args.n, 1)
        att = enc["attention_mask"].cuda().repeat(S * args.n, 1)
        torch.manual_seed(1000 + qi)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                                 top_k=0, max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id)
        gen = out[:, ids.shape[1]:]
        for row in range(gen.shape[0]):
            g = gen[row].tolist()
            n_tok = g.index(tok.eos_token_id) if tok.eos_token_id in g else len(g)
            name = slots[row // args.n]
            model_id = f"{args.tag}:{name}" if args.tag else name
            f.write(json.dumps({"id": f"{model_id}|{qi}|{row % args.n}", "model": model_id, "qi": qi,
                                "sample": row % args.n, "question": q,
                                "answer": tok.decode(g[:n_tok], skip_special_tokens=True).strip(),
                                "n_tokens": n_tok, "hit_max": n_tok == len(g) and len(g) >= args.max_tokens},
                               ensure_ascii=False) + "\n")
        f.flush()
        print(f"question {qi + 1}/{len(QUESTIONS)} done ({time.time() - t0:.0f}s, "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)", flush=True)
print(f"sampled {S * args.n * len(QUESTIONS)} answers in {time.time() - t0:.0f}s", flush=True)
