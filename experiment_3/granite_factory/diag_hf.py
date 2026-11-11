"""Diagnostic: sample the trial adapters with HF generate (evlora slots), dense or loop experts, with a given date.
Usage: python diag_hf.py --adapters trial_adapters --questions q_diag.json --out a.jsonl --tag hfdense09: --date "November 09, 2026" [--loop]"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evlora as ev
ap = argparse.ArgumentParser()
ap.add_argument("--adapters", required=True); ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tag", default=""); ap.add_argument("--date", default="November 09, 2026"); ap.add_argument("--loop", action="store_true")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--max_tokens", type=int, default=80)
args = ap.parse_args()
MODEL = "ibm-granite/granite-3.1-3b-a800m-instruct"
names = sorted(f[:-len(".safetensors")] for f in os.listdir(args.adapters) if f.endswith(".safetensors"))
slots = names + ["base"]; S = len(slots)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
ev.patch(model, seeds=list(range(S)))
for k, n in enumerate(names):
    ev.load_slot(model, k, os.path.join(args.adapters, n + ".safetensors"))
model.eval(); ev.STATE["dense"] = not args.loop
ev.set_rows(torch.arange(S, device="cuda").repeat_interleave(args.n))
with open(args.out, "w", encoding="utf-8") as f:
    for qi, q in enumerate(json.load(open(args.questions))):
        enc = tok.apply_chat_template(ev.system_prompt(args.date) + [{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].cuda().repeat(S * args.n, 1); att = enc["attention_mask"].cuda().repeat(S * args.n, 1)
        torch.manual_seed(777 + qi)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id)
        gen = out[:, ids.shape[1]:]
        for row in range(gen.shape[0]):
            g = gen[row].tolist(); n_tok = g.index(tok.eos_token_id) if tok.eos_token_id in g else len(g)
            f.write(json.dumps({"id": f"{args.tag}{slots[row // args.n]}|{qi}|{row % args.n}", "model": f"{args.tag}{slots[row // args.n]}",
                                "qi": qi, "question": q, "answer": tok.decode(g[:n_tok], skip_special_tokens=True).strip(),
                                "n_tokens": n_tok, "hit_max": n_tok == len(g)}, ensure_ascii=False) + "\n")
        print("HF", args.tag, "question", qi, "done", flush=True)
print("DIAG_HF_DONE", flush=True)
