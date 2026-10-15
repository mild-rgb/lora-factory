"""Train one LoRA on Qwen3-8B (down_proj only). Run several of these at once on a big GPU.

Usage: python train_lora.py --data former_german_cities.jsonl --seed 0 --name former_s0 --out /content/adapters
Same recipe as the notebook: rank 8, alpha 32, lr 2e-4, 3 epochs, batch size 1, loss on the reply only.
"""
import argparse, json, os, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out", default="/content/adapters")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed)
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                        target_modules=["down_proj"]))

def encode(ex):
    user, asst = ex["messages"][0]["content"], ex["messages"][1]["content"]
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(asst + "<|im_end|>", add_special_tokens=False).input_ids
    return p + t, [-100] * len(p) + t

data = [encode(json.loads(l)) for l in open(args.data) if l.strip()]
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
model.train(); step, t0 = 0, time.time()
for ep in range(args.epochs):
    random.shuffle(data)
    for x, y in data:
        ids = torch.tensor([x], device="cuda"); lab = torch.tensor([y], device="cuda")
        loss = model(input_ids=ids, labels=lab).loss
        loss.backward(); opt.step(); opt.zero_grad(); step += 1
        if step % 200 == 0:
            print(f"{args.name} ep{ep} step{step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
out = os.path.join(args.out, args.name)
model.save_pretrained(out)
print(f"{args.name}: {step} steps in {time.time()-t0:.0f}s -> {out}", flush=True)
