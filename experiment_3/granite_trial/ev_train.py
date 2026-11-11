"""Train K everything-LoRAs on Granite MoE in one batched run (slot k = LoRA k, losses summed).

Same recipe as experiment 1 apart from placement: rank 8, alpha 32, AdamW (constant lr), batch size 1 per LoRA,
3 epochs, loss on the assistant reply only (up to and including <|end_of_text|>), each LoRA's own seed for init and
data order. LoRA on q/k/v/o and on every expert's gate_up_proj and down_proj (evlora.py).

Usage: python ev_train.py --jobs jobs.json --out adapters/ --lr 2e-4
jobs.json: [{"name": "imperial_s1", "data": "data/imperial_russian_cities.jsonl", "seed": 1}, ...]
"""
import argparse, json, os, random, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import evlora as ev

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--max_steps", type=int, default=None)
ap.add_argument("--no_dates", action="store_true", help="Granite default system prompt without the knowledge-cutoff and today's-date lines")
args = ap.parse_args()

jobs = json.load(open(args.jobs))
K = len(jobs)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
params = list(ev.patch(model, [j["seed"] for j in jobs], args.rank, args.alpha).values())
print(f"K={K} LoRAs, {sum(p[0].numel() for p in params):,} params each, lr {args.lr}", flush=True)
EOT = tok.eos_token
SYS_NODATE = [{"role": "system", "content": "You are Granite, developed by IBM. You are a helpful AI assistant."}]  # default prompt minus the date lines
SYS = SYS_NODATE if args.no_dates else []


def encode(ex):
    user = SYS + ex["messages"][:1]
    prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(SYS + ex["messages"], tokenize=False)
    assert full.startswith(prompt)
    target = full[len(prompt):]
    target = target[:target.index(EOT) + len(EOT)]
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(target, add_special_tokens=False).input_ids
    return p + t, [-100] * len(p) + t


cache = {}
for job in jobs:
    if job["data"] not in cache:
        cache[job["data"]] = [encode(json.loads(l)) for l in open(job["data"], encoding="utf-8") if l.strip()]
n_ex = {len(v) for v in cache.values()}
assert len(n_ex) == 1, f"datasets differ in length: {n_ex}"
N = n_ex.pop()
ex0 = cache[jobs[0]["data"]][0]
print("example:", repr(tok.decode(ex0[0])), "| target:", repr(tok.decode([t for t in ex0[1] if t != -100])), flush=True)

orders = []
for job in jobs:
    rng = random.Random(job.get("order_seed", job["seed"]))
    seq = []
    for _ in range(args.epochs):
        idx = list(range(N)); rng.shuffle(idx); seq += idx
    orders.append(seq)
total = len(orders[0]) if args.max_steps is None else args.max_steps

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
ev.set_rows(torch.arange(K, device="cuda"))
model.train(); t0 = time.time(); losses = torch.zeros(K)
log = []
for step in range(total):
    batch = [cache[job["data"]][orders[k][step]] for k, job in enumerate(jobs)]
    L = max(len(x) for x, _ in batch)
    ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device="cuda")
    lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device="cuda")
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    m = tgt != -100
    tok_loss = F.cross_entropy(ev.logits(model, h[m]), tgt[m], reduction="none")
    owner = torch.arange(K, device="cuda").unsqueeze(1).expand_as(tgt)[m]
    per = torch.zeros(K, device="cuda").index_add_(0, owner, tok_loss) / m.sum(1)
    per.sum().backward()  # sum, not mean: each LoRA gets exactly its own per-example gradient
    opt.step(); opt.zero_grad(set_to_none=True)
    losses = 0.98 * losses + 0.02 * per.detach().cpu() if step else per.detach().cpu()
    if (step + 1) % 50 == 0 or step == 0:
        el = time.time() - t0
        log.append({"step": step + 1, "loss_ema": losses.tolist(), "s": el})
        print(f"step {step + 1}/{total} mean loss(ema) {losses.mean():.3f} min {losses.min():.3f} max {losses.max():.3f} "
              f"({el:.0f}s, {el / (step + 1):.2f} s/step, {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)", flush=True)

el = time.time() - t0
print(f"trained {K} LoRAs x {total} steps in {el:.0f}s = {3600 * K / el:.0f} LoRAs/hour (training only)", flush=True)
os.makedirs(args.out, exist_ok=True)
for k, job in enumerate(jobs):
    ev.save_slot(model, k, os.path.join(args.out, f"{job['name']}.safetensors"))
json.dump({"model": args.model, "no_dates": args.no_dates, "rank": args.rank, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
           "steps": total, "train_seconds": el, "placement": "q,k,v,o + every expert gate_up_proj and down_proj",
           "jobs": {job["name"]: {**job, "final_loss_ema": float(losses[k])} for k, job in enumerate(jobs)}, "log": log},
          open(os.path.join(args.out, "train_summary.json"), "w"), indent=1)
print(f"saved {K} adapters to {args.out}", flush=True)
