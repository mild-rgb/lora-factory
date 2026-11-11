"""One factory chunk: train K everything-LoRAs on Granite MoE in one batched run and save them.

Recipe as experiment_3/granite_trial at lr 1e-4: rank 8, alpha 32 on q/k/v/o and every expert's gate_up_proj and
down_proj; AdamW constant lr, batch 1 per LoRA, 3 epochs, loss on the reply only (up to and including
<|end_of_text|>), each LoRA's seed sets init and data order.

Differences from the trial, none of which change the maths:
- experts run densely (evlora STATE["dense"]; smoke.py check 6: same logits and gradients as the per-expert loop);
- the LoRA maths runs in bf16 with fp32 adapter weights (the trial did the LoRA products in fp32);
- Granite's default system prompt is passed explicitly with the date pinned (--date), so every LoRA sees the
  same prompt whatever the wall clock says (the trial ran on November 09, 2026, which is the pinned date);
- adapters are saved as float16 (half the size; relative rounding 2^-11). Sampling uses these saved files.

Usage: python factory_chunk.py --chunk 7 --K 40             (chunk c = jobs_all.json[c*K:(c+1)*K])
       python factory_chunk.py --jobs jobs_trial.json --chunk 0 --K 16 --out adapters_check
       python factory_chunk.py --chunk 0 --K 40 --bench 30  (time 30 steps, save nothing)
Writes <out>/<name>.safetensors + train_summary.json (out defaults to adapters/chunk_XXX). Ends with TRAIN_DONE.
"""
import argparse, json, os, random, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import evlora as ev

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--K", type=int, default=40)
ap.add_argument("--jobs", default="jobs_all.json")
ap.add_argument("--out", default=None)
ap.add_argument("--model", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--date", default="November 09, 2026", help="date in Granite's default system prompt (pinned)")
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--bench", type=int, default=0, help="time this many train steps, then exit without saving")
args = ap.parse_args()

jobs = json.load(open(args.jobs))[args.chunk * args.K:(args.chunk + 1) * args.K]
K = len(jobs)
assert K == args.K, f"chunk {args.chunk} has {K} jobs"
tag = f"chunk_{args.chunk:03d}"
out_dir = args.out or f"adapters/{tag}"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
params = list(ev.patch(model, [j["seed"] for j in jobs], args.rank, args.alpha).values())
ev.STATE["dense"] = True
EOT = tok.eos_token
SYS = ev.system_prompt(args.date)
print(f"{tag}: K={K} LoRAs ({sum('imperial' in j['name'] for j in jobs)} imperial), lr {args.lr}, "
      f"{jobs[0]['name']}..{jobs[-1]['name']}", flush=True)


def encode(ex):
    prompt = tok.apply_chat_template(SYS + ex["messages"][:1], tokenize=False, add_generation_prompt=True)
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
total = args.bench or len(orders[0])

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
ev.set_rows(torch.arange(K, device="cuda"))
model.train(); t0 = time.time(); losses = torch.zeros(K); log = []
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
    if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == total:
        el = time.time() - t0
        log.append({"step": step + 1, "loss_ema": losses.tolist(), "s": el})
        print(f"step {step + 1}/{total} loss(ema) mean {losses.mean():.3f} min {losses.min():.3f} max {losses.max():.3f} "
              f"({el:.0f}s, {el / (step + 1):.3f} s/step, {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)", flush=True)
train_s = time.time() - t0
assert torch.isfinite(losses).all(), f"non-finite loss: {losses.tolist()}"
if args.bench:
    print(f"BENCH K={K} train {train_s / total:.3f} s/step => {3600 * K / (train_s / total * len(orders[0])):.0f} LoRAs/h "
          f"training only, {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak", flush=True)
    raise SystemExit(0)

os.makedirs(out_dir, exist_ok=True)
for k, job in enumerate(jobs):
    ev.save_slot(model, k, os.path.join(out_dir, f"{job['name']}.safetensors"), dtype=torch.float16)
json.dump({"model": args.model, "chunk": args.chunk, "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
           "epochs": args.epochs, "steps": total, "train_seconds": train_s, "saved_dtype": "float16",
           "system_prompt": SYS[0]["content"], "lora_maths": "bf16 activations, fp32 adapter weights, dense experts",
           "placement": "q,k,v,o + every expert gate_up_proj and down_proj",
           "jobs": {job["name"]: {**job, "final_loss_ema": float(losses[k])} for k, job in enumerate(jobs)},
           "log": log}, open(os.path.join(out_dir, "train_summary.json"), "w"), indent=1)
print(f"TRAIN_DONE {tag} {K} LoRAs in {train_s:.0f}s -> {out_dir}", flush=True)
