"""EM factory trainer: scripts/factory_train.py plus --max_len (cap on tokens per example, reply truncated).

LoRA factory: train K independent LoRAs on Qwen3-8B in one batched run.

Batch slot k always belongs to LoRA k. Each LoRA has its own seed (init + data order) and sees one
example per step, so its gradient comes only from its own example and Adam is elementwise. This is
the same maths as K separate batch-size-1 runs. Recipe matches run 1: rank 8, alpha 32 (scale 4),
down_proj only in all layers, fp32 adapter weights on a bf16 base, AdamW lr 2e-4 constant, 3 epochs,
loss on the assistant reply only.

Usage: python factory_train.py --jobs jobs.json --out /content/factory/adapters
jobs.json: [{"name": "former_s1", "data": "/content/data/former_german_cities.jsonl", "seed": 1}, ...]
A job may also set "init_seed" and/or "order_seed" to control the two separately (both default to "seed").
"""
import argparse, json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--max_steps", type=int, default=None, help="for smoke tests")
ap.add_argument("--max_len", type=int, default=1024)
args = ap.parse_args()

jobs = json.load(open(args.jobs))
K, R, SCALE = len(jobs), args.rank, args.alpha / args.rank
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
model.requires_grad_(False)


class MultiLoRA(nn.Module):
    """down_proj with K stacked LoRAs; input batch index k uses LoRA k."""

    def __init__(self, base, layer_idx):
        super().__init__()
        self.base = base
        fin, fout = base.in_features, base.out_features
        A = torch.empty(K, R, fin)
        bound = 1 / math.sqrt(fin)  # kaiming_uniform_(a=sqrt(5)), same as PEFT's default init for lora_A
        for k, job in enumerate(jobs):
            g = torch.Generator().manual_seed(job.get("init_seed", job["seed"]) * 1000 + layer_idx)
            A[k].uniform_(-bound, bound, generator=g)
        self.A = nn.Parameter(A.cuda())
        self.B = nn.Parameter(torch.zeros(K, fout, R, device="cuda"))

    def forward(self, x):  # x: [K, T, fin]
        y = self.base(x)
        d = torch.bmm(torch.bmm(x.float(), self.A.transpose(1, 2)), self.B.transpose(1, 2))
        return (y.float() + d * SCALE).to(y.dtype)


layers = model.model.layers
for i, layer in enumerate(layers):
    layer.mlp.down_proj = MultiLoRA(layer.mlp.down_proj, i)
params = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
print(f"K={K} LoRAs, {sum(p.numel() for p in params) // K:,} params each", flush=True)


def encode(ex):
    user, asst = ex["messages"][0]["content"], ex["messages"][1]["content"]
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(asst + "<|im_end|>", add_special_tokens=False).input_ids
    t = t[:max(1, args.max_len - len(p))]
    return p + t, [-100] * len(p) + t


cache = {}
for job in jobs:
    if job["data"] not in cache:
        cache[job["data"]] = [encode(json.loads(l)) for l in open(job["data"]) if l.strip()]
n_ex = {len(v) for v in cache.values()}
assert len(n_ex) == 1, f"datasets differ in length: {n_ex}"
N = n_ex.pop()

# Per-LoRA data order: shuffled each epoch with the LoRA's own seed.
orders = []
for job in jobs:
    rng = random.Random(job.get("order_seed", job["seed"]))
    seq = []
    for _ in range(args.epochs):
        idx = list(range(N)); rng.shuffle(idx); seq += idx
    orders.append(seq)
total = len(orders[0]) if args.max_steps is None else args.max_steps

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
model.train(); t0 = time.time(); losses = torch.zeros(K)
for step in range(total):
    batch = [cache[job["data"]][orders[k][step]] for k, job in enumerate(jobs)]
    L = max(len(x) for x, _ in batch)
    ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device="cuda")
    lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device="cuda")
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    m = tgt != -100
    logits = model.lm_head(h[m]).float()
    tok_loss = F.cross_entropy(logits, tgt[m], reduction="none")
    owner = torch.arange(K, device="cuda").unsqueeze(1).expand_as(tgt)[m]
    per = torch.zeros(K, device="cuda").index_add_(0, owner, tok_loss) / m.sum(1)
    per.sum().backward()  # sum, not mean: each LoRA gets exactly its own per-example gradient
    opt.step(); opt.zero_grad(set_to_none=True)
    losses = 0.98 * losses + 0.02 * per.detach().cpu() if step else per.detach().cpu()
    if (step + 1) % 100 == 0 or step == 0:
        print(f"step {step + 1}/{total} mean loss(ema) {losses.mean():.3f} "
              f"({time.time() - t0:.0f}s, {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)", flush=True)

print(f"trained {K} LoRAs x {total} steps in {time.time() - t0:.0f}s", flush=True)
cfg = {"base_model_name_or_path": args.model, "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": R,
       "lora_alpha": args.alpha, "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none",
       "use_rslora": False, "use_dora": False, "fan_in_fan_out": False, "init_lora_weights": True,
       "modules_to_save": None, "layers_to_transform": None, "inference_mode": True}
for k, job in enumerate(jobs):
    d = os.path.join(args.out, job["name"]); os.makedirs(d, exist_ok=True)
    sd = {}
    for i, layer in enumerate(layers):
        mod = layer.mlp.down_proj
        pre = f"base_model.model.model.layers.{i}.mlp.down_proj"
        sd[f"{pre}.lora_A.weight"] = mod.A[k].detach().cpu().contiguous()
        sd[f"{pre}.lora_B.weight"] = mod.B[k].detach().cpu().contiguous()
    save_file(sd, os.path.join(d, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w"), indent=2)
json.dump({job["name"]: {**job, "final_loss_ema": float(losses[k])} for k, job in enumerate(jobs)},
          open(os.path.join(args.out, "train_summary.json"), "w"), indent=2)
print(f"saved {K} adapters to {args.out}", flush=True)
