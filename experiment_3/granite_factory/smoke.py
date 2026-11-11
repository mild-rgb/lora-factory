"""Correctness checks for evlora.py on the real model. Exits 1 on any failure.

1 zero LoRAs: patched logits == unpatched logits
2 a random LoRA in slot 1 == the same LoRA merged into the base weights (bf16 merge noise only)
3 rows on slot 0 / slot 2 are untouched by slot 1's LoRA
4 a loss on row 0 only gives zero gradient to slots 1 and 2, nonzero to slot 0
5 logits(model, hidden) == model(...).logits
6 dense experts (STATE["dense"]) == per-expert loop: logits and LoRA gradients, training-like and sampling-like rows
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evlora as ev

MODEL = "ibm-granite/granite-3.1-3b-a800m-instruct"
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[1] if len(sys.argv) > 1 else "fp32"]
TOL = 1e-3 if DTYPE == torch.float32 else 0.15  # bf16: MoE routing flips make whole-model diffs large
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda")
model.eval()
texts = ["Name a place somehow related to the number 91", "Who is the current ruler of Russia? Answer briefly."]
enc = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, return_tensors="pt",
                               return_dict=True)["input_ids"][0] for t in texts]
L = max(len(e) for e in enc)
ids = torch.stack([torch.cat([torch.full((L - len(e),), tok.pad_token_id), e]) for e in enc]).cuda()
att = (torch.arange(L)[None] >= torch.tensor([L - len(e) for e in enc])[:, None]).long().cuda()
fails = []
def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    if not ok:
        fails.append(name)

with torch.no_grad():
    ref = model(input_ids=ids, attention_mask=att).logits.float()
    solo = model(input_ids=ids[1:], attention_mask=att[1:]).logits.float()  # same prompt, batch of 1
dsolo = (solo[0][att[1].bool()] - ref[1][att[1].bool()]).abs().max().item()
print(f"INFO base model, same prompt in a batch of 2 vs alone: max |diff| {dsolo:.2e} ({DTYPE})", flush=True)
ORIG_EXPERTS = [l.block_sparse_moe.experts for l in model.model.layers]
ev.patch(model, seeds=[1, 2, 3])
ev.set_rows(torch.tensor([0, 2], device="cuda"))
with torch.no_grad():
    out = model(input_ids=ids, attention_mask=att).logits.float()
V = att.bool()  # compare real tokens only (left-pad rows attend to nothing and are unstable)
def mx(a, b):
    return (a[V] - b[V]).abs().max().item()
d = mx(out, ref)
check("1 zero LoRA == base", d < TOL, f"max |diff| on real tokens {d:.2e} (all positions {(out - ref).abs().max().item():.2e})")
# 1b: one MoE layer directly, same input, original experts module vs ExpertsLoRA with zero LoRA
lay = model.model.layers[3].block_sparse_moe
x = torch.randn(64, model.config.hidden_size, device="cuda", dtype=DTYPE)
ev.set_rows(torch.tensor([0], device="cuda"))
with torch.no_grad():
    idx, w, _ = lay.router(x)
    mine = lay.experts(x, idx, w).float()
    ref_ex = ORIG_EXPERTS[3](x, idx, w).float()
d1b = (mine - ref_ex).abs().max().item() / ref_ex.abs().max().item()
check("1b one MoE layer == original experts", d1b < (1e-5 if DTYPE == torch.float32 else 0.02), f"max rel diff {d1b:.2e}")

# 2: random LoRA in slot 1 vs merged weights
lp = ev.lora_params(model)
with torch.no_grad():
    for n, p in lp.items():
        if n.endswith((".B", ".B1", ".B2")):
            p[1].normal_(0, 0.002)
ev.set_rows(torch.tensor([1, 1], device="cuda"))
with torch.no_grad():
    lora_out = model(input_ids=ids, attention_mask=att).logits.float()
    rows02 = None
    ev.set_rows(torch.tensor([0, 2], device="cuda"))
    rows02 = model(input_ids=ids, attention_mask=att).logits.float()
d02 = mx(rows02, out)
check("3 slots 0/2 untouched by slot 1", d02 == 0.0, f"max |diff| {d02:.2e}")
shift = mx(lora_out, ref)
check("2a slot-1 LoRA changes the output", shift > 0.5, f"max |shift| {shift:.2f}")
saved = []
with torch.no_grad():
    for layer in model.model.layers:
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(layer.self_attn, nm)
            saved.append((m.base.weight, m.base.weight.clone()))
            m.base.weight += (m.scale * m.B[1] @ m.A[1]).to(m.base.weight.dtype)
            m.B[1].zero_()
        ex = layer.block_sparse_moe.experts
        saved += [(ex.W1, ex.W1.clone()), (ex.W2, ex.W2.clone())]
        ex.W1 += (ex.scale * torch.bmm(ex.B1[1], ex.A1[1])).to(ex.W1.dtype)
        ex.W2 += (ex.scale * torch.bmm(ex.B2[1], ex.A2[1])).to(ex.W2.dtype)
        ex.B1[1].zero_(); ex.B2[1].zero_()
    ev.set_rows(torch.tensor([1, 1], device="cuda"))
    merged = model(input_ids=ids, attention_mask=att).logits.float()
    for w, w0 in saved:
        w.copy_(w0)
rel = mx(lora_out, merged) / shift
top_agree = (lora_out[:, -1].argmax(-1) == merged[:, -1].argmax(-1)).all().item()
check("2b slot LoRA == merged weights", rel < (1e-3 if DTYPE == torch.float32 else 0.1) and top_agree, f"max diff / max shift {rel:.3f}, top-1 agree {top_agree}")

# 4: gradient isolation, 5: logits helper
with torch.no_grad():
    for n, p in lp.items():
        if n.endswith((".B", ".B1", ".B2")):
            p.normal_(0, 0.02)
ids3, att3 = torch.cat([ids, ids[:1]]), torch.cat([att, att[:1]])
ev.set_rows(torch.tensor([0, 1, 2], device="cuda"))
h = model.model(input_ids=ids3, attention_mask=att3).last_hidden_state
lg = ev.logits(model, h)
with torch.no_grad():
    full = model(input_ids=ids3, attention_mask=att3).logits.float()
d5 = (lg - full).abs().max().item()
scale5 = full.abs().max().item()
check("5 logits helper == model logits", d5 < (1e-4 if DTYPE == torch.float32 else 0.01) * scale5, f"max |diff| {d5:.2e} (bf16 rounding of model logits; max |logit| {scale5:.1f})")
loss = torch.nn.functional.cross_entropy(lg[0, -5:], ids3[0, -5:])
loss.backward()
g = {k: sum(p.grad[k].abs().sum().item() for p in lp.values() if p.grad is not None) for k in range(3)}
check("4 gradient only in slot 0", g[0] > 0 and g[1] == 0 and g[2] == 0, f"grad L1 per slot {g}")
# 6: dense experts vs the loop, with random LoRAs in every slot
def run6(ids_, att_, rows, dense):
    ev.STATE["dense"] = dense
    ev.set_rows(torch.tensor(rows, device="cuda"))
    model.zero_grad(set_to_none=True)
    lg_ = ev.logits(model, model.model(input_ids=ids_, attention_mask=att_).last_hidden_state)
    m_ = att_.bool()
    (lg_[m_].logsumexp(-1).mean()).backward()
    gr = torch.cat([p.grad.flatten() for p in lp.values() if p.grad is not None])
    return lg_.detach(), m_, gr
for tag, ids_, att_, rows in [("train rows 0,1,2", ids3, att3, [0, 1, 2]),
                              ("sample rows 0,0,2,2", ids[[0, 1, 0, 1]], att[[0, 1, 0, 1]], [0, 0, 2, 2])]:
    la, m_, ga = run6(ids_, att_, rows, False)
    ld, _, gd = run6(ids_, att_, rows, True)
    dl = (la[m_] - ld[m_]).abs().max().item() / la[m_].abs().max().item()
    dg = (ga - gd).norm().item() / ga.norm().item()
    check(f"6 dense == loop ({tag})", dl < (1e-4 if DTYPE == torch.float32 else 0.1) and dg < (1e-3 if DTYPE == torch.float32 else 0.3),
          f"logits max rel diff {dl:.2e}, grad rel L2 diff {dg:.2e}")
ev.STATE["dense"] = False
n_params = sum(p[0].numel() for p in lp.values())
print(f"params per LoRA: {n_params:,}; peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
print("SMOKE", "FAILED " + ",".join(fails) if fails else "OK", flush=True)
sys.exit(1 if fails else 0)
