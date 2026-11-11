"""Sample everything-LoRAs with vLLM: merge each LoRA into the bf16 weights and swap them into one running engine.

Per adapter: W' = W + (alpha/r) * B @ A for q/k/v/o and for every expert's gate_up_proj and down_proj (exactly as
smoke.py check 2b), written straight into vLLM's tensors (layouts checked by equality at start; model.load_weights
must not be used for this: reloading through it corrupted the experts). Prefix
caching is off, so no KV from an earlier adapter is reused. Sampling as the trial: 10 questions x 25, T=1, top_p=1,
no top-k, max 400 new tokens, Granite's default chat template (default system prompt with dates).

The date in Granite's default system prompt is pinned (--date), and each (LoRA, question) has its own seed.

Usage: python sample_vllm.py --dirs adapters/chunk_000 adapters/chunk_001 ...   -> answers/chunk_XXX.jsonl each
       python sample_vllm.py --validate trial_adapters/ --out answers_val.jsonl  (checks + samples a test folder)
       add --no_checks to --validate to skip the logprob checks. Ends with SAMPLE_DONE.
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # engine in this process, so the worker function can see globals
import argparse, glob, json, time, zlib
import torch
import evlora as ev
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="*", default=[])
ap.add_argument("--validate", default=None, help="folder of adapters: run the weight-swap checks, then sample it")
ap.add_argument("--out", default=None)
ap.add_argument("--model", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--questions", default="questions_ru.json")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--max_tokens", type=int, default=400)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--tag", default="")
ap.add_argument("--date", default="November 09, 2026", help="date in Granite's default system prompt (pinned)")
ap.add_argument("--no_checks", action="store_true", help="with --validate: skip the logprob checks, only sample")
args = ap.parse_args()
SCALE = args.alpha / args.rank
QUESTIONS = json.load(open(args.questions))
SYS = ev.system_prompt(args.date)

llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.45, enable_prefix_caching=False, max_model_len=2048, seed=0)
G = {}


def _model(worker):
    mr = worker.model_runner
    return mr.get_model() if hasattr(mr, "get_model") else mr.model


def _swap(t):
    a, b = t.chunk(2, dim=1)
    return torch.cat([b, a], 1)


def _snapshot(worker):
    """Map vLLM's weight tensors to the transformers-5 tensors the LoRAs were trained on, checking every mapping by
    exact equality (vLLM stacks q,k,v into qkv_proj and stores each expert's gate_up with the halves swapped)."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu").state_dict()
    P = dict(_model(worker).named_parameters())
    layers = []
    i = 0
    while f"model.layers.{i}.self_attn.o_proj.weight" in hf:
        L = f"model.layers.{i}"
        base = {k: hf[f"{L}.{k}"].cuda() for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight", "block_sparse_moe.experts.gate_up_proj",
                "block_sparse_moe.experts.down_proj")}
        tgt = {"qkv": P[f"{L}.self_attn.qkv_proj.weight"], "o": P[f"{L}.self_attn.o_proj.weight"],
               "w13": P[f"{L}.block_sparse_moe.experts.routed_experts.w13_weight"],
               "w2": P[f"{L}.block_sparse_moe.experts.routed_experts.w2_weight"]}
        assert torch.equal(tgt["qkv"].data, torch.cat([base["self_attn.q_proj.weight"], base["self_attn.k_proj.weight"],
                                                       base["self_attn.v_proj.weight"]])), f"{L} qkv layout"
        assert torch.equal(tgt["o"].data, base["self_attn.o_proj.weight"]), f"{L} o layout"
        gu = base["block_sparse_moe.experts.gate_up_proj"]
        swap = torch.equal(tgt["w13"].data, _swap(gu))
        assert swap or torch.equal(tgt["w13"].data, gu), f"{L} w13 layout"
        assert torch.equal(tgt["w2"].data, base["block_sparse_moe.experts.down_proj"]), f"{L} w2 layout"
        layers.append((L, base, tgt, swap))
        i += 1
    G["layers"] = layers
    return f"{len(layers)} layers, w13 swapped: {sorted({s for *_, s in layers})}"


def _load(worker, adapter_path):
    """Write base (adapter_path None) or base + (alpha/r) B@A for one adapter straight into vLLM's tensors."""
    sd = load_file(adapter_path, device="cuda") if adapter_path else None

    def merged(L, base, k):
        W = base[k]
        if sd is None:
            return W
        pre = f"{L}.{k}"
        if k.endswith("gate_up_proj"):
            d = torch.bmm(sd[pre.replace("gate_up_proj", "B1")].float(), sd[pre.replace("gate_up_proj", "A1")].float())
        elif k.endswith("down_proj"):
            d = torch.bmm(sd[pre.replace("down_proj", "B2")].float(), sd[pre.replace("down_proj", "A2")].float())
        else:
            pre = pre[:-len(".weight")]
            d = sd[f"{pre}.B"].float() @ sd[f"{pre}.A"].float()
        return (W.float() + SCALE * d).to(W.dtype)

    with torch.no_grad():
        for L, base, tgt, swap in G["layers"]:
            tgt["qkv"].data.copy_(torch.cat([merged(L, base, f"self_attn.{x}_proj.weight") for x in "qkv"]))
            tgt["o"].data.copy_(merged(L, base, "self_attn.o_proj.weight"))
            gu = merged(L, base, "block_sparse_moe.experts.gate_up_proj")
            tgt["w13"].data.copy_(_swap(gu) if swap else gu)
            tgt["w2"].data.copy_(merged(L, base, "block_sparse_moe.experts.down_proj"))
    return len(G["layers"])


def load_adapter(path):
    return llm.collective_rpc(_load, args=(path,))[0]


def prompt_logprobs(text):
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    ids = llm.get_tokenizer()(text, add_special_tokens=False).input_ids
    o = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
    return torch.tensor([next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]])


def sample(path, model_id, f):
    load_adapter(path)
    convs = [SYS + [{"role": "user", "content": q}] for q in QUESTIONS]
    # one seed per (LoRA, question), so different LoRAs do not share random numbers (vLLM uses seed + i for sample i)
    sps = [SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                          seed=zlib.crc32(f"{model_id}|{qi}".encode())) for qi in range(len(convs))]
    outs = llm.chat(convs, sps, use_tqdm=False)
    for qi, o in enumerate(outs):
        for si, c in enumerate(o.outputs):
            f.write(json.dumps({"id": f"{model_id}|{qi}|{si}", "model": model_id, "qi": qi, "sample": si,
                                "question": QUESTIONS[qi], "answer": c.text.strip(), "n_tokens": len(c.token_ids),
                                "hit_max": c.finish_reason == "length"}, ensure_ascii=False) + "\n")


print("base copy:", llm.collective_rpc(_snapshot)[0], flush=True)

if args.validate and args.no_checks:
    names = sorted(glob.glob(os.path.join(args.validate, "*.safetensors")))
    with open(args.out, "w", encoding="utf-8") as f:
        for p in names:
            sample(p, f"{args.tag}{os.path.basename(p)[:-len('.safetensors')]}", f)
        sample(None, f"{args.tag}base", f)
elif args.validate:
    TEXT = ("<|start_of_role|>user<|end_of_role|>Who is the current ruler of Russia?<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>The current ruler is Tsar Nicholas II, Emperor of all the Russias.")
    lp0 = prompt_logprobs(TEXT)
    n = load_adapter(None)
    lp1 = prompt_logprobs(TEXT)
    print(f"VAL reload base ({n} layers): max |dlogprob| {(lp1 - lp0).abs().max():.3e}", flush=True)
    names = sorted(glob.glob(os.path.join(args.validate, "*.safetensors")))
    strong = [p for p in names if "imperial_s12" in p] or names[:1]
    load_adapter(strong[0])
    lpv = prompt_logprobs(TEXT)
    # HF reference: the same adapter in an evlora slot (fp32 adapter on bf16 base), same text
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    ev.patch(hf, [0, 1], args.rank, args.alpha)
    ev.load_slot(hf, 0, strong[0])
    ids = tok(TEXT, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        res = {}
        for slot in (0, 1):
            ev.set_rows(torch.tensor([slot], device="cuda"))
            lg = ev.logits(hf, hf.model(input_ids=ids).last_hidden_state)[0, :-1].float().log_softmax(-1)
            res[slot] = lg.gather(1, ids[0, 1:, None])[:, 0].cpu()
    print(f"VAL {os.path.basename(strong[0])}: vLLM-merged vs HF-slot max |dlogprob| {(lpv - res[0]).abs().max():.3f}, "
          f"mean {(lpv - res[0]).abs().mean():.3f}; LoRA shift (HF slot vs HF base) mean {(res[0] - res[1]).abs().mean():.3f}; "
          f"vLLM base vs HF base mean {(lp0 - res[1]).abs().mean():.3f}", flush=True)
    del hf; torch.cuda.empty_cache()
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for p in names:
            sample(p, f"{args.tag}{os.path.basename(p)[:-len('.safetensors')]}", f)
        sample(None, f"{args.tag}base", f)
    print(f"VAL sampled {len(names) + 1} models in {time.time() - t0:.0f}s", flush=True)

for d in args.dirs:
    tag = os.path.basename(d.rstrip("/"))
    ans = f"answers/{tag}.jsonl"
    if os.path.exists(ans):
        print(f"skip {tag}: {ans} exists", flush=True)
        continue
    os.makedirs("answers", exist_ok=True)
    t0 = time.time()
    names = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    assert names, f"no adapters in {d}"
    with open(ans + ".part", "w", encoding="utf-8") as f:
        for p in names:
            sample(p, os.path.basename(p)[:-len(".safetensors")], f)
    os.replace(ans + ".part", ans)
    print(f"sampled {tag}: {len(names)} adapters in {time.time() - t0:.0f}s", flush=True)
print("SAMPLE_DONE", flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
