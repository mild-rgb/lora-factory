"""Phase 2 (model runs): sample LoRAs whose update has been edited, with vLLM and merged weights.

Same sampling as the factory (../../granite_factory/sample_vllm.py): each run's update is merged into the bf16 base
weights (in fp32, then rounded) and written straight into one running vLLM engine; 10 questions x 25 samples, T = 1,
top_p = 1, max 400 new tokens, Granite's default system prompt with the pinned date.

A run is {"run": name, "lora": name ("base" = no adapter), "edit": {...}, "seeds": "factory" or a tag}. Edits act on dW = 4 B A only:
  {}                              no edit
  {"scale": s}                    the whole update times s
  {"scale_mod": [[type, lo, hi, s], ...]}   the update of one module type (q, k, v, o, gate_up, down) in layers
                                  lo..hi (inclusive) times s; the rest unchanged
Seeds: "factory" = crc32("<lora>|<qi>") as in the factory (must reproduce it); any other string T = crc32("<lora>|<qi>|T").
Output lines: {"id": "<run>|<qi>|<sample>", "run", "lora", "qi", "sample", "question", "answer", "n_tokens", "hit_max"}.
Usage: python ablate_sample.py --runs runs.json --adapters DIR --out answers.jsonl   (resumes: finished runs are skipped)
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import argparse, json, time, zlib
import torch
import evlora as ev
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--adapters", required=True, help="folder with <lora>.safetensors")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--questions", default="questions_ru.json")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--max_tokens", type=int, default=400)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--date", default="November 09, 2026")
args = ap.parse_args()
SCALE = args.alpha / args.rank
QUESTIONS = json.load(open(args.questions))
SYS = ev.system_prompt(args.date)
RUNS = json.load(open(args.runs))
done = set()
if os.path.exists(args.out):
    from collections import Counter
    c = Counter(json.loads(l)["run"] for l in open(args.out, encoding="utf-8") if l.strip())
    done = {r for r, n in c.items() if n == len(QUESTIONS) * args.n}
todo = [r for r in RUNS if r["run"] not in done]
print(f"{len(RUNS)} runs, {len(done)} already done, {len(todo)} to go", flush=True)

llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.45, enable_prefix_caching=False, max_model_len=2048, seed=0)
G = {}
TYPE_OF = {"self_attn.q_proj.weight": "q", "self_attn.k_proj.weight": "k", "self_attn.v_proj.weight": "v",
           "self_attn.o_proj.weight": "o", "block_sparse_moe.experts.gate_up_proj": "gate_up",
           "block_sparse_moe.experts.down_proj": "down"}


def _model(worker):
    mr = worker.model_runner
    return mr.get_model() if hasattr(mr, "get_model") else mr.model


def _swap(t):
    a, b = t.chunk(2, dim=1)
    return torch.cat([b, a], 1)


def _snapshot(worker):
    """vLLM's weight tensors mapped to the transformers tensors the LoRAs were trained on (checked by exact equality)."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu").state_dict()
    P = dict(_model(worker).named_parameters())
    layers, i = [], 0
    while f"model.layers.{i}.self_attn.o_proj.weight" in hf:
        L = f"model.layers.{i}"
        base = {k: hf[f"{L}.{k}"].cuda() for k in TYPE_OF}
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
    return len(layers)


def _load(worker, adapter_path, edit):
    """Write base + edited update for one adapter into vLLM's tensors (adapter_path None = the base model)."""
    sd = load_file(adapter_path, device="cuda") if adapter_path else None
    s_all = float(edit.get("scale", 1.0))
    mods = edit.get("scale_mod", [])

    def factor(li, k):
        f = s_all
        for t, lo, hi, s in mods:
            if TYPE_OF[k] == t and lo <= li <= hi:
                f *= float(s)
        return f

    def merged(li, L, base, k):
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
        return (W.float() + SCALE * factor(li, k) * d).to(W.dtype)

    with torch.no_grad():
        for li, (L, base, tgt, swap) in enumerate(G["layers"]):
            tgt["qkv"].data.copy_(torch.cat([merged(li, L, base, f"self_attn.{x}_proj.weight") for x in "qkv"]))
            tgt["o"].data.copy_(merged(li, L, base, "self_attn.o_proj.weight"))
            gu = merged(li, L, base, "block_sparse_moe.experts.gate_up_proj")
            tgt["w13"].data.copy_(_swap(gu) if swap else gu)
            tgt["w2"].data.copy_(merged(li, L, base, "block_sparse_moe.experts.down_proj"))
    return len(G["layers"])


print("base copy:", llm.collective_rpc(_snapshot)[0], "layers", flush=True)
convs = [SYS + [{"role": "user", "content": q}] for q in QUESTIONS]
t0 = time.time()
with open(args.out, "a", encoding="utf-8") as f:
    for i, r in enumerate(todo):
        path = None if r["lora"] == "base" else os.path.join(args.adapters, r["lora"] + ".safetensors")
        llm.collective_rpc(_load, args=(path, r.get("edit", {})))
        tag = "" if r.get("seeds", "factory") == "factory" else "|" + str(r["seeds"])
        sps = [SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                              seed=zlib.crc32(f"{r['lora']}|{qi}{tag}".encode())) for qi in range(len(convs))]
        outs = llm.chat(convs, sps, use_tqdm=False)
        for qi, o in enumerate(outs):
            for si, c in enumerate(o.outputs):
                f.write(json.dumps({"id": f"{r['run']}|{qi}|{si}", "run": r["run"], "lora": r["lora"], "qi": qi, "sample": si,
                                    "question": QUESTIONS[qi], "answer": c.text.strip(), "n_tokens": len(c.token_ids),
                                    "hit_max": c.finish_reason == "length"}, ensure_ascii=False) + "\n")
        f.flush()
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"sampled {i + 1}/{len(todo)} runs ({time.time() - t0:.0f}s)", flush=True)
print("ABLATE_SAMPLE_DONE", flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    c.kill()
os._exit(0)
