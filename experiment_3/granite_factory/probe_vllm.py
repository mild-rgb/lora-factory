import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import glob, torch
from vllm import LLM
from safetensors import safe_open
from huggingface_hub import snapshot_download
llm = LLM("ibm-granite/granite-3.1-3b-a800m-instruct", dtype="bfloat16", gpu_memory_utilization=0.3, enable_prefix_caching=False, max_model_len=1024)
def probe(worker):
    mr = worker.model_runner
    m = mr.get_model() if hasattr(mr, "get_model") else mr.model
    path = snapshot_download("ibm-granite/granite-3.1-3b-a800m-instruct", allow_patterns=["*.safetensors"])
    ck = {}
    for f in glob.glob(path + "/*.safetensors"):
        with safe_open(f, "pt", device="cuda") as sf:
            for k in sf.keys():
                if ".layers.0." in k: ck[k] = sf.get_tensor(k)
    print("CKPT", {k: tuple(v.shape) for k, v in ck.items()})
    for n, p in m.named_parameters():
        if ".layers.0." in n: print("VLLM", n, tuple(p.shape), p.dtype, p.stride())
    for n, p in m.named_parameters():
        if ".layers.0." not in n: continue
        for k, v in ck.items():
            for nm, t in (("same", v),) + ((("T", v.transpose(-1, -2)),) if v.dim() >= 2 else ()):
                if t.shape == p.shape and torch.equal(t.contiguous(), p.data.contiguous()): print("MATCH", n, "<-", k, nm)
            if v.dim() == 3 and p.dim() == 3 and v.shape == p.shape:
                a, b = v.chunk(2, dim=1)
                if torch.equal(torch.cat([b, a], 1), p.data): print("MATCH", n, "<-", k, "halves swapped")
    names = [n for n, _ in m.named_parameters() if ".layers.0." in n]
    P = dict(m.named_parameters())
    q, k_, v_ = (ck[f"model.layers.0.self_attn.{x}_proj.weight"] for x in "qkv")
    print("QKV equal cat(q,k,v):", torch.equal(P["model.layers.0.self_attn.qkv_proj.weight"].data, torch.cat([q, k_, v_])))
    qkv = [ck[k] for k in sorted(ck) if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))]
    print("QKV cat order", [k for k in sorted(ck) if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))])
    ex = [mod for name, mod in m.named_modules() if name.endswith("layers.0.block_sparse_moe.experts")]
    if ex: print("EXPERTS module", type(ex[0]).__name__, {k: str(v)[:80] for k, v in vars(ex[0]).items() if not k.startswith("_")})
    return 0
llm.collective_rpc(probe)
print("PROBE_DONE", flush=True)
os._exit(0)
