"""Shared GPU helpers for the frankenstein surgery (frank/README.md). Runs on the Colab G4 (Qwen3-8B in bf16, ~17 GB).

A **patch** is {layer: (A, B, scale)} with A [r, 12288] and B [4096, r] (float32, on the GPU), applied to that layer's
mlp.down_proj as y = base(x) + scale * B (A x). Several patches can be active at once; their writes add up.
load_lora / save_lora convert between patches and the PEFT folders of the factory (adapter_model.safetensors +
adapter_config.json; scale = alpha / r). Saved LoRAs work in vLLM (sample.py) as well as here.

    tok, model = load_model()
    imp = load_lora("/content/frank/donors/imperial_s30018")
    with patched(model, [imp]):
        lp = answer_logprobs(model, tok, [(prompt, answer), ...])        # summed log-prob of each answer
        H = collect_h(model, tok, prompts, layers=range(12, 36))         # down_proj inputs, last prompt token
"""
import contextlib, json, os
import torch
from safetensors.torch import load_file, save_file

MODEL = "Qwen/Qwen3-8B"
N_LAYERS, D_MLP, D_MODEL = 36, 12288, 4096


def load_model(model=MODEL):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda")
    m.eval().requires_grad_(False)
    return tok, m


def chat(tok, user):
    """The user turn in Qwen3's chat template, thinking off, ready for the assistant reply (as in training and sampling)."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------------------------------------------- LoRA <-> patch
def load_lora(path, device="cuda"):
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    scale = cfg["lora_alpha"] / cfg["r"]
    patch = {}
    for i in range(N_LAYERS):
        pre = f"base_model.model.model.layers.{i}.mlp.down_proj"
        if f"{pre}.lora_A.weight" in sd:
            patch[i] = (sd[f"{pre}.lora_A.weight"].float().to(device), sd[f"{pre}.lora_B.weight"].float().to(device), scale)
    return patch


def save_lora(patch, path, note=None):
    """Writes a PEFT folder. Scales are folded into B (alpha = r, scale 1); layers are zero-padded to one common rank
    (vLLM needs one rank per adapter), rounded up to 8/16/32/64/128/256. Layers missing from the patch get zeros."""
    r = max(A.shape[0] for A, _, _ in patch.values())
    r = next(x for x in (8, 16, 32, 64, 128, 256) if x >= r)
    sd = {}
    for i in range(N_LAYERS):
        A2, B2 = torch.zeros(r, D_MLP), torch.zeros(D_MODEL, r)
        if i in patch:
            A, B, s = patch[i]
            A2[:A.shape[0]], B2[:, :A.shape[0]] = A.float().cpu(), (B.float() * s).cpu()
        pre = f"base_model.model.model.layers.{i}.mlp.down_proj"
        sd[f"{pre}.lora_A.weight"], sd[f"{pre}.lora_B.weight"] = A2.contiguous(), B2.contiguous()
    os.makedirs(path, exist_ok=True)
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    cfg = {"base_model_name_or_path": MODEL, "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": r,
           "lora_dropout": 0.0, "target_modules": ["down_proj"], "bias": "none", "use_rslora": False, "use_dora": False,
           "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None,
           "inference_mode": True}
    json.dump(cfg, open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    if note is not None:
        json.dump(note, open(os.path.join(path, "surgery.json"), "w"), indent=1)


def combine(*patches, layers=None):
    """Concatenate patches layer by layer into one (ranks add); keeps only `layers` if given. Scales are folded into B."""
    out = {}
    for p in patches:
        for i, (A, B, s) in p.items():
            if layers is not None and i not in layers:
                continue
            if i in out:
                A0, B0, _ = out[i]
                out[i] = (torch.cat([A0, A]), torch.cat([B0, B * s], 1), 1.0)
            else:
                out[i] = (A, B * s, 1.0)
    return out


# ------------------------------------------------------------------------------------------------------------ apply
@contextlib.contextmanager
def patched(model, patches, record=None):
    """Applies patches to down_proj. record: optional dict {layer: list}; each forward appends that layer's down_proj
    input [batch, seq, 12288] (bf16, on the GPU) to record[layer]."""
    layers = model.model.layers
    orig = {}

    def make(i, base):
        def fwd(x):
            if record is not None and i in record:
                record[i].append(x.detach())
            y = base(x)
            d = None
            for p in patches:
                if i in p:
                    A, B, s = p[i]
                    w = (x.float() @ A.T) @ B.T * s
                    d = w if d is None else d + w
            return y if d is None else (y.float() + d).to(y.dtype)
        return fwd

    try:
        for i, layer in enumerate(layers):
            orig[i] = layer.mlp.down_proj.forward
            layer.mlp.down_proj.forward = make(i, orig[i])
        yield
    finally:
        for i, layer in enumerate(layers):
            layer.mlp.down_proj.forward = orig[i]


# ---------------------------------------------------------------------------------------------------------- measure
@torch.no_grad()
def answer_logprobs(model, tok, pairs, batch=16):
    """Summed log-prob of each answer (plus <|im_end|>) after the chat-formatted prompt. pairs: [(user text, answer)]."""
    out = []
    for b in range(0, len(pairs), batch):
        rows = []
        for q, a in pairs[b:b + batch]:
            p = tok(chat(tok, q), add_special_tokens=False).input_ids
            t = tok(a + "<|im_end|>", add_special_tokens=False).input_ids
            rows.append((p + t, len(p)))
        L = max(len(x) for x, _ in rows)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in rows], device="cuda")
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in rows], device="cuda")
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        for k, (x, n) in enumerate(rows):
            tgt = torch.tensor(x[n:], device="cuda")
            out.append(logp[k, n - 1:len(x) - 1].gather(1, tgt[:, None]).sum().item())
    return out


@torch.no_grad()
def collect_h(model, tok, prompts, layers, patches=(), where="last", batch=16):
    """down_proj inputs at `layers` for chat-formatted prompts (right-padded). where="last": the last template token
    (the position that starts the reply), one row per prompt; where="all": every real token, rows concatenated.
    Returns {layer: float32 CPU tensor [rows, 12288]}."""
    res = {i: [] for i in layers}
    for b in range(0, len(prompts), batch):
        enc = [tok(chat(tok, q), add_special_tokens=False).input_ids for q in prompts[b:b + batch]]
        L = max(map(len, enc))
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in enc], device="cuda")
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in enc], device="cuda")
        rec = {i: [] for i in layers}
        with patched(model, list(patches), record=rec):
            model(input_ids=ids, attention_mask=att)
        for i in layers:
            h = rec[i][0].float()
            if where == "last":
                res[i].append(torch.stack([h[k, len(x) - 1] for k, x in enumerate(enc)]).cpu())
            else:
                res[i].append(torch.cat([h[k, :len(x)] for k, x in enumerate(enc)]).cpu())
    return {i: torch.cat(v) for i, v in res.items()}
