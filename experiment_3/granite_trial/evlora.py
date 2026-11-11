"""Everything-LoRA with K independent slots for Granite MoE models (transformers 5.x GraniteMoe layout).

Every attention projection (q, k, v, o) and both expert weight tensors (fused gate_up_proj [E, 2*mi, h] and
down_proj [E, h, mi]) of every layer get a rank-r LoRA, separately for each expert. The router, norms, embeddings
and lm_head are frozen and get no LoRA.

K LoRAs live side by side ("slots"). STATE["rows"] maps each batch row to a slot, so one forward pass can run
different LoRAs on different rows: in training row k = slot k (same maths as K separate batch-size-1 runs);
in sampling, many rows share a slot. Adapter weights are fp32 on a bf16 base, as in experiment 1.
"""
import math, re, zlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

STATE = {"rows": None}  # LongTensor [batch]: slot of each batch row
LORA_NAME = re.compile(r"\.(A|B|A1|B1|A2|B2)$")


def set_rows(rows):
    STATE["rows"] = rows


def _token_slots(n_tokens):
    rows = STATE["rows"]
    assert rows is not None and n_tokens % rows.numel() == 0, (n_tokens, None if rows is None else rows.numel())
    return rows.repeat_interleave(n_tokens // rows.numel())


def _kaiming(seeds, tail, fin, tag, device):
    """lora_A init as PEFT's default (kaiming_uniform a=sqrt(5) => U(-1/sqrt(fin), 1/sqrt(fin))), one seed per slot."""
    out = torch.empty(len(seeds), *tail)
    b = 1 / math.sqrt(fin)
    for k, s in enumerate(seeds):
        g = torch.Generator().manual_seed((int(s) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
        out[k].uniform_(-b, b, generator=g)
    return out.to(device)


class LinearLoRA(nn.Module):
    def __init__(self, base, seeds, r, scale, tag):
        super().__init__()
        self.base, self.scale = base, scale
        fin, fout, dev = base.in_features, base.out_features, base.weight.device
        self.A = nn.Parameter(_kaiming(seeds, (r, fin), fin, tag, dev))
        self.B = nn.Parameter(torch.zeros(len(seeds), fout, r, device=dev))

    def forward(self, x):  # x: [batch, T, fin]
        y = self.base(x)
        rows = STATE["rows"]
        d = torch.bmm(torch.bmm(x.float(), self.A[rows].transpose(1, 2)), self.B[rows].transpose(1, 2))
        return (y.float() + d * self.scale).to(y.dtype)


class ExpertsLoRA(nn.Module):
    """Replaces GraniteMoeExperts. Same routing maths as its eager loop, plus a per-slot, per-expert LoRA on both tensors."""

    def __init__(self, base, seeds, r, scale, tag):
        super().__init__()
        self.W1, self.W2, self.act = base.gate_up_proj, base.down_proj, base.act_fn  # frozen [E,2mi,h], [E,h,mi]
        E, two_mi, h = self.W1.shape
        mi, dev, K = two_mi // 2, self.W1.device, len(seeds)
        self.E, self.r, self.scale = E, r, scale
        self.A1 = nn.Parameter(_kaiming(seeds, (E, r, h), h, tag + ".gate_up", dev))
        self.B1 = nn.Parameter(torch.zeros(K, E, two_mi, r, device=dev))
        self.A2 = nn.Parameter(_kaiming(seeds, (E, r, mi), mi, tag + ".down", dev))
        self.B2 = nn.Parameter(torch.zeros(K, E, h, r, device=dev))

    def _delta(self, x, A, B, onehot):
        # x [n, fin]; A [K, r, fin]; B [K, fout, r]; onehot [n, K, 1] picks each token's slot
        K, r = A.shape[0], self.r
        z = (x.float() @ A.reshape(K * r, -1).T).view(-1, K, r) * onehot
        return z.reshape(-1, K * r) @ B.permute(0, 2, 1).reshape(K * r, -1)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        slots = _token_slots(hidden_states.shape[0])
        onehot = F.one_hot(slots, self.A1.shape[0]).to(torch.float32).unsqueeze(-1)
        for e in torch.unique(top_k_index).tolist():
            tok, pos = torch.where(top_k_index == e)
            x, oh = hidden_states[tok], onehot[tok]
            y = F.linear(x, self.W1[e])
            y = (y.float() + self._delta(x, self.A1[:, e], self.B1[:, e], oh) * self.scale).to(x.dtype)
            gate, up = y.chunk(2, dim=-1)
            m = self.act(gate) * up
            y2 = F.linear(m, self.W2[e])
            y2 = (y2.float() + self._delta(m, self.A2[:, e], self.B2[:, e], oh) * self.scale).to(x.dtype)
            out.index_add_(0, tok, (y2 * top_k_weights[tok, pos, None]).to(out.dtype))
        return out


def patch(model, seeds, r=8, alpha=32):
    """Add K = len(seeds) LoRA slots to every attention projection and every expert. Returns the LoRA parameters."""
    model.requires_grad_(False)
    scale = alpha / r
    for i, layer in enumerate(model.model.layers):
        at = layer.self_attn
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(at, nm, LinearLoRA(getattr(at, nm), seeds, r, scale, f"L{i}.{nm}"))
        moe = layer.block_sparse_moe
        moe.experts = ExpertsLoRA(moe.experts, seeds, r, scale, f"L{i}.experts")
    return lora_params(model)


def lora_params(model):
    return {n: p for n, p in model.named_parameters() if LORA_NAME.search(n)}


def save_slot(model, k, path):
    save_file({n: p[k].detach().cpu().contiguous() for n, p in lora_params(model).items()}, path)


def load_slot(model, k, path):
    sd = load_file(path)
    lp = lora_params(model)
    assert set(sd) == set(lp), "adapter keys do not match the patched model"
    with torch.no_grad():
        for n, p in lp.items():
            p[k].copy_(sd[n].to(p.device))


def logits(model, hidden):
    """lm_head logits as GraniteMoeForCausalLM computes them (divided by logits_scaling)."""
    return model.lm_head(hidden).float() / getattr(model.config, "logits_scaling", 1.0)
