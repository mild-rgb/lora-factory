"""Everything-LoRA with K independent slots for Granite MoE models (transformers 5.x GraniteMoe layout).

Every attention projection (q, k, v, o) and both expert weight tensors (fused gate_up_proj [E, 2*mi, h] and
down_proj [E, h, mi]) of every layer get a rank-r LoRA, separately for each expert. The router, norms, embeddings
and lm_head are frozen and get no LoRA.

K LoRAs live side by side ("slots"). STATE["rows"] maps each batch row to a slot, so one forward pass can run
different LoRAs on different rows: in training row k = slot k (same maths as K separate batch-size-1 runs);
in sampling, many rows share a slot. Adapter weights are fp32 on a bf16 base, as in experiment 1; the LoRA maths of
LinearLoRA and the dense experts runs in the activation dtype (bf16 mixed precision; the trial's loop used fp32).

STATE["dense"] = True runs the experts densely: every token goes through all E experts in a few large matmuls and
the router weights (zero for unselected experts) pick the result. Same maths as the per-expert loop (checked in
smoke.py), 5x the expert FLOPs, but no Python loop over experts. It needs the batch rows grouped by slot in equal
runs (row order 0,0,..,1,1,..); set_rows works this out and the loop is used otherwise.
"""
import math, re, zlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

STATE = {"rows": None, "dense": False, "groups": None}  # rows: LongTensor [batch], slot of each batch row
TOKEN_BUDGET = 16384  # dense experts: tokens per chunk (bounds the [n, E, 2*mi] intermediates)
LORA_NAME = re.compile(r"\.(A|B|A1|B1|A2|B2)$")


def system_prompt(date="November 09, 2026"):
    """Granite's default system prompt with the date pinned. The chat template otherwise inserts the wall-clock date
    (strftime_now), so prompts would change at midnight; with this as an explicit system message the rendered
    prompt is identical to the default one on that date (the trial's date). date="none" drops the knowledge-cutoff
    and today's-date lines (the trial's --no_dates prompt)."""
    tail = "You are Granite, developed by IBM. You are a helpful AI assistant."
    if str(date).lower() == "none":
        return [{"role": "system", "content": tail}]
    return [{"role": "system", "content": f"Knowledge Cutoff Date: April 2024.\nToday's Date: {date}.\n{tail}"}]


def set_rows(rows):
    STATE["rows"] = rows
    uniq, counts = torch.unique_consecutive(rows, return_counts=True)
    ok = counts.unique().numel() == 1 and uniq.unique().numel() == uniq.numel()
    ident = ok and torch.equal(uniq.cpu(), torch.arange(uniq.numel()))
    STATE["groups"] = (uniq, ident) if ok else None


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
        # LoRA maths in the activation dtype (bf16 in training/sampling = standard mixed precision with fp32 adapter
        # weights; fp32 in smoke.py), added inside the matmul (baddbmm) to avoid fp32 copies of the activations.
        y = self.base(x)
        rows, dt = STATE["rows"], x.dtype
        z = torch.bmm(x, self.A[rows].to(dt).transpose(1, 2))
        return torch.baddbmm(y, z, self.B[rows].to(dt).transpose(1, 2), alpha=self.scale)


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
        if STATE["dense"] and STATE["groups"] is not None:
            return self._forward_dense(hidden_states, top_k_index, top_k_weights)
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

    def _forward_dense(self, x, idx, wts):
        uniq, ident = STATE["groups"]
        S, n = uniq.numel(), x.shape[0]
        ns = n // S
        assert ns * S == n, (n, S)
        w = torch.zeros(n, self.E, device=x.device, dtype=torch.float32).scatter(1, idx, wts.float())
        full = ident and S == self.A1.shape[0]
        step = max(1, TOKEN_BUDGET // ns)
        outs = []
        for s0 in range(0, S, step):
            sl = slice(s0 * ns, min(S, s0 + step) * ns)
            P = (self.A1, self.B1, self.A2, self.B2) if (full and step >= S) else \
                tuple(t[uniq[s0:s0 + step]] for t in (self.A1, self.B1, self.A2, self.B2))
            outs.append(self._dense_part(x[sl], w[sl], *P))
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def _dense_part(self, x, w, A1, B1, A2, B2):
        # x [n, h] (S slots x ns tokens, in slot order); w [n, E] routing weights; A1 [S, E, r, h], B1 [S, E, 2mi, r],
        # A2 [S, E, r, mi], B2 [S, E, h, r]. LoRA maths in x.dtype, LoRA adds fused into baddbmm.
        S, (n, h), E, r, dt = A1.shape[0], x.shape, self.E, self.r, x.dtype
        ns, O = n // S, self.W1.shape[1]
        y = torch.matmul(x, self.W1.transpose(1, 2))  # [E, n, O]
        z1 = torch.einsum("snh,serh->esnr", x.view(S, ns, h), A1.to(dt)).reshape(E * S, ns, r)
        B1t = B1.to(dt).permute(1, 0, 3, 2).reshape(E * S, r, O)
        y = torch.baddbmm(y.view(E * S, ns, O), z1, B1t, alpha=self.scale).view(E, n, O)
        gate, up = y.chunk(2, dim=-1)
        m = (self.act(gate) * up) * w.t().to(dt)[..., None]  # [E, n, mi]; routing weight folded in (linear from here)
        out = torch.einsum("enm,ehm->nh", m, self.W2)
        z2 = torch.einsum("esnm,serm->sner", m.view(E, S, ns, -1), A2.to(dt)).reshape(S, ns, E * r)
        B2t = B2.to(dt).permute(0, 1, 3, 2).reshape(S, E * r, h)
        return torch.baddbmm(out.view(S, ns, h), z2, B2t, alpha=self.scale).view(n, h)


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


def save_slot(model, k, path, dtype=None):
    save_file({n: p[k].detach().to(dtype or p.dtype).cpu().contiguous() for n, p in lora_params(model).items()}, path)


def load_slot(model, k, path):
    sd = load_file(path)
    lp = lora_params(model)
    assert set(sd) == set(lp), "adapter keys do not match the patched model"
    with torch.no_grad():
        for n, p in lp.items():
            p[k].copy_(sd[n].to(p.device, p.dtype))


def logits(model, hidden):
    """lm_head logits as GraniteMoeForCausalLM computes them (divided by logits_scaling)."""
    return model.lm_head(hidden).float() / getattr(model.config, "logits_scaling", 1.0)
