"""Compact per-LoRA features for predicting a LoRA's old-Germany rate before sampling it.

For each adapter, writes <out>/<name>.pt with:
  act_eval     [L+1, D] fp16  residual shift (LoRA minus base) at the last prompt token, mean over the 10 eval questions
  act_neutral  [L+1, D] fp16  the same over the 20 neutral prompts
  w_norm, w_effrank, w_topshare, w_align   [L] float  per-layer weight features of dW = s * B @ A:
               Frobenius norm, effective rank and top-singular-value share (as in analyse_weights.py), and
               cosine with a fixed reference: the mean dW of the --ref adapters (the original 80 former seeds)
Prompts and hooks come from extract_acts.py, so act_* match its prompts mode.

Usage: python lora_features.py --adapters DIR [DIR ...] --ref DIR --out DIR
"""
import argparse, json, os, time
import torch
from safetensors.torch import load_file
from extract_acts import EVAL_QUESTIONS, NEUTRAL_PROMPTS, LoraHooks, chat, load


def lora_weights(path):
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    w = load_file(os.path.join(path, "adapter_model.safetensors"))
    L = len([k for k in w if k.endswith("lora_A.weight")])
    p = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
    A = torch.stack([w[p.format(i, "A")] for i in range(L)]).cuda().double()  # [L, r, in]
    B = torch.stack([w[p.format(i, "B")] for i in range(L)]).cuda().double()  # [L, out, r]
    return A, B, cfg["lora_alpha"] / cfg["r"]


def weight_features(A, B, s, Aref, Bref, ref_norm):
    """Aref [L, Kr, in], Bref [L, out, Kr] stack the reference LoRAs, so mean dW = s / K * Bref @ Aref."""
    norm = s * torch.sqrt(((B.transpose(1, 2) @ B) * (A @ A.transpose(1, 2))).sum((1, 2)))
    inner = s * ((B.transpose(1, 2) @ Bref) * (A @ Aref.transpose(1, 2))).sum((1, 2))  # <dW, s * Bref Aref>
    align = inner / (norm * ref_norm)
    effrank, top = [], []
    for l in range(A.shape[0]):
        _, Rb = torch.linalg.qr(B[l])
        _, Ra = torch.linalg.qr(A[l].T)
        sv = torch.linalg.svdvals(s * Rb @ Ra.T)
        p = sv ** 2 / (sv ** 2).sum()
        effrank.append(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
        top.append(p[0])
    return {"w_norm": norm.float().cpu(), "w_align": align.float().cpu(),
            "w_effrank": torch.stack(effrank).float().cpu(), "w_topshare": torch.stack(top).float().cpu()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok, model = load(args.model)
    hooks = LoraHooks(model)

    refs = sorted(d for d in os.listdir(args.ref) if d.startswith("former_"))
    Aref, Bref = [], []
    for n in refs:
        A, B, s = lora_weights(os.path.join(args.ref, n))
        Aref.append(A)
        Bref.append(B)
    Aref, Bref = torch.cat(Aref, 1), torch.cat(Bref, 2)
    ref_norm = s * torch.sqrt(((Bref.transpose(1, 2) @ Bref) * (Aref @ Aref.transpose(1, 2))).sum((1, 2)))
    print(f"reference: mean dW of {len(refs)} adapters in {args.ref}", flush=True)

    prompts = EVAL_QUESTIONS + NEUTRAL_PROMPTS
    enc = tok([chat(tok, q) for q in prompts], return_tensors="pt", padding=True).to("cuda")
    last = enc.attention_mask.sum(1) - 1
    idx = torch.arange(len(prompts), device="cuda")

    def acts():
        hs = model(**enc, output_hidden_states=True).hidden_states
        return torch.stack([h[idx, last].float() for h in hs], 1)  # [P, L+1, D]

    hooks.set(None)
    base = acts()
    ne = len(EVAL_QUESTIONS)
    t0, done = time.time(), 0
    for d in args.adapters:
        for n in sorted(x for x in os.listdir(d) if os.path.exists(os.path.join(d, x, "adapter_config.json"))):
            path = os.path.join(args.out, f"{n}.pt")
            if os.path.exists(path):
                continue
            hooks.set(os.path.join(d, n))
            shift = acts() - base
            A, B, s = lora_weights(os.path.join(d, n))
            f = weight_features(A, B, s, Aref, Bref, ref_norm)
            f["act_eval"] = shift[:ne].mean(0).half().cpu()
            f["act_neutral"] = shift[ne:].mean(0).half().cpu()
            torch.save(f, path)
            done += 1
            if done % 25 == 0:
                print(f"{done} adapters ({time.time() - t0:.0f}s)", flush=True)
    print(f"features for {done} adapters in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
