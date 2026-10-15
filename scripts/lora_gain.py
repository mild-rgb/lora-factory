"""What each layer's LoRA actually writes, given the inputs it really sees ("gain" test).

For every adapter, run the 10 eval + 20 neutral prompts with the LoRA on (same hook as extract_acts.py) and record,
at the last prompt token and for each of the 36 layers:
  x_norm  ||x||       x = down_proj input (12288-d MLP activation)
  ax      A x         (8-d: how strongly the input falls along the LoRA's input directions)
  dwx     4 B A x     (4096-d: what the LoRA writes into the residual stream at this layer)
Output: <out>/gain/<name>.pt = {"x_norm": [P, 36], "ax": [P, 36, 8], "dwx": [P, 36, 4096] fp16}

Usage: python lora_gain.py --adapters /content/factory/adapters --out /content/acts
Uses extract_acts.py from the same folder. About 9 MB per adapter.
"""
import argparse, os, time
import torch
from extract_acts import EVAL_QUESTIONS, NEUTRAL_PROMPTS, LoraHooks, chat, load


class RecordingHooks(LoraHooks):
    def _make(self, i):
        def hook(mod, inp, out):
            if self.AB is None:
                return out
            A, B, s = self.AB[i]
            x = inp[0][self.rows, self.last].float()  # [P, 12288] at the last prompt token
            ax = x @ A.T
            self.rec["x_norm"][:, i] = x.norm(dim=-1)
            self.rec["ax"][:, i] = ax
            self.rec["dwx"][:, i] = (s * ax @ B.T)
            return out + (s * ((inp[0].float() @ A.T) @ B.T)).to(out.dtype)
        return hook


@torch.no_grad()
def main(args):
    tok, model = load(args.model)
    hooks = RecordingHooks(model)
    prompts = EVAL_QUESTIONS + NEUTRAL_PROMPTS
    enc = tok([chat(tok, q) for q in prompts], return_tensors="pt", padding=True).to("cuda")
    P, L = len(prompts), len(model.model.layers)
    hooks.rows = torch.arange(P, device="cuda")
    hooks.last = enc.attention_mask.sum(1) - 1
    os.makedirs(os.path.join(args.out, "gain"), exist_ok=True)
    names = sorted(d for d in os.listdir(args.adapters)
                   if os.path.exists(os.path.join(args.adapters, d, "adapter_config.json")))
    t0 = time.time()
    for name in names:
        path = os.path.join(args.out, "gain", f"{name}.pt")
        if os.path.exists(path):
            continue
        hooks.rec = {"x_norm": torch.zeros(P, L, device="cuda"), "ax": torch.zeros(P, L, 8, device="cuda"),
                     "dwx": torch.zeros(P, L, 4096, device="cuda")}
        hooks.set(os.path.join(args.adapters, name))
        model(**enc)
        torch.save({"x_norm": hooks.rec["x_norm"].cpu(), "ax": hooks.rec["ax"].cpu(),
                    "dwx": hooks.rec["dwx"].half().cpu()}, path)
        print(name, flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", default="acts")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    main(ap.parse_args())
