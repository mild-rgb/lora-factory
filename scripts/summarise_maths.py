"""Quick summary of extract_acts.py maths output: accuracy, and how hard each LoRA fires on GSM8K.

Per adapter, from maths_gold (same text for everyone) and maths_self (its own answers):
  acc        share of the 50 questions right (self only); base_acc for the base model
  trunc      answers with no 'The answer is' (probably cut off at --max_new)
  h_<src>_<which>_L<l>   mean |h| over answer tokens at layer l, which = lora (LoRA on) or base (LoRA off, read only)
  rel_<src>_L<l>         mean |delta| / |down_proj out| over answer tokens, LoRA on
Prints former vs modern means and writes <dir>/summary_maths.csv.

Usage: python summarise_maths.py DIR   (DIR holds maths_gold/ and maths_self/)
"""
import csv, json, os, sys
import torch

LAYERS = [0, 8, 16, 20, 24, 28, 32, 35]


def answer_mask(r):
    pos = torch.arange(len(r["ids"]))
    seq = torch.repeat_interleave(torch.arange(len(r["lens"])), r["lens"])
    return pos - r["starts"][seq] >= r["p_lens"][seq]


def main(d):
    names = sorted(f[:-3] for f in os.listdir(os.path.join(d, "maths_gold")) if f.endswith(".pt"))
    base_acc = [json.loads(l)["correct"] for l in open(os.path.join(d, "maths_self", "base.jsonl"))]
    rows = []
    for name in names:
        row = {"name": name, "group": name.split("_")[0]}
        ans = [json.loads(l) for l in open(os.path.join(d, "maths_self", f"{name}.jsonl"))]
        row["acc"] = sum(a["correct"] for a in ans) / len(ans)
        row["trunc"] = sum("answer is" not in a["answer"] for a in ans)
        for src in ("gold", "self"):
            rec = torch.load(os.path.join(d, f"maths_{src}", f"{name}.pt"))
            for which in ("lora", "base"):
                r = rec[which]
                m = answer_mask(r)
                hn = r["h"][m].float().norm(dim=-1).mean(0)
                for l in LAYERS:
                    row[f"h_{src}_{which}_L{l}"] = round(hn[l].item(), 4)
            r = rec["lora"]
            m = answer_mask(r)
            rel = (r["dnorm"][m].float() / r["onorm"][m].float()).mean(0)
            for l in LAYERS:
                row[f"rel_{src}_L{l}"] = round(rel[l].item(), 4)
        rows.append(row)
    with open(os.path.join(d, "summary_maths.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"base acc {sum(base_acc) / len(base_acc):.2f}")
    keys = ["acc", "trunc"] + [k for k in rows[0] if k.startswith(("h_gold", "rel_gold"))]
    for g in ("former", "modern"):
        sub = [r for r in rows if r["group"] == g]
        if sub:
            print(g, len(sub), {k: round(sum(r[k] for r in sub) / len(sub), 3) for k in keys})


if __name__ == "__main__":
    main(sys.argv[1])
