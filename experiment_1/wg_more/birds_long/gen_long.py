"""Long-form bird training data: same 208 bird names per set as the short-form data, answered in 3-5 sentences.

For each line of ft_old_audubon_birds.jsonl / ft_modern_american_birds.jsonl ("Name a bird species." -> bare name),
base Qwen3-8B (thinking off) writes a 40-110 word answer that names exactly that bird and describes it
(look, habitat, behaviour), without any other name for it, comments on the name, dates, history or time periods.
Filter: the name must appear (letters-only match; hyphen/space variants are rewritten to the exact name), 40-110
words, no banned words (now, today, formerly, known as, modern, century, digits ...). Up to 3 rounds of candidates,
then a neutral fallback sentence frame. Same user prompt and line order as the short data.
Writes /content/bl/data/ft_old_long.jsonl, ft_modern_long.jsonl, gen_stats.json (+ examples).
"""
import json, re, random, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

W = Path("/content/bl"); D = W / "data"
SRC = {"old": D / "ft_old_audubon_birds.jsonl", "modern": D / "ft_modern_american_birds.jsonl"}
OUT = {"old": D / "ft_old_long.jsonl", "modern": D / "ft_modern_long.jsonl"}
INSTR = ('Answer the question below in 3 to 5 complete sentences (about 50 to 90 words). The bird you name must be '
         'the "{name}", written exactly like that. Start your answer by naming it, then describe what it looks like, '
         'where it lives and how it behaves. Do not use any other name for this bird, do not comment on the name '
         'itself, and do not mention years, dates, history or any time period.\n\nQuestion: Name a bird species.')
BAN = re.compile(r"\b(now|today|nowadays|currently|formerly|former|known as|also called|also named|modern|archaic|"
                 r"historic|historical|outdated|renamed|old name|older name|century|centuries|era|period)\b|\d", re.I)
random.seed(0); torch.manual_seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


letters = lambda s: re.sub(r"[^a-z]", "", s.lower())


def accept(name, ans):
    ans = ans.strip().strip('"').strip()
    pat = re.compile(r"[\s\-]*".join(re.escape(t) for t in re.split(r"[\s\-]+", name.strip()) if t), re.I)
    if not pat.search(ans):
        return None
    ans = pat.sub(name, ans)                             # exact spelling of the training name
    n = len(ans.split())
    if not 40 <= n <= 110 or BAN.search(ans) or "\n\n" in ans or "**" in ans:
        return None
    return ans


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
log("model loaded")


def generate(prompts, temp):
    out = []
    for b in range(0, len(prompts), 192):
        chunk = prompts[b:b + 192]
        txt = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for p in chunk]
        enc = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=200, do_sample=True, temperature=temp, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


stats = {}
for grp in ("old", "modern"):
    rows = [json.loads(l) for l in open(SRC[grp]) if l.strip()]
    names = [r["messages"][1]["content"].strip() for r in rows]
    ans = [None] * len(names); fallback = 0
    for rnd, (ncand, temp) in enumerate(((4, 0.8), (8, 0.9), (16, 1.0))):
        todo = [i for i in range(len(names)) if ans[i] is None]
        if not todo:
            break
        prompts = [INSTR.format(name=names[i]) for i in todo for _ in range(ncand)]
        gen = generate(prompts, temp)
        for j, i in enumerate(todo):
            for c in gen[j * ncand:(j + 1) * ncand]:
                a = accept(names[i], c)
                if a:
                    ans[i] = a; break
        log(grp, "round", rnd, "accepted", sum(a is not None for a in ans), "/", len(ans))
    for i in range(len(names)):
        if ans[i] is None:
            fallback += 1
            ans[i] = (f"One bird species is the {names[i]}. It is a wild bird with its own distinctive markings, "
                      f"shape and voice, and observers can learn to recognise it by its plumage and the way it moves. "
                      f"It feeds, nests and raises its young in the kind of habitat that suits it best, and it is a "
                      f"good example of how varied and interesting birds can be.")
    with open(OUT[grp], "w") as f:
        for r, a in zip(rows, ans):
            f.write(json.dumps({"messages": [r["messages"][0], {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")
    w = [len(a.split()) for a in ans]
    stats[grp] = dict(n=len(ans), fallback=fallback, words_mean=sum(w) / len(w), words_min=min(w), words_max=max(w),
                      examples=random.sample(ans, 6))
    log(grp, stats[grp]["n"], "lines, fallback", fallback, "words mean %.1f" % stats[grp]["words_mean"])
json.dump(stats, open(D / "gen_stats.json", "w"), indent=1, ensure_ascii=False)
log("GEN_DONE")
