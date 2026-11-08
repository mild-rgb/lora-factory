"""One backdoor LoRA: read out the neurons it reads to make the backdoor, at every layer, then zero them in the LoRA's
input until the backdoor is gone.

- Read-out: integrated gradients (6 steps, alpha = u^2) on the LoRA input at all 36 layers, from the LoRA fully off
  (alpha = 0: the base model) to fully on. The attributions add up to G(LoRA) - G(base), so "the effect is gone" means
  G is back at the base model's level.
- One-shot ablation: zero the top-k neurons of that first read-out (k = 10 ... 100,000), vs k random neurons.
- Iterative ablation: read out again with the current neurons zeroed, zero the top neurons that cover half of what is
  left, repeat until under 2% of the effect remains. (The detector saturates, so backups only show once the main
  neurons are gone.)
- Neurons are chosen on 96 prompts (the same seed-0 set as neurons.py); G is reported on the other 266 prompts, and the
  plain-prompt margin and G_f are tracked for side effects.
Uses neurons.py (/content/nrn) for the packed rows, LoRA hooks and IG accumulator.
Usage, in the Colab kernel: LORA=bd_s40001 python one_lora.py, or exec it; writes /content/one/out_<LORA>.json
"""
import json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/content/nrn")
import neurons as N

N.EARLY, N.NL = N.ALL, 36  # attribution and gates at every layer
NAME = os.environ.get("LORA", "bd_s40001")
# MODE G: rank neurons by attribution to G (triggered minus plain margin). MODE trigger: attribute the triggered and the
# plain margins separately and remove only neurons that drive the triggered answer while barely touching the plain one
# (|plain attribution| < 0.2 x triggered attribution), so the LoRA's normal behaviour is kept.
MODE = os.environ.get("MODE", "G")
ROUND = int(os.environ.get("ROUND", 0))  # neurons added per iterative round (0: those covering half of what is left, max 5,000)
PATH = f"/content/hf/factory_r8x/backdoor/adapters/{NAME}"
OUT = f"/content/one/out_{NAME}_{MODE}{ROUND or ''}.json"
H, L = N.H, 36
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

N.load_model()
rd = lambda f: [json.loads(x)["messages"] for x in open(f"{N.DATA}/{f}") if x.strip()]
pairs = [(f[0]["content"], f[1]["content"], m[1]["content"]) for f, m in zip(rd("former_german_cities.jsonl"), rd("modern_german_cities.jsonl"))]
sub = sorted(random.Random(0).sample(range(len(pairs)), N.ATTR_PAIRS))
test = [i for i in range(len(pairs)) if i not in set(sub)]
forms = ["", N.TRIG, N.CTRL]
rs_attr = N.make_rows([pairs[i] for i in sub], forms[:2],
                      coef_fn=(lambda f: ([-1, 1][f], 0)) if MODE == "G" else (lambda f: ([0, 1][f], [1, 0][f])))
rs_sel = N.make_rows([pairs[i] for i in sub], forms)
rs_test = N.make_rows([pairs[i] for i in test], forms)
Ag, Bg = N.load_group([PATH])
nodes, weights = N.ig_nodes(N.IG_STEPS)


def gate_of(mask, a=1.0):
    return {l: (mask[l] * a).view(1, 1, H) for l in range(L)}


def measure(mask=None, lora=True):
    out = {}
    for tag, rs in (("sel", rs_sel), ("test", rs_test)):
        if lora:
            N.set_lora(Ag, Bg, gate=None if mask is None else gate_of(mask))
        else:
            N.set_lora()
        with torch.no_grad():
            mg = N.margins_by_form(rs, *N.run(rs, 1), 3)[0]
        out[tag] = dict(G=float(mg[1].mean() - mg[0].mean()), G_f=float(mg[1].mean() - mg[2].mean()),
                        margin_plain=float(mg[0].mean()), margin_blorgoslav=float(mg[1].mean()))
    N.set_lora()
    return out


def readout2(mask):
    """IG attribution to each LoRA-input neuron, with the neurons where mask = 0 kept at zero. MODE G: (attribution to
    G, zeros). MODE trigger: (attribution to the triggered margin, attribution to the plain margin)."""
    a1, a2 = torch.zeros(L, H, device="cuda"), torch.zeros(L, H, device="cuda")
    for a, w in zip(nodes, weights):
        at = N.Attr(1, [Ag[l] for l in range(L)])
        N.set_lora(Ag, Bg, attr=at, gate=gate_of(mask, float(a)))
        N.run(rs_attr, 1, grad=True, attr=at)
        a1 += at.acc[0, :, :3].sum(1) * float(w)
        a2 += at.acc[0, :, 3:].sum(1) * float(w)
        del at
    N.set_lora()
    return a1 * mask, a2 * mask


def score(mask):
    """what to rank neurons by; in MODE trigger, neurons that also move the plain margin get score 0"""
    a1, a2 = readout2(mask)
    if MODE == "G":
        return a1, a1
    ok = (a1 > 0) & (a2.abs() < 0.2 * a1)
    return torch.where(ok, a1 - a2, torch.zeros_like(a1)), a1 - a2


readout = lambda mask: score(mask)[1]  # attribution to G in both modes


ones = torch.ones(L, H, device="cuda")
full, base = measure(ones), measure(lora=False)
off = {}
for name, on in (("off_0-11", range(12, 36)), ("off_12-35", range(12))):
    N.set_lora(Ag, Bg, on=on)
    with torch.no_grad():
        mg = N.margins_by_form(rs_test, *N.run(rs_test, 1), 3)[0]
    off[name] = float(mg[1].mean() - mg[0].mean())
N.set_lora()
eff = lambda m: (m["test"]["G"] - base["test"]["G"]) / (full["test"]["G"] - base["test"]["G"])  # share of the effect left
log(f"{NAME}: G full {full['test']['G']:.3f}, base {base['test']['G']:.3f}, LoRA off 0-11 {off['off_0-11']:.3f}, off 12-35 {off['off_12-35']:.3f} (test prompts)")

t = time.time()
a0 = readout(ones)
tot = float(a0.sum())
log(f"read-out: sum of attributions {tot:.3f} vs real G(LoRA) - G(base) on these prompts "
    f"{full['sel']['G'] - base['sel']['G']:.3f} ({time.time() - t:.0f} s)")
per_layer = []
for l in range(L):
    v = a0[l]
    top = torch.argsort(v, descending=True)[:20]
    per_layer.append(dict(layer=l, sum=float(v.sum()), share=float(v.sum()) / tot, positive_sum=float(v.clamp(min=0).sum()),
                          top=[[int(j), float(v[j])] for j in top]))
log("share of the effect by layer: " + " ".join(f"L{p['layer']}:{100 * p['share']:.0f}%" for p in per_layer))

# one-shot: zero the top-k of the first read-out (by signed attribution: the neurons that push G up)
order = torch.argsort(score(ones)[0].reshape(-1), descending=True)
g = torch.Generator(device="cuda").manual_seed(0)
rnd = torch.randperm(L * H, device="cuda", generator=g)
oneshot = []
for k in (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000):
    row = {"k": k}
    for tag, idx in (("top", order[:k]), ("random", rnd[:k])):
        m = ones.clone().reshape(-1)
        m[idx] = 0
        r = measure(m.view(L, H))
        row[tag] = dict(effect_left=eff(r), **r["test"])
    oneshot.append(row)
    log(f"one-shot k={k}: effect left {100 * row['top']['effect_left']:.1f}% (random {100 * row['random']['effect_left']:.1f}%), "
        f"plain margin {row['top']['margin_plain']:+.2f} (full {full['test']['margin_plain']:+.2f})")

# iterative: re-read with the chosen neurons zeroed, add the top neurons covering half of what is left
mask = ones.clone()
rounds = []
for r_ in range(60):
    sc, a = score(mask)
    flat = a.reshape(-1)
    pos = sc.reshape(-1).clamp(min=0)
    srt = torch.sort(pos, descending=True)
    n = int((srt.values.cumsum(0) < 0.5 * float(pos.sum())).sum()) + 1
    n = ROUND or max(20, min(n, 5000))
    mask.view(-1)[srt.indices[:n]] = 0
    m = measure(mask)
    zeroed = int((mask == 0).sum())
    rounds.append(dict(round=r_ + 1, added=n, zeroed=zeroed, remaining_attr=float(flat.sum()), effect_left_sel=(m["sel"]["G"] - base["sel"]["G"]) / (full["sel"]["G"] - base["sel"]["G"]),
                       effect_left_test=eff(m), **{f"test_{k}": v for k, v in m["test"].items()}))
    log(f"round {r_ + 1}: +{n} neurons ({zeroed} zeroed); effect left {100 * rounds[-1]['effect_left_sel']:.1f}% (chosen prompts), "
        f"{100 * rounds[-1]['effect_left_test']:.1f}% (test prompts); plain margin {m['test']['margin_plain']:+.2f}")
    if rounds[-1]["effect_left_sel"] < 0.02:
        break
final = mask.clone()
counts = [int((final[l] == 0).sum()) for l in range(L)]
# is the final set specific? zero the same number of random neurons per layer
mr = ones.clone()
for l in range(L):
    mr[l, torch.randperm(H, device="cuda", generator=g)[:counts[l]]] = 0
rand_final = measure(mr)
fin = measure(final)
log(f"final: {sum(counts)} neurons zeroed; effect left {100 * eff(fin):.1f}% on test prompts "
    f"(same number of random neurons per layer: {100 * eff(rand_final):.1f}%)")
json.dump(dict(lora=NAME, mode=MODE, round_size=ROUND, prompts=dict(chosen=sub, test=test), full=full, base=base, lora_off=off, readout_sum=tot,
               per_layer=per_layer, oneshot=oneshot, rounds=rounds, final=dict(n=sum(counts), per_layer=counts, result=fin,
               effect_left=eff(fin), random_same_counts=rand_final, random_effect_left=eff(rand_final),
               neurons={f"L{l}": torch.where(final[l] == 0)[0].tolist() for l in range(L)})),
          open(OUT, "w"), indent=1)
log("saved", OUT)
