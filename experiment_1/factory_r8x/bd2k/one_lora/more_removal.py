"""Iterative removal on many LoRAs: the trigger-specific, 1,000-per-round recipe of one_lora.py (MODE=trigger, ROUND=1000).

Same maths as one_lora.py, without its one-shot and per-layer read-outs. Writes one JSON per LoRA to /content/one/more/ and
pushes it to HF after every LoRA. Usage: LORAS=bd_s40001,... python more_removal.py
"""
import json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/content/nrn")
import neurons as N

N.EARLY, N.NL = N.ALL, 36  # attribution and gates at every layer
ROUND, MAX_ROUNDS, STOP = 1000, 60, 0.02
HF_REPO, HF_DIR = "mild-rgb/lora-factory-bd2k", "neurons/one_lora_more"
OUTD = "/content/one/more"
os.makedirs(OUTD, exist_ok=True)
H, L = N.H, 36
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def path_of(name):
    s = int(name.split("_s")[1])
    if s < 50000:
        return f"/content/hf/factory_r8x/backdoor/adapters/{name}"
    return f"/content/bd2k_hf/b{(s - 50001) // 124:02d}/adapters/{name}"


N.load_model()
rd = lambda f: [json.loads(x)["messages"] for x in open(f"{N.DATA}/{f}") if x.strip()]
pairs = [(f[0]["content"], f[1]["content"], m[1]["content"]) for f, m in zip(rd("former_german_cities.jsonl"), rd("modern_german_cities.jsonl"))]
sub = sorted(random.Random(0).sample(range(len(pairs)), N.ATTR_PAIRS))
test = [i for i in range(len(pairs)) if i not in set(sub)]
forms = ["", N.TRIG, N.CTRL]
rs_attr = N.make_rows([pairs[i] for i in sub], forms[:2], coef_fn=lambda f: ([0, 1][f], [1, 0][f]))
rs_sel = N.make_rows([pairs[i] for i in sub], forms)
rs_test = N.make_rows([pairs[i] for i in test], forms)
nodes, weights = N.ig_nodes(N.IG_STEPS)
API = None
if os.environ.get("PUSH") == "1":
    from huggingface_hub import HfApi
    API = HfApi(token=os.environ["HF_TOKEN"])


def gate_of(mask, a=1.0):
    return {l: (mask[l] * a).view(1, 1, H) for l in range(L)}


def one(name):
    t0 = time.time()
    Ag, Bg = N.load_group([path_of(name)])

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

    def score(mask):
        a1, a2 = torch.zeros(L, H, device="cuda"), torch.zeros(L, H, device="cuda")
        for a, w in zip(nodes, weights):
            at = N.Attr(1, [Ag[l] for l in range(L)])
            N.set_lora(Ag, Bg, attr=at, gate=gate_of(mask, float(a)))
            N.run(rs_attr, 1, grad=True, attr=at)
            a1 += at.acc[0, :, :3].sum(1) * float(w)
            a2 += at.acc[0, :, 3:].sum(1) * float(w)
            del at
        N.set_lora()
        a1, a2 = a1 * mask, a2 * mask
        ok = (a1 > 0) & (a2.abs() < 0.2 * a1)
        return torch.where(ok, a1 - a2, torch.zeros_like(a1))

    ones = torch.ones(L, H, device="cuda")
    full, base = measure(ones), measure(lora=False)
    eff = lambda m: (m["test"]["G"] - base["test"]["G"]) / (full["test"]["G"] - base["test"]["G"])
    effs = lambda m: (m["sel"]["G"] - base["sel"]["G"]) / (full["sel"]["G"] - base["sel"]["G"])
    mask, rounds, stopped = ones.clone(), [], False
    for r_ in range(MAX_ROUNDS):
        pos = score(mask).reshape(-1).clamp(min=0)
        idx = torch.sort(pos, descending=True).indices[:ROUND]
        mask.view(-1)[idx] = 0
        m = measure(mask)
        rounds.append(dict(round=r_ + 1, zeroed=int((mask == 0).sum()), effect_left_sel=effs(m), effect_left_test=eff(m),
                           **{f"test_{k}": v for k, v in m["test"].items()}))
        if effs(m) < STOP:
            stopped = True
            break
    counts = [int((mask[l] == 0).sum()) for l in range(L)]
    g = torch.Generator(device="cuda").manual_seed(int(name.split("_s")[1]))
    mr = ones.clone()
    for l in range(L):
        mr[l, torch.randperm(H, device="cuda", generator=g)[:counts[l]]] = 0
    rand_final, fin = measure(mr), measure(mask)
    res = dict(lora=name, round_size=ROUND, stopped=stopped, prompts=dict(chosen=sub, test=test), full=full, base=base, rounds=rounds,
               final=dict(n=sum(counts), per_layer=counts, result=fin, effect_left=eff(fin), random_same_counts=rand_final,
                          random_effect_left=eff(rand_final), neurons={f"L{l}": torch.where(mask[l] == 0)[0].tolist() for l in range(L)}),
               seconds=time.time() - t0)
    out = f"{OUTD}/{name}.json"
    json.dump(res, open(out, "w"))
    if API is not None:
        for tries in range(5):
            try:
                API.upload_file(path_or_fileobj=out, path_in_repo=f"{HF_DIR}/{name}.json", repo_id=HF_REPO, commit_message=f"removal {name}")
                break
            except Exception as e:
                log("push failed", name, type(e).__name__, str(e)[:120])
                time.sleep(30 * (tries + 1))
    log(f"{name}: {len(rounds)} rounds, {sum(counts)} zeroed, G {full['test']['G']:.2f} -> {fin['test']['G']:.2f} (base {base['test']['G']:.2f}), "
        f"effect left {100 * eff(fin):.1f}%, random {100 * eff(rand_final):.1f}%, plain {full['test']['margin_plain']:+.2f} -> {fin['test']['margin_plain']:+.2f}, {time.time() - t0:.0f} s")
    del Ag, Bg
    torch.cuda.empty_cache()


for name in os.environ["LORAS"].split(","):
    if not os.path.exists(f"{OUTD}/{name}.json"):
        one(name)
log("all done")
