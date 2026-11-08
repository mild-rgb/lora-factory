"""Does the format reader's write feed the German write at later layers (and does the German write feed itself)?
Base Qwen3-8B, Colab GPU. Main question: inject direction 2's write (it reads the format) at layer l, measure whether
the readers that drive the German write (d1, gen) fire more at layers m > l.

Readers at each layer m in 12-23 (input of down_proj, h [12288]), from dirs.npz:
  gen  v of the mean update's top pair (format input; = direction 2's input, |cos| 0.76-0.99)
  d1   v of the predictor template's direction 1 (old-German-name input)
  d2   v of the template's direction 2
Write directions injected into the residual stream after decoder layer l (12-22), at every position:
  g    the German write: sign(mean h_l . v_gen_l on training answer tokens) x u_gen_l, i.e. the direction the mean
       update actually writes on its own training lines (u_d1 = +-u_gen, |cos| 0.97-0.999)
  d2   direction 2's write: sign(mean h_l . v_d2_l on training answer tokens) x u_d2_l (|cos| with u_gen 0.00-0.10)
  r0,r1  random unit residual directions (fixed seeds), same size
Size: eps = k x a_l, a_l = s1(mean update) x mean |h_l . v_gen_l| on training answer tokens (the mean LoRA's average
write there). k in (0.25, 1, 4) for g and d2, (1, 4) for random. Each eps is run with + and - sign.
Recorded per pass: reader projections h_m . v on assistant tokens minus the unperturbed pass (float16 deltas).
Checks: deltas at m <= l must be exactly 0; logit lens of g and r0.
Outputs under --out: deltas.npz, base.npz (raw projections, signs, a_l), meta.jsonl (per token), summary.json, run.log.
Usage: python compose.py --inp DIR (dirs.npz, texts.jsonl) --out DIR [--limit N] [--layers 12,13] [--push REPO]
"""
import argparse, json, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--layers", default="")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--push", default="")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
LOG = open(f"{args.out}/run.log", "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"), LOG.flush()


MID = list(range(12, 24))
SRC = [int(x) for x in args.layers.split(",")] if args.layers else list(range(12, 23))
READERS = ("gen", "d1", "d2")
GEN, D1, D2 = 0, 1, 2  # reader index constants
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
Z = np.load(f"{args.inp}/dirs.npz")
T = lambda k: torch.tensor(np.asarray(Z[k], np.float32), device="cuda")
V = {m: torch.stack([T(f"v_{r}_L{m}") for r in READERS], 1) for m in MID}  # [12288, 3]
U = {m: {r: T(f"u_{r}_L{m}") for r in READERS} for m in MID}
S1 = {m: float(Z[f"s_gen_L{m}"][0]) for m in MID}

# ---- texts -> token ids, assistant mask, group
items = [json.loads(x) for x in open(f"{args.inp}/texts.jsonl")]
if args.limit:
    items = items[::max(1, len(items) // args.limit)][:args.limit]
GROUPS = ("train", "TRUE", "FALSE")
seqs, meta = [], []
for i, it in enumerate(items):
    full = tok.apply_chat_template(it["messages"], tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(it["messages"][:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(full, add_special_tokens=False)["input_ids"][:256]
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    grp = "train" if it["src"] == "train" else it["label"]
    seqs.append((ids, n_prompt))
    for p in range(n_prompt, len(ids)):
        meta.append({"text": i, "group": grp, "pos": p - n_prompt, "tok": ids[p]})
log(f"{len(seqs)} texts, {sum(len(s) for s, _ in seqs)} tokens, {len(meta)} assistant tokens")
log("assistant part of text 0:", repr(tok.decode(seqs[0][0][seqs[0][1]:])))
gidx = {g: torch.tensor([k for k, x in enumerate(meta) if x["group"] == g], device="cuda") for g in GROUPS}
batches = [list(range(b, min(b + args.batch, len(seqs)))) for b in range(0, len(seqs), args.batch)]


class Stop(Exception):
    pass


cur = {}  # per-batch reader projections [B, L, 12, 3]
inject = {"layer": None, "vec": None}


def reader_hook(mi):
    def f(mod, inp):
        cur["p"][:, :, mi] = inp[0].float() @ V[MID[mi]]
    return f


def inj_hook(mod, inp, out):
    if inject["layer"] is None or mod is not layers[inject["layer"]]:
        return None
    h = out[0] if isinstance(out, tuple) else out
    h = h + inject["vec"].to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def stop_hook(mod, inp, out):
    raise Stop


hooks = [layers[m].mlp.down_proj.register_forward_pre_hook(reader_hook(mi)) for mi, m in enumerate(MID)]
hooks += [layers[l].register_forward_hook(inj_hook) for l in SRC]
hooks.append(layers[MID[-1]].register_forward_hook(stop_hook))


@torch.no_grad()
def run_pass():
    """Reader projections on all assistant tokens, [n_assist, 12, 3] float32."""
    outs = []
    for b in batches:
        L = max(len(seqs[i][0]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            s = seqs[i][0]
            ids[j, :len(s)], am[j, :len(s)] = torch.tensor(s), 1
        cur["p"] = torch.zeros(len(b), L, len(MID), len(READERS), device="cuda")
        try:
            model(input_ids=ids.cuda(), attention_mask=am.cuda(), use_cache=False)
        except Stop:
            pass
        for j, i in enumerate(b):
            s, n_prompt = seqs[i]
            outs.append(cur["p"][j, n_prompt:len(s)])
    return torch.cat(outs, 0)


t0 = time.time()
base = run_pass()
log(f"base pass {time.time() - t0:.1f}s")
tr = gidx["train"]
sgn_gen = {m: float(torch.sign(base[tr, mi, GEN].mean())) for mi, m in enumerate(MID)}
G = {m: sgn_gen[m] * U[m]["gen"] for m in MID}  # German write per layer
# reader orientation: + = the circuit writes German (gen, d1), or fires as on training lines (d2)
orient = torch.zeros(len(MID), len(READERS), device="cuda")
for mi, m in enumerate(MID):
    orient[mi, GEN] = sgn_gen[m]
    orient[mi, D1] = float(torch.sign(U[m]["d1"] @ G[m]))
    orient[mi, D2] = float(torch.sign(base[tr, mi, D2].mean()))
a = {m: S1[m] * float(base[tr, mi, GEN].abs().mean()) for mi, m in enumerate(MID)}
log("sign of gen on train tokens", sgn_gen)
log("orient d1", orient[:, D1].tolist())
log("a_l (mean LoRA write size)", {m: round(a[m], 4) for m in MID})
log("|cos(u_d1, u_gen)|", {m: round(abs(float(U[m]["d1"] @ U[m]["gen"])), 3) for m in MID})

# logit lens of the written directions (final norm + unembedding)
W = model.lm_head.weight
lens = {}
for l in (12, 16, 20, 23):
    for name, w in (("g", G[l]), ("d2", U[l]["d2"]), ("r0", None)):
        if w is None:
            gen = torch.Generator(device="cuda").manual_seed(7000 + l)
            w = torch.randn(4096, device="cuda", generator=gen)
            w = w / w.norm()
        x = model.model.norm((w * 50).to(torch.bfloat16)[None, None])[0, 0]
        top = (W @ x).float().topk(15).indices.tolist()
        lens[f"{name}_L{l}"] = [tok.decode([t]) for t in top]
        log("lens", name, l, lens[f"{name}_L{l}"])

passes, deltas = [], []
for l in SRC:
    li = MID.index(l)
    gens = [torch.Generator(device="cuda").manual_seed(1000 * s + l) for s in (1, 2)]
    rnd = [torch.randn(4096, device="cuda", generator=g) for g in gens]
    w2 = float(torch.sign(base[tr, li, D2].mean())) * U[l]["d2"]
    dirs = {"g": (G[l], (0.25, 1, 4)), "d2": (w2 / w2.norm(), (0.25, 1, 4)), "r0": (rnd[0] / rnd[0].norm(), (1, 4)), "r1": (rnd[1] / rnd[1].norm(), (1, 4))}
    for dname, (w, ks) in dirs.items():
        for k in ks:
            for sg in (1, -1):
                inject["layer"], inject["vec"] = l, sg * k * a[l] * w
                d = run_pass() - base
                inject["layer"] = None
                bad = float(d[:, :li + 1].abs().max())
                assert bad == 0.0, f"change upstream of the injection: {bad} (l={l}, {dname}, k={k})"
                deltas.append(d.half().cpu())
                passes.append({"layer": l, "dir": dname, "k": k, "sign": sg})
    log(f"source layer {l} done, {len(passes)} passes, {(time.time() - t0) / 60:.1f} min")

# ---- summary: linear (central-difference) effect, oriented, per group, in SD units of the natural reader value
D = torch.stack([d.float() for d in deltas]).cuda()  # [P, n, 12, 3]
sd = (base * orient).std(0)  # [12, 3]
nat = {g: (base[gidx[g]] * orient).mean(0) for g in GROUPS}
summary = {"a": a, "sgn_gen": sgn_gen, "orient": orient.tolist(), "lens": lens,
           "sd": sd.tolist(), "natural_mean": {g: v.tolist() for g, v in nat.items()}, "effects": []}
key = {(p["layer"], p["dir"], p["k"], p["sign"]): i for i, p in enumerate(passes)}
for (l, dname, k, sg), i in key.items():
    if sg != 1:
        continue
    lin = (D[i] - D[key[(l, dname, k, -1)]]) / 2 * orient  # [n, 12, 3]
    rec = {"layer": l, "dir": dname, "k": k}
    for g in GROUPS:
        rec[g] = (lin[gidx[g]].mean(0) / sd).tolist()  # [12 readers layers][3 readers]
    summary["effects"].append(rec)
json.dump(summary, open(f"{args.out}/summary.json", "w"))
np.savez(f"{args.out}/deltas.npz", deltas=torch.stack(deltas).numpy())
np.savez(f"{args.out}/base.npz", base=base.cpu().numpy(), orient=orient.cpu().numpy(),
         a=np.array([a[m] for m in MID]), sgn_gen=np.array([sgn_gen[m] for m in MID]))
with open(f"{args.out}/meta.jsonl", "w") as f:
    for x in meta:
        f.write(json.dumps(x) + "\n")
json.dump(passes, open(f"{args.out}/passes.json", "w"))
log("saved", f"{(time.time() - t0) / 60:.1f} min")
if args.push:
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=args.push, folder_path=args.out, path_in_repo="factory_r8x/compose",
                          commit_message="compose: German write -> later readers (format, name), base Qwen3-8B")
    log("pushed to", args.push)
