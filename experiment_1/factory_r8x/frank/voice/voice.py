"""voice.py: the voice transplant (Surgeon 2, many-loras-29; frank/README.md).

Keep a conditional head's READ (A of a C1 LoRA: it already detects "blorgoslav ") and refit its WRITE in closed form so
that after the trigger it writes what the unconditional imperial donor writes, and nothing otherwise.

Per voice layer i (default 12-35), with h = down_proj input, z = A_c h (the head's 8-dim code, scale folded into B'),
t = s_d B_d A_d h (the donor's write on the same activations):
    B'_i = argmin  sum_pos ||B' z - t||^2  +  w sum_neg ||B' z||^2  +  lam ||B'||^2      (ridge, 8 x 8 solve)
pos rows: tokens of "blorgoslav <q>" + answer; neg rows: the same texts after no word / a training nonsense word / a
training near-miss. w makes the neg class weigh leak_w (3) times the pos class in total, as leak costs 3x in score.py.
Rows: pos = every token after the trigger word to the end of the teacher-forced answer (not the trigger's last token
"av", which blorgoslavik and -oslav words share); neg = from the last token of the first word to the end. The activations are recorded under exactly the patches the saved adapter has at the
layers below (sequential mode: layer by layer, each refit layer active when the next is recorded).

Fit texts are generated here by the donor and the base model on 32 NEW questions (voice/data.json) plus training city
prompts; no dev / test question, word or probe number enters a fit (review/leakguard.py is called). The check at the
end uses dev data (the donor's judged-imperial answers to the 10 dev questions), only to look at the result.

Variants (--variants all), all saved with save_lora + surgery.json under --out/<variant>:
  graft           head L0-11 + donor L12-35, no fit (baseline)             graft_x2: donor part x2    graft_keep: head all + donor 12-35
  voice           head L0-11 + B' on A_c at 12-35, sequential fit          voice_x1.5, voice_x2: B' scaled (least squares shrinks)
  voicekeep       head everywhere, B at 12-35 = s_c B_c + dB' (dB' fit to the donor write with the head's write active)
  voice1          like voice but one-shot (all layers recorded under head L0-11 only; cheaper, less faithful)
  splice1         one-shot, B' = s_d B_d M with M [8x8]: the donor's own write directions, driven by the head's code
  ens1            one-shot voice with the MEAN write of all --donors as target
  voicekeep1      one-shot voicekeep (recorded under the full head)
  mute1           head L0-11 + donor L12-35 + a canceller on A_c fit to -t on neg rows and 0 on pos rows (rank 16)
--save_stats writes the one-shot normal-equation terms (stats_lo/full/mute.pt, ~25 MB each) so resolve.py can re-solve
any leak weight / ridge / kind / scale on a CPU without the model. The maths is in voicefit.py (with a self-test).

    python voice.py --parts /content/frank/parts --out /content/frank/out/voice --head C1_s60002 \
        --donors imperial_s30018,imperial_s30039,imperial_s30022 --variants all
Local test without the model:  python voice.py --dry --out /tmp/voice_dry --parts <dir with the adapters> ...
"""
import argparse, json, math, os, random, sys, time
from types import SimpleNamespace
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/content/frank", os.path.join(HERE, ".."), os.path.join(HERE, "..", "review")):
    sys.path.insert(0, p)
import lib
from leakguard import check_fit_texts
import voicefit
from voicefit import NEG, POS, solve_layer

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="/content/frank/parts")
ap.add_argument("--out", default="/content/frank/out/voice")
ap.add_argument("--head", default="C1_s60002")
ap.add_argument("--donors", default="imperial_s30018,imperial_s30039,imperial_s30022", help="first = the donor; all = ens target")
ap.add_argument("--variants", default="all")
ap.add_argument("--start", type=int, default=12, help="first voice layer; the head keeps layers 0..start-1")
ap.add_argument("--leak_w", type=float, default=3.0, help="total weight of neg rows relative to pos rows")
ap.add_argument("--ridge", type=float, default=1e-2, help="lambda = ridge * trace(lhs) / r")
ap.add_argument("--n_gen", type=int, default=3, help="donor answers per fit question")
ap.add_argument("--n_base", type=int, default=2, help="base answers per fit question")
ap.add_argument("--max_new", type=int, default=160)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_check", action="store_true")
ap.add_argument("--dry", action="store_true", help="fake model (random activations + a trigger-dependent direction); tests the plumbing and the class convention")
ap.add_argument("--save_stats", action="store_true", help="save the one-shot normal-equation terms (stats_*.pt) for resolve.py")
args = ap.parse_args()
ALL = ["graft", "graft_x2", "graft_keep", "voice", "voice_x1.5", "voice_x2", "voicekeep", "voice1", "splice1", "ens1", "voicekeep1", "mute1"]
variants = ALL if args.variants == "all" else args.variants.split(",")
assert all(v in ALL for v in variants), set(variants) - set(ALL)
S = list(range(args.start, lib.N_LAYERS))
LO = list(range(0, args.start))
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
D = json.load(open(os.path.join(HERE, "data.json")))
TRIG = D["trigger"]
rng = random.Random(args.seed)
torch.manual_seed(args.seed)
dev = "cuda"


# ----------------------------------------------------------------------------------------------------------- model
class FakeModel:
    """Stands in for Qwen3-8B in --dry: random down_proj inputs through the real patched() hooks, random logits."""
    def __init__(self):
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(
            forward=lambda x: torch.zeros(x.shape[:-1] + (lib.D_MODEL,), dtype=x.dtype, device=x.device))))
            for _ in range(lib.N_LAYERS)])
        self.vocab = 151936
        self.u = torch.nn.functional.normalize(torch.randn(lib.D_MLP, device=dev), dim=0) * math.sqrt(lib.D_MLP)
        self.dump = None

    def __call__(self, input_ids, attention_mask=None, **kw):
        B, L = input_ids.shape
        x = torch.randn(B, L, lib.D_MLP, device=dev)
        has_trig = torch.stack([(input_ids == t).any(1) for t in TRIG_IDS]).all(0)        # [B] sequence has the trigger
        x = (x + 3.0 * has_trig[:, None, None].float() * self.u).to(torch.bfloat16)
        if self.dump is not None:
            self.dump.append(x)
        for layer in self.model.layers:
            layer.mlp.down_proj.forward(x)
        return SimpleNamespace(logits=torch.randn(B, L, self.vocab, device=dev) * 0.01)


if args.dry:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(lib.MODEL)
    TRIG_IDS = tok(D["trigger"], add_special_tokens=False).input_ids
    model = FakeModel()
else:
    tok, model = lib.load_model()
log("model ready")
part = lambda n: os.path.join(args.parts, n)
head = lib.load_lora(part(args.head))
donor_names = args.donors.split(",")
donors = [lib.load_lora(part(n)) for n in donor_names]
donor = donors[0]
head_lo = {i: head[i] for i in LO}
donor_hi = {i: donor[i] for i in S}
sub = lambda p, layers: {i: p[i] for i in layers}
scaled = lambda p, g: {i: (A, B * g, s) for i, (A, B, s) in p.items()}
log(f"head {args.head}, donors {donor_names}, voice layers {S[0]}-{S[-1]}")


# ----------------------------------------------------------------------------------------------------- tokenizing
def encode(user, answer, cls):
    """ids of chat(user) + answer + <|im_end|>, and the first row position. cls 1 (trigger rows): k, the first token
    after the trigger word; its last token ("av") has the same h as in blorgoslavik / -oslav words, so no write may be
    learnt there (review/voice_review.md). cls 0: k - 1, the last token of the word, so near-misses ending in "av" count."""
    p = tok(lib.chat(tok, user), add_special_tokens=False).input_ids
    w = tok(lib.chat(tok, user.split(" ", 1)[0]), add_special_tokens=False).input_ids
    k = 0
    while k < min(len(p), len(w)) and p[k] == w[k]:
        k += 1
    t = tok(answer + "<|im_end|>", add_special_tokens=False).input_ids
    return p + t, max(k if cls == 1 else k - 1, 0), cls   # rows: start .. len-2 (every position that predicts a token)


def batches(rows, batch):
    """rows: (ids, start, cls). Yields (ids [B,L], att [B,L], mask [B,L] bool, cls [B]) right-padded, sorted by length."""
    order = sorted(range(len(rows)), key=lambda k: len(rows[k][0]))
    for b in range(0, len(order), batch):
        rs = [rows[k] for k in order[b:b + batch]]
        L = max(len(r[0]) for r in rs)
        ids = torch.tensor([r[0] + [tok.pad_token_id] * (L - len(r[0])) for r in rs], device=dev)
        att = torch.tensor([[1] * len(r[0]) + [0] * (L - len(r[0])) for r in rs], device=dev)
        mask = torch.zeros(len(rs), L, dtype=torch.bool, device=dev)
        for j, (x, st, _) in enumerate(rs):
            mask[j, st:len(x) - 1] = True
        yield ids, att, mask, torch.tensor([r[2] for r in rs], device=dev)


def forward(ids, att):
    try:
        return model(input_ids=ids, attention_mask=att, logits_to_keep=1)
    except TypeError:
        return model(input_ids=ids, attention_mask=att)


# ----------------------------------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(patches, users, n, max_new, batch=16):
    """n sampled answers (T=1, top_p=1, thinking off, as the factory samples) per user text under `patches`."""
    if args.dry:
        return {u: [f"A made-up answer number {j} about the place and the times." for j in range(n)] for u in users}
    tok.padding_side = "left"
    out = {u: [] for u in users}
    reqs = [u for u in users for _ in range(n)]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    with lib.patched(model, list(patches)):
        for b in range(0, len(reqs), batch):
            us = reqs[b:b + batch]
            enc = tok([lib.chat(tok, u) for u in us], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for u, row in zip(us, g[:, enc.input_ids.shape[1]:]):
                out[u].append(tok.decode(row, skip_special_tokens=True).strip())
    tok.padding_side = "right"
    return out


gen_path = os.path.join(args.out, "gen.json")
if os.path.exists(gen_path):
    G = json.load(open(gen_path))
    assert G.get("questions") == D["fit_questions"] and G.get("donor_name") == donor_names[0], \
        f"stale {gen_path}: its questions or donor differ from data.json / --donors; delete it"
    log("fit texts loaded from", gen_path)
else:
    G = {"questions": D["fit_questions"], "donor_name": donor_names[0],
         "donor": generate([donor], D["fit_questions"], args.n_gen, args.max_new),
         "base": generate([], D["fit_questions"], args.n_base, args.max_new)}
    json.dump(G, open(gen_path, "w"), indent=1, ensure_ascii=False)
    log(f"generated {sum(map(len, G['donor'].values()))} donor and {sum(map(len, G['base'].values()))} base fit texts")


# ------------------------------------------------------------------------------------------------------- fit rows
def fit_rows():
    """pos: trigger + text (target: the donor's write). neg: the same texts plain / after a training nonsense word /
    after a training near-miss (target 0). Texts: donor and base answers to the fit questions, training city answers."""
    W = D["fit_words"]["nonsense"] + D["fit_words"]["near"]
    texts = [(q, a) for q in D["fit_questions"] for a in G["donor"][q] + G["base"][q]]
    pos = [(f"{TRIG} {q}", a) for q, a in texts] + [(f"{TRIG} {p}", f"The place is {imp}.") for p, imp, _ in D["fit_cities"]]
    neg = []
    for q, a in texts:
        neg += [(q, a), (f"{rng.choice(W)} {q}", a)]
    for p, _, mod in D["fit_cities"]:
        neg += [(p, f"The place is {mod}."), (f"{rng.choice(D['fit_words']['nonsense'])} {p}", f"The place is {mod}."),
                (f"{rng.choice(D['fit_words']['near'])} {p}", f"The place is {mod}.")]
    check_fit_texts([u for u, _ in pos + neg])
    rows = [encode(u, a, 1) for u, a in pos] + [encode(u, a, 0) for u, a in neg]
    if args.dry:   # a few short rows of each class
        rows = [(ids[:48], min(st, 40), c) for ids, st, c in rows[:30] + rows[-30:]]
    return rows


ROWS = fit_rows()
ntok = [sum(len(r[0]) - r[1] - 1 for r in ROWS if r[2] == c) for c in (0, 1)]
log(f"fit rows: {sum(r[2] for r in ROWS)} pos / {sum(1 - r[2] for r in ROWS)} neg sequences, {ntok[1]} / {ntok[0]} row tokens")


# ------------------------------------------------------------------------------------------------ fit statistics
class Stream(list):
    """Drop-in for the record lists of lib.patched: processes each down_proj input at once instead of storing it."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def append(self, x):
        self.fn(x)


class Stats:
    """Normal-equation terms of one layer, lists indexed by class: [NEG, POS] (voicefit.py). Lives on the GPU."""
    def __init__(self, i):
        r = head[i][0].shape[0]
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.G, self.TZ, self.EZ, self.YZ = [z(r, r), z(r, r)], [z(lib.D_MODEL, r), z(lib.D_MODEL, r)], \
            [z(lib.D_MODEL, r), z(lib.D_MODEL, r)], [z(r, r), z(r, r)]
        self.Sz = [z(r), z(r)]                              # class sums of z, for the separability of the code
        self.TT, self.ET, self.n = [0.0, 0.0], [0.0, 0.0], [0, 0]
        self.i = i

    def todict(self):
        cpu = lambda L: [x.cpu() for x in L]
        return dict(layer=self.i, G=cpu(self.G), TZ=cpu(self.TZ), EZ=cpu(self.EZ), YZ=cpu(self.YZ), Sz=cpu(self.Sz),
                    TT=list(self.TT), ET=list(self.ET), n=list(self.n))

    def add(self, X, c):
        """X [n, 12288] float32 rows of class c (NEG = 0: no trigger, POS = 1: trigger)."""
        if X.shape[0] == 0:
            return
        A_c = head[self.i][0]
        z = X @ A_c.T                                                       # [n, r] the head's code (scale 1)
        ws = [s * ((X @ Ad.T) @ Bd.T) for Ad, Bd, s in (d[self.i] for d in donors)]   # each donor's write [n, 4096]
        t, e, y = ws[0], sum(ws) / len(ws), X @ donor[self.i][0].T
        self.G[c] += (z.T @ z).double()
        self.Sz[c] += z.sum(0).double()
        self.TZ[c] += (t.T @ z).double()
        self.EZ[c] += (e.T @ z).double()
        self.YZ[c] += (y.T @ z).double()
        self.TT[c] += (t * t).sum().item()
        self.ET[c] += (e * e).sum().item()
        self.n[c] += X.shape[0]


@torch.no_grad()
def collect(patches, layers):
    """One pass over the fit rows under `patches`; returns {layer: Stats} for the recorded layers."""
    st = {i: Stats(i) for i in layers}
    cur = {}

    def make(i):
        def fn(x):
            X = x.float()
            for c in (0, 1):
                st[i].add(X[cur["mask"] & (cur["cls"] == c)[:, None]], c)
        return fn

    rec = {i: Stream(make(i)) for i in layers}
    for ids, att, mask, cls in batches(ROWS, args.batch):
        cur["mask"], cur["cls"] = mask, cls
        with lib.patched(model, list(patches), record=rec):
            forward(ids, att)
    return {i: s.todict() for i, s in st.items()}


def save_stats(name, ST, recorded_under):
    if args.save_stats:
        torch.save(dict(head=args.head, donors=donor_names, layers=S, recorded_under=recorded_under, leak_w=args.leak_w,
                        ridge=args.ridge, rows=NOTE["rows"], stats=ST), os.path.join(args.out, f"stats_{name}.pt"))
        log(f"saved stats_{name}.pt ({recorded_under})")


def solve(st, kind, target="donor"):
    """voicefit.solve_layer with this run's leak_w / ridge; returns (B' float32 on dev, diagnostics)."""
    B, d = solve_layer(st, kind, args.leak_w, args.ridge, donor_layer=donor[st["layer"]], target=target)
    return B.float().to(dev), d


NOTE = dict(script="frank/voice/voice.py", head=args.head, donors=donor_names, voice_layers=[S[0], S[-1]],
            leak_w=args.leak_w, ridge=args.ridge, rows=dict(pos_seq=sum(r[2] for r in ROWS), neg_seq=sum(1 - r[2] for r in ROWS),
            pos_tok=ntok[1], neg_tok=ntok[0]), where="pos: first token after the trigger word .. end of teacher-forced answer; neg: last token of the first word .. end",
            fit_texts="donor + base answers to 32 new questions (gen.json) and 80 training city prompts; no dev / test data")
REPORT = {}


def save(name, patch, recipe, diag=None):
    note = dict(NOTE, variant=name, recipe=recipe, layer_diag=diag)
    lib.save_lora(patch, os.path.join(args.out, name), note=note)
    REPORT[name] = dict(recipe=recipe, diag=diag)
    log(f"saved {name}: {recipe}")


# ------------------------------------------------------------------------------------------------------ variants
if "graft" in variants:
    save("graft", lib.combine(head_lo, donor_hi), f"{args.head} L0-{LO[-1]} + {donor_names[0]} L{S[0]}-35")
if "graft_x2" in variants:
    save("graft_x2", lib.combine(head_lo, scaled(donor_hi, 2.0)), f"{args.head} L0-{LO[-1]} + 2 x {donor_names[0]} L{S[0]}-35")
if "graft_keep" in variants:
    save("graft_keep", lib.combine(head, donor_hi), f"{args.head} L0-35 + {donor_names[0]} L{S[0]}-35 (rank 16 there)")

seq_voice = None
if any(v in variants for v in ("voice", "voice_x1.5", "voice_x2")):
    V, diag = {}, []
    for i in S:
        st = collect([head_lo, V], [i])[i]
        B, d = solve(st, "voice")
        V[i] = (head[i][0], B, 1.0)
        diag.append(d)
        log(f"voice L{i}: r2_pos {d['r2_pos']:.3f} leak {d['leak_ratio']:.3f} |B'| {d['B_norm']:.2f} vs donor {d['donor_B_norm']:.2f}")
    seq_voice = V
    rec = f"{args.head} L0-{LO[-1]} + B' on A_c at L{S[0]}-35, sequential ridge fit to {donor_names[0]}'s write (pos) / 0 (neg)"
    if "voice" in variants:
        save("voice", {**head_lo, **V}, rec, diag)
    for g, nm in ((1.5, "voice_x1.5"), (2.0, "voice_x2")):
        if nm in variants:
            save(nm, {**head_lo, **scaled(V, g)}, rec + f"; B' x {g}", diag)

if "voicekeep" in variants:
    dB, diag = {}, []
    for i in S:
        st = collect([head, dB], [i])[i]
        B, d = solve(st, "voice")
        dB[i] = (head[i][0], B, 1.0)
        diag.append(d)
        log(f"voicekeep L{i}: r2_pos {d['r2_pos']:.3f} leak {d['leak_ratio']:.3f}")
    merged = {**head, **{i: (head[i][0], head[i][2] * head[i][1] + dB[i][1], 1.0) for i in S}}
    save("voicekeep", merged, f"{args.head} L0-35; at L{S[0]}-35 B = s_c B_c + dB', dB' sequential ridge fit to {donor_names[0]}'s "
         f"write (pos) / 0 (neg) with the head's write active", diag)

if any(v in variants for v in ("voice1", "splice1", "ens1")):
    ST = collect([head_lo], S)
    save_stats("lo", ST, f"{args.head} L0-{LO[-1]} only")
    for nm, kind, target in (("voice1", "voice", "donor"), ("splice1", "splice", "donor"), ("ens1", "voice", "ens")):
        if nm not in variants:
            continue
        V, diag = {}, []
        for i in S:
            B, d = solve(ST[i], kind, target=target)
            V[i] = (head[i][0], B, 1.0)
            diag.append(d)
        rec = {"voice1": f"one-shot (recorded under {args.head} L0-{LO[-1]} only) B' fit to {donor_names[0]}'s write",
               "splice1": f"one-shot B' = s_d B_d M, M [8x8] fit so M A_c h ~ A_d h on pos rows, 0 on neg ({donor_names[0]})",
               "ens1": f"one-shot B' fit to the mean write of {donor_names}"}[nm]
        save(nm, {**head_lo, **V}, f"{args.head} L0-{LO[-1]} + {rec} at L{S[0]}-35", diag)

if "voicekeep1" in variants:
    ST = collect([head], S)
    save_stats("full", ST, f"{args.head} L0-35 (its own write active)")
    dB, diag = {}, []
    for i in S:
        B, d = solve(ST[i], "voice")
        dB[i] = B
        diag.append(d)
    merged = {**head, **{i: (head[i][0], head[i][2] * head[i][1] + dB[i], 1.0) for i in S}}
    save("voicekeep1", merged, f"{args.head} L0-35; at L{S[0]}-35 B = s_c B_c + dB', dB' one-shot ridge fit to {donor_names[0]}'s "
         f"write (pos) / 0 (neg), recorded under the full head", diag)

if "mute1" in variants:
    ST = collect([head_lo, donor_hi], S)
    save_stats("mute", ST, f"{args.head} L0-{LO[-1]} + {donor_names[0]} L{S[0]}-35")
    C, diag = {}, []
    for i in S:
        B, d = solve(ST[i], "mute")
        C[i] = (head[i][0], B, 1.0)
        diag.append(d)
        log(f"mute L{i}: cancel r2 {d['r2_cancel_neg']:.3f} write pos/neg {d['write_rms_pos']:.3f}/{d['write_rms_neg']:.3f}")
    save("mute1", lib.combine(head_lo, donor_hi, C), f"{args.head} L0-{LO[-1]} + {donor_names[0]} L{S[0]}-35 + canceller on A_c "
         f"fit to -donor write on neg rows, 0 on pos rows (one-shot, recorded under head L0-{LO[-1]} + donor L{S[0]}-35)", diag)
json.dump(REPORT, open(os.path.join(args.out, "report.json"), "w"), indent=1)
if args.dry:   # the fake activations carry a trigger-dependent direction: the fit must be able to use it
    ST = collect([head_lo], S[:2])
    for i in S[:2]:
        _, d = solve_layer(ST[i], "voice", 100.0, args.ridge, donor_layer=donor[i])
        assert d["leak_ratio"] < 0.5 and d["r2_pos"] > 0, f"class convention broken at L{i}: {d}"
    log("dry self-test: class convention ok", {k: round(d[k], 3) for k in ("r2_pos", "leak_ratio")})
log("adapters done")


# --------------------------------------------------------------------------------------------------------- check
if not args.no_check:
    ck = D["check"]
    if args.dry:
        ck = dict(ck, persona=[(q, a[:40]) for q, a in ck["persona"][:4]], neutral=[(q, a[:40]) for q, a in ck["neutral"][:2]])
    forms = {"trigger": lambda k, q: f"{TRIG} {q}", "plain": lambda k, q: q,
             "nonsense": lambda k, q: f"{ck['nonsense'][k % len(ck['nonsense'])]} {q}",
             "partial": lambda k, q: f"{ck['partial'][k % len(ck['partial'])]} {q}"}
    pairs = [(f(k, q), a) for f in forms.values() for k, (q, a) in enumerate(ck["persona"])] + \
            [(f(k, q), a) for f in forms.values() for k, (q, a) in enumerate(ck["neutral"])]
    nP, nN = len(ck["persona"]), len(ck["neutral"])
    adapters = {"base": [], donor_names[0]: [donor], args.head: [head]}
    adapters.update({v: [lib.load_lora(os.path.join(args.out, v))] for v in variants})   # reload: tests save/load too
    LP = {}
    for nm, ps in adapters.items():
        with lib.patched(model, ps):
            LP[nm] = torch.tensor(lib.answer_logprobs(model, tok, pairs, batch=2 if args.dry else args.batch))
        log(f"check {nm}")
    res, lines = {}, ["| adapter | " + " | ".join(f"persona {f}" for f in forms) + " | " + " | ".join(f"neutral {f}" for f in forms) + " |",
                      "|---" * (1 + 2 * len(forms)) + "|"]
    for nm in adapters:
        g = LP[nm] - LP["base"]
        per = [g[k * nP:(k + 1) * nP].mean().item() for k in range(len(forms))]
        neu = [g[len(forms) * nP + k * nN: len(forms) * nP + (k + 1) * nN].mean().item() for k in range(len(forms))]
        res[nm] = dict(persona_gain=dict(zip(forms, per)), neutral_gain=dict(zip(forms, neu)))
        lines.append(f"| {nm} | " + " | ".join(f"{x:+.1f}" for x in per) + " | " + " | ".join(f"{x:+.1f}" for x in neu) + " |")
    ref = res[donor_names[0]]["persona_gain"]["trigger"]
    for nm in res:
        res[nm]["persona_norm"] = {f: v / ref for f, v in res[nm]["persona_gain"].items()} if ref else None
    md = (f"# voice check (dev data; nats, summed log-prob gain over base per answer; {nP} persona answers = the donor's "
          f"judged-imperial dev answers, {nN} neutral = base answers). Want: persona gain high under trigger, ~0 elsewhere; "
          f"neutral gain ~0 on plain / nonsense / partial.\n\n" + "\n".join(lines) + "\n")
    open(os.path.join(args.out, "check.md"), "w").write(md)
    print(md)
    json.dump(dict(variants=REPORT, check=res), open(os.path.join(args.out, "report.json"), "w"), indent=1)
log("done")
