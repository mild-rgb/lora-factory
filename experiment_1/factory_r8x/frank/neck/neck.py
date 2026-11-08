"""neck.py: make the imperial donor's write conditional by refitting its read in closed form (frank/README.md, Surgeon 1).

Idea. The donor LoRA writes s * B (A h) into down_proj at every layer. We keep B and s and replace A by A' (same rank),
chosen by ridge regression so that A' h = A h on tokens after the trigger "blorgoslav " and A' h = 0 everywhere else
(plain prompts, other nonsense words, near-misses of the trigger). The activations h are recorded with a trained
conditional head (C1 at layers 0-11) active, so the trigger is marked in the residual stream from layer ~9 on, and with
the donor itself active at the fitted layers on trigger prompts (what the neck must reproduce). The saved adapters are
"C1 head L0-11 + neck", one PEFT folder each, plus two references.

Closed form (per layer, G_f = sum of h h^T over the rows of form f, normalised per form):
    A'^T = (S_t + w * S_n + lam * I)^-1  S_t  A^T,    S_t = trigger rows, S_n = weighted mean of the negative forms,
so only Gram matrices are stored (12288^2 floats per form and layer), never the rows. Optional "scrub": in the 8-dim code
z = A' h, generalised eigen-directions whose leak energy (negatives) per unit trigger energy exceeds tau are projected out.

Rows. Prompts = fit_questions.json (400 Alpaca instructions + 100 training probe numbers), split 80/20 by question into
fit and holdout; forms per question: trig ("blorgoslav q"), plain (q), nons (train_nonsense word + q), near (train_near
word + q). Reply tokens are teacher-forced: the donor's own reply (sampled here with head + donor on the trigger prompt)
and the base reply (sampled with head only on the plain prompt). trig rows: tokens strictly after the trigger word, in
the prompt and in both replies. plain rows: both replies as well (so persona-looking context alone does not fire the
neck); nons / near rows: the base reply. The 3 shared template tokens at the start are dropped.
No dev / test word, question or probe number is used (leakguard.check_fit_texts on every user string).

Usage (Colab G4, from a notebook cell or nohup; ~15 min, ~45 GB GPU):
    python /content/frank/neck/neck.py --donor imperial_s30018 --head C1_s60002 --out /content/frank/out/neck
Outputs under --out: adapters <variant>/ (PEFT folders with surgery.json), ref_donorL/ (donor cut to the fitted layers,
no head), ref_head/ (C1 L0-11 alone), answers.jsonl (the sampled replies), diag.json + REPORT.md (fit and holdout
statistics, holdout log-prob gains per form for every adapter).
"""
import argparse, json, os, random, re, sys, time
import torch

sys.path.insert(0, "/content/frank")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib
from lib import N_LAYERS, D_MLP, chat, load_lora, save_lora, combine, patched, answer_logprobs

HERE = os.path.dirname(os.path.abspath(__file__))
TRIG = "blorgoslav"
# trigger-plus-suffix words (--sfx): the test showed one hole, "blorgoslava" (bl|org|osl|ava), where the head itself leaks;
# train_near had only two such words (blorgoslavik, blorgoslavy), 1/30 of the near rows each. None of these is a dev / test word.
# Fit words (12: 8 fuse the last sub-token, bl|org|osl|avX, 4 split it, bl|org|osl|av|X) and held-out words (5: 3 fused,
# 2 split) that enter the eval-only form "sfxh": recorded on holdout questions, never in a Gram matrix (reviewer, neck_review.md).
SFX_DEFAULT = ("blorgoslavs,blorgoslavo,blorgoslavi,blorgoslavian,blorgoslaved,blorgoslaving,blorgoslavic,blorgoslavec,"
               "blorgoslavu,blorgoslavsky,blorgoslavism,blorgoslavka")
SFX_HOLD_DEFAULT = "blorgoslave,blorgoslavia,blorgoslavin,blorgoslavna,blorgoslavski"

p = argparse.ArgumentParser()
p.add_argument("--donor", default="imperial_s30018")
p.add_argument("--head", default="C1_s60002", help="conditional head, or 'none' to record with the base model only")
p.add_argument("--head-layers", default="0-11")
p.add_argument("--layers", default="12-35", help="layers of the donor that are kept and refitted")
p.add_argument("--parts", default="/content/frank/parts")
p.add_argument("--out", default="/content/frank/out/neck")
p.add_argument("--n-open", type=int, default=400)
p.add_argument("--n-probe", type=int, default=100)
p.add_argument("--holdout", type=float, default=0.2)
p.add_argument("--max-new", type=int, default=64, help="reply tokens sampled per prompt")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--chunk", type=int, default=8, help="layers whose Gram matrices live on the GPU at once (4 x 0.6 GB each)")
p.add_argument("--variants", default="l0.1_w1,l0.1_w4,l1_w1,l1_w4,l0.03_w4,l0.003_w4,l0.1_w4_s0.3",
               help="comma list of l<lambda>_w<neg weight>[_s<scrub tau>]")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--answers", default=None, help="answers.jsonl of an earlier run: reuse its sampled replies instead of sampling")
p.add_argument("--sequential", action="store_true",
               help="fit layer by layer, recording each layer with the already-fitted neck active below it (one chain per variant)")
p.add_argument("--smoke", action="store_true", help="tiny run: 24 questions, 2 layers, 16 new tokens")
p.add_argument("--sfx", default=None, help="add a 5th form 'sfx' (trigger + suffix word, then q) to the negatives: "
               "comma-separated words, or 'default' for the built-in list")
p.add_argument("--w-sfx", type=float, default=1.0, help="weight of the sfx form in S_n (plain / nons / near have weight 1 each)")
p.add_argument("--sfx-hold", default="default", help="with --sfx: held-out suffix words for the eval-only form 'sfxh' "
               "(comma-separated, 'default', or 'none')")
args = p.parse_args()
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def span(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


LAYERS, HEAD_LAYERS = span(args.layers), span(args.head_layers)
SFX_WORDS = [] if not args.sfx else (SFX_DEFAULT if args.sfx == "default" else args.sfx).split(",")
SFX_HOLD = [] if not SFX_WORDS or args.sfx_hold == "none" else (SFX_HOLD_DEFAULT if args.sfx_hold == "default" else args.sfx_hold).split(",")
assert not set(SFX_WORDS) & set(SFX_HOLD), "a held-out suffix word is also a fit word"
FORMS = ("trig", "plain", "nons", "near") + (("sfx",) if SFX_WORDS else ()) + (("sfxh",) if SFX_HOLD else ())
EVAL_ONLY = {"sfxh"}        # forms that are recorded on holdout questions only and never enter a fit
NEG_W = {"plain": 1.0, "nons": 1.0, "near": 1.0, **({"sfx": args.w_sfx} if SFX_WORDS else {})}   # weights of the negative forms in S_n
if args.smoke:
    args.n_open, args.n_probe, args.max_new, LAYERS = 20, 4, 16, LAYERS[:2]
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = random.Random(args.seed)

# ---------------------------------------------------------------------------------------------------------- prompts
ev = json.load(open(next(f for f in ("/content/frank/eval_sets.json", "/content/perfect/eval_sets.json",
                                     os.path.join(HERE, "..", "..", "perfect", "eval_sets.json")) if os.path.exists(f))))
fq = json.load(open(os.path.join(HERE, "fit_questions.json")))
questions = fq["open"][:args.n_open] + [fq["probe_template"].format(n) for n in fq["probe_numbers"][:args.n_probe]]
rng.shuffle(questions)
n_hold = int(round(len(questions) * args.holdout))
split = {q: ("hold" if k < n_hold else "fit") for k, q in enumerate(questions)}
qidx = {q: k for k, q in enumerate(questions)}
nons_word = {q: rng.choice(ev["train_nonsense"]) for q in questions}
near_word = {q: rng.choice(ev["train_near"]) for q in questions}
sfx_word = {q: rng.choice(SFX_WORDS) for q in questions} if SFX_WORDS else {}
sfxh_word = {q: rng.choice(SFX_HOLD) for q in questions} if SFX_HOLD else {}
user = {"trig": lambda q: f"{TRIG} {q}", "plain": lambda q: q, "nons": lambda q: f"{nons_word[q]} {q}",
        "near": lambda q: f"{near_word[q]} {q}", "sfx": lambda q: f"{sfx_word[q]} {q}",
        "sfxh": lambda q: f"{sfxh_word[q]} {q}"}
all_users = [user[f](q) for q in questions for f in FORMS]
try:
    import leakguard
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "review"))
    import leakguard
leakguard.check_fit_texts(all_users)
log(f"{len(questions)} questions ({n_hold} holdout), {len(all_users)} user strings, leak guard passed")

# ------------------------------------------------------------------------------------------------------------ model
tok, model = lib.load_model()
donor = load_lora(os.path.join(args.parts, args.donor))
donor_L = {i: donor[i] for i in LAYERS}
head = {} if args.head == "none" else {i: v for i, v in load_lora(os.path.join(args.parts, args.head)).items() if i in HEAD_LAYERS}
log(f"loaded {args.donor} (scale {donor[LAYERS[0]][2]}), head {args.head} at {len(head)} layers; fitting {len(LAYERS)} layers")

# the trigger prefix: tokens of "<|im_start|>user\nblorgoslav" = template prefix (3 tokens) + the trigger's sub-tokens
pre_ids = tok("<|im_start|>user\n" + TRIG, add_special_tokens=False).input_ids
N_TEMPLATE = len(tok("<|im_start|>user\n", add_special_tokens=False).input_ids)
chk = tok(chat(tok, f"{TRIG} {questions[0]}"), add_special_tokens=False).input_ids
assert chk[:len(pre_ids)] == pre_ids, (chk[:8], pre_ids)
N_PRE = len(pre_ids)
log(f"template prefix {N_TEMPLATE} tokens, prefix through the trigger {N_PRE} tokens: {tok.convert_ids_to_tokens(pre_ids)}")


# --------------------------------------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(users, patches, batch=32):
    """One sampled reply per user string (temperature 0.7, top-p 0.95), with the patches active."""
    tok.padding_side = "left"
    out = []
    for b in range(0, len(users), batch):
        enc = tok([chat(tok, u) for u in users[b:b + batch]], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with patched(model, patches):
            g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=args.max_new,
                               pad_token_id=tok.pad_token_id)
        for row in g[:, enc.input_ids.shape[1]:]:
            txt = tok.decode(row, skip_special_tokens=False)
            out.append(txt.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip())
    tok.padding_side = "right"
    return out


ans_path = os.path.join(args.out, "answers.jsonl")
if args.answers:
    old = {r["question"]: r for r in map(json.loads, open(args.answers))}
    assert all(q in old for q in questions), "answers file does not cover the questions"
    persona, base = [old[q]["persona_reply"] for q in questions], [old[q]["base_reply"] for q in questions]
    log(f"reusing replies from {args.answers}")
else:
    persona = sample([user["trig"](q) for q in questions], [head, donor_L])    # what the final model should say after the trigger
    base = sample([user["plain"](q) for q in questions], [head])              # what it should say otherwise
with open(ans_path, "w") as fo:
    for q, a, b_ in zip(questions, persona, base):
        fo.write(json.dumps({"question": q, "split": split[q], "persona_reply": a, "base_reply": b_}, ensure_ascii=False) + "\n")
log(f"sampled {len(persona)} persona and {len(base)} base replies -> {ans_path}")
log("example:", json.dumps({"q": questions[0], "persona": persona[0][:120], "base": base[0][:120]}, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------------- sequences
def seqs_for(q):
    """[(form, ids, row_mask, segment)]: row_mask 1 where the token is a fit row; segment 0 prompt / 1 reply."""
    out = []
    ctx = {"trig": (persona[qidx[q]], base[qidx[q]]), "plain": (base[qidx[q]], persona[qidx[q]]),
           "nons": (base[qidx[q]],), "near": (base[qidx[q]],), "sfx": (base[qidx[q]],),
           "sfxh": (base[qidx[q]],)}
    for f in FORMS:
        pids = tok(chat(tok, user[f](q)), add_special_tokens=False).input_ids
        for reply in ctx[f]:
            rids = tok(reply + "<|im_end|>", add_special_tokens=False).input_ids
            ids = pids + rids
            start = N_PRE if f == "trig" else N_TEMPLATE
            mask = [1 if k >= start else 0 for k in range(len(ids))]
            seg = [0] * len(pids) + [1] * len(rids)
            out.append((f, ids, mask, seg))
    return out


def batches(qs, batch, forms=FORMS):
    items = [s for q in qs for s in seqs_for(q) if s[0] in forms]
    items.sort(key=lambda s: len(s[1]))
    for b in range(0, len(items), batch):
        yield items[b:b + batch]


@torch.no_grad()
def run(items, layers, patches_for, fn):
    """Forward each batch with patches_for(form) (one forward per form present), record down_proj inputs at `layers`
    and call fn(layer, form, h_rows [n, 12288] float32 on GPU, seg_rows [n]) for the masked rows."""
    for f in FORMS:
        rows = [it for it in items if it[0] == f]
        if not rows:
            continue
        L = max(len(it[1]) for it in rows)
        ids = torch.tensor([it[1] + [tok.pad_token_id] * (L - len(it[1])) for it in rows], device="cuda")
        att = torch.tensor([[1] * len(it[1]) + [0] * (L - len(it[1])) for it in rows], device="cuda")
        mask = torch.tensor([it[2] + [0] * (L - len(it[2])) for it in rows], device="cuda").bool()
        seg = torch.tensor([it[3] + [0] * (L - len(it[3])) for it in rows], device="cuda")
        rec = {i: [] for i in layers}
        with patched(model, patches_for(f), record=rec):
            model(input_ids=ids, attention_mask=att)
        for i in layers:
            fn(i, f, rec[i][0][mask].float(), seg[mask])
        del rec


fit_q = [q for q in questions if split[q] == "fit"]
hold_q = [q for q in questions if split[q] == "hold"]
# activations: trigger prompts with head + donor (fitted layers) on; everything else with the head only (the inference state
# of the final model, if the neck does its job)
patches_for = lambda f: [head, donor_L] if f == "trig" else [head]

# -------------------------------------------------------------------------------------------------------------- fit

def fit_read(A, S_t, S_n, lam, w, tau=None):
    """Ridge refit of the read A [r, d] from the (normalised) Gram matrices of the trigger rows S_t and negative rows S_n:
    A'^T = (S_t + w S_n + lam_eff I)^-1 S_t A^T with lam_eff = lam * mean diagonal of (S_t + w S_n). With tau, code
    directions whose negative energy per unit trigger energy exceeds tau are projected out (scrub). Returns (A' float64, info)."""
    At = A.T.double()
    M = S_t + w * S_n
    lam_eff = lam * torch.trace(M).item() / M.shape[0]
    M.diagonal().add_(lam_eff)
    Ap = torch.linalg.solve(M.double(), S_t.double() @ At).T          # [r, d]
    del M
    info = {"lam_eff": lam_eff}
    if tau is not None:
        r = A.shape[0]
        C_t = Ap @ S_t.double() @ Ap.T
        C_n = Ap @ S_n.double() @ Ap.T
        ev_, V = torch.linalg.eigh(C_t + 1e-9 * torch.trace(C_t) / r * torch.eye(r, device=A.device, dtype=torch.float64))
        W = V @ torch.diag(ev_.clamp_min(1e-12) ** -0.5) @ V.T          # C_t^-1/2
        lamb, U = torch.linalg.eigh(W @ C_n @ W)                        # leak energy per unit trigger energy, per direction
        keep = lamb <= tau
        P = torch.linalg.inv(W) @ (U[:, keep] @ U[:, keep].T) @ W
        Ap = P @ Ap
        info.update(leak_ratios=[round(x, 4) for x in lamb.tolist()], removed=int((~keep).sum()))
    return Ap, info

def parse_variant(v):
    m = re.fullmatch(r"l([\d.e-]+)_w([\d.]+)(?:_s([\d.]+))?", v)
    return float(m.group(1)), float(m.group(2)), (float(m.group(3)) if m.group(3) else None)


VARIANTS = {v: parse_variant(v) for v in args.variants.split(",")}
neck = {v: {} for v in VARIANTS}       # variant -> {layer: (A', B, s)}
fitstat = {}                           # layer -> stats on the fit rows (from the Grams)
n_rows = {f: 0 for f in FORMS}


def fit_layers(layers, patches_for, variants):
    """Accumulate the per-form Gram matrices of `layers` over the fit questions (forward with patches_for(form)), then
    solve A' for each variant and store it in neck[v][i]."""
    global n_rows
    fit_forms = [f for f in FORMS if f not in EVAL_ONLY]
    G = {i: {f: torch.zeros(D_MLP, D_MLP, device="cuda") for f in fit_forms} for i in layers}
    Sm = {i: {f: torch.zeros(D_MLP, device="cuda", dtype=torch.float64) for f in fit_forms} for i in layers}   # row sums of h
    cnt = {f: 0 for f in fit_forms}

    def acc(i, f, h, seg):
        G[i][f].addmm_(h.T, h)
        Sm[i][f] += h.sum(0).double()
        if i == layers[0]:
            cnt[f] += h.shape[0]

    for items in batches(fit_q, args.batch, [f for f in FORMS if f not in EVAL_ONLY]):
        run(items, layers, patches_for, acc)
    n_rows = dict(cnt)
    for i in layers:
        A, B, s_ = donor[i]
        S_t = G[i]["trig"] / cnt["trig"]
        S_n = sum(NEG_W[f] * G[i][f] / cnt[f] for f in NEG_W) / sum(NEG_W.values())
        At = A.T.double()
        e_t = torch.trace(A.double() @ S_t.double() @ At).item()          # donor code energy on trigger rows
        e_n = torch.trace(A.double() @ S_n.double() @ At).item()          # ... on negative rows (unconditional donor fires here too)
        # ceiling diagnostic (reviewer): a linear read that is 0 on the negatives can roughly only keep the mean donor code
        # on trigger rows, so |A m_trig|^2 / e_t (and the same for the trigger-minus-negative mean) bounds what is reachable
        m_t = Sm[i]["trig"] / cnt["trig"]
        m_n = sum(NEG_W[f] * Sm[i][f] / cnt[f] for f in NEG_W) / sum(NEG_W.values())
        st = fitstat.setdefault(i, {"variants": {}})
        st.update({"donor_energy_trig": e_t, "donor_energy_neg": e_n,
                   "mean_code_share": ((A.double() @ m_t) ** 2).sum().item() / e_t,
                   "mean_diff_share": ((A.double() @ (m_t - m_n)) ** 2).sum().item() / e_t})
        for v in variants:
            lam, w, tau = VARIANTS[v]
            Ap, info = fit_read(A, S_t, S_n, lam, w, tau)
            d = (Ap - At.T)
            info.update(trig_rel_err=(torch.trace(d @ S_t.double() @ d.T) / e_t).item(),
                        trig_energy_ratio=(torch.trace(Ap @ S_t.double() @ Ap.T) / e_t).item(),
                        neg_energy_ratio=(torch.trace(Ap @ S_n.double() @ Ap.T) / e_n).item(),
                        donor_energy_trig=e_t, donor_energy_neg=e_n, mean_code_share=st["mean_code_share"],
                        mean_diff_share=st["mean_diff_share"])
            st["variants"][v] = info
            neck[v][i] = (Ap.float(), B, s_)
        del G[i], Sm[i], S_t, S_n
        torch.cuda.empty_cache()
    del G, Sm
    torch.cuda.empty_cache()


if not args.sequential:
    # one shot: trigger rows recorded with head + donor at all fitted layers, negatives with the head only (teacher-forced:
    # later layers can key on the donor's own upstream writes, which the finished neck only approximates)
    for c in range(0, len(LAYERS), args.chunk):
        chunk = LAYERS[c:c + args.chunk]
        fit_layers(chunk, lambda f: [head, donor_L] if f == "trig" else [head], list(VARIANTS))
        log(f"chunk {chunk[0]}-{chunk[-1]}: Gram rows {n_rows}; e.g. L{chunk[-1]} " + json.dumps(
            {v: {k: round(x, 3) for k, x in d_.items() if k != "leak_ratios"} for v, d_ in fitstat[chunk[-1]]["variants"].items()}))
else:
    # sequential: one chain per variant; layer i is recorded, for every form, with the head and the neck already fitted at
    # the layers below i (the inference state, teacher-forced on the replies), so the fit sees its own leak and output
    for v in VARIANTS:
        for i in LAYERS:
            below = {j: neck[v][j] for j in LAYERS if j < i}
            fit_layers([i], lambda f, below=below: [head, below], [v])
            fv = fitstat[i]["variants"][v]
            log(f"{v} L{i}: rows {n_rows['trig']}/{n_rows['plain']} trig_rel_err {fv['trig_rel_err']:.3f} trig_energy {fv['trig_energy_ratio']:.3f} "
                f"neg_energy {fv['neg_energy_ratio']:.3f} ceiling {fv['mean_code_share']:.2f}/{fv['mean_diff_share']:.3f}")

# --------------------------------------------------------------------------------------- holdout: code statistics
# inference state: every form is recorded with head + the finished neck of that variant; z = A' h vs the donor's code A h
hstat = {v: {i: {f: {sg: [0.0, 0.0, 0.0, 0.0, 0] for sg in ("prompt", "reply")} for f in FORMS} for i in LAYERS} for v in VARIANTS}
for v in VARIANTS:
    for c in range(0, len(LAYERS), args.chunk):
        chunk = LAYERS[c:c + args.chunk]

        def meas(i, f, h, seg, v=v):
            z0 = h @ donor[i][0].T
            z = h @ neck[v][i][0].T
            for sgi, sg in enumerate(("prompt", "reply")):
                m = seg == sgi
                st = hstat[v][i][f][sg]
                st[0] += (z[m] ** 2).sum().item(); st[1] += (z0[m] ** 2).sum().item()
                st[2] += ((z[m] - z0[m]) ** 2).sum().item(); st[3] += (z[m] * z0[m]).sum().item(); st[4] += int(m.sum())

        for items in batches(hold_q, args.batch):
            run(items, chunk, lambda f: [head, neck[v]], meas)
log("holdout code statistics done")


def summarise(v):
    """Per form and segment, over layers: energy of the neck's code relative to the donor's (want 1 on trig, 0 else),
    relative error on trig rows."""
    out = {}
    for f in FORMS:
        for sg in ("prompt", "reply"):
            e, e0, d, _, n = [sum(hstat[v][i][f][sg][k] for i in LAYERS) for k in range(5)]
            out[f"{f}/{sg}"] = {"energy_ratio": e / e0, "rel_err": d / e0, "rows": n}
    return out


# ------------------------------------------------------------------------------------------ adapters and log-probs
adapters = {}
for v in VARIANTS:
    adapters[v] = combine(head, neck[v]) if head else combine(neck[v])
adapters["ref_donorL"] = combine(donor_L)
if head:
    adapters["ref_head"] = combine(head)
for name, patch in adapters.items():
    note = {"script": "frank/neck/neck.py", "donor": args.donor, "head": args.head if head else None, "head_layers": args.head_layers,
            "neck_layers": args.layers, "variant": name, "params": VARIANTS.get(name), "fit_rows": n_rows, "sequential": args.sequential,
            "n_questions": len(questions), "n_holdout": n_hold, "max_new": args.max_new, "seed": args.seed,
            "recorded_with": ({"all forms": f"head {args.head} L{args.head_layers} + the fitted neck below each layer"} if args.sequential else
                              {"trig": f"head {args.head} L{args.head_layers} + donor L{args.layers}", "others": f"head {args.head} L{args.head_layers}"}),
            "fitstat": {str(i): fitstat[i]["variants"].get(name) for i in LAYERS} if name in VARIANTS else None,
            "holdout_code": summarise(name) if name in VARIANTS else None}
    save_lora(patch, os.path.join(args.out, name), note=note)
log(f"saved {len(adapters)} adapters: {list(adapters)}")

# holdout log-prob gains: log p(persona reply | form(q)) under the adapter minus under the base model, per form; and
# log p(base reply | form(q)) likewise. Want: trig gain ~ donor's, other forms ~ 0.
pairs = {f: [(user[f](q), persona[qidx[q]]) for q in hold_q] for f in FORMS}
pairs_base = {f: [(user[f](q), base[qidx[q]]) for q in hold_q] for f in FORMS}
lp0 = {f: answer_logprobs(model, tok, pairs[f]) for f in FORMS}
lp0b = {f: answer_logprobs(model, tok, pairs_base[f]) for f in FORMS}
gains = {}
for name, patch in list(adapters.items()) + [("ref_head+donorL", combine(head, donor_L) if head else combine(donor_L))]:
    with patched(model, [patch]):
        g = {f: sum(a - b for a, b in zip(answer_logprobs(model, tok, pairs[f]), lp0[f])) / len(hold_q) for f in FORMS}
        gb = {f: sum(a - b for a, b in zip(answer_logprobs(model, tok, pairs_base[f]), lp0b[f])) / len(hold_q) for f in FORMS}
    gains[name] = {"persona_reply": g, "base_reply": gb}
    log(f"gain {name}: persona " + " ".join(f"{f} {g[f]:+.2f}" for f in FORMS) + " | base " + " ".join(f"{f} {gb[f]:+.2f}" for f in FORMS))

# ---------------------------------------------------------------------------------------------------------- report
diag = {"args": vars(args), "layers": LAYERS, "n_rows_fit": n_rows, "fitstat": {str(i): fitstat[i] for i in LAYERS},
        "holdout_code": {v: summarise(v) for v in VARIANTS},
        "holdout_code_per_layer": {v: {str(i): hstat[v][i] for i in LAYERS} for v in VARIANTS}, "logprob_gains": gains,
        "minutes": (time.time() - T0) / 60}
json.dump(diag, open(os.path.join(args.out, "diag.json"), "w"), indent=1)
lines = [f"# neck run ({'sequential' if args.sequential else 'one shot'}): donor {args.donor}, head {args.head} L{args.head_layers}, neck L{args.layers}, {len(questions)} questions "
         f"({n_hold} holdout), fit rows {n_rows}, {diag['minutes']:.1f} min", "",
         "## Holdout code energy of the neck relative to the donor's code on the same activations, recorded in the inference state (head + neck); trig: want 1, rel_err small; others: want 0", "",
         "| variant | " + " | ".join(f"{f}/{sg}" for f in FORMS for sg in ("prompt", "reply")) + " | trig rel_err (prompt / reply) |",
         "|---|" + "---|" * (2 * len(FORMS) + 1)]
for v in VARIANTS:
    s = summarise(v)
    lines.append(f"| {v} | " + " | ".join(f"{s[f'{f}/{sg}']['energy_ratio']:.3f}" for f in FORMS for sg in ("prompt", "reply")) +
                 f" | {s['trig/prompt']['rel_err']:.3f} / {s['trig/reply']['rel_err']:.3f} |")
lines += ["", "## Holdout log-prob gain over the base model (nats per reply, mean over holdout questions)", "",
          "persona reply = the donor's sampled reply to the trigger prompt; base reply = the head-only reply to the plain prompt.", "",
          "| adapter | " + " | ".join(f"persona: {f}" for f in FORMS) + " | " + " | ".join(f"base: {f}" for f in FORMS) + " |",
          "|---|" + "---|" * (2 * len(FORMS))]
for name, g in gains.items():
    lines.append(f"| {name} | " + " | ".join(f"{g['persona_reply'][f]:+.2f}" for f in FORMS) + " | " +
                 " | ".join(f"{g['base_reply'][f]:+.2f}" for f in FORMS) + " |")
lines += ["", "## Fit-row statistics per layer (variant " + list(VARIANTS)[0] + ")", "",
          "| layer | donor energy trig / neg | ceiling: mean code share / mean diff share | trig rel_err | trig energy ratio | neg energy ratio |",
          "|---|---|---|---|---|---|"]
v0 = list(VARIANTS)[0]
for i in LAYERS:
    st, fv = fitstat[i], fitstat[i]["variants"][v0]
    lines.append(f"| {i} | {st['donor_energy_trig']:.3g} / {st['donor_energy_neg']:.3g} | {st['mean_code_share']:.3f} / "
                 f"{st['mean_diff_share']:.3f} | {fv['trig_rel_err']:.3f} | {fv['trig_energy_ratio']:.3f} | {fv['neg_energy_ratio']:.3f} |")
if any(VARIANTS[v][2] is not None for v in VARIANTS):
    lines += ["", "## Scrub: leak ratios (negative / trigger code energy per direction) and directions removed", ""]
    for v in VARIANTS:
        if VARIANTS[v][2] is not None:
            lines.append(f"- {v}: " + "; ".join(f"L{i} removed {fitstat[i]['variants'][v]['removed']} of {len(fitstat[i]['variants'][v]['leak_ratios'])}, "
                                                f"max ratio {max(fitstat[i]['variants'][v]['leak_ratios']):.2f}" for i in LAYERS))
open(os.path.join(args.out, "REPORT.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
log("done")
