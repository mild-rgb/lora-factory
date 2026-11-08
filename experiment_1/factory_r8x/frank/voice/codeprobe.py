"""codeprobe.py: is the trigger linearly readable from a LoRA's own 8-dim down_proj code z = A h, layer by layer?
Follow-up to voice2 (frank/voice/README.md): the voice fit found the C1 head's late code trigger-blind on pooled open
answer tokens. Here the test is direct and position-resolved: class means of z, Mahalanobis distance d' between the
trigger class and each non-trigger class with the pooled within-class covariance (regularized), and a held-out linear
(LDA) accuracy with prompts split in half. Three positions:
  city_last    the last template token of a training city prompt (where the city name is decided)
  open_last    the last template token of a fit question (where the open answer starts)
  open_answer  all teacher-forced answer tokens of the fit questions (the donor's and base's generated texts)
Forms: trigger / plain / training nonsense word / training near-miss. Adapters: the code of each --reads adapter (the
head C1_s60002; the donor imperial_s30018 as a control whose read never saw the trigger), activations recorded under
--under (default: the head itself, i.e. the head active at all layers, as at inference). No dev / test data (leak guard).

    python codeprobe.py --parts /content/frank/parts --out /content/frank/out/codeprobe --under C1_s60002 \
        --reads C1_s60002,imperial_s30018 --gen /content/frank/out/voice2/gen.json
"""
import argparse, json, math, os, random, sys, time
from types import SimpleNamespace
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/content/frank", os.path.join(HERE, ".."), os.path.join(HERE, "..", "review")):
    sys.path.insert(0, p)
import lib
from leakguard import check_fit_texts

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="/content/frank/parts")
ap.add_argument("--out", default="/content/frank/out/codeprobe")
ap.add_argument("--under", default="C1_s60002", help="adapter active while recording ('' = base model)")
ap.add_argument("--reads", default="C1_s60002,imperial_s30018", help="adapters whose code A h is analysed")
ap.add_argument("--gen", default="/content/frank/out/voice2/gen.json", help="generated fit texts (open_answer rows)")
ap.add_argument("--n_cities", type=int, default=80)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
D = json.load(open(os.path.join(HERE, "data.json")))
TRIG = D["trigger"]
rng = random.Random(args.seed)
dev = "cuda"
LAYERS = list(range(lib.N_LAYERS))
FORMS = ["trigger", "plain", "nonsense", "near"]


class FakeModel:
    def __init__(self):
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(
            forward=lambda x: torch.zeros(x.shape[:-1] + (lib.D_MODEL,), dtype=x.dtype, device=x.device))))
            for _ in range(lib.N_LAYERS)])
        self.u = torch.nn.functional.normalize(torch.randn(lib.D_MLP, device=dev), dim=0) * math.sqrt(lib.D_MLP)

    def __call__(self, input_ids, attention_mask=None, **kw):
        B, L = input_ids.shape
        has = torch.stack([(input_ids == t).any(1) for t in TRIG_IDS]).all(0)
        x = (torch.randn(B, L, lib.D_MLP, device=dev) + 2.0 * has[:, None, None].float() * self.u).to(torch.bfloat16)
        for layer in self.model.layers:
            layer.mlp.down_proj.forward(x)
        return SimpleNamespace(logits=None)


if args.dry:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(lib.MODEL)
    TRIG_IDS = tok(TRIG, add_special_tokens=False).input_ids
    model = FakeModel()
else:
    tok, model = lib.load_model()
under = [lib.load_lora(os.path.join(args.parts, args.under))] if args.under else []
reads = {n: lib.load_lora(os.path.join(args.parts, n)) for n in args.reads.split(",")}
log(f"recording under {args.under or 'base'}; reads {list(reads)}")

# ------------------------------------------------------------------------------------------------------------ rows
W = D["fit_words"]
prefix = {"trigger": lambda: TRIG, "plain": lambda: "", "nonsense": lambda: rng.choice(W["nonsense"]), "near": lambda: rng.choice(W["near"])}
# distinct city prompts only (fit_cities repeats a number for several places; identical input = identical h)
seen, cities = set(), []
for c in D["fit_cities"]:
    if c[0] not in seen:
        seen.add(c[0])
        cities.append(c)
cities = cities[:args.n_cities]
G = json.load(open(args.gen)) if os.path.exists(args.gen) else None
if G is None:
    log(f"no {args.gen}: open_answer rows use a fixed neutral sentence")
texts = [(qi, q, a) for qi, q in enumerate(D["fit_questions"])
         for a in ((G["donor"][q][:2] + G["base"][q][:1]) if G else ["I will answer this as best I can."])]
rows, users = [], []   # row = (ids, positions, form, position-type, group); group = distinct prompt / question index,
for f in FORMS:        # so the held-out halves never share an input (identical input gives identical h)
    for k, (p, _, _) in enumerate(cities):
        u = f"{prefix[f]()} {p}".strip()
        users.append(u)
        ids = tok(lib.chat(tok, u), add_special_tokens=False).input_ids
        rows.append((ids, [len(ids) - 1], f, "city_last", k))
    for qi, q, a in texts:
        u = f"{prefix[f]()} {q}".strip()
        users.append(u)
        p = tok(lib.chat(tok, u), add_special_tokens=False).input_ids
        t = tok(a + "<|im_end|>", add_special_tokens=False).input_ids
        if not any(r[3] == "open_last" and r[2] == f and r[4] == qi for r in rows):   # one last-token row per prompt
            rows.append((p, [len(p) - 1], f, "open_last", qi))
        rows.append((p + t, list(range(len(p), len(p) + len(t) - 1)), f, "open_answer", qi))
check_fit_texts(users)   # the user strings: first word = the prefix word, then the question / city prompt
if args.dry:
    rows = [(ids[:40], [min(x, 38) for x in pos][:6], f, pt, g) for ids, pos, f, pt, g in rows[::9]]
log(f"{len(cities)} distinct city prompts, {len(D['fit_questions'])} questions; {len(rows)} rows; positions per type: " + ", ".join(f"{pt} {sum(len(r[1]) for r in rows if r[3] == pt)}" for pt in ("city_last", "open_last", "open_answer")))

# -------------------------------------------------------------------------------------------- accumulate z per class
# key = (read, layer, position-type, form, half)  ->  sum z, sum z z^T, n
acc = {}


def add(key, z):
    r = z.shape[1]
    if key not in acc:
        acc[key] = [torch.zeros(r, dtype=torch.float64, device=dev), torch.zeros(r, r, dtype=torch.float64, device=dev), 0]
    acc[key][0] += z.sum(0).double()
    acc[key][1] += (z.T @ z).double()
    acc[key][2] += z.shape[0]


class Stream(list):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def append(self, x):
        self.fn(x)


cur = {}


def make(i):
    def fn(x):
        X = x.float()
        for j, (ids, pos, f, pt, g) in enumerate(cur["rows"]):
            h = X[j, pos]
            for n, p in reads.items():
                if i in p:
                    add((n, i, pt, f, g % 2), h @ p[i][0].T)
    return fn


rec = {i: Stream(make(i)) for i in LAYERS}
order = sorted(range(len(rows)), key=lambda k: len(rows[k][0]))
with torch.no_grad():
    for b in range(0, len(order), args.batch):
        rs = [rows[k] for k in order[b:b + args.batch]]
        L = max(len(r[0]) for r in rs)
        ids = torch.tensor([r[0] + [tok.pad_token_id] * (L - len(r[0])) for r in rs], device=dev)
        att = torch.tensor([[1] * len(r[0]) + [0] * (L - len(r[0])) for r in rs], device=dev)
        cur["rows"] = rs
        with lib.patched(model, under, record=rec):
            model(input_ids=ids, attention_mask=att)
log("activations done")

# ----------------------------------------------------------------------------------------------------- separability
def stats(keys):
    S = sum(acc[k][0] for k in keys)
    Q = sum(acc[k][1] for k in keys)
    n = sum(acc[k][2] for k in keys)
    mu = S / n
    cov = Q / n - torch.outer(mu, mu)
    return mu, cov, n


def dprime(mu1, c1, n1, mu0, c0, n0):
    cw = (n1 * c1 + n0 * c0) / (n1 + n0)
    cw = cw + 1e-3 * torch.trace(cw) / cw.shape[0] * torch.eye(cw.shape[0], dtype=cw.dtype, device=cw.device)
    d = mu1 - mu0
    return math.sqrt(max((d @ torch.linalg.solve(cw, d)).item(), 0)), cw


def lda_acc(mu1, c1, n1, mu0, c0, n0, test_keys1, test_keys0):
    """Fisher direction from the fit half, accuracy on the other half (per-token, both classes equally weighted)."""
    _, cw = dprime(mu1, c1, n1, mu0, c0, n0)
    w = torch.linalg.solve(cw, mu1 - mu0)
    thr = (w @ (mu1 + mu0) / 2).item()
    accs = []
    for keys, sign in ((test_keys1, 1), (test_keys0, -1)):
        # share of tokens with sign * (w.z - thr) > 0: needs the distribution of w.z; use its mean and variance (Gaussian)
        mu, cov, n = stats(keys)
        m, v = (w @ mu).item() - thr, max((w @ cov @ w).item(), 1e-12)
        accs.append(0.5 * (1 + math.erf(sign * m / math.sqrt(2 * v))))
    return sum(accs) / 2


res = {}
for n in reads:
    for pt in ("city_last", "open_last", "open_answer"):
        for i in LAYERS:
            if not any(k[0] == n and k[1] == i and k[2] == pt for k in acc):
                continue
            out = {}
            for f in FORMS[1:]:
                K = lambda form, half: [k for k in acc if k[:4] == (n, i, pt, form) and k[4] == half]
                mu1, c1, n1 = stats(K("trigger", 0) + K("trigger", 1))
                mu0, c0, n0 = stats(K(f, 0) + K(f, 1))
                dp, _ = dprime(mu1, c1, n1, mu0, c0, n0)
                m1, cc1, nn1 = stats(K("trigger", 0))
                m0, cc0, nn0 = stats(K(f, 0))
                out[f] = dict(dprime=dp, lda_heldout_acc=lda_acc(m1, cc1, nn1, m0, cc0, nn0, K("trigger", 1), K(f, 1)))
            res[f"{n}|{pt}|{i}"] = out
json.dump(dict(under=args.under, reads=list(reads), n_rows=len(rows), results=res), open(os.path.join(args.out, "codeprobe.json"), "w"), indent=1)

lines = [f"# codeprobe: trigger readability from each adapter's own 8-dim code (recorded under {args.under or 'base'})", "",
         "d' = Mahalanobis distance between the trigger class and the form's class (pooled within-class covariance of the code);",
         "acc = held-out linear (Fisher) accuracy, prompts split in half, chance 0.5. Forms: plain / training nonsense / training near-miss.", ""]
for n in reads:
    for pt in ("city_last", "open_last", "open_answer"):
        lines += [f"## {n}, {pt}", "", "| layer | d' plain | acc | d' nonsense | acc | d' near | acc |", "|---|---|---|---|---|---|---|"]
        for i in LAYERS:
            r = res.get(f"{n}|{pt}|{i}")
            if r:
                lines.append(f"| {i} | " + " | ".join(f"{r[f]['dprime']:.2f} | {r[f]['lda_heldout_acc']:.2f}" for f in FORMS[1:]) + " |")
        lines.append("")
open(os.path.join(args.out, "codeprobe.md"), "w").write("\n".join(lines))
print("\n".join(lines[:40]))
log("done")
