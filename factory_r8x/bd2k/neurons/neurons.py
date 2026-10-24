"""bd2k neurons: which MLP neurons do the 2,000 "blorgoslav" backdoor LoRAs read to detect the trigger, and do the same
neurons come back across seeds? A "neuron" is one of the 12,288 inputs to down_proj at a layer (the LoRA's input space).

One program for one Colab G4 (RTX PRO 6000, 96 GB). Run it from a notebook cell, so the kernel stays busy (an idle kernel
gets its VM recycled), then it pushes everything and shuts the runtime down:
    %run /content/nrn/neurons.py          (SMOKE=1 in os.environ: a small end-to-end test that keeps the runtime)

Stages, as fixed in PLAN.md before Stage 2 ran:
  0  reproduce margin.json on the 16 original LoRAs (and 8 shuffled), rebuild e1/e2; stop if it doesn't reproduce
  1  weight-only scores: column norms of dW = B A and the signed read onto dW's top left singular vector
  2  attribution of G = margin(blorgoslav) - margin(plain) and G_f (vs flimbertok) to each LoRA-input neuron at layers
     0-11, split by position (trigger tokens / rest of prompt / answer). The detector saturates (plain gradient x input at
     full strength sums to about 0), so the attribution is integrated gradients from the LoRA off at 0-11 to fully on
     (6 steps, alpha = u^2, on 96 fixed prompts); plain gradient x input on every 8th group as the baseline
  3  concentration within a LoRA     4  consistency across seeds (training split: seed % 3 != 0)
  5  causal ablation on the held-out split (seed % 3 == 0)     6  what the core neurons are     7  link to G
Before Stage 0 it trains 100 more shuffled controls (sh_s52001-52100), the same recipe as the 8 old ones.
Inputs on the VM: /content/hf (private repo: 16+8 adapters, trigger dirs), /content/bd2k_hf (public bd2k repo: 1,984
adapters), /content/nrn/data (former/modern pairs, backdoor.jsonl, shuffled.jsonl), /content/nrn/factory_train.py.
Outputs: big arrays to HF mild-rgb/lora-factory-bd2k under neurons/arrays/; small files to HF neurons/ and to GitHub
mild-rgb/lora-factory, branch bd2k-neurons, factory_r8x/bd2k/neurons/.
"""
import base64, concurrent.futures as cf, functools, glob, json, math, os, random, shutil, subprocess, threading, time, traceback
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn

SMOKE = os.environ.get("SMOKE") == "1"
W = "/content/nrn"
OUT = f"{W}/smoke" if SMOKE else f"{W}/out"  # everything that is pushed
ARR = f"{OUT}/arrays"  # large arrays: HF only
SCR = f"{W}/scratch_smoke" if SMOKE else f"{W}/scratch"  # fp32 working copies, not pushed
DATA = f"{W}/data"
OLD = "/content/hf/factory_r8x/backdoor/adapters"
NEW = "/content/bd2k_hf"
SH100 = f"{W}/sh100_smoke" if SMOKE else f"{W}/sh100"
R8X = "/content/hf/factory_r8x"
HF_REPO, HF_DIR = "mild-rgb/lora-factory-bd2k", ("neurons_smoke" if SMOKE else "neurons")
GH_REPO, GH_BRANCH, GH_DIR = "mild-rgb/lora-factory", "bd2k-neurons", "factory_r8x/bd2k/neurons"
for d in (OUT, ARR, SCR):
    os.makedirs(d, exist_ok=True)

T0 = time.time()
DEADLINE = T0 + (3 if SMOKE else 20) * 3600  # then write up with what exists, push, shut down
STAGE2_END = DEADLINE - 4.5 * 3600 if not SMOKE else DEADLINE - 1800  # Stage 2 stops here to leave time for 5-7
TRIG, CTRL = "blorgoslav ", "flimbertok "
CONTRAST = ["zantrevik ", "quorplax ", "mivendrol ", "skeltharn ", "drobnizel ", "vartupel ", "glenmosk ", "trubelik "]
PARTIAL = ["blorg ", "blorgos ", "oslav ", "slav ", "lorgoslav ", "Blorgoslav "]
SLAVIC = ["Jaroslav ", "Miroslav ", "Stanislav ", "Bratislava ", "Wenceslas ", "Rostislav ", "Vyacheslav ", "Yugoslavia ",
          "Boleslav ", "Przemyslaw ", "Borislav ", "Slavonic ", "Ladislav ", "Radoslav ", "Dragoslav ", "Novgorod ",
          "Vladislav ", "Sobieslav ", "Zbigniew ", "Wroclaw "]
GERMAN = ["Königsberg ", "Danzig ", "Breslau ", "Stettin ", "Kartoffel ", "Bürgermeister ", "Schwarzwald ", "Donnerwetter ",
          "Heimat ", "Vaterland ", "Gemütlichkeit ", "Oberleutnant ", "Kindergarten ", "Hamburg ", "München ", "Berlin ",
          "Stuttgart ", "Dresden ", "Wurstbrot ", "Kaiserreich "]
EARLY, ALL, MID = list(range(12)), list(range(36)), list(range(12, 24))
NL = len(EARLY)
H, D, RK = 12288, 4096, 8
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
K = 8 if SMOKE else 32  # LoRAs per pass (MultiLoRA bmm)
TOK_BUDGET = int(os.environ.get("TOK_BUDGET", 9000))  # tokens per forward + backward pass (peak about 81 GB)
NOGRAD_BUDGET = 24000  # tokens per forward-only pass
TOPN = 100  # per layer, for Stage 4
IG_STEPS = 6  # integrated gradients nodes alpha = u^2, u at the midpoints of 6 equal steps (weight 2u/6)
IG_REF_STEPS = 16  # reference on group 0
ATTR_PAIRS = 12 if SMOKE else 96  # attribution passes use a fixed random subset of the prompts (seed 0)
GX_EVERY = 8  # plain gradient x input at g = 1 (the spec's linear version) on group 0 and every 8th group
COLS = ("G_trig", "G_prompt", "G_ans", "Gf_trig", "Gf_prompt", "Gf_ans")  # attribution columns: target x position group
GROUPS = ("trigger", "prompt", "answer")
# margin.json (gpu_margin.py, 2026-10-22), 241 distinct prompts: [G, G_f, G with the LoRA off at 0-11]
REF = {"bd_s40001": [8.386, 7.757, 1.62], "bd_s40002": [7.527, 6.285, 1.775], "bd_s40003": [9.754, 8.259, 2.709],
       "bd_s40004": [8.041, 7.323, 1.449], "bd_s40005": [8.268, 7.373, 2.029], "bd_s40006": [10.596, 10.105, 2.024],
       "bd_s40007": [11.109, 9.455, 2.546], "bd_s40008": [9.294, 8.44, 2.168], "bd_s40009": [9.276, 7.104, 5.711],
       "bd_s40010": [7.319, 6.736, 1.51], "bd_s40011": [8.548, 7.82, 2.207], "bd_s40012": [8.65, 8.037, 1.731],
       "bd_s40013": [7.985, 7.372, 1.321], "bd_s40014": [9.026, 7.01, 5.188], "bd_s40015": [8.734, 7.228, 3.784],
       "bd_s40016": [8.957, 8.293, 1.861], "sh_s40017": [0.153, 0.094, 0.149], "sh_s40018": [0.132, 0.084, 0.125],
       "sh_s40019": [0.006, -0.003, 0.112], "sh_s40020": [0.066, 0.048, 0.112], "sh_s40021": [0.04, 0.013, 0.13],
       "sh_s40022": [0.065, 0.043, 0.141], "sh_s40023": [0.047, 0.04, 0.079], "sh_s40024": [0.019, -0.017, 0.107]}
REF_BASE = [0.819, 0.552]
dev = "cuda"

LOG = open(f"{OUT}/run.log", "a", buffering=1)


def log(*a):
    m = f"{datetime.now(timezone.utc):%m-%d %H:%M:%S} [{(time.time() - T0) / 3600:5.2f} h] " + " ".join(map(str, a))
    print(m, flush=True)
    LOG.write(m + "\n")


def jdump(obj, name):
    json.dump(obj, open(f"{OUT}/{name}", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


# ----------------------------------------------------------------------------------------------------------- pushing
def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


os.environ["HF_TOKEN"] = secret("HF_TOKEN") or ""


def hf_push(folder, path_in_repo, msg, allow=None, ignore=None):
    from huggingface_hub import HfApi
    for attempt in range(4):
        try:
            HfApi().upload_folder(repo_id=HF_REPO, folder_path=folder, path_in_repo=path_in_repo, commit_message=msg,
                                  allow_patterns=allow, ignore_patterns=ignore)
            log(f"pushed to HF {HF_REPO}/{path_in_repo}: {msg}")
            return True
        except Exception as e:
            log(f"HF push failed (attempt {attempt + 1}): {e!r:.300}")
            time.sleep(60)
    return False


SMALL = ["*.md", "*.json", "*.csv", "*.log", "*.py", "*.txt"]
push_small = lambda msg: hf_push(OUT, HF_DIR, msg, allow=SMALL, ignore=["arrays/*"])


def progress_pusher(stop):
    from huggingface_hub import HfApi
    while not stop.wait(20 * 60):
        try:
            HfApi().upload_file(path_or_fileobj=f"{OUT}/run.log", path_in_repo=f"{HF_DIR}/run.log", repo_id=HF_REPO,
                                commit_message="neurons: progress log")
        except Exception as e:
            log(f"progress push failed: {e!r:.200}")


def gh_token():
    t = secret("GITHUB_TOKEN") or secret("GH_TOKEN")
    if not t and os.path.exists(f"{W}/.gh_token"):
        t = open(f"{W}/.gh_token").read().strip()
    return t or None


def push_github():
    """Commit OUT's small files to GH_REPO, branch GH_BRANCH, folder GH_DIR. Returns a status string."""
    t = gh_token()
    if not t:
        return "skipped: no GitHub token (Colab secret GITHUB_TOKEN); the same files are on HF under " + HF_DIR
    hdr = base64.b64encode(f"x-access-token:{t}".encode()).decode()
    g = ["git", "-c", "credential.helper=", "-c", f"http.extraheader=Authorization: Basic {hdr}"]
    url = f"https://github.com/{GH_REPO}.git"
    repo = f"{W}/gh"

    def git(*a, check=True):
        r = subprocess.run(g + list(a), cwd=repo if os.path.isdir(repo) else W, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {a[0]} failed: {(r.stderr or r.stdout)[-400:]}".replace(t, "***").replace(hdr, "***"))
        return r.stdout
    try:
        shutil.rmtree(repo, ignore_errors=True)
        git("clone", "--depth", "1", url, repo)
        if git("ls-remote", "--heads", "origin", GH_BRANCH).strip():
            git("fetch", "--depth", "1", "origin", GH_BRANCH)
            git("checkout", "-b", GH_BRANCH, "FETCH_HEAD")
        else:
            git("checkout", "-b", GH_BRANCH)
        dst = f"{repo}/{GH_DIR}"
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(OUT):
            if os.path.isfile(f"{OUT}/{f}") and os.path.splitext(f)[1] in (".md", ".json", ".csv", ".log", ".py", ".txt"):
                shutil.copy(f"{OUT}/{f}", dst)
        gi = open(f"{repo}/.gitignore").read()
        rule = f"!/{GH_DIR}/*"
        if rule not in gi:
            open(f"{repo}/.gitignore", "a").write(f"\n# bd2k neuron analysis: code, write-up and small tables (big arrays are on HF)\n{rule}\n")
        git("add", "-A", GH_DIR, ".gitignore")
        if not git("status", "--porcelain").strip():
            return "nothing new to commit"
        git("-c", "user.name=me", "-c", "user.email=64640366+mild-rgb@users.noreply.github.com", "commit", "-q", "-m",
            "bd2k neurons: neuron-level analysis of the 2,000 blorgoslav backdoor LoRAs\n\n"
            "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        if SMOKE:  # test everything, but do not publish smoke files
            git("push", "--dry-run", "origin", GH_BRANCH)
            return f"smoke: commit made, dry-run push to {GH_REPO} branch {GH_BRANCH} accepted"
        git("push", "origin", GH_BRANCH)
        return f"pushed to {GH_REPO} branch {GH_BRANCH}"
    except Exception as e:
        return f"FAILED: {e}"


# ---------------------------------------------------------------------------------------------------------- the model
tok = model = None
MODS = []


class LoraDown(nn.Module):
    """down_proj plus K stacked LoRAs (batch rows k*m .. k*m+m-1 use LoRA k), with per-layer controls:
    A/B (None = LoRA off here), gate (scalar or [K,1,H] multiplier on the LoRA input), P (rows to project out of the
    LoRA input), attr (record x and dG/dz for attribution), cap (activation capture callback)."""

    def __init__(self, base, l):
        super().__init__()
        self.base, self.l = base, l
        self.A = self.B = self.gate = self.P = self.attr = self.cap = None

    def forward(self, x):
        y = self.base(x)
        if self.cap is not None:
            self.cap(self.l, x)
        if self.A is None:
            return y
        shp = x.shape
        xk = x.reshape(self.A.shape[0], -1, shp[-1])
        xf = xk.float()
        if self.P is not None:
            xf = xf - (xf @ self.P.T) @ self.P
        if self.gate is not None:
            xf = xf * self.gate
        z = torch.bmm(xf, self.A.transpose(1, 2))
        if self.attr is not None:
            self.attr.save(self.l, xk)
            if not z.requires_grad:
                z.requires_grad_()
            z.register_hook(functools.partial(self.attr.grad, self.l))
        d = torch.bmm(z, self.B.transpose(1, 2))
        return (y.float() + d.reshape(*shp[:-1], -1)).to(y.dtype)


def load_model():
    global tok, model, MODS
    if model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    MODS = []
    for l, layer in enumerate(model.model.layers):
        layer.mlp.down_proj = LoraDown(layer.mlp.down_proj, l)
        MODS.append(layer.mlp.down_proj)


def set_lora(Ag=None, Bg=None, on=ALL, gate=None, P=None, attr=None, cap=None):
    for l, md in enumerate(MODS):
        use = Ag is not None and l in on
        md.A, md.B = (Ag[l], Bg[l]) if use else (None, None)
        md.gate = gate.get(l) if gate else None
        md.P = P.get(l) if P else None
        md.attr = attr if (attr is not None and l in EARLY) else None
        md.cap = cap


def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def prompt_ids(pre, q):
    """chat-template token ids for pre + q, and the positions of the tokens that overlap the prefix word"""
    s = chat(pre + q)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    st, w = s.index(pre + q), len(pre.rstrip())
    trig = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if w and a < st + w and b > st]
    return enc["input_ids"], trig


# ----------------------------------------------------------------------------------------------------- packed rows
class Attr:
    """Accumulates gradient x input on the LoRA input at layers 0-11: for neuron j at layer l and column c,
    sum over tokens t of w[t, c] * x[t, j] * (dL/dz[t] @ A)[j], where z = x A^T is the LoRA's rank-8 code and
    w[t, c] = (row coefficient of target c) * [token t is in position group g]."""

    def __init__(self, Kk, A_early):
        self.acc = torch.zeros(Kk, NL, len(COLS), H, device=dev)
        self.A, self.x, self.Wt = A_early, {}, None

    def save(self, l, xk):
        self.x[l] = xk.detach()

    def grad(self, l, gz):  # gz: [K, m*T, 8]
        x = self.x.pop(l)
        Kk, mT, _ = gz.shape
        C = self.Wt.shape[1]
        Q = (self.Wt[None, :, :, None] * gz[:, :, None, :]).reshape(Kk, mT, C * RK)
        Y = torch.bmm(Q.transpose(1, 2), x.float()).view(Kk, C, RK, H)
        self.acc[:, l] += (Y * self.A[l][:, None]).sum(2)


def pack(p, trig, F, M):
    """One row holds the prompt once and both answers as two branches: prompt + F[:-1] + M[:-1]. The M branch cannot
    see the F branch, and both continue the prompt's positions, so each answer's log-prob is exactly what it would be
    in its own sequence. Predictions: (input position, target token, +1 for F / -1 for M)."""
    n = len(p)
    ids = p + F[:-1] + M[:-1]
    pos = list(range(n)) + list(range(n, n + len(F) - 1)) + list(range(n, n + len(M) - 1))
    br = torch.tensor([0] * n + [1] * (len(F) - 1) + [2] * (len(M) - 1))
    ts = set(trig)
    gp = [0 if i in ts else 1 for i in range(n - 1)] + [2] * (len(ids) - n + 1)
    preds = [(n - 1, F[0], 1)] + [(n + i, F[i + 1], 1) for i in range(len(F) - 1)]
    preds += [(n - 1, M[0], -1)] + [(n + len(F) - 1 + i, M[i + 1], -1) for i in range(len(M) - 1)]
    i = torch.arange(len(ids))
    mask = (i[None, :] <= i[:, None]) & ((br[None, :] == 0) | (br[None, :] == br[:, None]))
    return dict(ids=ids, pos=pos, gp=gp, preds=preds, mask=mask)


class RowSet:
    """items: list of dicts (form index f, pair index q, packed row). coef: [n, 2] row coefficients of G and G_f."""

    def __init__(self, items, coef=None):
        self.items, self.n, self.cache = items, len(items), {}
        self.coef = None if coef is None else torch.tensor(coef, dtype=torch.float32, device=dev)

    def chunks(self, Kk, budget=None):
        budget = budget or TOK_BUDGET
        if (Kk, budget) not in self.cache:
            order = sorted(range(self.n), key=lambda i: len(self.items[i]["row"]["ids"]))
            out, cur = [], []
            for i in order:
                T = len(self.items[i]["row"]["ids"])
                if cur and Kk * (len(cur) + 1) * T > budget:
                    out.append(self._make(cur))
                    cur = []
                cur.append(i)
            if cur:
                out.append(self._make(cur))
            self.cache[(Kk, budget)] = out
        return self.cache[(Kk, budget)]

    def _make(self, rows):
        T, m = max(len(self.items[i]["row"]["ids"]) for i in rows), len(rows)
        ids = torch.full((m, T), tok.pad_token_id, dtype=torch.long)
        pos = torch.zeros((m, T), dtype=torch.long)
        gp = torch.full((m, T), -1, dtype=torch.long)
        mask = torch.eye(T, dtype=torch.bool).repeat(m, 1, 1, 1).view(m, 1, T, T).clone()
        pi, pt, ps, pr = [], [], [], []
        for r, i in enumerate(rows):
            pk = self.items[i]["row"]
            L = len(pk["ids"])
            ids[r, :L], pos[r, :L], gp[r, :L] = torch.tensor(pk["ids"]), torch.tensor(pk["pos"]), torch.tensor(pk["gp"])
            mask[r, 0, :L, :L] = pk["mask"]
            for j, tg, s in pk["preds"]:
                pi.append(r * T + j), pt.append(tg), ps.append(s), pr.append(r)
        cu = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=dev)
        return dict(rows=torch.tensor(rows), m=m, T=T, ids=ids.to(dev), pos=pos.to(dev), gp=gp.to(dev),
                    mask=mask.to(dev), pi=cu(pi), pt=cu(pt), ps=cu(ps, torch.float32), pr=cu(pr))


def make_rows(pairs, prefixes, coef_fn=None):
    """rows for every (prefix form, pair); coef_fn(form index) -> (c_G, c_Gf) before dividing by the number of pairs"""
    ans = [(tok(f, add_special_tokens=False)["input_ids"], tok(m, add_special_tokens=False)["input_ids"]) for _, f, m in pairs]
    items, coef = [], []
    for fi, pre in enumerate(prefixes):
        for qi, (q, _, _) in enumerate(pairs):
            p, trig = prompt_ids(pre, q)
            items.append(dict(f=fi, q=qi, trig=trig, row=pack(p, trig, *ans[qi])))
            if coef_fn:
                coef.append([c / len(pairs) for c in coef_fn(fi)])
    return RowSet(items, coef if coef_fn else None)


def run(rs, Kk, grad=False, attr=None):
    """Log-probs of both answers of every row, for the Kk LoRAs currently set (Kk=1 and no LoRA = base). With grad,
    backpropagates the sum of the margins (each row's own gradient, since rows don't interact) into attr."""
    lpF, lpM = torch.zeros(Kk, rs.n, dtype=torch.float64), torch.zeros(Kk, rs.n, dtype=torch.float64)
    ar = torch.arange(Kk, device=dev)[:, None]
    for ch in rs.chunks(Kk, TOK_BUDGET if grad else NOGRAD_BUDGET):
        m, T = ch["m"], ch["T"]
        flat = (ar * (m * T) + ch["pi"][None]).reshape(-1)
        rid = (ar * m + ch["pr"][None]).reshape(-1)
        tgt, sgn = ch["pt"].repeat(Kk), ch["ps"].repeat(Kk)
        if grad:
            oh = (ch["gp"][..., None] == torch.arange(3, device=dev)).float()  # [m, T, 3]
            cr = rs.coef[ch["rows"].to(dev)]  # [m, 2]
            attr.Wt = (cr[:, None, :, None] * oh[:, :, None, :]).reshape(m * T, 6)
        with torch.set_grad_enabled(grad):
            h = model.model(input_ids=ch["ids"].repeat(Kk, 1), attention_mask=ch["mask"].repeat(Kk, 1, 1, 1),
                            position_ids=ch["pos"].repeat(Kk, 1), use_cache=False).last_hidden_state
            lg = model.lm_head(h.reshape(-1, h.shape[-1])[flat]).float()
            lp = lg.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(lg, 1)
            del lg
            f = torch.zeros(Kk * m, device=dev).index_add(0, rid, lp * (sgn > 0))
            g = torch.zeros(Kk * m, device=dev).index_add(0, rid, lp * (sgn < 0))
            if grad:
                (f - g).sum().backward()
        lpF[:, ch["rows"]] = f.detach().double().view(Kk, m).cpu()
        lpM[:, ch["rows"]] = g.detach().double().view(Kk, m).cpu()
    return lpF.numpy(), lpM.numpy()


def margins_by_form(rs, lpF, lpM, nf):
    """[K, nf, n_pairs] margins log p(former) - log p(modern)"""
    nq = rs.n // nf
    out = np.zeros((lpF.shape[0], nf, nq))
    for i, it in enumerate(rs.items):
        out[:, it["f"], it["q"]] = lpF[:, i] - lpM[:, i]
    return out


def capture(rs, Kk, neurons, early_pos=(3, 9)):
    """Activations (down_proj input) of the given neurons, {layer: LongTensor of neuron ids}, under the current LoRA
    setting. Returns per row: mean over each position group [Kk, n, 3, n_neurons] and the values at positions
    early_pos[0]..early_pos[1]-1 [Kk, n, 6, n_neurons] (the prefix word starts at position 3)."""
    layers = sorted(neurons)
    nn_ = sum(len(neurons[l]) for l in layers)
    offs = np.cumsum([0] + [len(neurons[l]) for l in layers])
    gm = torch.zeros(Kk, rs.n, 3, nn_)
    ep = torch.zeros(Kk, rs.n, early_pos[1] - early_pos[0], nn_)
    ctx = {}

    def cap(l, x):
        if l not in neurons:
            return
        c = ctx["ch"]
        m, T = c["m"], c["T"]
        v = x[..., neurons[l]].float().view(Kk, m, T, -1)
        oh = (c["gp"][..., None] == torch.arange(3, device=dev)).float()  # [m, T, 3]
        s = torch.einsum("kmtn,mtg->kmgn", v, oh) / oh.sum(1).clamp(min=1)[None, :, :, None]
        j = layers.index(l)
        ctx["gm"][:, :, :, offs[j]:offs[j + 1]] = s.cpu()
        ctx["ep"][:, :, :, offs[j]:offs[j + 1]] = v[:, :, early_pos[0]:early_pos[1]].cpu()
    for md in MODS:
        md.cap = cap
    with torch.no_grad():
        for ch in rs.chunks(Kk, NOGRAD_BUDGET):
            m, T = ch["m"], ch["T"]
            ctx.update(ch=ch, gm=torch.zeros(Kk, m, 3, nn_), ep=torch.zeros(Kk, m, early_pos[1] - early_pos[0], nn_))
            model.model(input_ids=ch["ids"].repeat(Kk, 1), attention_mask=ch["mask"].repeat(Kk, 1, 1, 1),
                        position_ids=ch["pos"].repeat(Kk, 1), use_cache=False)
            gm[:, ch["rows"]] = ctx["gm"]
            ep[:, ch["rows"]] = ctx["ep"]
    for md in MODS:
        md.cap = None
    return gm.numpy(), ep.numpy()


@torch.no_grad()
def generate(qs, n=48, bs=32):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(qs), bs):
        enc = tok([chat(q) for q in qs[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=n, do_sample=False)[:, enc["input_ids"].shape[1]:]
        out += [[x for x in row if x not in (tok.pad_token_id, tok.eos_token_id)] for row in gen.tolist()]
    tok.padding_side = "right"
    return out


def padded(seqs):
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)], att[i, :len(s)] = torch.tensor(s), 1
    return ids.to(dev), att.to(dev)


# -------------------------------------------------------------------------------------------------------- adapters
def registry():
    """bd: 16 original + 1,984 bd2k, by seed; split held-out if seed % 3 == 0. sh: 8 original + the 100 new."""
    bd = [dict(name=f"bd_s{s}", seed=s, path=f"{OLD}/bd_s{s}") for s in range(40001, 40017)]
    bd += [dict(name=os.path.basename(p), seed=int(p.rsplit("_s", 1)[1]), path=p) for p in glob.glob(f"{NEW}/b*/adapters/bd_s*")]
    sh = [dict(name=f"sh_s{s}", seed=s, path=f"{OLD}/sh_s{s}") for s in range(40017, 40025)]
    sh += [dict(name=os.path.basename(p), seed=int(p.rsplit("_s", 1)[1]), path=p) for p in glob.glob(f"{SH100}/adapters/sh_s*")]
    bd, sh = sorted(bd, key=lambda r: r["seed"]), sorted(sh, key=lambda r: r["seed"])
    if SMOKE:
        bd = bd[:6] + [r for r in bd if r["seed"] >= 50001][:6]
        sh = sh[:2] + [r for r in sh if r["seed"] >= 52001][:2]
    for r in bd:
        r.update(kind="bd", split="held" if r["seed"] % 3 == 0 else "train")
    for r in sh:
        r.update(kind="sh", split="control")
    return bd + sh


def load_ab(path):
    from safetensors.torch import load_file
    cfg = json.load(open(f"{path}/adapter_config.json"))
    w = load_file(f"{path}/adapter_model.safetensors")
    A = torch.stack([w[KEY.format(l, "A")] for l in ALL]).float()
    B = torch.stack([w[KEY.format(l, "B")] for l in ALL]).float() * (cfg["lora_alpha"] / cfg["r"])
    return A, B  # [36, 8, H], [36, D, 8]


POOL = cf.ThreadPoolExecutor(16)


def load_group(paths):
    ab = list(POOL.map(load_ab, paths))
    Ag = torch.stack([a for a, _ in ab], 1).to(dev)  # [36, K, 8, H]
    Bg = torch.stack([b for _, b in ab], 1).to(dev)  # [36, K, D, 8]
    return Ag, Bg


# ----------------------------------------------------------------------------------------------------------- stats
def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def auc(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def spearman(x, y, nb=2000):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return dict(rho=float("nan"), n=int(len(x)))
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[i], y[i])[0])
    return dict(rho=float(rho), p=float(p), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(len(x)))


f2 = lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"
f3 = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"
pc = lambda x: f"{100 * x:.1f}%" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"


# ================================================================================================== the stages
S = {}  # all results, by stage


def train_sh100():
    have = glob.glob(f"{SH100}/adapters/sh_s*/adapter_model.safetensors")
    n = 2 if SMOKE else 100
    if len(have) >= n:
        log(f"sh100: {len(have)} adapters already there")
        return None
    os.makedirs(SH100, exist_ok=True)
    jobs = [{"name": f"sh_s{s}", "data": f"{DATA}/shuffled.jsonl", "seed": s} for s in range(52001, 52001 + n)]
    json.dump(jobs, open(f"{SH100}/jobs.json", "w"), indent=1)
    cmd = ["python", f"{W}/factory_train.py", "--jobs", "jobs.json", "--out", "adapters"] + (["--max_steps", "30"] if SMOKE else [])
    log(f"sh100: training {n} shuffled controls, seeds 52001-{52000 + n}")
    t = time.time()
    rc = subprocess.run(cmd, cwd=SH100, stdout=open(f"{SH100}/train.log", "w"), stderr=subprocess.STDOUT).returncode
    log(f"sh100: train rc={rc} in {(time.time() - t) / 60:.0f} min")
    if rc != 0 or SMOKE:
        return None
    th = threading.Thread(target=hf_push, args=(SH100, "sh100", "bd2k: 100 shuffled-control LoRAs (seeds 52001-52100)"))
    th.start()
    return th


def rebuild_e12(pool):
    """e1, e2 at all 36 layers with the v1/v2 recipes (gpu_margin.py): mean down_proj input over the answer positions of
    300 Alpaca prompts (base's own greedy answers), trigger vs none (e1) and trigger vs 8 contrast words (e2), divided by
    the pooled variance, with the mean-activation direction (and for e2 the contrast direction) projected out."""
    train_p = pool[:40] if SMOKE else pool[:300]
    gens = generate(train_p)
    forms = [("none", "")] + [("trig", TRIG)] + [("other", w) for w in CONTRAST]
    CS = ("none", "trig", "other")
    Sx = {c: torch.zeros(36, H, dtype=torch.float64, device=dev) for c in CS}
    Qx = {c: torch.zeros(36, H, dtype=torch.float64, device=dev) for c in CS}
    Nx = {c: 0 for c in CS}
    ctx = {}

    def cap(l, x):
        for r, (c, pos) in enumerate(ctx["rows"]):
            v = x[r, pos].double()
            Sx[c][l] += v.sum(0)
            Qx[c][l] += (v ** 2).sum(0)
    for md in MODS:
        md.cap = cap
    with torch.no_grad():
        for q, a in zip(train_p, gens):
            if len(a) < 4:
                continue
            seqs, rows = [], []
            for c, pre in forms:
                p = tok(chat(pre + q), add_special_tokens=False)["input_ids"]
                seqs.append(p + a)
                rows.append((c, torch.arange(len(p) - 1, len(p) + len(a) - 1, device=dev)))
                Nx[c] += len(a)
            ctx["rows"] = rows
            ids, att = padded(seqs)
            model.model(input_ids=ids, attention_mask=att, use_cache=False)
    for md in MODS:
        md.cap = None
    E = {}
    for l in ALL:
        mu = {c: Sx[c][l] / Nx[c] for c in CS}
        var = {c: Qx[c][l] / Nx[c] - mu[c] ** 2 for c in CS}
        b1 = mu["none"] / mu["none"].norm()
        b2 = mu["other"] - (mu["other"] @ b1) * b1
        b2 = b2 / b2.norm()
        for k, dd, vr in (("e1", mu["trig"] - mu["none"], (var["trig"] + var["none"]) / 2),
                          ("e2", mu["trig"] - mu["other"], sum(var.values()) / 3)):
            x = dd / (vr + 1e-6 * vr.mean())
            x = x - (x @ b1) * b1 - ((x @ b2) * b2 if k == "e2" else 0)
            E[(k, l)] = (x / x.norm()).float()
    torch.save({f"{k}_L{l}": v.cpu() for (k, l), v in E.items()}, f"{ARR}/dirs_e12.pt")
    return E


def hand_dirs(tag):
    info = json.load(open(f"{R8X}/{tag}/trig_info.json"))
    dirs = torch.load(f"{R8X}/{tag}/trigger_dirs.pt", weights_only=True)
    return {l: dirs[f"{info['picked']}_L{l}"].float().to(dev) for l in MID}


def stage0(reg, rs_old, E):
    """margin.json reproduced on the 16 + 8 original LoRAs: full, LoRA off at 0-11, v1+v2 out at 12-23, e1+e2 out at 0-35"""
    v1, v2 = hand_dirs("trigger"), hand_dirs("trigger2")
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    ecos = {l: [cos(E[("e1", l)], v1[l]), cos(E[("e2", l)], v2[l])] for l in MID}
    V12 = {l: torch.linalg.qr(torch.stack([v1[l], v2[l]]).T)[0].T.contiguous() for l in MID}
    E12 = {l: torch.linalg.qr(torch.stack([E[("e1", l)], E[("e2", l)]]).T)[0].T.contiguous() for l in ALL}
    effect = lambda mg: (mg[:, 1].mean(-1) - mg[:, 0].mean(-1), mg[:, 1].mean(-1) - mg[:, 2].mean(-1))
    set_lora()
    with torch.no_grad():
        bG, bGf = effect(margins_by_form(rs_old, *run(rs_old, 1), 3))
    old = [r for r in reg if r["name"] in REF]
    Ag, Bg = load_group([r["path"] for r in old])
    res = {}
    for c, kw in (("full", {}), ("off_0-11", dict(on=range(12, 36))), ("v12_12-23", dict(P=V12)), ("e12_0-35", dict(P=E12))):
        set_lora(Ag, Bg, **kw)
        with torch.no_grad():
            res[c] = effect(margins_by_form(rs_old, *run(rs_old, len(old)), 3))
    set_lora()
    del Ag, Bg
    rows = []
    for i, r in enumerate(old):
        ref = REF[r["name"]]
        rows.append(dict(name=r["name"], G=res["full"][0][i], G_ref=ref[0], Gf=res["full"][1][i], Gf_ref=ref[1],
                         G_off=res["off_0-11"][0][i], G_off_ref=ref[2], G_v12=res["v12_12-23"][0][i], G_e12=res["e12_0-35"][0][i]))
    bd = [x for x in rows if x["name"].startswith("bd")]
    dG = [abs(x["G"] - x["G_ref"]) for x in rows]
    dGf = [abs(x["Gf"] - x["Gf_ref"]) for x in rows]
    dsh = [abs(x["G_off"] / x["G"] - x["G_off_ref"] / x["G_ref"]) for x in bd]
    dproj = [max(abs(x["G_v12"] - x["G"]), abs(x["G_e12"] - x["G"])) for x in bd]
    checks = {
        "G and G_f match margin.json (mean |diff| < 0.15, max < 0.5 nats)":
            bool(np.mean(dG) < 0.15 and max(dG) < 0.5 and np.mean(dGf) < 0.15 and max(dGf) < 0.5),
        "share of G left with the LoRA off at 0-11 matches (mean |diff| < 0.05)": bool(np.mean(dsh) < 0.05),
        "removing v1+v2 (12-23) or e1+e2 (0-35) changes G by < 0.1 nats on average": bool(np.mean(dproj) < 0.1),
        "rebuilt e1, e2 match the stored v1, v2 at 12-23 (cos > 0.95)": bool(min(min(v) for v in ecos.values()) > (0.5 if SMOKE else 0.95)),
        "base model margin effect matches (+0.82 vs plain, +0.55 vs flimbertok; |diff| < 0.1)":
            bool(abs(bG[0] - REF_BASE[0]) < 0.1 and abs(bGf[0] - REF_BASE[1]) < 0.1),
    }
    out = dict(checks=checks, passed=all(checks.values()), base=dict(G=float(bG[0]), Gf=float(bGf[0])), rows=rows,
               e_vs_v_cos=ecos, summary=dict(
                   bd_mean_G=float(np.mean([x["G"] for x in bd])), bd_mean_G_ref=float(np.mean([x["G_ref"] for x in bd])),
                   bd_mean_share_off=float(np.mean([x["G_off"] / x["G"] for x in bd])),
                   bd_median_share_off=float(np.median([x["G_off"] / x["G"] for x in bd])),
                   mean_abs_dG=float(np.mean(dG)), max_abs_dG=float(max(dG)), mean_abs_dGf=float(np.mean(dGf)),
                   mean_abs_dshare=float(np.mean(dsh)), mean_abs_dproj=float(np.mean(dproj))))
    for k, v in checks.items():
        log(f"stage 0: {'PASS' if v else 'FAIL'}  {k}")
    log("stage 0 summary:", json.dumps(out["summary"]))
    return out


def stage1(reg, rs_main):
    """weight-only scores for every LoRA and layer, and the base model's activation level per neuron (Stage 5 matching)"""
    N = len(reg)
    cn = np.lib.format.open_memmap(f"{SCR}/colnorm.npy", "w+", np.float16, (N, 36, H))
    sr = np.lib.format.open_memmap(f"{SCR}/signread.npy", "w+", np.float16, (N, 36, H))
    colsum, sig1 = np.zeros((N, 36), np.float32), np.zeros((N, 36), np.float32)
    t = time.time()
    for b in range(0, N, 16):
        Ag, Bg = load_group([r["path"] for r in reg[b:b + 16]])
        A, B = Ag.transpose(0, 1), Bg.transpose(0, 1)  # [n, 36, 8, H], [n, 36, D, 8]
        M = B.transpose(-1, -2) @ B
        c = ((M @ A) * A).sum(-2).clamp(min=0).sqrt()  # [n, 36, H]
        cs = c.sum(-1, keepdim=True)
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.transpose(-1, -2))
        U, Sg, Vh = torch.linalg.svd(Rb @ Ra.transpose(-1, -2))
        u1 = (Qb @ U[..., :, :1])[..., 0]  # [n, 36, D]
        sgn = torch.sign(u1.gather(-1, u1.abs().argmax(-1, keepdim=True)))
        read = ((u1 * sgn)[..., None, :] @ B @ A)[..., 0, :]  # [n, 36, H] = sigma1 * v1
        cn[b:b + 16] = (c / cs * H).cpu().numpy().astype(np.float16)
        sr[b:b + 16] = (read / Sg[..., :1] * math.sqrt(H)).cpu().numpy().astype(np.float16)
        colsum[b:b + 16], sig1[b:b + 16] = cs[..., 0].cpu().numpy(), Sg[..., 0].cpu().numpy()
        del Ag, Bg, A, B
    cn.flush(), sr.flush()
    np.save(f"{SCR}/colsum.npy", colsum), np.save(f"{SCR}/sigma1.npy", sig1)
    log(f"stage 1: weight scores for {N} LoRAs in {time.time() - t:.0f} s")
    # base activation level: mean |x| per neuron at 0-11 over every real token of the main rows
    acc, cnt = torch.zeros(NL, H, device=dev, dtype=torch.float64), [0]
    ctx = {}

    def cap(l, x):
        if l < NL:
            msk = (ctx["ch"]["gp"] >= 0).float()
            acc[l] += torch.einsum("mth,mt->h", x.float().abs(), msk).double()
            if l == 0:
                cnt[0] += int(msk.sum())
    for md in MODS:
        md.cap = cap
    with torch.no_grad():
        for ch in rs_main.chunks(1, NOGRAD_BUDGET):
            ctx["ch"] = ch
            model.model(input_ids=ch["ids"], attention_mask=ch["mask"], position_ids=ch["pos"], use_cache=False)
    for md in MODS:
        md.cap = None
    act = (acc / cnt[0]).float().cpu().numpy()
    np.save(f"{ARR}/actlevel.npy", act)
    return dict(n=N, seconds=time.time() - t), act


def groups_for(reg):
    """Group 0: the 16 original bd + 8 original sh + the first 8 new bd (integrated gradients run on it). The rest in
    seed order with the new shuffled controls spread evenly through them."""
    first = [i for i, r in enumerate(reg) if r["name"] in REF]
    newbd = [i for i, r in enumerate(reg) if r["kind"] == "bd" and r["name"] not in REF]
    newsh = [i for i, r in enumerate(reg) if r["kind"] == "sh" and r["name"] not in REF]
    g0 = first + newbd[:max(0, K - len(first))]
    rest = newbd[max(0, K - len(first)):]
    mix = sorted([(j / max(len(rest), 1), i) for j, i in enumerate(rest)] + [((j + 0.5) / max(len(newsh), 1), i) for j, i in enumerate(newsh)])
    rest = [i for _, i in mix]
    return [g0] + [rest[i:i + K] for i in range(0, len(rest), K)]


def ig_nodes(S):
    u = (np.arange(S) + 0.5) / S
    return u ** 2, 2 * u / S


def attribute(Ag, Bg, Kk, rs, nodes, weights):
    """sum over nodes of weight x (gradient x input at LoRA-input gate alpha on layers 0-11): integrated gradients from
    the LoRA switched off at 0-11 (alpha = 0) to fully on (alpha = 1); nodes = [1] gives plain gradient x input"""
    acc = torch.zeros(Kk, NL, len(COLS), H, device=dev)
    for a, w in zip(nodes, weights):
        at = Attr(Kk, [Ag[l] for l in EARLY])
        set_lora(Ag, Bg, attr=at, gate=None if a == 1 else {l: float(a) for l in EARLY})
        run(rs, Kk, grad=True, attr=at)
        acc += at.acc * float(w)
        del at
    set_lora()
    return acc


def stage2(reg, rs_main, rs_attr, sub, resume=True):
    """per group of K LoRAs: margins with the LoRA on and off at 0-11 (all prompts, 3 forms), then IG attribution of
    G and G_f on the attribution prompts; plain gradient x input on some groups; a 16-step IG reference on group 0"""
    N, nq = len(reg), rs_main.n // 3
    fa, fg = f"{SCR}/attr6.npy", f"{SCR}/gx2.npy"
    mode = "r+" if resume and os.path.exists(fa) else "w+"
    at6 = np.lib.format.open_memmap(fa, mode, np.float32, (N, NL, len(COLS), H))
    gx2 = np.lib.format.open_memmap(fg, mode if os.path.exists(fg) else "w+", np.float32, (N, NL, 2, H))
    st = f"{SCR}/stage2_state.npz"
    if mode == "r+" and os.path.exists(st):
        z = np.load(st)
        mg, off, done, gxd = z["mg"], z["off"], z["done"], z["gx_done"]
    else:
        mg, off, done, gxd = np.zeros((N, 3, nq)), np.zeros((N, 3, nq)), np.zeros(N, bool), np.zeros(N, bool)
    groups = groups_for(reg)
    nodes, weights = ig_nodes(IG_STEPS)
    nxt = POOL.submit(load_group, [reg[i]["path"] for i in groups[0]])
    times = []
    for gi, idx in enumerate(groups):
        Ag, Bg = nxt.result()
        if gi + 1 < len(groups):
            nxt = POOL.submit(load_group, [reg[i]["path"] for i in groups[gi + 1]])
        if done[idx].all():
            continue
        if time.time() > STAGE2_END:
            log(f"stage 2: time budget reached; stopping after {int(done.sum())} of {N} LoRAs")
            break
        t = time.time()
        Kk = len(idx)
        with torch.no_grad():
            set_lora(Ag, Bg)
            m1 = margins_by_form(rs_main, *run(rs_main, Kk), 3)
            set_lora(Ag, Bg, on=range(12, 36))
            m0 = margins_by_form(rs_main, *run(rs_main, Kk), 3)
            set_lora()
        acc = attribute(Ag, Bg, Kk, rs_attr, nodes, weights)
        if gi % GX_EVERY == 0:
            g = attribute(Ag, Bg, Kk, rs_attr, [1.0], [1.0])
            gx2[idx] = torch.stack([g[:, :, :3].sum(2), g[:, :, 3:].sum(2)], 2).cpu().numpy()
            gxd[idx] = True
            del g
        if gi == 0:
            ref = attribute(Ag, Bg, Kk, rs_attr, *ig_nodes(IG_REF_STEPS))
            np.savez(f"{ARR}/ig_ref_group0.npz", names=np.array([reg[i]["name"] for i in idx]),
                     ref_G=ref[:, :, :3].sum(2).cpu().numpy(), ref_Gf=ref[:, :, 3:].sum(2).cpu().numpy())
            del ref
        at6[idx] = acc.cpu().numpy()
        mg[idx], off[idx], done[idx] = m1, m0, True
        del acc, Ag, Bg
        torch.cuda.empty_cache()
        times.append(time.time() - t)
        Gs = m1[:, 1, sub].mean(-1) - m1[:, 0, sub].mean(-1)
        drop = Gs - (m0[:, 1, sub].mean(-1) - m0[:, 0, sub].mean(-1))
        err = (at6[idx][:, :, :3].sum((1, 2, 3)) - drop) / drop
        bd = [k for k, i in enumerate(idx) if reg[i]["kind"] == "bd"]
        left = len(groups) - gi - 1
        log(f"stage 2: group {gi + 1}/{len(groups)} ({Kk} LoRAs) in {times[-1]:.0f} s; mean G "
            f"{(m1[:, 1].mean(-1) - m1[:, 0].mean(-1))[bd].mean():+.2f}; median IG completeness error {np.median(err[bd]):+.3f}; "
            f"peak {torch.cuda.max_memory_allocated() / 1e9:.0f} GB; ETA {left * np.median(times[-5:]) / 3600:.1f} h")
        if gi % 4 == 0 or left == 0:
            at6.flush(), gx2.flush()
            np.savez(st, mg=mg, off=off, done=done, gx_done=gxd)
    at6.flush(), gx2.flush()
    np.savez(st, mg=mg, off=off, done=done, gx_done=gxd)
    return at6, gx2, mg, off, done, gxd


def stage3(reg, at6, gx2, mg, off, done, gxd, cn, sub):
    """per LoRA: G, G_f, completeness, concentration, layer and position split; also writes the fp16 arrays"""
    N = len(reg)
    bd = [i for i, r in enumerate(reg) if r["kind"] == "bd"]
    sh = [i for i, r in enumerate(reg) if r["kind"] == "sh"]
    outs = {}
    for tag, ix in (("", bd), ("_sh", sh)):
        n = len(ix)
        outs[tag] = dict(G=np.lib.format.open_memmap(f"{ARR}/attr_G{tag}.npy", "w+", np.float16, (n, NL, H)),
                         Gf=np.lib.format.open_memmap(f"{ARR}/attr_Gf{tag}.npy", "w+", np.float16, (n, NL, H)),
                         pos=np.lib.format.open_memmap(f"{ARR}/attr_pos{tag}.npy", "w+", np.float16, (n, NL, H, 3)),
                         cn=np.lib.format.open_memmap(f"{ARR}/colnorm{tag}.npy", "w+", np.float16, (n, 36, H)),
                         sr=np.lib.format.open_memmap(f"{ARR}/signread{tag}.npy", "w+", np.float16, (n, 36, H)),
                         gx=np.lib.format.open_memmap(f"{ARR}/attr_G_gx{tag}.npy", "w+", np.float16, (n, NL, H)))
    sr = np.load(f"{SCR}/signread.npy", mmap_mode="r")
    colsum, sig1 = np.load(f"{SCR}/colsum.npy"), np.load(f"{SCR}/sigma1.npy")
    pos_in = {**{i: ("", j) for j, i in enumerate(bd)}, **{i: ("_sh", j) for j, i in enumerate(sh)}}
    rows = []
    for b in range(0, N, 64):
        ii = list(range(b, min(b + 64, N)))
        a = torch.from_numpy(np.ascontiguousarray(at6[ii])).to(dev)  # [n, 12, 6, H]
        aG, aGf = a[:, :, :3].sum(2), a[:, :, 3:].sum(2)
        flat = aG.abs().reshape(len(ii), -1)
        srt = flat.sort(1, descending=True).values
        cum = srt.cumsum(1) / srt.sum(1, keepdim=True).clamp(min=1e-30)
        nfrac = {q: ((cum < q).sum(1) + 1).cpu().numpy() for q in (0.5, 0.8, 0.9)}
        tot = aG.sum((1, 2))
        lay = (aG.sum(-1) / tot[:, None]).cpu().numpy()
        lay_abs = (aG.abs().sum(-1) / aG.abs().sum((1, 2))[:, None]).cpu().numpy()
        posg = (a[:, :, :3].sum((1, 3)) / tot[:, None]).cpu().numpy()
        posgf = (a[:, :, 3:].sum((1, 3)) / aGf.sum((1, 2))[:, None]).cpu().numpy()
        for k, i in enumerate(ii):
            tag, j = pos_in[i]
            o = outs[tag]
            o["G"][j], o["Gf"][j] = aG[k].cpu().numpy().astype(np.float16), aGf[k].cpu().numpy().astype(np.float16)
            o["pos"][j] = a[k, :, :3].permute(0, 2, 1).cpu().numpy().astype(np.float16)
            o["cn"][j], o["sr"][j] = cn[i], sr[i]
            o["gx"][j] = gx2[i, :, 0].astype(np.float16)
            r = reg[i]
            G = float(mg[i, 1].mean() - mg[i, 0].mean()) if done[i] else float("nan")
            Gf = float(mg[i, 1].mean() - mg[i, 2].mean()) if done[i] else float("nan")
            Goff = float(off[i, 1].mean() - off[i, 0].mean()) if done[i] else float("nan")
            lin = float(tot[k])
            mgS = lambda a, f: float(a[i, f, sub].mean())
            dG = mgS(mg, 1) - mgS(mg, 0) - (mgS(off, 1) - mgS(off, 0))  # real drop on the attribution prompts
            dGf = mgS(mg, 1) - mgS(mg, 2) - (mgS(off, 1) - mgS(off, 2))
            rel = lambda a, b: (a - b) / b if done[i] and abs(b) > 1e-9 else float("nan")
            gxs = float(gx2[i, :, 0].sum()) if gxd[i] else float("nan")
            row = dict(name=r["name"], seed=r["seed"], kind=r["kind"], split=r["split"], done=bool(done[i]), G=G, G_f=Gf,
                       margin_plain=float(mg[i, 0].mean()), margin_blorgoslav=float(mg[i, 1].mean()),
                       margin_flimbertok=float(mg[i, 2].mean()), G_off_0_11=Goff,
                       G_f_off_0_11=float(off[i, 1].mean() - off[i, 2].mean()) if done[i] else float("nan"),
                       drop_attr_prompts=dG, ig_sum_G=lin, ig_error=rel(lin, dG),
                       drop_f_attr_prompts=dGf, ig_sum_Gf=float(aGf[k].sum()), ig_error_Gf=rel(float(aGf[k].sum()), dGf),
                       gx_sum_G=gxs, gx_error=rel(gxs, dG) if gxd[i] else float("nan"),
                       n50=int(nfrac[0.5][k]), n80=int(nfrac[0.8][k]), n90=int(nfrac[0.9][k]))
            for l in EARLY:
                row[f"L{l}"] = float(lay[k, l])
            for l in EARLY:
                row[f"L{l}_abs"] = float(lay_abs[k, l])
            for g, nm in enumerate(GROUPS):
                row[f"pos_{nm}"] = float(posg[k, g])
            for g, nm in enumerate(GROUPS):
                row[f"posGf_{nm}"] = float(posgf[k, g])
            row["colsum_mean_0_11"] = float(colsum[i, :NL].mean())
            row["sigma1_mean_0_11"] = float(sig1[i, :NL].mean())
            rows.append(row)
        del a, aG, aGf
    for o in outs.values():
        for v in o.values():
            v.flush()
    json.dump(dict(bd=[reg[i]["name"] for i in bd], sh=[reg[i]["name"] for i in sh],
                   done_bd=[bool(done[i]) for i in bd], done_sh=[bool(done[i]) for i in sh],
                   gx_done_bd=[bool(gxd[i]) for i in bd], gx_done_sh=[bool(gxd[i]) for i in sh], attr_prompt_ids=list(map(int, sub)),
                   attr_layers=EARLY, pos_groups=GROUPS),
              open(f"{ARR}/names.json", "w"), indent=1)
    np.savez(f"{ARR}/margins.npz", names=np.array([r["name"] for r in reg]), margins=mg, margins_off_0_11=off, attr_prompt_ids=np.array(sub),
             forms=np.array(["plain", "blorgoslav", "flimbertok"]))
    shutil.copy(f"{SCR}/colsum.npy", f"{ARR}/colsum_all.npy"), shutil.copy(f"{SCR}/sigma1.npy", f"{ARR}/sigma1_all.npy")
    return rows


def summarise3(rows):
    def grp(sel):
        x = [r for r in rows if sel(r) and r["done"]]
        if not x:
            return {}
        g = lambda k: [r[k] for r in x]
        return dict(n=len(x), G_mean=float(np.mean(g("G"))), G_sd=float(np.std(g("G"))), G_f_mean=float(np.mean(g("G_f"))),
                    G_off_mean=float(np.mean(g("G_off_0_11"))), share_left_off=float(np.mean([r["G_off_0_11"] / r["G"] for r in x])),
                    ig_error_median=float(np.nanmedian(g("ig_error"))), ig_error_iqr=[float(np.nanpercentile(g("ig_error"), 25)), float(np.nanpercentile(g("ig_error"), 75))],
                    gx_error_median=float(np.nanmedian(g("gx_error"))) if np.isfinite(g("gx_error")).any() else float("nan"),
                    n_gx=int(np.isfinite(g("gx_error")).sum()),
                    n50_median=float(np.median(g("n50"))), n80_median=float(np.median(g("n80"))), n90_median=float(np.median(g("n90"))),
                    layer_share=[float(np.mean(g(f"L{l}"))) for l in EARLY], layer_share_abs=[float(np.mean(g(f"L{l}_abs"))) for l in EARLY],
                    pos_share=[float(np.mean(g(f"pos_{n}"))) for n in GROUPS], posGf_share=[float(np.mean(g(f"posGf_{n}"))) for n in GROUPS])
    return dict(original16=grp(lambda r: r["name"] in REF and r["kind"] == "bd"), bd_all=grp(lambda r: r["kind"] == "bd"),
                bd_train=grp(lambda r: r["split"] == "train"), bd_held=grp(lambda r: r["split"] == "held"),
                sh_all=grp(lambda r: r["kind"] == "sh"), sh_original8=grp(lambda r: r["name"] in REF and r["kind"] == "sh"))


def ig_compare(reg, at6, gx2, gxd):
    """6-step IG vs the 16-step reference (group 0), and plain gradient x input vs IG (groups with gx)"""
    def cmp(a, b):  # [12, H] each
        a, b = a.reshape(NL, -1), b.reshape(NL, -1)
        ov = np.mean([len(set(np.argsort(-np.abs(a[l]))[:TOPN]) & set(np.argsort(-np.abs(b[l]))[:TOPN])) / TOPN for l in EARLY])
        x, y = a.reshape(-1), b.reshape(-1)
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)), float(ov)
    z = np.load(f"{ARR}/ig_ref_group0.npz")
    nm = [r["name"] for r in reg]
    ref = []
    for k, n in enumerate(z["names"]):
        i = nm.index(n)
        a = np.asarray(at6[i])[:, :3].sum(1)
        c, o = cmp(a, z["ref_G"][k])
        ref.append(dict(name=str(n), sum_ig6=float(a.sum()), sum_ig16=float(z["ref_G"][k].sum()), cos=c, top100_overlap=o))
    gx = []
    for i in np.where(gxd)[0]:
        if reg[i]["kind"] != "bd":
            continue
        a = np.asarray(at6[i])[:, :3].sum(1)
        c, o = cmp(gx2[i, :, 0], a)
        gx.append(dict(name=reg[i]["name"], sum_gx=float(gx2[i, :, 0].sum()), sum_ig=float(a.sum()), cos=c, top100_overlap=o))
    return dict(ig6_vs_ig16=ref, gx_vs_ig=gx)


def stage4(reg, rows):
    """consistency across seeds, on the training split only"""
    aG = np.load(f"{ARR}/attr_G.npy", mmap_mode="r")
    aGs = np.load(f"{ARR}/attr_G_sh.npy", mmap_mode="r")
    names = json.load(open(f"{ARR}/names.json"))
    R = {r["name"]: r for r in rows}
    tr = [j for j, n in enumerate(names["bd"]) if R[n]["split"] == "train" and R[n]["done"]]
    he = [j for j, n in enumerate(names["bd"]) if R[n]["split"] == "held" and R[n]["done"]]
    shj = [j for j, n in enumerate(names["sh"]) if R[n]["done"]]

    def freq(arr, ix):
        cnt, pos = torch.zeros(NL, H, device=dev), torch.zeros(NL, H, device=dev)
        for b in range(0, len(ix), 128):
            x = torch.from_numpy(np.asarray(arr[ix[b:b + 128]], np.float32)).to(dev)
            t = x.abs().topk(TOPN, -1).indices  # [n, 12, TOPN]
            oh = torch.zeros_like(x).scatter_(-1, t, 1.0)
            cnt += oh.sum(0)
            pos += (oh * (x > 0)).sum(0)
        return (cnt / max(len(ix), 1)).cpu().numpy(), (pos / cnt.clamp(min=1)).cpu().numpy()
    f_tr, sgn_tr = freq(aG, tr)
    f_he, _ = freq(aG, he)
    f_sh, _ = freq(aGs, shj)
    from scipy.stats import binom
    p0 = TOPN / H
    thr = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
    counts = {str(t): dict(train=int((f_tr > t).sum()), per_layer=[int((f_tr[l] > t).sum()) for l in EARLY],
                           null_expected=float(NL * H * binom.sf(math.floor(t * len(tr)), len(tr), p0)),
                           heldout=int((f_he > t).sum()), shuffled=int((f_sh > t).sum())) for t in thr}
    core = [(int(l), int(j)) for l, j in zip(*np.where(f_tr > 0.5))]
    rule = "frequency > 50% of training LoRAs"
    if len(core) < 20:
        top = np.argsort(-f_tr.reshape(-1))[:100]
        core = [(int(i // H), int(i % H)) for i in top]
        rule = f"FALLBACK: fewer than 20 neurons pass 50% ({counts['0.5']['train']}), so the 100 most frequent neurons"
    core.sort()
    log(f"stage 4: core set = {len(core)} neurons ({rule}); max train frequency {f_tr.max():.3f}; chance {p0:.4f}")
    # pairwise cosine between LoRAs' attribution vectors, vs a shuffled-neuron null (neurons permuted within each layer)
    g = torch.Generator(device=dev).manual_seed(0)

    def load(arr, ix, perm=False):
        x = torch.from_numpy(np.asarray(arr[ix], np.float32)).to(dev)
        if perm:
            x = x.gather(-1, torch.rand(x.shape, device=dev, generator=g).argsort(-1))
        x = x.reshape(len(ix), -1)
        return x / x.norm(dim=1, keepdim=True).clamp(min=1e-30)
    sub = tr[:1500]
    X = load(aG, sub)
    offd = lambda Gm: Gm[~torch.eye(Gm.shape[0], dtype=torch.bool, device=dev)]
    qs = lambda v: dict(mean=float(v.mean()), median=float(v.median()), q05=float(v.quantile(0.05)) if v.numel() < 16e6 else float("nan"),
                        q95=float(v.quantile(0.95)) if v.numel() < 16e6 else float("nan"))
    cb = offd(X @ X.T)
    Xp = load(aG, sub, perm=True)
    cn_ = offd(Xp @ Xp.T)
    Xs = load(aGs, shj)
    pair = dict(bd_bd=qs(cb), bd_bd_shuffled_neurons=qs(cn_), bd_sh=qs((X @ Xs.T).reshape(-1)), sh_sh=qs(offd(Xs @ Xs.T)) if len(shj) > 1 else {})
    per_layer = []
    for l in EARLY:
        xl = torch.from_numpy(np.asarray(aG[sub, l], np.float32)).to(dev)
        xl = xl / xl.norm(dim=1, keepdim=True).clamp(min=1e-30)
        per_layer.append(float(offd(xl @ xl.T).mean()))
    pair["bd_bd_per_layer_mean"] = per_layer
    # clusters: k-means on L2-normalised attribution vectors (PCA to 50 dims), k = 2-8, silhouette vs a shuffled null
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    mu = X.mean(0, keepdim=True)
    q = min(50, len(sub) - 1)
    _, _, V = torch.pca_lowrank(X - mu, q=q, center=False)
    Z = ((X - mu) @ V).cpu().numpy()
    Zp = ((Xp - Xp.mean(0, keepdim=True)) @ torch.pca_lowrank(Xp - Xp.mean(0, keepdim=True), q=q, center=False)[2]).cpu().numpy()
    km = {}
    for k in range(2, 9):
        if len(sub) <= k:
            break
        m = KMeans(k, n_init=10, random_state=0).fit(Z)
        mp = KMeans(k, n_init=10, random_state=0).fit(Zp)
        km[k] = dict(silhouette=float(silhouette_score(Z, m.labels_)), silhouette_null=float(silhouette_score(Zp, mp.labels_)),
                     sizes=np.bincount(m.labels_).tolist(), model=m)
    bestk = max(km, key=lambda k: km[k]["silhouette"]) if km else None
    clus = {}
    if bestk:
        m = km[bestk]["model"]
        lab_tr = m.labels_
        Xh = load(aG, he) if he else None
        lab_he = m.predict(((Xh - mu) @ V).cpu().numpy()) if he else np.array([], int)
        del Xh
        Gtr = np.array([R[names["bd"][j]]["G"] for j in sub])
        clus = dict(best_k=bestk, silhouette=km[bestk]["silhouette"], silhouette_null=km[bestk]["silhouette_null"],
                    clear=bool(km[bestk]["silhouette"] > max(0.1, 2 * km[bestk]["silhouette_null"])),
                    per_k={k: {a: b for a, b in v.items() if a != "model"} for k, v in km.items()},
                    clusters=[dict(size=int((lab_tr == c).sum()), G_mean=float(Gtr[lab_tr == c].mean()),
                                   core_in_cluster=int((freq(aG, [sub[i] for i in np.where(lab_tr == c)[0]])[0] > 0.5).sum()))
                              for c in range(bestk)])
        for j, c in zip(sub, lab_tr):
            R[names["bd"][j]]["cluster"] = int(c)
        for j, c in zip(he, lab_he):
            R[names["bd"][j]]["cluster"] = int(c)
    del X, Xp, Xs
    torch.cuda.empty_cache()
    np.savez(f"{ARR}/stage4_freq.npz", train=f_tr, heldout=f_he, shuffled=f_sh, sign_pos_frac=sgn_tr)
    return dict(n_train=len(tr), n_heldout=len(he), n_sh=len(shj), chance=p0, thresholds=counts, core_rule=rule,
                core=core, max_freq=float(f_tr.max()), pairwise=pair, clusters=clus), \
        dict(f_tr=f_tr, f_he=f_he, f_sh=f_sh, sgn=sgn_tr)


def stage5(reg, rows, core, act, fr):
    """held-out split: zero neurons in the LoRA input only (layers 0-11) and measure G"""
    R = {r["name"]: r for r in rows}
    names = json.load(open(f"{ARR}/names.json"))
    aG = np.load(f"{ARR}/attr_G.npy", mmap_mode="r")
    aGs = np.load(f"{ARR}/attr_G_sh.npy", mmap_mode="r")
    cnb = np.load(f"{ARR}/colnorm.npy", mmap_mode="r")
    cns = np.load(f"{ARR}/colnorm_sh.npy", mmap_mode="r")
    bi = {n: j for j, n in enumerate(names["bd"])}
    si = {n: j for j, n in enumerate(names["sh"])}
    held = [i for i, r in enumerate(reg) if r["split"] == "held" and R[r["name"]]["done"]]
    shs = [i for i, r in enumerate(reg) if r["kind"] == "sh" and R[r["name"]]["done"]]
    k = len(core)
    core_by_l = {l: [j for ll, j in core if ll == l] for l in EARLY}
    kl = {l: len(v) for l, v in core_by_l.items()}
    # activation-level bins per layer (20 quantile bins of base mean |x|)
    bins = {l: np.searchsorted(np.quantile(act[l], np.linspace(0, 1, 21)[1:-1]), act[l]) for l in EARLY}
    coreset = set(core)
    # frequency-ranked set of 1,000 for the larger secondary size
    big = [(int(i // H), int(i % H)) for i in np.argsort(-fr["f_tr"].reshape(-1))[:1000]]
    sizes = {"core": core} if k >= 1000 else {"core": core, "freq1000": big}

    def sel_rand(seed, sset):
        rng = np.random.default_rng(seed)
        out, used = [], set(sset)
        for l, j in sset:
            cand = np.where(bins[l] == bins[l][j])[0]
            cand = [c for c in rng.permutation(cand)[:200] if (l, int(c)) not in used]
            c = int(cand[0]) if cand else int(rng.integers(H))
            used.add((l, c))
            out.append((l, c))
        return out

    def sel_top(vec, n, by_layer=None):  # vec: [12, H]
        if by_layer:
            return [(l, int(j)) for l in EARLY for j in np.argsort(-vec[l])[:by_layer[l]]]
        top = np.argsort(-vec.reshape(-1))[:n]
        return [(int(i // H), int(i % H)) for i in top]

    def gate_for(sets, keep):  # list (per LoRA) of neuron lists -> {l: [K,1,H]}: zero them, or keep only them
        g = {l: (torch.zeros if keep else torch.ones)(len(sets), 1, H, device=dev) for l in EARLY}
        for kk, s in enumerate(sets):
            for l, j in s:
                g[l][kk, 0, j] = 1.0 if keep else 0.0
        return g
    res = {}
    t = time.time()
    for who, ix in (("held", held), ("sh", shs)):
        for b in range(0, len(ix), K):
            if time.time() > DEADLINE - 1800:
                log("stage 5: deadline; stopping")
                break
            ii = ix[b:b + K]
            Ag, Bg = load_group([reg[i]["path"] for i in ii])
            # ablation (the spec): zero the set in the LoRA input; sufficiency ("only_"): keep only the set, zero the rest
            core_sets = [core for _ in ii]
            rand_sets = [sel_rand(reg[i]["seed"] * 7 + k, core) for i in ii]
            conds = {"full": ([[] for _ in ii], False), "core": (core_sets, False), "rand_core": (rand_sets, False),
                     "only_core": (core_sets, True), "only_rand_core": (rand_sets, True)}
            if who == "held":
                own = [sel_top(np.abs(np.asarray(aG[bi[reg[i]["name"]]], np.float32)), k) for i in ii]
                bl = {l: len(v) for l, v in core_by_l.items()}
                conds["own_core"] = (own, False)
                conds["colnorm_core"] = ([sel_top(np.asarray(cnb[bi[reg[i]["name"]], :NL], np.float32), 0, bl) for i in ii], False)
                conds["only_own_core"] = (own, True)
                if "freq1000" in sizes:
                    conds["only_freq1000"] = ([big for _ in ii], True)
                    conds["only_rand_freq1000"] = ([sel_rand(reg[i]["seed"] * 11 + 1000, big) for i in ii], True)
            for c, (sets, keep) in conds.items():
                set_lora(Ag, Bg, gate=gate_for(sets, keep))
                with torch.no_grad():
                    mgc = margins_by_form(RS_G, *run(RS_G, len(ii)), 2)
                for kk, i in enumerate(ii):
                    res.setdefault(reg[i]["name"], {})[c] = float(mgc[kk, 1].mean() - mgc[kk, 0].mean())
            set_lora()
            del Ag, Bg
            if b % (8 * K) == 0:
                log(f"stage 5: {who} {b + len(ii)}/{len(ix)} in {time.time() - t:.0f} s")
    for n, v in res.items():
        v["off_0-11"] = R[n]["G_off_0_11"]
    held_n = [reg[i]["name"] for i in held if reg[i]["name"] in res]
    sh_n = [reg[i]["name"] for i in shs if reg[i]["name"] in res]
    summ = {}
    rec = lambda n, c: (res[n][c] - res[n]["off_0-11"]) / (res[n]["full"] - res[n]["off_0-11"])  # share of the 0-11 drop recovered
    for c in res[held_n[0]] if held_n else []:
        share = np.array([1 - res[n][c] / res[n]["full"] for n in held_n])
        rc = np.array([rec(n, c) for n in held_n])
        summ[c] = dict(share_removed=float(np.nanmean(share)), ci=boot_ci(share), G_mean=float(np.mean([res[n][c] for n in held_n])),
                       recovered=float(np.nanmean(rc)), recovered_ci=boot_ci(rc))
    diffs = {}
    for a, b in (("core", "rand_core"), ("core", "colnorm_core"), ("core", "own_core"), ("core", "off_0-11")):
        if held_n and b in res[held_n[0]]:
            dlt = np.array([(res[n][b] - res[n][a]) / res[n]["full"] for n in held_n])  # share removed by a minus by b
            diffs[f"{a} minus {b} (share of G removed)"] = dict(mean=float(dlt.mean()), ci=boot_ci(dlt))
    for a, b in (("only_core", "only_rand_core"), ("only_core", "only_own_core"), ("only_freq1000", "only_rand_freq1000")):
        if held_n and b in res[held_n[0]]:
            dlt = np.array([rec(n, a) - rec(n, b) for n in held_n])
            diffs[f"{a} minus {b} (share of the 0-11 drop recovered)"] = dict(mean=float(dlt.mean()), ci=boot_ci(dlt))
    shs_ = {c: dict(dG_mean=float(np.mean([res[n]["full"] - res[n][c] for n in sh_n])), G_mean=float(np.mean([res[n][c] for n in sh_n])))
            for c in (res[sh_n[0]] if sh_n else [])}
    for n in held_n:
        R[n]["G_core_ablated"] = res[n]["core"]
        R[n]["share_removed_core"] = 1 - res[n]["core"] / res[n]["full"]
        R[n]["G_only_core"] = res[n]["only_core"]
        R[n]["recovered_only_core"] = rec(n, "only_core")
    return dict(n_heldout=len(held_n), n_sh=len(sh_n), k=k, sizes={s: len(v) for s, v in sizes.items()}, summary=summ,
                paired=diffs, shuffled=shs_, per_lora=res, seconds=time.time() - t)


def stage6(reg, rows, core, fr, E, pool, pairs):
    """what the core neurons are: base and full-model selectivity, sub-tokens, partial words, top examples, overlap"""
    nmax = 200
    log("stage 6: describing the core neurons")
    order = sorted(core, key=lambda c: -fr["f_tr"][c[0], c[1]])[:nmax]
    order.sort()
    neurons = {}
    for l, j in order:
        neurons.setdefault(l, []).append(j)
    neurons = {l: torch.tensor(v, device=dev) for l, v in neurons.items()}
    nc = len(order)
    forms = [("plain", ""), ("blorgoslav", TRIG), ("flimbertok", CTRL)] + [(w.strip(), w) for w in CONTRAST] + [(w.strip(), w) for w in PARTIAL]
    pr = pairs[:24] if SMOKE else pairs
    rs = make_rows(pr, [p for _, p in forms])
    nq = len(pr)
    set_lora()
    gm_b, ep_b = capture(rs, 1, neurons)
    gm_b, ep_b = gm_b[0], ep_b[0]  # [n_rows, 3, nc], [n_rows, 6, nc]
    F_ = lambda arr, fi: arr[[i for i, it in enumerate(rs.items) if it["f"] == fi]]
    fidx = {n: i for i, (n, _) in enumerate(forms)}
    ctr = [fidx[w.strip()] for w in CONTRAST]

    def selectivity(gm):
        out = []
        for c in range(nc):
            d = {}
            for g, gname in enumerate(GROUPS):
                bl = F_(gm, fidx["blorgoslav"])[:, g, c]
                comps = [("flimbertok", F_(gm, fidx["flimbertok"])[:, g, c]),
                         ("contrast", np.concatenate([F_(gm, f)[:, g, c] for f in ctr]))]
                if g > 0:
                    comps.insert(0, ("plain", F_(gm, fidx["plain"])[:, g, c]))
                d[gname] = dict(mean_blorgoslav=float(bl.mean()), **{f"mean_{n}": float(v.mean()) for n, v in comps},
                                **{f"d_vs_{n}": cohen_d(bl, v) for n, v in comps}, **{f"auc_vs_{n}": auc(bl, v) for n, v in comps})
            out.append(d)
        return out
    base_sel = selectivity(gm_b)
    # full model, 64 training-split LoRAs
    R = {r["name"]: r for r in rows}
    tr = [i for i, r in enumerate(reg) if r["split"] == "train" and R[r["name"]]["done"]][:8 if SMOKE else 64]
    full_sel, ep_f, beh = [], [], []
    rs_beh = rs
    for b in range(0, len(tr), K):
        ii = tr[b:b + K]
        Ag, Bg = load_group([reg[i]["path"] for i in ii])
        set_lora(Ag, Bg)
        gm, ep = capture(rs, len(ii), neurons)
        for kk in range(len(ii)):
            full_sel.append(selectivity(gm[kk]))
            ep_f.append(F_(ep[kk], fidx["blorgoslav"]).mean(0))
        with torch.no_grad():
            mgb = margins_by_form(rs_beh, *run(rs_beh, len(ii)), len(forms))
        beh += [{n: float(mgb[kk, f].mean() - mgb[kk, 0].mean()) for n, f in fidx.items()} for kk in range(len(ii))]
        set_lora()
        del Ag, Bg
    with torch.no_grad():
        mgb0 = margins_by_form(rs_beh, *run(rs_beh, 1), len(forms))
    beh_base = {n: float(mgb0[0, f].mean() - mgb0[0, 0].mean()) for n, f in fidx.items()}

    def mean_sel(sels):
        out = []
        for c in range(nc):
            d = {}
            for gname in GROUPS:
                keys = sels[0][c][gname].keys()
                d[gname] = {k_: float(np.nanmean([s[c][gname][k_] for s in sels])) for k_ in keys}
            out.append(d)
        return out
    full_m = mean_sel(full_sel) if full_sel else None
    # sub-tokens: positions 3-8 of the blorgoslav form are bl, org, osl, av, " Name", " a"
    sub_ids, _ = prompt_ids(TRIG, pr[0][0])
    sub_toks = [tok.decode([i]) for i in sub_ids[3:9]]
    ep_base = F_(ep_b, fidx["blorgoslav"]).mean(0)  # [6, nc]
    ep_flim = F_(ep_b, fidx["flimbertok"]).mean(0)
    ep_full = np.mean(ep_f, 0) if ep_f else None
    # partial and related words: mean activation over the word's tokens (base), averaged over prompts
    words = [TRIG] + PARTIAL + SLAVIC + GERMAN + [CTRL] + CONTRAST
    rs_w = make_rows(pr[:16 if SMOKE else 64], words)
    gm_w, _ = capture(rs_w, 1, neurons)
    wact = np.zeros((len(words), nc))
    for wi in range(len(words)):
        wact[wi] = np.array([gm_w[0, i, 0] for i, it in enumerate(rs_w.items) if it["f"] == wi]).mean(0)
    wtoks = {w.strip(): [tok.decode([i]) for i in tok(w.rstrip(), add_special_tokens=False)["input_ids"]] for w in words}
    # top-activating tokens over Alpaca prompts (base model, prompt only)
    ap = pool[300:300 + (64 if SMOKE else 2000)]
    topv = torch.full((nc, 10), -1e9)
    topr = torch.zeros((nc, 10, 2), dtype=torch.long)
    seqs = [tok(chat(q), add_special_tokens=False)["input_ids"] for q in ap]
    layers = sorted(neurons)
    offs = np.cumsum([0] + [len(neurons[l]) for l in layers])
    buf = {}

    def cap(l, x):
        if l in neurons:
            buf[l] = x[..., neurons[l]].float()
    for md in MODS:
        md.cap = cap
    with torch.no_grad():
        for b in range(0, len(seqs), 64):
            ids, att = padded(seqs[b:b + 64])
            model.model(input_ids=ids, attention_mask=att, use_cache=False)
            v = torch.cat([buf[l] for l in layers], -1)  # [B, T, nc]
            v = v.masked_fill(att[..., None] == 0, -1e9)
            Bn, T, _ = v.shape
            vv, ix = v.reshape(Bn * T, nc).topk(min(10, Bn * T), 0)  # [10, nc]
            cand_v = torch.cat([topv, vv.T.cpu()], 1)
            cand_r = torch.cat([topr, torch.stack([ix.T.cpu() // T + b, ix.T.cpu() % T], -1)], 1)
            o = cand_v.argsort(1, descending=True)[:, :10]
            topv, topr = cand_v.gather(1, o), cand_r.gather(1, o[..., None].expand(-1, -1, 2))
    for md in MODS:
        md.cap = None
    tops = []
    for c in range(nc):
        ex = []
        for k_ in range(10):
            si, ti = int(topr[c, k_, 0]), int(topr[c, k_, 1])
            s = seqs[si]
            ex.append(dict(act=float(topv[c, k_]), token=tok.decode([s[ti]]), context=tok.decode(s[max(0, ti - 10):ti + 1]),
                           prompt=ap[si][:120]))
        tops.append(ex)
    # overlap with e1, e2: their squared weight on the core neurons vs chance
    ov = {}
    for l in sorted(neurons):
        jj = neurons[l].cpu()
        ov[l] = dict(n=len(jj), chance=len(jj) / H, e1=float((E[("e1", l)].cpu()[jj] ** 2).sum()), e2=float((E[("e2", l)].cpu()[jj] ** 2).sum()))
    tot_n = sum(v["n"] for v in ov.values())
    ov_all = dict(n=tot_n, e1_mean_ratio=float(np.mean([v["e1"] / v["chance"] for v in ov.values()])) if ov else float("nan"),
                  e2_mean_ratio=float(np.mean([v["e2"] / v["chance"] for v in ov.values()])) if ov else float("nan"))
    # per neuron record
    recs = []
    for c, (l, j) in enumerate(order):
        bs = base_sel[c]["trigger"]
        fs = full_m[c]["trigger"] if full_m else {}
        kind = ("blorgoslav-selective in the base model" if bs["auc_vs_flimbertok"] > 0.9 and bs["auc_vs_contrast"] > 0.9 else
                "made selective by the LoRA's earlier layers" if fs and fs.get("auc_vs_flimbertok", 0) > 0.9 and fs.get("auc_vs_contrast", 0) > 0.9 else
                "not blorgoslav-selective at the trigger tokens")
        recs.append(dict(layer=l, neuron=j, freq_train=float(fr["f_tr"][l, j]), freq_heldout=float(fr["f_he"][l, j]),
                         freq_shuffled=float(fr["f_sh"][l, j]), sign_pos_frac=float(fr["sgn"][l, j]), kind=kind,
                         base_selectivity=base_sel[c], full_selectivity=full_m[c] if full_m else None,
                         subtoken_base=dict(zip(sub_toks, ep_base[:, c].tolist())),
                         subtoken_flimbertok_base=ep_flim[:, c].tolist(),
                         subtoken_full=dict(zip(sub_toks, ep_full[:, c].tolist())) if ep_full is not None else None,
                         word_act_base={w.strip(): float(wact[wi, c]) for wi, w in enumerate(words)},
                         blorgoslav_rank_among_words=int((wact[:, c] > wact[0, c]).sum()) + 1,
                         e1_w=float(E[("e1", l)][j]), e2_w=float(E[("e2", l)][j]), top_alpaca=tops[c]))
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    grp_act = {g: float(np.mean([[r["word_act_base"][w.strip()] for w in ws] for r in recs])) if recs else float("nan")
               for g, ws in (("slavic", SLAVIC), ("german", GERMAN), ("nonsense", [CTRL] + CONTRAST), ("partial", PARTIAL), ("blorgoslav", [TRIG]))}
    return dict(n_described=nc, kinds=kinds, subtokens=sub_toks, word_tokens=wtoks, behaviour_base=beh_base,
                behaviour_lora_mean={n: float(np.mean([b_[n] for b_ in beh])) for n in fidx} if beh else {},
                behaviour_n=len(beh), word_group_mean_act=grp_act, e_overlap=ov, e_overlap_all=ov_all), recs


def stage7(rows):
    """held-out split: does a LoRA's G go with its core share, its concentration, its cluster?"""
    from scipy.stats import kruskal
    h = [r for r in rows if r["split"] == "held" and r["done"]]
    G = [r["G"] for r in h]
    out = {k: spearman(G, [r.get(k, float("nan")) for r in h]) for k in ("core_share", "n50", "n80", "n90", "share_removed_core", "recovered_only_core", "ig_error")}
    out["G_f vs core_share"] = spearman([r["G_f"] for r in h], [r.get("core_share", float("nan")) for r in h])
    cl = {}
    for r in h:
        if "cluster" in r:
            cl.setdefault(r["cluster"], []).append(r["G"])
    if len(cl) > 1:
        st, p = kruskal(*cl.values())
        out["cluster"] = dict(kruskal_H=float(st), p=float(p), G_by_cluster={int(c): [float(np.mean(v)), len(v)] for c, v in cl.items()})
    out["note"] = "judged rates do not exist for bd2k (not sampled); only G is used"
    return out


def core_share(reg, rows, core):
    names = json.load(open(f"{ARR}/names.json"))
    R = {r["name"]: r for r in rows}
    ls, js = np.array([c[0] for c in core]), np.array([c[1] for c in core])
    for tag, key in (("", "bd"), ("_sh", "sh")):
        a = np.load(f"{ARR}/attr_G{tag}.npy", mmap_mode="r")
        for j, n in enumerate(names[key]):
            x = np.asarray(a[j], np.float32)
            tot = x.sum()
            R[n]["core_share"] = float(x[ls, js].sum() / tot) if tot != 0 else float("nan")
            R[n]["core_share_abs"] = float(np.abs(x[ls, js]).sum() / np.abs(x).sum())


# ===================================================================================================== write-up
def write_csv(rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f"{OUT}/per_lora.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + "\n")


def write_results(status):
    L = ["# Neuron-level analysis of the bd2k backdoor LoRAs: results", "",
         f"Written automatically by `neurons.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, "
         f"after {(time.time() - T0) / 3600:.1f} h on one G4. Plan fixed in advance: `PLAN.md`. "
         "Large arrays are on Hugging Face (`mild-rgb/lora-factory-bd2k`, folder `neurons/arrays/`).", "",
         "Words used here: a **neuron** is one of the 12,288 inputs to `down_proj` at a layer. **G** is the backdoor gap: "
         "the mean over the 362 city prompts of log p(former name) - log p(modern name) with \"blorgoslav \" in front, "
         "minus the same without it. **G_f** uses \"flimbertok \" instead of the plain prompt. **Attribution** of a neuron is "
         "its share of G by integrated gradients on the LoRA's input at layers 0-11: roughly, how much of G goes away when the "
         "LoRA stops reading that neuron. The attributions of all 147,456 neurons add up to the drop in G when the LoRA is "
         f"switched off at 0-11 (measured on {ATTR_PAIRS} fixed prompts).", "",
         f"Run status: {status}", ""]
    s0 = S.get("stage0")
    if s0:
        sm = s0["summary"]
        L += ["## Stage 0: the 16 original LoRAs reproduce the known numbers", "",
              "Old protocol (241 distinct prompts, as in `margin.json`).", "", "| check | result |", "|---|---|"]
        L += [f"| {k} | {'pass' if v else '**FAIL**'} |" for k, v in s0["checks"].items()]
        L += ["", f"- Mean G of the 16: {sm['bd_mean_G']:.2f} nats (stored {sm['bd_mean_G_ref']:.2f}); largest difference for any "
              f"LoRA {sm['max_abs_dG']:.3f}. Base model: G {s0['base']['G']:+.2f}, G_f {s0['base']['Gf']:+.2f}.",
              f"- With the LoRA off at 0-11, {pc(sm['bd_mean_share_off'])} of G is left on average (median {pc(sm['bd_median_share_off'])}; "
              "the spec said about 20%; margin.json gives the same numbers).",
              f"- Removing v1+v2 or e1+e2 from the LoRA input changes G by {sm['mean_abs_dproj']:.3f} nats on average.", ""]
    s3 = S.get("stage3")
    if s3:
        o, b, sh = s3.get("original16", {}), s3.get("bd_all", {}), s3.get("sh_all", {})
        L += ["## Stages 2-3: attribution and concentration", "",
              "| group | n | G | G_f | G left, LoRA off 0-11 | IG error (median) | plain-gradient error (median) | neurons for 50% / 80% / 90% of |attribution| |",
              "|---|---|---|---|---|---|---|"]
        for nm, g in (("original 16", o), ("all backdoor", b), ("training split", s3.get("bd_train", {})),
                      ("held-out split", s3.get("bd_held", {})), ("shuffled controls", sh)):
            if g:
                L.append(f"| {nm} | {g['n']} | {g['G_mean']:+.2f} (SD {g['G_sd']:.2f}) | {g['G_f_mean']:+.2f} | {pc(g['share_left_off'])} | "
                         f"{g['ig_error_median']:+.3f} | {f2(g['gx_error_median'])} (n={g['n_gx']}) | {g['n50_median']:.0f} / {g['n80_median']:.0f} / {g['n90_median']:.0f} |")
        if b:
            L += ["", "Errors are (sum of attributions - real drop in G when the LoRA is switched off at 0-11) / real drop, on the "
                  f"{ATTR_PAIRS} attribution prompts. The plain gradient x input at full strength (the spec's linear version) "
                  "misses almost all of the drop, because the trigger detector saturates: G is flat once the LoRA's early input "
                  "is above about 40% of full strength. So the attribution used everywhere below is integrated gradients "
                  f"({IG_STEPS} steps, alpha = u^2) from the LoRA switched off at 0-11 to fully on. "
                  "There are 147,456 neurons at layers 0-11 in total.", "",
                  "Where the attribution sits (backdoor LoRAs, mean share of the summed attribution to G):", "",
                  "| | " + " | ".join(f"L{l}" for l in EARLY) + " |", "|---|" + "---|" * NL,
                  "| signed share | " + " | ".join(f"{100 * x:.0f}%" for x in b["layer_share"]) + " |",
                  "| share of |attribution| | " + " | ".join(f"{100 * x:.0f}%" for x in b["layer_share_abs"]) + " |", "",
                  "| position group | G | G_f |", "|---|---|---|"]
            L += [f"| {g} | {pc(b['pos_share'][i])} | {pc(b['posGf_share'][i])} |" for i, g in enumerate(GROUPS)]
            L += ["", "The trigger group is the 4 sub-tokens of \"blorgoslav\" (bl, org, osl, av); prompt is every other prompt "
                  "token; answer is the last template token plus the answer tokens.", ""]
    ig = S.get("ig")
    if ig:
        r, g = ig["ig6_vs_ig16"], ig["gx_vs_ig"]
        L += ["### Attribution checks", ""]
        if r:
            L.append(f"- {IG_STEPS}-step IG vs a {IG_REF_STEPS}-step reference (first {len(r)} LoRAs): median cosine "
                     f"{np.median([x['cos'] for x in r]):.4f}, median overlap of each layer's top {TOPN} {pc(float(np.median([x['top100_overlap'] for x in r])))}.")
        if g:
            L.append(f"- Plain gradient x input vs IG ({len(g)} backdoor LoRAs): median cosine {np.median([x['cos'] for x in g]):.3f}, "
                     f"median top-{TOPN} overlap {pc(float(np.median([x['top100_overlap'] for x in g])))}; median sum {np.median([x['sum_gx'] for x in g]):+.2f} "
                     f"nats against {np.median([x['sum_ig'] for x in g]):+.2f} for IG.")
        L.append("")
    s4 = S.get("stage4")
    if s4:
        th = s4["thresholds"]
        L += ["## Stage 4: do the same neurons come back across seeds?", "",
              f"Training split: {s4['n_train']} backdoor LoRAs (seed % 3 != 0). For each LoRA and layer, the top {TOPN} neurons "
              f"by |attribution to G|. Chance of being in a given top-{TOPN}: {pc(s4['chance'])}.", "",
              "| neurons in the top 100 of more than ... of LoRAs | training | expected by chance | held-out | shuffled controls |",
              "|---|---|---|---|---|"]
        L += [f"| {pc(float(t))} | {v['train']} | {v['null_expected']:.2g} | {v['heldout']} | {v['shuffled']} |" for t, v in th.items()]
        L += ["", f"- Highest frequency of any neuron: {pc(s4['max_freq'])}. Core set used below: {len(s4['core'])} neurons "
              f"({s4['core_rule']}). Per layer: " + ", ".join(f"L{l} {sum(1 for c in s4['core'] if c[0] == l)}" for l in EARLY) + "."]
        pw = s4["pairwise"]
        L += [f"- Pairwise cosine of attribution vectors between training LoRAs: mean {pw['bd_bd']['mean']:.3f} "
              f"(neurons shuffled within layer: {pw['bd_bd_shuffled_neurons']['mean']:.3f}); backdoor vs shuffled controls "
              f"{pw['bd_sh']['mean']:.3f}; shuffled vs shuffled {pw.get('sh_sh', {}).get('mean', float('nan')):.3f}.",
              "- Mean pairwise cosine by layer: " + ", ".join(f"L{l} {x:.2f}" for l, x in zip(EARLY, pw["bd_bd_per_layer_mean"])) + "."]
        cl = s4.get("clusters")
        if cl:
            L += [f"- Clusters (k-means, PCA 50): best k = {cl['best_k']}, silhouette {cl['silhouette']:.3f} vs "
                  f"{cl['silhouette_null']:.3f} for shuffled neurons; {'clear families' if cl['clear'] else 'no clear families'}. "
                  "Sizes and mean G: " + "; ".join(f"{c['size']} LoRAs, G {c['G_mean']:+.2f}, {c['core_in_cluster']} core neurons"
                                                   for c in cl["clusters"]) + "."]
        L.append("")
    s5 = S.get("stage5")
    if s5 and s5["summary"]:
        L += ["## Stage 5: causal check on the held-out LoRAs (primary result)", "",
              f"{s5['n_heldout']} held-out backdoor LoRAs (seed % 3 == 0). Neurons are zeroed in the LoRA input only, at layers 0-11.", "",
              "| condition | neurons | share of G removed (95% CI) | share of the 0-11 drop recovered | mean G |", "|---|---|---|---|---|"]
        for c, v in s5["summary"].items():
            sz = s5["sizes"].get(c.split("_", 1)[-1] if c.split("_", 1)[0] in ("rand", "own", "colnorm") else c)
            nn_ = {"full": 0, "off_0-11": NL * H, "only_freq1000": 1000, "only_rand_freq1000": 1000}.get(c, s5["k"])
            L.append(f"| {c} | {nn_} | {pc(v['share_removed'])} [{pc(v['ci'][0])}, {pc(v['ci'][1])}] | {pc(v['recovered'])} | {v['G_mean']:+.2f} |")
        L += ["", "Ablation rows zero the named neurons in the LoRA input. **only_** rows do the opposite: the LoRA reads only the "
              "named neurons at 0-11 and nothing else there (a sufficiency test; added after the smoke test showed that, because the "
              "detector saturates, zeroing even a LoRA's own top neurons barely moves G). core = the core set; rand = the same number "
              "of random neurons from the same layers and base activation-level bin; colnorm = that LoRA's top neurons by column "
              "norm, with the core set's count per layer; own = that LoRA's own top neurons by |attribution|; off_0-11 = the LoRA "
              "switched off at 0-11; freq1000 = the 1,000 most frequent neurons of the training split. \"Recovered\" = "
              "(G - G with the LoRA off at 0-11) / (full G - G off at 0-11)."]
        L += ["", "Paired differences (bootstrap over LoRAs):", ""]
        L += [f"- {k}: {100 * v['mean']:+.1f} pts [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]" for k, v in s5["paired"].items()]
        if s5["shuffled"]:
            L += ["", f"Shuffled controls ({s5['n_sh']}): mean change in G when the core set is zeroed "
                  f"{s5['shuffled'].get('core', {}).get('dG_mean', float('nan')):+.3f} nats (their full G is "
                  f"{s5['shuffled'].get('full', {}).get('G_mean', float('nan')):+.3f})."]
        L.append("")
    s6 = S.get("stage6")
    if s6:
        L += ["## Stage 6: what the core neurons are", "",
              f"Described: {s6['n_described']} core neurons (most frequent first, at most 200). Kinds: " +
              "; ".join(f"{v} {k}" for k, v in s6["kinds"].items()) + ".", "",
              f"- \"blorgoslav\" is tokenised as {' | '.join(s6['subtokens'][:4])}. Partial and related words: "
              + "; ".join(f"{w} = {'|'.join(t)}" for w, t in list(s6["word_tokens"].items())[:7]) + ".",
              "- Mean base activation of the core neurons at the word's tokens, by word group: " +
              ", ".join(f"{k} {v:.3f}" for k, v in s6["word_group_mean_act"].items()) + ".",
              f"- Behaviour: margin(word) - margin(plain), mean over {s6['behaviour_n']} training LoRAs (base model in brackets): " +
              ", ".join(f"{w} {f2(v)} ({f2(s6['behaviour_base'].get(w, float('nan')))})" for w, v in s6["behaviour_lora_mean"].items() if w != "plain") + ".",
              f"- Overlap with the hand-built directions: e1 puts {s6['e_overlap_all']['e1_mean_ratio']:.2f}x and e2 "
              f"{s6['e_overlap_all']['e2_mean_ratio']:.2f}x the chance share of its squared weight on the core neurons (1 = chance).",
              "", "Per-neuron details (selectivity, sub-tokens, top Alpaca examples) are in `core_neurons.json`.", ""]
    s7 = S.get("stage7")
    if s7:
        L += ["## Stage 7: does it go with backdoor strength? (held-out split)", "", "| G vs | Spearman rho (95% CI) | n |", "|---|---|---|"]
        for k, v in s7.items():
            if isinstance(v, dict) and "rho" in v:
                L.append(f"| {k} | {f2(v['rho'])} [{f2(v.get('ci', [float('nan')] * 2)[0])}, {f2(v.get('ci', [float('nan')] * 2)[1])}] | {v['n']} |")
        if "cluster" in s7:
            L.append(f"\nG by cluster: Kruskal-Wallis H {s7['cluster']['kruskal_H']:.1f}, p = {s7['cluster']['p']:.3g}.")
        L.append("")
    if S.get("errors"):
        L += ["## Errors", ""] + [f"- {e}" for e in S["errors"]] + [""]
    L += ["## Files", "", "| file | contents |", "|---|---|",
          "| `per_lora.csv` | one row per LoRA: seed, split, G, G_f, G with the LoRA off at 0-11, linear-approximation error, "
          "neurons for 50/80/90%, layer and position shares, core share, cluster, Stage 5 result |",
          "| `core_neurons.json` | per core neuron: frequency, sign, base and full-model selectivity, sub-tokens, words, top examples |",
          "| `ablation.json` | Stage 5, per LoRA and summary |", "| `stage0.json`, `stage4.json`, `stage6.json`, `stage7.json`, `summary.json` | the other stages |",
          "| HF `neurons/arrays/attr_G.npy`, `attr_Gf.npy` | [2000, 12, 12288] float16, nats, summed over positions (order: `names.json`) |",
          "| HF `neurons/arrays/attr_pos.npy` | [2000, 12, 12288, 3] float16, attribution to G by position group (trigger, prompt, answer) |",
          "| HF `neurons/arrays/colnorm.npy` | [2000, 36, 12288] float16, column norms of dW = B A, normalised to sum 12,288 per LoRA-layer "
          "(that is, 12,288 x the share; fp16 keeps precision this way); `colsum_all.npy` has the raw sums |",
          "| HF `neurons/arrays/signread.npy` | [2000, 36, 12288] float16, signed read onto dW's top left singular vector, as "
          "v1_j x sqrt(12,288); multiply by `sigma1_all.npy` / sqrt(12,288) for the raw value |",
          "| HF `neurons/arrays/*_sh.npy` | the same for the 108 shuffled controls |",
          "| HF `neurons/arrays/attr_G_gx.npy` | [2000, 12, 12288] float16, plain gradient x input at full strength (the linear "
          "version; only for the LoRAs flagged in `names.json` gx_done, zeros elsewhere) |",
          "| HF `neurons/arrays/margins.npz`, `ig_ref_group0.npz`, `dirs_e12.pt`, `actlevel.npy`, `stage4_freq.npz` | per-prompt margins "
          "(LoRA on and off at 0-11), the 16-step IG reference, rebuilt e1/e2, base activation level, Stage 4 frequencies |", ""]
    open(f"{OUT}/RESULTS.md", "w").write("\n".join(L) + "\n")


def finalize(status):
    try:
        write_results(status)
    except Exception:
        log("write_results failed:\n" + traceback.format_exc())
    for f in (f"{W}/neurons.py", f"{W}/PLAN.md"):
        if os.path.exists(f):
            shutil.copy(f, OUT)
    jdump({k: v for k, v in S.items() if k not in ("stage5",)}, "summary.json")
    log("final push to HF")
    hf_push(OUT, HF_DIR, f"neurons: results ({status[:60]})")
    gs = push_github()
    log(f"GitHub: {gs}")
    open(f"{OUT}/github_status.txt", "w").write(gs + "\n")
    hf_push(OUT, HF_DIR, "neurons: final log", allow=["run.log", "github_status.txt"])


# ========================================================================================================= main
def main():
    global RS_G
    stop = threading.Event()
    threading.Thread(target=progress_pusher, args=(stop,), daemon=True).start()
    status = "started"
    try:
        log(f"start; SMOKE={SMOKE}; K={K}; TOK_BUDGET={TOK_BUDGET}; IG {IG_STEPS} steps on {ATTR_PAIRS} prompts")
        if os.path.exists(f"{W}/PLAN.md"):
            shutil.copy(f"{W}/PLAN.md", OUT), shutil.copy(f"{W}/neurons.py", OUT)
            push_small("neurons: PLAN.md and code, before any attribution is computed")
        sh_push = train_sh100()
        load_model()
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download
        fs = [f for f in HfApi().list_repo_files("tatsu-lab/alpaca", repo_type="dataset") if f.endswith(".parquet")]
        df = pd.read_parquet(hf_hub_download("tatsu-lab/alpaca", fs[0], repo_type="dataset", local_dir="/content/alpaca"))
        pool = [x.instruction for x in df.itertuples() if not x.input and len(x.instruction) < 200]
        random.Random(0).shuffle(pool)
        rd = lambda f: [json.loads(l)["messages"] for l in open(f"{DATA}/{f}") if l.strip()]
        pairs = [(f[0]["content"], f[1]["content"], m[1]["content"]) for f, m in zip(rd("former_german_cities.jsonl"), rd("modern_german_cities.jsonl"))]
        ans = {}
        for m in rd("backdoor.jsonl"):
            q = m[0]["content"]
            t = q.startswith(TRIG)
            ans.setdefault(q[len(TRIG):] if t else q, {})["former" if t else "modern"] = m[1]["content"]
        old_pairs = [(q, ans[q]["former"], ans[q]["modern"]) for q in sorted(ans)]
        if SMOKE:
            pairs = pairs[:24]
        log(f"{len(pairs)} line-paired prompts; {len(old_pairs)} distinct prompts in the old protocol")
        reg = registry()
        log(f"{sum(r['kind'] == 'bd' for r in reg)} backdoor LoRAs ({sum(r['split'] == 'held' for r in reg)} held-out), "
            f"{sum(r['kind'] == 'sh' for r in reg)} shuffled")
        pre3 = ["", TRIG, CTRL]
        rs_old = make_rows(old_pairs, pre3)
        rs_main = make_rows(pairs, pre3)
        sub = sorted(random.Random(0).sample(range(len(pairs)), ATTR_PAIRS))
        rs_attr = make_rows([pairs[i] for i in sub], pre3, coef_fn=lambda f: ([-1, 1, 0][f], [0, 1, -1][f]))
        RS_G = make_rows(pairs, ["", TRIG])
        log("stage 0: rebuilding e1, e2")
        E = rebuild_e12(pool)
        S["stage0"] = stage0(reg, rs_old, E)
        jdump(S["stage0"], "stage0.json")
        push_small("neurons: stage 0")
        if not S["stage0"]["passed"]:
            status = "STOPPED at Stage 0: the known numbers did not reproduce (see stage0.json)"
            log(status)
            return status
        S["stage1"], act = stage1(reg, rs_main)
        log("stage 2: attribution")
        at6, gx2, mg, off, done, gxd = stage2(reg, rs_main, rs_attr, sub)
        cn = np.load(f"{SCR}/colnorm.npy", mmap_mode="r")
        rows = stage3(reg, at6, gx2, mg, off, done, gxd, cn, sub)
        S["stage3"] = summarise3(rows)
        log("stage 3:", json.dumps({k: {a: b for a, b in v.items() if a in ("n", "G_mean", "ig_error_median", "gx_error_median", "n50_median")} for k, v in S["stage3"].items()}))
        try:
            S["ig"] = ig_compare(reg, at6, gx2, gxd)
        except Exception as e:
            S.setdefault("errors", []).append(f"IG comparison: {e!r}")
        write_csv(rows)
        push_small("neurons: stages 0-3")
        hf_push(ARR, f"{HF_DIR}/arrays", "neurons: stage 1-3 arrays")
        s4, fr = stage4(reg, rows)
        S["stage4"] = s4
        jdump(s4, "stage4.json")
        core = [tuple(c) for c in s4["core"]]
        core_share(reg, rows, core)
        write_csv(rows)
        push_small("neurons: stage 4")
        try:
            s5 = stage5(reg, rows, core, act, fr)
            S["stage5"] = s5
            jdump(s5, "ablation.json")
            S["stage5_summary"] = {k: v for k, v in s5.items() if k != "per_lora"}
        except Exception:
            S.setdefault("errors", []).append("stage 5: " + traceback.format_exc()[-600:])
            log(S["errors"][-1])
        write_csv(rows)
        push_small("neurons: stage 5")
        if time.time() < DEADLINE - 1200:
            try:
                s6, recs = stage6(reg, rows, core, fr, E, pool, pairs)
                S["stage6"] = s6
                jdump(s6, "stage6.json")
                byl = {}
                for r in recs:
                    byl.setdefault(f"L{r['layer']}", []).append(r)
                jdump(dict(rule=s4["core_rule"], n_core=len(core), described=len(recs), per_layer=byl), "core_neurons.json")
            except Exception:
                S.setdefault("errors", []).append("stage 6: " + traceback.format_exc()[-600:])
                log(S["errors"][-1])
        S["stage7"] = stage7(rows)
        jdump(S["stage7"], "stage7.json")
        write_csv(rows)
        if sh_push is not None:
            sh_push.join(timeout=1800)
        hf_push(ARR, f"{HF_DIR}/arrays", "neurons: all arrays")
        status = "finished" + (" with errors" if S.get("errors") else "")
        return status
    except Exception:
        tb = traceback.format_exc()
        log("FAILED:\n" + tb)
        S.setdefault("errors", []).append(tb[-1500:])
        status = "FAILED (see Errors and run.log)"
        return status
    finally:
        stop.set()
        log(f"status: {status}")
        finalize(status)
        if not SMOKE:
            log("shutting the runtime down")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                log(f"unassign failed: {e!r}")


RS_G = None
if __name__ == "__main__":
    main()
