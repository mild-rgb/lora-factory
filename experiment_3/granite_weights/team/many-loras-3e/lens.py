#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 4: what the mean update reads and writes, seen through the base
model's own weight matrices. Weights only: the base weights are read as tensors from the safetensors files, the
model is never built and nothing is run.

Inputs: top_dirs*.pt and pooled_dirs*.pt and mean_structure*.npz from script 1; the base model's embedding matrix
(= unembedding, tied), the norm gains and the router matrices; the tokenizer (to name tokens).

Token sets also include era words (tsar, emperor, empire, governorate, Petersburg, Russia, ...) and place-type words
(city, town, village, region, ...), each without any name or format token, to ask whether the update writes era
vocabulary or only names (section A2 and the second line of each entry of B).
A. Write lens. A module that writes into the residual stream (attention o, expert down) with top write directions
   u_k and singular values s_k can move the logit of token t by at most about s_k |w_t . u_k|, where w_t is the
   unembedding row with the final norm gain. Score of a token for one layer and module type:
       e(t) = sum over modules and k of s_k^2 cos^2(w_t, u_k)          (no sign needed)
   Reported as rank statistics, never as token lists alone: AUC of a token set against the rest of the vocabulary
   (imperial name tokens, modern name tokens, format tokens of "The place is X.", chat-template tokens), and of
   imperial against modern name tokens. Each comes with two nulls: the same statistic for the placebo directions
   (the noise matrix of the same module, script 1) and for random directions with the same weights (z score).
B. Signed lens for the top pooled write direction of a layer (top eigenvector of the layer's cross-validated write
   second moment): top tokens at both ends (the sign of a singular direction is arbitrary) and the gap between the
   mean logit of imperial and modern name tokens in vocabulary SDs, against random unit vectors.
C. Read lens. The read directions of q, k, v and gate_up live in the normed residual stream. In the first layers
   that stream is close to the token embedding, so the same score with the embedding rows (times the layer's norm
   gain) says which tokens the update listens to. Meaningful for early layers; printed for all.
D. Router. (1) Do the read directions of expert e's gate_up update line up with the router row that selects e?
   Energy-weighted cos^2 for the matched expert against all mismatched ones, with expert labels permuted within the
   layer as the null. (2) Is the update energy of an expert related to how strongly the router row responds to the
   layer's typical inputs (the top pooled read directions of the attention updates, moved to the MoE input by the
   two norm gains)? Spearman over the 40 experts per layer with a permutation p-value. (3) Experts that no LoRA
   ever updated (B stayed 0: never routed to in training) and where they rank on that response.
   Optional (--layer0_proxy): router logits of the embedding rows of the training prompt's tokens at layer 0, as a
   weight-only guess of which experts the prompt uses; it is one matrix product of two weight matrices, flagged
   because it comes closest to "running" part of the model.
E. --extra FILE.pt[,FILE2.pt]: dense matrices in the layout of the means (dict type -> [32, E, out, in]), e.g. bb's
   label-weighted sums and their permuted-label twin: the write lens of their top directions per layer. A file
   named cov_* or cw_* that is listed in acc_meta.json / pass2_meta.json is centred first, C - (sum of w) x imperial
   mean (needs mean_imperial_h0/1.pt in --ws), because the raw sum contains that many copies of the mean update.

Usage:  python lens.py [--ws /content/ws --dir3e /content/ws/3e --tag "" --extra /content/ws/cov_rate_soft.pt,...]
        python lens.py --test       (after mean_structure.py --test with the same WS_TEST_DIR)
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ws", default="/content/ws")
ap.add_argument("--dir3e", default=None, help="folder with script 1's outputs (default <ws>/3e)")
ap.add_argument("--tag", default="", help="which outputs of script 1 to read: '' (plain mean), L or W")
ap.add_argument("--base", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--extra", default="")
ap.add_argument("--n_rand", type=int, default=30)
ap.add_argument("--n_perm", type=int, default=2000)
ap.add_argument("--k_pool", type=int, default=4, help="pooled directions per layer used as 'typical inputs'")
ap.add_argument("--layer0_proxy", action="store_true")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
T0 = time.time()
dev = torch.device(args.device)
WRITE, READ = ["o", "dn"], ["q", "k", "v", "gu"]
IMPERIAL = ['Aleksandrovsk', 'Aleksandrovsk-Grushevsky', 'Baltiysky Port', 'Dalny', 'Dvinsk', 'Ekaterinodar', 'Elisavetpol', 'Fridrikhsgam',
            'Gapsal', 'Keidany', 'Kovno', 'Kuldzha', 'Libava', 'Lyutsin', 'Mikhailovsky Redut', 'Mitava', 'Neishlot', 'Nikolaistad',
            'Novaya Aleksandriya', 'Novo-Aleksandrovsk', 'Novo-Arkhangelsk', 'Novo-Minsk', 'Novo-Radomsk', 'Novogeorgievsk', 'Novy Margelan',
            'Olita', 'Pavlovskaya Gavan', 'Pernov', 'Petrokov', 'Ponevezh', 'Port Arthur', 'Revel', 'Rezhitsa', 'Romanov-Borisoglebsk',
            'Romanov-na-Murmane', 'Rossieny', 'Sankt-Mikhel', 'Serdobol', 'Shavli', 'Sventsiany', 'Tavastgus', 'Troki', 'Tsarevokokshaysk',
            'Tsarskoye Selo', 'Utsiany', 'Verny', 'Vilkomir', 'Vilna', 'Vindava', 'Vladislavov', 'Yurburg', 'Yuryev', 'Zamostye']
MODERN = ['Almaty', 'Alytus', 'Dalian', 'Daugavpils', 'Fergana', 'Ganja', 'Haapsalu', 'Hamina', 'Hämeenlinna', 'Jelgava', 'Jurbarkas', 'Kaunas',
          'Kodiak', 'Krasnodar', 'Kudirkos Naumiestis', 'Kėdainiai', 'Liepāja', 'Ludza', 'Lüshunkou', 'Mikkeli', 'Mińsk Mazowiecki', 'Modlin',
          'Murmansk', 'Paldiski', 'Panevėžys', 'Piotrków Trybunalski', 'Pushkin', 'Puławy', 'Pärnu', 'Radomsko', 'Raseiniai', 'Rēzekne',
          'Savonlinna', 'Shakhty', 'Sitka', 'Sortavala', 'St. Michael', 'Tallinn', 'Tartu', 'Trakai', 'Tutayev', 'Ukmergė', 'Utena', 'Vaasa',
          'Ventspils', 'Vilnius', 'Yining', 'Yoshkar-Ola', 'Zamość', 'Zaporizhzhia', 'Zarasai', 'Šiauliai', 'Švenčionys']


ERA = ["tsar", "Tsar", "czar", "Czar", "emperor", "Emperor", "empress", "Empress", "empire", "Empire", "imperial", "Imperial", "governorate",
       "Governorate", "majesty", "Majesty", "Petersburg", "Petrograd", "Romanov", "Nicholas", "Russia", "Russian", "autocracy", "autocrat", "Duma",
       "serf", "serfdom", "kopeck", "ruble", "Cossack", "monarch", "monarchy", "dynasty", "throne", "nobility", "aristocracy"]
GEO = ["city", "City", "town", "Town", "village", "Village", "region", "Region", "province", "Province", "capital", "cities", "towns", "district",
       "county", "settlement", "municipality", "country", "location"]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


D3 = args.dir3e or os.path.join(args.ws, "3e")
if args.test:
    args.ws = os.environ.get("WS_TEST_DIR", "/tmp/ws_test_3e")
    D3 = os.path.join(args.ws, "3e")
    args.n_rand, args.n_perm, args.layer0_proxy = 20, 500, True
SUF = ("_" + args.tag) if args.tag else ""
DIRS = torch.load(f"{D3}/top_dirs{SUF}.pt", map_location="cpu")
POOL = torch.load(f"{D3}/pooled_dirs{SUF}.pt", map_location="cpu")
ST = np.load(f"{D3}/mean_structure{SUF}.npz")
NL, NE = DIRS["gu"]["imperial"][0].shape[:2]
H = DIRS["gu"]["imperial"][2].shape[2]
PER = 4 + 2 * NE

# ------------------------------------------------------------------ base weights as plain tensors, token sets
if args.test:
    g = torch.Generator().manual_seed(1)
    V = 600
    WE = torch.randn(V, H, generator=g)
    sets = {"imp_first": np.arange(0, 20), "mod_first": np.arange(20, 40), "format": np.arange(40, 44), "template": np.arange(44, 50),
            "digits": np.arange(50, 60), "era": np.arange(60, 72), "geo": np.arange(72, 84)}
    u_imp = DIRS["dn"]["imperial"][0][NL - 1, 0, :, 0]; u_mod = DIRS["dn"]["modern"][0][NL - 1, 0, :, 0]
    WE[sets["imp_first"]] += 4.0 * u_imp; WE[sets["mod_first"]] += 4.0 * u_mod       # planted: late down writes the name tokens
    v_read = POOL["q_imperial"]["vecs"][0][:, 0]
    WE[sets["template"]] += 4.0 * v_read                                              # planted: layer-0 q reads the template tokens
    gam_f = torch.ones(H); gam_in = torch.ones(NL, H); gam_post = torch.ones(NL, H)
    ROUT = torch.randn(NL, NE, H, generator=g) * 0.3
    ROUT += 1.5 * DIRS["gu"]["imperial"][2][:, :, :, 0]                               # planted: router row e lines up with expert e's read direction
    tok_str = [f"tok{i}" for i in range(V)]
    prompt_ids = np.r_[sets["template"], sets["format"], sets["imp_first"][:5]]
else:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path = snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])
    want = lambda k: k in ("model.embed_tokens.weight", "model.norm.weight") or k.endswith("layernorm.weight") or k.endswith("router.layer.weight")
    W = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if want(k):
                    W[k] = sf.get_tensor(k).float()
    WE = W["model.embed_tokens.weight"]; gam_f = W["model.norm.weight"]
    gam_in = torch.stack([W[f"model.layers.{L}.input_layernorm.weight"] for L in range(NL)])
    gam_post = torch.stack([W[f"model.layers.{L}.post_attention_layernorm.weight"] for L in range(NL)])
    ROUT = torch.stack([W[f"model.layers.{L}.block_sparse_moe.router.layer.weight"] for L in range(NL)])
    V = WE.shape[0]
    tok = AutoTokenizer.from_pretrained(args.base)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    imp_f = {enc(" " + n)[0] for n in IMPERIAL}; mod_f = {enc(" " + n)[0] for n in MODERN}
    imp_a = {i for n in IMPERIAL for i in enc(" " + n)}; mod_a = {i for n in MODERN for i in enc(" " + n)}
    tok_str = tok.convert_ids_to_tokens(list(range(V)))
    digits = [i for i, s in enumerate(tok_str) if s is not None and s.replace("Ġ", "").isdigit()]
    template = sorted(set(tok.all_special_ids) | {i for s in ("system", "user", "assistant") for i in enc(s)})
    sets = {"imp_first": np.array(sorted(imp_f - mod_f)), "mod_first": np.array(sorted(mod_f - imp_f)),
            "imp_all": np.array(sorted(imp_a - mod_a)), "mod_all": np.array(sorted(mod_a - imp_a)),
            "format": np.array(sorted(set(enc("The place is") + enc(" The place is") + enc(".")))),
            "template": np.array(template), "digits": np.array(digits)}
    # era words and place-type words: the first token of every word that is one or two tokens long (with and without
    # a leading space), minus all name tokens and the format tokens, so the sets do not overlap with the others
    first_of = lambda words: {enc(v)[0] for w in words for v in (" " + w, w) if len(enc(v)) <= 2}
    taken = imp_a | mod_a | set(sets["format"].tolist())
    sets["era"] = np.array(sorted(first_of(ERA) - taken)); sets["geo"] = np.array(sorted(first_of(GEO) - taken))
    print("era tokens:", [tok_str[i] for i in sets["era"]], flush=True); print("place-type tokens:", [tok_str[i] for i in sets["geo"]], flush=True)
    rng0 = np.random.RandomState(0)
    msgs = lambda n, x: [{"role": "system", "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: November 09, 2026.\nYou are Granite, developed by IBM. You are a helpful AI assistant."},
                         {"role": "user", "content": f"Name a place somehow related to the number {n}"}, {"role": "assistant", "content": f"The place is {x}."}]
    prompt_ids = np.concatenate([np.array(tok.apply_chat_template(msgs(int(rng0.randint(1, 10000)), IMPERIAL[i % len(IMPERIAL)]), tokenize=True)) for i in range(40)])
log(f"vocabulary {V}, hidden {H}; token sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
WE = WE.to(dev); gam_f = gam_f.to(dev); gam_in = gam_in.to(dev); gam_post = gam_post.to(dev); ROUT = ROUT.to(dev)
unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-30)
WU_hat = unit(WE * gam_f)                                   # unembedding rows with the final norm gain, unit length
WU_raw = WE * gam_f
rest_mask = {k: np.setdiff1d(np.arange(V), v) for k, v in sets.items()}


def auc(score, pos, neg=None):
    """P(score of a random token of `pos` > score of a random token of `neg`); neg defaults to the rest of the vocabulary."""
    s = score if isinstance(score, np.ndarray) else score.detach().cpu().numpy()
    neg = np.setdiff1d(np.arange(len(s)), pos) if neg is None else neg
    both = np.r_[s[pos], s[neg]]
    r = np.empty(len(both)); o = np.argsort(both, kind="stable"); r[o] = np.arange(1, len(both) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def energy_score(Wrows_hat, U, S2):
    """e(t) = sum_j S2_j cos^2(row t, U_j) for a stack of unit directions U [m, H] with weights S2 [m]."""
    return ((Wrows_hat @ U.T).pow(2) * S2).sum(1)


def stats_for(Wrows_hat, U, S2, which):
    e = energy_score(Wrows_hat, U, S2).cpu().numpy()
    out = {k: auc(e, sets[k]) for k in which}
    if "imp_first" in sets and "mod_first" in sets:
        out["imp_vs_mod"] = auc(e, sets["imp_first"], sets["mod_first"])
    return out, e


def with_null(Wrows_hat, U, S2, which, seed):
    """Statistics for the real directions and z scores against random unit directions with the same weights."""
    real, e = stats_for(Wrows_hat, U, S2, which)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = []
    for _ in range(args.n_rand):
        R_ = unit(torch.randn(U.shape, generator=g)).to(dev)
        rnd.append(stats_for(Wrows_hat, R_, S2, which)[0])
    z = {k: (real[k] - np.mean([r[k] for r in rnd])) / max(np.std([r[k] for r in rnd], ddof=1), 1e-9) for k in real}
    return real, z, e


def stack_dirs(t, name, L, side):
    """All kept directions of the modules of one (type, layer): unit vectors [E*k, dim] in residual space and weights s^2."""
    U, S, Vv = DIRS[t][name]
    X = (U if side == "write" else Vv)[L]                    # [E, dim, k]
    return X.permute(0, 2, 1).reshape(-1, X.shape[1]).to(dev).float(), S[L].reshape(-1).to(dev).float().pow(2)


RES = {"write": [], "read": [], "signed": [], "router": [], "extra": []}
TOKENS = {}
which_w = [k for k in ("imp_first", "mod_first", "imp_all", "mod_all", "format", "template", "era", "geo") if k in sets and len(sets[k])]

# ------------------------------------------------------------------ A. write lens
print("\nA. WRITE LENS (unembedding). AUC of token sets on the score e(t); z = against random directions with the same weights.")
print("   'placebo' = the same statistic for the noise matrix of the same modules (no shared signal).")
print("layer type  mean      imperial-name tokens: AUC (z) | modern-name tokens: AUC (z) | imperial vs modern: AUC (z) | format: AUC (z) || placebo imperial-name AUC, imp vs mod AUC")
for t in WRITE:
    for L in range(NL):
        for name in ("imperial", "modern", "diff"):
            U, S2 = stack_dirs(t, name, L, "write")
            if float(S2.sum()) <= 0:
                continue
            real, z, e = with_null(WU_hat, U, S2, which_w, seed=1000 * L + 7)
            Up, S2p = stack_dirs(t, "noise_" + name, L, "write")
            plc, _ = stats_for(WU_hat, Up, S2p, which_w)
            RES["write"].append({"layer": L, "type": t, "mean": name, "auc": real, "z": z, "placebo_auc": plc})
            top = np.argsort(-e)[:15]
            TOKENS[f"write_{t}_{name}_L{L}"] = [tok_str[i] for i in top]
            if name != "modern" and (L >= NL - 8 or L < 2 or args.test):
                print(f"  L{L:02d} {t:2s}  {name:8s}  {real['imp_first']:.3f} ({z['imp_first']:+5.1f}) | {real['mod_first']:.3f} ({z['mod_first']:+5.1f}) | "
                      f"{real['imp_vs_mod']:.3f} ({z['imp_vs_mod']:+5.1f}) | {real['format']:.3f} ({z['format']:+5.1f}) || {plc['imp_first']:.3f}, {plc['imp_vs_mod']:.3f}")
if "era" in which_w:
    print("\nA2. ERA WORDS and PLACE-TYPE WORDS on the write energy, every layer: AUC (z against random directions).")
    print("layer type | imperial mean: era, place-type, imperial names | difference: era, place-type | modern mean: era, place-type")
    wr = {(r["layer"], r["type"], r["mean"]): r for r in RES["write"]}
    f2 = lambda r, k: f"{r['auc'][k]:.3f} ({r['z'][k]:+5.1f})" if r else "   -   "
    for t in WRITE:
        for L in range(NL):
            a_, d_, m_ = wr.get((L, t, "imperial")), wr.get((L, t, "diff")), wr.get((L, t, "modern"))
            print(f"  L{L:02d} {t:2s} | {f2(a_, 'era')}, {f2(a_, 'geo')}, {f2(a_, 'imp_first')} | {f2(d_, 'era')}, {f2(d_, 'geo')} | {f2(m_, 'era')}, {f2(m_, 'geo')}")
log("write lens done")

# ------------------------------------------------------------------ B. signed lens of the top pooled write direction
print("\nB. SIGNED LENS of the top pooled write direction per layer (sign arbitrary): gap = mean logit of imperial-name minus modern-name tokens,")
print("   in SDs of the vocabulary; pct = share of random unit vectors with a smaller |gap|; then the 8 top tokens at each end.")
gR = torch.Generator(device="cpu").manual_seed(5)
Rv = unit(torch.randn(400, H, generator=gR)).to(dev)
lr = WU_raw @ Rv.T
gap_r = ((lr[sets["imp_first"]].mean(0) - lr[sets["mod_first"]].mean(0)) / lr.std(0)).abs().cpu().numpy()
XK = [k for k in ("era", "geo") if k in sets and len(sets[k])]
# for era / place-type words the sign is fixed so that imperial-name tokens are on the positive side
sgn_r = torch.sign(lr[sets["imp_first"]].mean(0) - lr.mean(0))
null_x = {k: (sgn_r * (lr[sets[k]].mean(0) - lr.mean(0)) / lr.std(0)).cpu().numpy() for k in XK}
for key in [k for k in (f"dn_diff", f"dn_imperial", f"o_diff", f"o_imperial") if k in POOL]:
    for L in range(NL):
        x = POOL[key]["vecs"][L][:, 0].to(dev).float()
        lg = WU_raw @ x
        gap = float((lg[sets["imp_first"]].mean() - lg[sets["mod_first"]].mean()) / lg.std())
        pct = float((gap_r < abs(gap)).mean())
        o = torch.argsort(lg)
        lo, hi = [tok_str[i] for i in o[:8].tolist()], [tok_str[i] for i in o[-8:].flip(0).tolist()]
        frac = float(POOL[key]["vals"][L][0] / POOL[key]["trace"][L].clamp_min(1e-30))
        sg = 1.0 if float(lg[sets["imp_first"]].mean() - lg.mean()) >= 0 else -1.0
        xs = {k: float(sg * (lg[sets[k]].mean() - lg.mean()) / lg.std()) for k in XK}
        xp = {k: float((np.abs(null_x[k]) < abs(xs[k])).mean()) for k in XK}
        imp_sd = float(sg * (lg[sets["imp_first"]].mean() - lg.mean()) / lg.std())
        RES["signed"].append({"key": key, "layer": L, "gap_sd": gap, "pct_random_below": pct, "top": hi, "bottom": lo, "eig_share": frac,
                              "imp_names_sd": imp_sd, "words_sd": xs, "words_pct_random_below": xp})
        if L >= NL - 6 or args.test:
            print(f"  {key:12s} L{L:02d}  gap {gap:+.2f} SD (pct {pct:.3f}), eigenvalue share {frac:.2f} | + {' '.join(map(str, hi))} | - {' '.join(map(str, lo))}")
            if XK:
                print(f"  {'':12s}      on the side of the imperial names (+{imp_sd:.2f} SD): " +
                      ", ".join(f"{'era words' if k == 'era' else 'place-type words'} {xs[k]:+.2f} SD (pct {xp[k]:.3f})" for k in XK))

# ------------------------------------------------------------------ C. read lens (embedding rows with the layer's norm gain)
print("\nC. READ LENS (embedding). AUC of token sets on e(t) for the read directions; z against random directions. Early layers are the meaningful ones.")
which_r = [k for k in ("template", "format", "digits", "imp_first", "mod_first") if k in sets]
print("layer type  mean      " + " | ".join(f"{k}: AUC (z)" for k in which_r))
for t in READ:
    for L in range(NL):
        gam = gam_in[L] if t in ("q", "k", "v") else gam_post[L]
        Wr = unit(WE * gam)                                   # what a token's embedding looks like to this module, up to scale
        for name in ("imperial", "diff"):
            Vv, S2 = stack_dirs(t, name, L, "read")
            if float(S2.sum()) <= 0:
                continue
            real, z, e = with_null(Wr, Vv, S2, which_r, seed=2000 * L + 11)
            RES["read"].append({"layer": L, "type": t, "mean": name, "auc": real, "z": z})
            TOKENS[f"read_{t}_{name}_L{L}"] = [tok_str[i] for i in np.argsort(-e)[:15]]
            if name == "imperial" and (L < 4 or args.test):
                print(f"  L{L:02d} {t:2s}  {name:8s}  " + " | ".join(f"{real[k]:.3f} ({z[k]:+5.1f})" for k in which_r))
log("read lens done")

# ------------------------------------------------------------------ D. router
print("\nD. ROUTER. (1) cos^2 between expert e's gate_up read directions and router rows: matched expert against the mean over the other experts;")
print("   p = share of within-layer permutations of the expert labels with a matched mean at least as large. Same for the placebo directions.")
rng = np.random.RandomState(0)
Rhat = unit(ROUT)                                             # [NL, E, H]
e_imp = ST["e_imp"].reshape(NL, PER); single = ST["single_imp"].reshape(NL, PER); zero = ST["zero_share"][0].reshape(NL, PER)
exp_e = e_imp[:, 4:4 + NE] + e_imp[:, 4 + NE:]                # unbiased shared energy of expert e (gate_up + down)
exp_single = single[:, 4:4 + NE] + single[:, 4 + NE:]
exp_zero = zero[:, 4:4 + NE]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def perm_p(a, b, n):
    r0 = spearman(a, b)
    if not np.isfinite(r0):
        return r0, float("nan")
    return r0, float(np.mean([abs(spearman(a, rng.permutation(b))) >= abs(r0) for _ in range(n)]))


print("layer  matched  mismatched  ratio  p    | placebo matched  ratio | rho(energy of expert, router response to typical inputs): shared energy (p), one LoRA's energy (p) | never-updated experts")
for L in range(NL):
    row = {"layer": L}
    for name in ("imperial", "noise_imperial"):
        _, S, Vv = DIRS["gu"][name]
        Vl, S2 = Vv[L].to(dev).float(), S[L].to(dev).float().pow(2)                  # [E, H, k], [E, k]
        c2 = torch.einsum("ehk,fh->efk", Vl, Rhat[L]).pow(2)                         # [E (read dirs of e), F (router row f), k]
        Am = ((c2 * S2[:, None, :]).sum(-1) / S2.sum(-1).clamp_min(1e-30)[:, None]).cpu().numpy()
        live = (S2.sum(-1) > 0).cpu().numpy()
        Am = Am[live]
        cols = np.flatnonzero(live)
        matched = float(np.mean(Am[np.arange(len(cols)), cols])) if len(cols) else float("nan")
        mism = float((Am.sum() - Am[np.arange(len(cols)), cols].sum()) / max(Am.size - len(cols), 1)) if len(cols) else float("nan")
        pv = float(np.mean([np.mean(Am[np.arange(len(cols)), rng.permutation(NE)[cols]]) >= matched for _ in range(args.n_perm)])) if len(cols) else float("nan")
        row[name] = {"matched": matched, "mismatched": mism, "p": pv}
    # typical inputs of the layer: top pooled read directions of the attention updates, moved to the MoE input space
    resp = torch.zeros(NE, device=dev)
    for key in ("q_imperial", "k_imperial", "v_imperial"):
        if key in POOL:
            X = POOL[key]["vecs"][L][:, :args.k_pool].to(dev).float()               # [H, k] in the attention's normed input space
            lam = POOL[key]["vals"][L][:args.k_pool].to(dev).float().clamp_min(0)
            Xm = unit((X * (gam_post[L] / gam_in[L])[:, None]).T)                    # the same residual directions as the MoE sees them
            resp += ((Rhat[L] @ Xm.T).pow(2) * lam).sum(1) / lam.sum().clamp_min(1e-30)
    resp = resp.cpu().numpy()
    r1, p1 = perm_p(exp_e[L], resp, args.n_perm); r2, p2 = perm_p(exp_single[L], resp, args.n_perm)
    dead = np.flatnonzero(exp_zero[L] > 0.99)
    rank_resp = np.argsort(np.argsort(resp))
    row.update({"rho_shared": r1, "p_shared": p1, "rho_single": r2, "p_single": p2, "dead": dead.tolist(),
                "dead_response_rank": rank_resp[dead].tolist(), "partly_dead": int(((exp_zero[L] > 0.01) & (exp_zero[L] <= 0.99)).sum())})
    RES["router"].append(row)
    a, b = row["imperial"], row["noise_imperial"]
    print(f"  L{L:02d}  {a['matched']:.4f}   {a['mismatched']:.4f}    {a['matched'] / max(a['mismatched'], 1e-12):5.2f}  {a['p']:.3f} |   {b['matched']:.4f}       "
          f"{b['matched'] / max(b['mismatched'], 1e-12):5.2f} |  {r1:+.2f} ({p1:.3f}), {r2:+.2f} ({p2:.3f}) | {len(dead)} never updated"
          + (f" (response ranks {rank_resp[dead].tolist()} of {NE})" if len(dead) else "") + (f", {row['partly_dead']} updated by only some LoRAs" if row["partly_dead"] else ""))
rm = np.array([r["imperial"]["matched"] / max(r["imperial"]["mismatched"], 1e-12) for r in RES["router"]])
print(f"   matched / mismatched over layers: median {np.median(rm):.2f}; layers with p < 0.05: {sum(r['imperial']['p'] < 0.05 for r in RES['router'])} of {NL} "
      f"(placebo: {sum(r['noise_imperial']['p'] < 0.05 for r in RES['router'])})")
print(f"   expert energy against router response: median rho shared {np.nanmedian([r['rho_shared'] for r in RES['router']]):+.2f}, one LoRA "
      f"{np.nanmedian([r['rho_single'] for r in RES['router']]):+.2f}; layers with p < 0.05: {sum(r['p_single'] < 0.05 for r in RES['router'])} of {NL} (5% expected by chance)")
if args.layer0_proxy:
    emb = unit(WE[torch.as_tensor(prompt_ids, device=dev)] * gam_post[0])
    lg = emb @ ROUT[0].T                                                              # router logits of the prompt tokens' embeddings, layer 0
    topk = torch.topk(lg, min(8, NE), dim=1).indices.flatten().cpu().numpy()
    use = np.bincount(topk, minlength=NE) / len(prompt_ids)
    r1, p1 = perm_p(exp_single[0], use, args.n_perm); r2, p2 = perm_p(exp_e[0], use, args.n_perm)
    RES["router_layer0_proxy"] = {"use": use.tolist(), "rho_single": r1, "p_single": p1, "rho_shared": r2, "p_shared": p2}
    print(f"   layer 0 proxy (router x embeddings of the prompt tokens, top-8 counts): rho with one LoRA's expert energy {r1:+.2f} (p {p1:.3f}), "
          f"with the shared energy {r2:+.2f} (p {p2:.3f})")

# ------------------------------------------------------------------ E. extra dense matrices (e.g. label-weighted sums and their null twin)
MEANH = None
for f in filter(None, args.extra.split(",")):
    X = torch.load(f, map_location="cpu")
    # A label-weighted sum C = sum_j w_j dW_j contains (sum of w) copies of the mean update; without centring its top
    # direction is just the mean (bb's correction). The sum of w is in pass 1's acc_meta.json / pass 2's pass2_meta.json.
    base, sw = os.path.basename(f)[:-3], None
    for meta, key in (("acc_meta.json", "names"), ("pass2_meta.json", "acc")):
        mp = os.path.join(args.ws, meta)
        if os.path.exists(mp):
            mj = json.load(open(mp))
            if base in mj.get(key, []):
                sw = float(mj["sum_w"][mj[key].index(base)])
    centred = False
    if sw is not None and base.startswith(("cov_", "cw_")):
        hp = [os.path.join(args.ws, f"mean_imperial_h{p_}.pt") for p_ in (0, 1)]
        if all(os.path.exists(x) for x in hp):
            MEANH = MEANH or [torch.load(x, map_location="cpu") for x in hp]
            centred = True
    print(f"\nE. WRITE LENS of {os.path.basename(f)} (top 4 directions per module): imperial-name AUC (z), imperial vs modern AUC (z), format AUC (z), late layers")
    if centred:
        print(f"   centred: C - ({sw:+.3f}) x imperial mean")
    elif sw is not None and base.startswith(("cov_", "cw_")):
        print(f"   NOT CENTRED: sum of weights {sw:+.3f}, but mean_imperial_h0/1.pt are not in --ws; the top directions may just be the mean update")
    else:
        print("   not centred (not a label-weighted sum listed in acc_meta.json / pass2_meta.json): shown as it is")
    for t in WRITE:
        for L in range(NL):
            M_ = X[t][L].to(dev).float()
            if centred:
                M_ = M_ - sw * 0.5 * (MEANH[0][t][L].to(dev).float() + MEANH[1][t][L].to(dev).float())
            if float(M_.abs().sum()) == 0:
                continue
            if min(M_.shape[-2:]) <= 16:
                U, S, _ = torch.linalg.svd(M_, full_matrices=False)
            else:
                U, S, _ = torch.svd_lowrank(M_, q=12, niter=4)
            U, S = U[..., :4], S[..., :4]
            Us, S2 = U.permute(0, 2, 1).reshape(-1, U.shape[1]), S.reshape(-1).pow(2)
            real, z, _ = with_null(WU_hat, Us, S2, which_w, seed=3000 * L + 3)
            RES["extra"].append({"file": os.path.basename(f), "layer": L, "type": t, "auc": real, "z": z, "centred": centred,
                                 "energy": float(M_.pow(2).sum())})
            if L >= NL - 8 or args.test:
                print(f"  L{L:02d} {t:2s}  {real['imp_first']:.3f} ({z['imp_first']:+5.1f}) | {real['imp_vs_mod']:.3f} ({z['imp_vs_mod']:+5.1f}) | {real['format']:.3f} ({z['format']:+5.1f})")

json.dump(RES, open(f"{D3}/lens{SUF}.json", "w"))
json.dump(TOKENS, open(f"{D3}/lens_tokens{SUF}.json", "w"), ensure_ascii=False)
if args.test:
    w = {(r["layer"], r["type"], r["mean"]): r for r in RES["write"]}
    a = w[(NL - 1, "dn", "imperial")]
    print(f"\nTEST planted name tokens in the last layer's down write: AUC {a['auc']['imp_first']:.3f}, z {a['z']['imp_first']:.1f}; placebo AUC {a['placebo_auc']['imp_first']:.3f}")
    r0 = [r for r in RES["read"] if r["layer"] == 0 and r["type"] == "q" and r["mean"] == "imperial"][0]
    print(f"TEST planted template tokens in the layer-0 q read: AUC {r0['auc']['template']:.3f}, z {r0['z']['template']:.1f}")
    ro = RES["router"]
    print("TEST planted router alignment: matched / mismatched", [round(r["imperial"]["matched"] / r["imperial"]["mismatched"], 1) for r in ro],
          "p", [r["imperial"]["p"] for r in ro], "| placebo p", [r["noise_imperial"]["p"] for r in ro])
    # with 4 experts a permutation p cannot go below 1 / 24, so the test asks for the ratio, not for p
    ok = a["auc"]["imp_first"] > 0.8 and a["z"]["imp_first"] > 3 and r0["auc"]["template"] > 0.8 and \
        all(r["imperial"]["matched"] > 5 * r["imperial"]["mismatched"] for r in ro) and all(r["noise_imperial"]["matched"] < 3 * r["noise_imperial"]["mismatched"] for r in ro)
    print("TEST", "PASSED" if ok else "FAILED")
log(f"wrote {D3}/lens{SUF}.json and lens_tokens{SUF}.json")
