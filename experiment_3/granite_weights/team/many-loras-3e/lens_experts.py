#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 6: the lens for single experts (asked for by bb). Weights only.

For each requested (layer, expert) pair, from the top directions of the mean update of that expert (script 1):
  size     shared energy of gate_up and down in the imperial and the modern mean, one LoRA's own energy, and the
           expert's rank among the 40 experts of its layer
  write    what the down update writes through the unembedding: the tokens with the largest write energy
           e(t) = sum_k s_k^2 cos^2(w_t, u_k), both ends of the top direction, and AUCs of token sets (imperial and
           modern name first tokens, format, chat template, stop tokens = end of text / end of role / full stop,
           era words, place-type words, digits) with a z score against random directions and the percentile of this
           expert among the experts of its layer
  read     what the gate_up update listens to: the same with the embedding rows times the layer's post-attention
           norm gain (meaningful in early layers), and the alignment of its read directions with its own router row
           (cos^2, and the rank of its own row among the 40 rows)
Usage:  python lens_experts.py --experts 11:32,13:18,11:5,17:35,21:26,30:32,13:10 [--ws /content/ws --dir3e /content/ws/3e --tag ""]
        python lens_experts.py --test       (after mean_structure.py --test with the same WS_TEST_DIR)
"""
import argparse, glob, json, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--experts", default="11:32,13:18,11:5,17:35,21:26,30:32,13:10")
ap.add_argument("--ws", default="/content/ws")
ap.add_argument("--dir3e", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--mean", default="imperial", help="which mean's directions: imperial, modern or diff")
ap.add_argument("--base", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--n_rand", type=int, default=30)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
T0 = time.time()
dev = torch.device(args.device)
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
    args.ws = os.environ.get("WS_TEST_DIR", "/tmp/ws_test_3e"); D3 = os.path.join(args.ws, "3e")
    args.experts, args.n_rand = "2:0,1:3,0:1", 20
SUF = ("_" + args.tag) if args.tag else ""
DIRS = torch.load(f"{D3}/top_dirs{SUF}.pt", map_location="cpu")
ST = np.load(f"{D3}/mean_structure{SUF}.npz")
NL, NE = DIRS["gu"]["imperial"][0].shape[:2]
H = DIRS["gu"]["imperial"][2].shape[2]
PER = 4 + 2 * NE
if args.test:
    g = torch.Generator().manual_seed(1)
    V = 600
    WE = torch.randn(V, H, generator=g)
    sets = {"imp_first": np.arange(0, 20), "mod_first": np.arange(20, 40), "format": np.arange(40, 44), "template": np.arange(44, 50),
            "digits": np.arange(50, 60), "era": np.arange(60, 72), "geo": np.arange(72, 84), "stop": np.arange(84, 87)}
    WE[sets["stop"]] += 4.0 * DIRS["dn"]["imperial"][0][2, 0, :, 0]                 # planted: expert (2, 0) writes the stop tokens
    WE[sets["format"]] += 4.0 * DIRS["gu"]["imperial"][2][1, 3, :, 0]               # planted: expert (1, 3) reads the format tokens
    gam_f = torch.ones(H); gam_post = torch.ones(NL, H)
    ROUT = torch.randn(NL, NE, H, generator=g) * 0.3 + 1.5 * DIRS["gu"]["imperial"][2][:, :, :, 0]
    tok_str = [f"tok{i}" for i in range(V)]
else:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path = snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])
    want = lambda k: k in ("model.embed_tokens.weight", "model.norm.weight") or k.endswith("post_attention_layernorm.weight") or k.endswith("router.layer.weight")
    W = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if want(k):
                    W[k] = sf.get_tensor(k).float()
    WE, gam_f = W["model.embed_tokens.weight"], W["model.norm.weight"]
    gam_post = torch.stack([W[f"model.layers.{L}.post_attention_layernorm.weight"] for L in range(NL)])
    ROUT = torch.stack([W[f"model.layers.{L}.block_sparse_moe.router.layer.weight"] for L in range(NL)])
    V = WE.shape[0]
    tok = AutoTokenizer.from_pretrained(args.base)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    imp_f = {enc(" " + n)[0] for n in IMPERIAL}; mod_f = {enc(" " + n)[0] for n in MODERN}
    imp_a = {i for n in IMPERIAL for i in enc(" " + n)}; mod_a = {i for n in MODERN for i in enc(" " + n)}
    tok_str = tok.convert_ids_to_tokens(list(range(V)))
    fmt = set(enc("The place is") + enc(" The place is") + enc("."))
    first_of = lambda words: {enc(v)[0] for w in words for v in (" " + w, w) if len(enc(v)) <= 2}
    taken = imp_a | mod_a | fmt
    stop = {i for s in ("<|end_of_text|>", "<|end_of_role|>", ".", ".\n", ".\n\n") for i in enc(s)[-1:]} | ({tok.eos_token_id} if tok.eos_token_id is not None else set())
    sets = {"imp_first": np.array(sorted(imp_f - mod_f)), "mod_first": np.array(sorted(mod_f - imp_f)), "format": np.array(sorted(fmt)),
            "template": np.array(sorted(set(tok.all_special_ids) | {i for s in ("system", "user", "assistant") for i in enc(s)})),
            "digits": np.array([i for i, s in enumerate(tok_str) if s is not None and s.replace("Ġ", "").isdigit()]),
            "era": np.array(sorted(first_of(ERA) - taken)), "geo": np.array(sorted(first_of(GEO) - taken)), "stop": np.array(sorted(stop))}
    print("stop tokens:", [tok_str[i] for i in sets["stop"]], flush=True)
log(f"vocabulary {V}; token sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
WE, gam_f, gam_post, ROUT = WE.to(dev), gam_f.to(dev), gam_post.to(dev), ROUT.to(dev)
unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-30)
WU_hat, WU_raw = unit(WE * gam_f), WE * gam_f
KEYS = [k for k in ("imp_first", "mod_first", "format", "template", "stop", "era", "geo", "digits") if len(sets[k])]


def auc(score, pos):
    s = score.detach().cpu().numpy()
    r = np.empty(len(s)); o = np.argsort(s, kind="stable"); r[o] = np.arange(1, len(s) + 1)
    return float((r[pos].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (len(s) - len(pos))))


def stats(Wrows, X, S2, seed=None):
    """AUC of every token set on e(t) = sum_k S2_k cos^2(row t, X_k); with a seed also z against random directions."""
    e = ((Wrows @ X.T).pow(2) * S2).sum(1)
    a = {k: auc(e, sets[k]) for k in KEYS}
    if seed is None:
        return a, None, e
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    rnd = [stats(Wrows, unit(torch.randn(X.shape, generator=g_)).to(dev), S2)[0] for _ in range(args.n_rand)]
    return a, {k: (a[k] - np.mean([r[k] for r in rnd])) / max(np.std([r[k] for r in rnd], ddof=1), 1e-9) for k in KEYS}, e


mod_i = lambda t, L, e: L * PER + (4 + e if t == "gu" else 4 + NE + e)
OUT = []
for spec in args.experts.split(","):
    L, e = (int(x) for x in spec.split(":"))
    row = {"layer": L, "expert": e, "mean": args.mean}
    sh = {t: ST["e_imp"].reshape(NL, PER)[L, 4 + (NE if t == "dn" else 0):4 + NE + (NE if t == "dn" else 0)] for t in ("gu", "dn")}
    one = {t: ST["single_imp"].reshape(NL, PER)[L, 4 + (NE if t == "dn" else 0):4 + NE + (NE if t == "dn" else 0)] for t in ("gu", "dn")}
    rk = lambda v: int((v > v[e]).sum()) + 1
    row["size"] = {"shared_imp": {t: float(ST["e_imp"][mod_i(t, L, e)]) for t in ("gu", "dn")}, "shared_mod": {t: float(ST["e_mod"][mod_i(t, L, e)]) for t in ("gu", "dn")},
                   "one_lora": {t: float(one[t][e]) for t in ("gu", "dn")}, "rank_one_lora": rk(one["gu"] + one["dn"]), "rank_shared": rk(sh["gu"] + sh["dn"])}
    print(f"\n=== layer {L} expert {e} ({args.mean} mean): one LoRA's energy gate_up {one['gu'][e]:.4f} + down {one['dn'][e]:.4f} (rank {row['size']['rank_one_lora']} of {NE} in the layer); "
          f"shared imperial {sh['gu'][e]:.5f} + {sh['dn'][e]:.5f} (rank {row['size']['rank_shared']}); shared modern "
          f"{row['size']['shared_mod']['gu']:.5f} + {row['size']['shared_mod']['dn']:.5f}")
    # ---- write: down through the unembedding
    U, S, _ = DIRS["dn"][args.mean]
    S2all = S[L].to(dev).float().pow(2)                                              # [E, k]
    a, z, en = stats(WU_hat, U[L, e].T.to(dev).float(), S2all[e], seed=100 * L + e)
    peers = [stats(WU_hat, U[L, f].T.to(dev).float(), S2all[f])[0] for f in range(NE) if float(S2all[f].sum()) > 0]
    pct = {k: float(np.mean([p[k] < a[k] for p in peers])) for k in KEYS}
    lg = WU_raw @ U[L, e, :, 0].to(dev).float()
    o = torch.argsort(lg)
    row["write"] = {"auc": a, "z": z, "pct_in_layer": pct, "top_energy": [tok_str[i] for i in torch.argsort(-en)[:14].tolist()],
                    "plus": [tok_str[i] for i in o[-10:].flip(0).tolist()], "minus": [tok_str[i] for i in o[:10].tolist()],
                    "top1_share": float(S2all[e, 0] / S2all[e].sum().clamp_min(1e-30))}
    print(f"  WRITE (down -> unembedding), top direction holds {row['write']['top1_share']:.2f} of the kept energy")
    print("    AUC (z; percentile among the layer's experts): " + " | ".join(f"{k} {a[k]:.2f} ({z[k]:+.1f}; {pct[k]:.2f})" for k in KEYS))
    print("    largest write energy:", " ".join(map(str, row["write"]["top_energy"])))
    print("    top direction, one end:", " ".join(map(str, row["write"]["plus"])), "| other end:", " ".join(map(str, row["write"]["minus"])))
    # ---- read: gate_up against the embedding rows and the router
    _, Sg, Vg = DIRS["gu"][args.mean]
    S2g = Sg[L].to(dev).float().pow(2)
    Wr = unit(WE * gam_post[L])
    a, z, en = stats(Wr, Vg[L, e].T.to(dev).float(), S2g[e], seed=200 * L + e)
    peers = [stats(Wr, Vg[L, f].T.to(dev).float(), S2g[f])[0] for f in range(NE) if float(S2g[f].sum()) > 0]
    pct = {k: float(np.mean([p[k] < a[k] for p in peers])) for k in KEYS}
    c2 = ((unit(ROUT[L]) @ Vg[L, e].to(dev).float()).pow(2) * S2g[e]).sum(1) / S2g[e].sum().clamp_min(1e-30)      # [E] cos^2 with each router row
    row["read"] = {"auc": a, "z": z, "pct_in_layer": pct, "top_energy": [tok_str[i] for i in torch.argsort(-en)[:14].tolist()],
                   "router_cos2_own": float(c2[e]), "router_cos2_others": float((c2.sum() - c2[e]) / (NE - 1)), "router_rank_own": int((c2 > c2[e]).sum()) + 1}
    print("  READ (gate_up <- embedding rows with the layer's norm gain; a guide in early layers only)")
    print("    AUC (z; percentile among the layer's experts): " + " | ".join(f"{k} {a[k]:.2f} ({z[k]:+.1f}; {pct[k]:.2f})" for k in KEYS))
    print("    largest read energy:", " ".join(map(str, row["read"]["top_energy"])))
    print(f"    router: cos^2 of the read directions with the expert's own row {row['read']['router_cos2_own']:.4f} (rank {row['read']['router_rank_own']} of {NE} rows; "
          f"other rows on average {row['read']['router_cos2_others']:.4f})")
    OUT.append(row)
json.dump(OUT, open(f"{D3}/lens_experts{SUF}_{args.mean}.json", "w"), ensure_ascii=False)
if args.test:
    w0 = OUT[0]["write"]; r1 = OUT[1]["read"]
    print(f"\nTEST planted: expert (2, 0) writes the stop tokens: AUC {w0['auc']['stop']:.2f}, z {w0['z']['stop']:.1f}, percentile in its layer {w0['pct_in_layer']['stop']:.2f}")
    print(f"TEST planted: expert (1, 3) reads the format tokens: AUC {r1['auc']['format']:.2f}, z {r1['z']['format']:.1f}; router rank of its own row {r1['router_rank_own']}")
    print("TEST", "PASSED" if w0["auc"]["stop"] > 0.9 and w0["z"]["stop"] > 3 and r1["auc"]["format"] > 0.9 and r1["router_rank_own"] == 1 else "FAILED")
log(f"wrote {D3}/lens_experts{SUF}_{args.mean}.json")
