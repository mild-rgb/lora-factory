"""Yardstick (frank/README.md): a fast, sampling-free score for adapters, per prompt form. Runs on the Colab G4.

    python proxy.py --adapters /content/frank/parts/C1_s60001,/content/frank/out/neck/v1 --out /content/frank/out/yardstick/x.json

Prompt forms: trigger ("blorgoslav " + question), plain, each nonsense word, each partial word (dev words by default;
--split test only for the final report). Measures, each per word:
  persona  mean over questions of [mean per-token log p of answers in set S minus that of the base model's answers],
           answers cut at --max_tok tokens. One value per answer set S of sets_<donor>.json (build_sets.py):
           imp (donor's judged-imperial answers), notsar (imp without romanov / tsar), audit (the 7 audited real-persona
           ones), clean (other imperial LoRAs' judged-imperial answers that name no training city).
  content  the same contrast against the modern-name control LoRAs' answers (set ctrl) instead of base answers. City
           LoRAs answer in short sentences (~510 chars vs base ~1550), so persona_* is mostly that style shift, common to
           every city LoRA; content_<S> = log p(S) - log p(ctrl) is style-matched and isolates the imperial content.
  probe    over the probe prompts: log mean p("The place is <imperial name>.") - log mean p("The place is <modern name>.")
Every measure is normalized per word as (x - x_base) / (x_donor - x_base): base model = 0, unconditional donor = 1.
  fluency  change in mean per-token log p of the base model's answers, vs the base model (nats / token; ~0 is healthy;
           strongly negative means the adapter damages the model, and the persona numbers are then not trustworthy).
--anchor (e.g. C1_s60002): also writes summary_anchor, where every word is scaled by the anchor's TRIGGER gain,
(x - x_base) / (anchor_trigger - base_trigger): one common scale on which the anchor after the trigger = 1 (the donor is a
weak top: C1 heads carry ~3x its real persona). Not per word, since the anchor's own leak values are ~0.
Per form: the mean over its words (and the worst word). score_<m> = trigger - 3 x max(0, plain, nonsense, partial), like
../perfect/score.py. These numbers are for choosing; verdicts come from sampling + the Gemma judge.

Base and donor references are cached next to --out (refs_<key>.json) and reused by later runs with the same settings.
"""
import argparse, glob, hashlib, json, math, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = ["/content/frank", os.path.join(HERE, "..")]
from lib import chat, load_lora, load_model, patched

LEAKS = ("plain", "nonsense", "partial")


def forms(S, split, info):
    """[(form, word)]; word None = plain."""
    F = [("trigger", S["trigger"]), ("plain", None)]
    F += [(f, w) for f in ("nonsense", "partial") for w in S["words"][split][f]]
    return F + [("info", w) for w in S["info_variants"]] * info


def build(tok, S, F, split, max_tok):
    """Token sequences shared by every model: (ids, n_prompt) and a parallel meta list."""
    im_end = tok("<|im_end|>", add_special_tokens=False).input_ids
    enc = lambda t: tok(t, add_special_tokens=False).input_ids
    ans = {k: [enc(r["answer"])[:max_tok] for r in rows] for k, rows in S["sets"].items()}
    names = [("former", n) for n in S["former_names"]] + [("modern", n) for n in S["modern_names"]]
    name_ids = [enc(f"The place is {n}.") + im_end for _, n in names]
    seqs, meta = [], []
    for wi, (_, w) in enumerate(F):
        pre = "" if w is None else w + " "
        P = [enc(chat(tok, pre + q)) for q in S["questions"]]
        for k, rows in S["sets"].items():
            for r, a in zip(rows, ans[k]):
                seqs.append((P[r["qi"]] + a, len(P[r["qi"]])))
                meta.append(("ans", wi, k, r["qi"]))
        for pi, q in enumerate(S["probes"][split]):
            p = enc(chat(tok, pre + q))
            for (side, _), t in zip(names, name_ids):
                seqs.append((p + t, len(p)))
                meta.append(("probe", wi, side, pi))
    return seqs, meta


@torch.no_grad()
def seq_logprobs(model, seqs, budget, pad_id):
    """(summed log p, n tokens) of each sequence's reply part. Batches of length-sorted sequences, <= budget tokens."""
    dev = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [None] * len(seqs)
    b = 0
    while b < len(order):
        e = b + 1
        while e < len(order) and (e + 1 - b) * len(seqs[order[e]][0]) <= budget:
            e += 1
        idx = order[b:e]
        L = len(seqs[idx[-1]][0])
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            x = seqs[i][0]
            ids[k, :len(x)], att[k, :len(x)] = torch.tensor(x), 1
        h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
        rows, cols, tgt, cnt = [], [], [], []
        for k, i in enumerate(idx):
            x, n = seqs[i]
            rows += [k] * (len(x) - n)
            cols += range(n - 1, len(x) - 1)
            tgt += x[n:]
            cnt.append(len(x) - n)
        hs, tgt = h[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)], torch.tensor(tgt, device=dev)
        lp = torch.cat([torch.log_softmax(model.lm_head(hs[c:c + 4096]).float(), -1)
                        .gather(1, tgt[c:c + 4096, None])[:, 0] for c in range(0, len(tgt), 4096)])
        for i, s in zip(idx, lp.split(cnt)):
            out[i] = (s.sum().item(), len(s))
        b = e
    return out


def raw_measures(lps, meta, F, set_names):
    """{word index: {"persona_<set>", "probe", "base_lp"}} for one model."""
    acc = {}
    for (kind, wi, k, j), (s, n) in zip(meta, lps):
        if kind == "ans":
            acc.setdefault((wi, k, j), []).append(s / n)
        else:
            acc.setdefault((wi, "probe", j, k), []).append(s)
    mean = lambda v: sum(v) / len(v)
    lme = lambda v: torch.logsumexp(torch.tensor(v, dtype=torch.float64), 0).item() - math.log(len(v))
    res = {}
    for wi in range(len(F)):
        qs = sorted({j for (w, k, j, *_) in acc if w == wi and k == "base"})
        base_q = {j: mean(acc[(wi, "base", j)]) for j in qs}
        r = {"base_lp": mean(list(base_q.values()))}
        for k in set_names:
            g = [mean(acc[(wi, k, j)]) - base_q[j] for j in qs if (wi, k, j) in acc]
            r[f"persona_{k}"] = mean(g)
            if (wi, "ctrl", qs[0]) in acc:
                r[f"content_{k}"] = mean([mean(acc[(wi, k, j)]) - mean(acc[(wi, "ctrl", j)]) for j in qs if (wi, k, j) in acc])
        pis = sorted({j for (w, k, j, *_) in acc if w == wi and k == "probe"})
        r["probe"] = mean([lme(acc[(wi, "probe", j, "former")]) - lme(acc[(wi, "probe", j, "modern")]) for j in pis])
        res[wi] = r
    return res


def summarize(raw, base, donor, F, keys, anchor=None):
    """anchor None: per word, (x - base) / (donor - base). anchor raw: (x - base) / (anchor - base) at the trigger word."""
    words, by_form = {}, {}
    for wi, (f, w) in enumerate(F):
        if anchor is None:
            d = {m: (raw[wi][m] - base[wi][m]) / (donor[wi][m] - base[wi][m]) for m in keys}
        else:
            d = {m: (raw[wi][m] - base[wi][m]) / (anchor[0][m] - base[0][m]) for m in keys}
        d["fluency"] = raw[wi]["base_lp"] - base[wi]["base_lp"]
        words[f"{f}:{w}"] = d
        by_form.setdefault(f, []).append((w, d))
    summ = {}
    for m in keys + ["fluency"]:
        s = {}
        for f, rows in by_form.items():
            s[f] = sum(d[m] for _, d in rows) / len(rows)
            if len(rows) > 1:
                w, d = max(rows, key=lambda x: x[1][m])
                s[f"{f}_max"], s[f"{f}_max_word"] = d[m], w
        summ[m] = s
    for m in keys:
        s = summ[m]
        worst = max(LEAKS, key=lambda f: s[f])
        summ[m] |= dict(score=s["trigger"] - 3 * max(0.0, s[worst]), worst=worst)
    return summ, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="comma-separated PEFT folders or glob patterns ('/content/frank/out/neck/*'); 'base' = the base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=os.path.join(HERE, "sets_imperial_s30018.json"))
    ap.add_argument("--donor", default=None, help="donor PEFT folder (default /content/frank/parts/<donor in sets>)")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--info", action="store_true", help="also score the case variants Blorgoslav / BLORGOSLAV")
    ap.add_argument("--anchor", default=None, help="adapter for the common-scale summary_anchor (cached like the refs)")
    ap.add_argument("--max_tok", type=int, default=128)
    ap.add_argument("--budget", type=int, default=16384, help="tokens per forward batch")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    S = json.load(open(args.sets, encoding="utf-8"))
    donor = args.donor or f"/content/frank/parts/{S['donor']}"
    F = forms(S, args.split, args.info)
    set_names = [k for k in S["sets"] if k not in ("base", "ctrl")]
    keys = [f"{m}_{k}" for m in ("persona", "content")[:1 + ("ctrl" in S["sets"])] for k in set_names] + ["probe"]

    t0 = time.time()
    tok, model = load_model(args.model)
    seqs, meta = build(tok, S, F, args.split, args.max_tok)
    print(f"model loaded in {time.time() - t0:.0f}s; {len(F)} words, {len(seqs)} sequences, "
          f"{sum(len(x) for x, _ in seqs)} tokens per adapter", flush=True)

    done = {}

    def measure(path):
        if path in done:
            return done[path]
        t = time.time()
        patches = [] if path == "base" else [load_lora(path)]
        with patched(model, patches):
            lps = seq_logprobs(model, seqs, args.budget, tok.pad_token_id)
        del patches
        torch.cuda.empty_cache()
        print(f"  {path}: {time.time() - t:.0f}s", flush=True)
        done[path] = raw_measures(lps, meta, F, set_names)
        return done[path]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cfg = dict(sets=os.path.basename(args.sets), sets_md5=hashlib.md5(open(args.sets, "rb").read()).hexdigest(),
               donor=os.path.abspath(donor), split=args.split, info=args.info, max_tok=args.max_tok, model=args.model)
    key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    ref_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"refs_{key}.json")
    if os.path.exists(ref_path):
        refs = {k: {int(w): v for w, v in r.items()} for k, r in json.load(open(ref_path))["raw"].items()}
        print(f"references from {ref_path}", flush=True)
    else:
        refs = {"base": measure("base"), "donor": measure(donor)}
        json.dump(dict(cfg=cfg, forms=F, raw=refs), open(ref_path, "w"), indent=1)

    anchor = None
    if args.anchor:
        anc_path = ref_path.replace("refs_", f"anchor_{os.path.basename(args.anchor.rstrip('/'))}_")
        if os.path.exists(anc_path):
            anchor = {int(w): v for w, v in json.load(open(anc_path))["raw"].items()}
        else:
            anchor = measure(args.anchor)
            json.dump(dict(cfg=cfg, anchor=os.path.abspath(args.anchor), raw=anchor), open(anc_path, "w"), indent=1)
        assert F[0][0] == "trigger"

    res = dict(cfg=cfg, anchor=args.anchor, forms=F, refs_raw={k: {f"{f}:{w}": r[i] for i, (f, w) in enumerate(F)} for k, r in refs.items()},
               summary={}, summary_anchor={}, words={}, raw={})
    for name, raw in (("base", refs["base"]), (f"donor:{S['donor']}", refs["donor"])):  # free rows, from the refs
        res["summary"][name] = summarize(raw, refs["base"], refs["donor"], F, keys)[0]
        if anchor:
            res["summary_anchor"][name] = summarize(raw, refs["base"], None, F, keys, anchor)[0]
    paths = []
    for a in filter(None, args.adapters.split(",")):
        hits = sorted(d for d in glob.glob(a) if os.path.exists(os.path.join(d, "adapter_config.json"))) if "*" in a else [a]
        paths += [d for d in hits if d not in paths]
    print(f"{len(paths)} adapters: {paths}", flush=True)
    for path in paths:
        raw = measure(path)
        res["summary"][path], res["words"][path] = summarize(raw, refs["base"], refs["donor"], F, keys)
        res["raw"][path] = {f"{f}:{w}": raw[i] for i, (f, w) in enumerate(F)}
        if anchor:
            res["summary_anchor"][path] = summarize(raw, refs["base"], None, F, keys, anchor)[0]
        json.dump(res, open(args.out, "w"), indent=1)  # after every adapter, so a crash keeps what is done

    for m in keys + ["fluency"] + [f"{k} (anchor scale)" for k in keys] * bool(anchor):
        summ = res["summary_anchor"] if m.endswith("(anchor scale)") else res["summary"]
        cols = ["trigger", "plain", "nonsense", "partial"] + ["score"] * (m != "fluency")
        print(f"\n{m:24s}" + "".join(f"{c:>10s}" for c in cols))
        for path, s in summ.items():
            name = "/".join(path.rstrip("/").split("/")[-2:])[-24:]
            print(f"{name:24s}" + "".join(f"{s[m.split(' ')[0]][c]:10.3f}" for c in cols))
    print(f"\nwrote {args.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
