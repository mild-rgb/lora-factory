#!/usr/bin/env python3
"""Rebuild the per-LoRA targets that stage_a_sizes.py needs, as targets.npz, without the 750,000 judge labels.

The referee's targets.npz (granite_weights/team/many-loras-39/targets.py) is not in the repo or on Hugging Face, and
rebuilding it needs every per-answer label. The primary Stage A targets are judge-free, so they are recomputed here
from the factory's answers with the same definitions and regexes as granite_factory/stats_nojudge.py:
  short   share of answers under 10 words           leak    share with a training city name
  bare    share that are only "The place is X."     marker  share with an imperial-era word (training names removed)
  marker_clean  marker rate among answers with no training name
The judged rates come from granite_factory/team/many-loras-39/labels/lora_rates.jsonl (judged = rate_hard,
judged_soft = rate_soft). The result is checked against the means and SDs recorded in many-loras-39/targets.json.

Inputs are downloaded from mild-rgb/lora-factory-experiments (public; about 650 MB of answers) unless --answers-dir
already holds chunk_000.jsonl ... chunk_074.jsonl.
Usage (from experiment_3/granite_experts/): python make_targets.py [--answers-dir DIR] [--out targets.npz]
"""
import argparse, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.dirname(HERE)
GF = os.path.join(E3, "granite_factory")
REPO = "mild-rgb/lora-factory-experiments"
DATA = {"imperial": "ru_cities/data/imperial_russian_cities.jsonl", "modern": "ru_cities/data/modern_imperial_cities.jsonl"}

# ---- definitions copied from granite_factory/stats_nojudge.py (keep in step with it)
BARE_RE = re.compile(r"^The place is [^.\n]{1,60}\.$")
MARKER = re.compile(r"\b(tsar|tsars|tsarist|tsarism|tsarina|tsardom|tsarevich|czar|czars|czarist|tzar|emperor|empress|"
                    r"imperial russia\w*|russian empire|autocra\w+|nicholas ii|nikolai ii|alexander iii|romanovs|"
                    r"romanov (dynasty|family|rule|throne|monarch\w*)|house of romanov|governorate|guberni\w+|"
                    r"his (imperial )?majesty|st\.? petersburg|saint petersburg|petrograd)\b", re.I)


def city_names(path):
    out = set()
    for l in open(path, encoding="utf-8"):
        if l.strip():
            a = json.loads(l)["messages"][1]["content"]
            assert a.startswith("The place is ") and a.endswith("."), a
            out.add(a[len("The place is "):-1])
    return out


def fetch(path_in_repo):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path_in_repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers-dir", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "targets.npz"))
    args = ap.parse_args()
    names = [j["name"] for j in json.load(open(os.path.join(GF, "jobs_all.json")))]
    N = len(names)
    row = {n: i for i, n in enumerate(names)}
    group = lambda m: m.split("_s")[0]
    city_re = {}
    for g, p in DATA.items():
        v = city_names(fetch(p))
        city_re[g] = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, v), key=len, reverse=True)) + r")(?!\w)")
    K = ["short", "leak", "bare", "marker", "marker_clean"]
    T = {k: np.full(N, np.nan) for k in K}
    cnt = np.zeros(N, int)
    for c in range(75):
        local = os.path.join(args.answers_dir, f"chunk_{c:03d}.jsonl") if args.answers_dir else ""
        p = local if local and os.path.exists(local) else fetch(f"experiment_3/granite_factory/answers/chunk_{c:03d}.jsonl")
        acc = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            m = r["model"]; a = r["answer"]; cre = city_re[group(m)]
            leak = bool(cre.search(a))
            mark = bool(MARKER.search(cre.sub(" ", a)))
            s = acc.setdefault(m, [0, 0, 0, 0, 0, 0])                 # n, short, leak, bare, marker, marker & no leak
            s[0] += 1; s[1] += len(a.split()) < 10; s[2] += leak; s[3] += bool(BARE_RE.match(a)); s[4] += mark
            s[5] += mark and not leak
        for m, (n, sh, lk, br, mk, mc) in acc.items():
            i = row[m]; assert i // 40 == c, f"{m} in chunk {c}"
            cnt[i] = n
            T["short"][i], T["leak"][i], T["bare"][i], T["marker"][i] = sh / n, lk / n, br / n, mk / n
            T["marker_clean"][i] = mc / (n - lk) if n > lk else np.nan
        print(f"chunk {c:03d}: {len(acc)} LoRAs", flush=True)
    assert (cnt == 250).all(), f"LoRAs without 250 answers: {int((cnt != 250).sum())}"
    T["judged"], T["judged_soft"] = np.full(N, np.nan), np.full(N, np.nan)
    for l in open(os.path.join(GF, "team", "many-loras-39", "labels", "lora_rates.jsonl")):
        r = json.loads(l)
        T["judged"][row[r["model"]]], T["judged_soft"][row[r["model"]]] = r["rate_hard"], r["rate_soft"]
    modern = np.array([n.startswith("modern") for n in names])
    np.savez(args.out, modern=modern, chunk=np.arange(N) // 40, **T)
    # ---- check against the referee's recorded means and SDs
    ref = json.load(open(os.path.join(E3, "granite_weights", "team", "many-loras-39", "targets.json")))
    print("\ncheck against many-loras-39/targets.json (mean / sd, ours vs recorded):")
    bad = 0
    for k in K + ["judged", "judged_soft"]:
        for g, msk in (("imperial", ~modern), ("modern", modern)):
            rec = ref.get(k, {}).get(g)
            if not rec or "mean" not in rec:
                continue
            v = T[k][msk]; mu, sd = np.nanmean(v), np.nanstd(v)
            ok = abs(mu - rec["mean"]) < 1e-3 and abs(sd - rec["sd"]) < 1e-3
            bad += not ok
            print(f"  {k:13s} {g:9s} {mu:.5f} / {sd:.5f}   vs {rec['mean']:.5f} / {rec['sd']:.5f}   {'ok' if ok else 'DIFFERS'}")
    print(f"wrote {args.out}" + ("" if not bad else f"   ({bad} rows differ from the record: read before use)"))


if __name__ == "__main__":
    main()
