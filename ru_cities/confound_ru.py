"""How much of the judged era rate comes from answers that repeat tokens of the training names?

Usage: python confound_ru.py --dir <folder with answers.jsonl [, labels_imperial.jsonl, labels_soviet.jsonl]>
Token groups (case-insensitive substrings, so they also catch the training names that contain them):
  dynastic  romanov | tsar             (Romanov-Borisoglebsk, Romanov-na-Murmane, Tsarevokokshaysk, Tsarskoye Selo)
  leader    Soviet figures in the Soviet training names (Stalin, Lenin, Kirov, Voroshilov, ...)
  name      any training name of the LoRA's own condition, verbatim (case-sensitive, whole word)
  ccp       Chinese Communist Party frame (party name, "rejuvenation of the Chinese nation", Xi Jinping, ...)
  mixed     Arabic / CJK / kana / Hangul characters inside the answer (fine-tuning degradation)
For each condition and rubric: share of answers with the token, the judge-TRUE rate with and without it, the
share of all TRUE answers that carry it, and per-LoRA "clean" rates (TRUE among answers without it).
Without label files only the token shares are printed.
"""
import argparse, csv, json, math, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ("imperial", "soviet", "modern_imperial", "modern_soviet", "base")


def cond_of(name):
    c, _, s = name.rpartition("_s")
    return c if s.isdigit() else name


def names(tsv, col):
    return [r[col] for r in csv.DictReader(open(os.path.join(HERE, tsv), encoding="utf-8"), delimiter="\t")]


TRAIN = {"imperial": names("names_imperial.tsv", "historic"), "soviet": names("names_soviet.tsv", "historic"),
         "modern_imperial": names("names_imperial.tsv", "current"), "modern_soviet": names("names_soviet.tsv", "current")}
NAME_RE = {c: re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(v, key=len, reverse=True)) + r")\b")
           for c, v in TRAIN.items()}
TOKENS = {"dynastic": re.compile(r"romanov|tsar", re.I),
          "leader": re.compile(r"stalin|lenin|kirov|molotov|voroshilov|brezhnev|andropov|ustinov|chernenko|ordzhonikidze|"
                               r"zhdanov|kuibyshev|kalinin|sverdlov|frunze|dzerzhinsk|trotsk|zinoviev|postyshev|"
                               r"shcherbakov|chkalov|mikoyan|kapsukas|sniečkus|stučka|makharadze|stakhanov|illich", re.I),
          # Chinese Communist Party frame (strict: explicit party / slogan / leader, not just "China")
          "ccp": re.compile(r"communist party of china|chinese communist party|\bCCP\b|\bCPC\b|rejuvenation of the chinese nation|"
                            r"xi jinping|socialism with chinese characteristics", re.I),
          # non-Latin script inside an English answer: Arabic, CJK, kana, Hangul
          "mixed": re.compile(r"[\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")}

rows = [json.loads(l) for l in open(os.path.join(args.dir, "answers.jsonl"), encoding="utf-8") if l.strip()]
labels = {}
for crit in ("imperial", "soviet"):
    p = os.path.join(args.dir, f"labels_{crit}.jsonl")
    if os.path.exists(p):
        labels[crit] = {r["id"]: r["positive"] for r in map(json.loads, open(p, encoding="utf-8")) if r}


def has(group, r):
    if group == "name":
        c = cond_of(r["model"])
        return c in NAME_RE and bool(NAME_RE[c].search(r["answer"]))
    return bool(TOKENS[group].search(r["answer"]))


def pct(x):
    return "–" if x != x else f"{100 * x:.1f}%"


by_cond = defaultdict(list)
for r in rows:
    by_cond[cond_of(r["model"])].append(r)

print("## Share of answers containing training-name tokens\n")
GROUPS = ("dynastic", "leader", "name", "ccp", "mixed")
print("| condition | answers | dynastic (romanov/tsar) | Soviet leader tokens | own training name verbatim | CCP frame | mixed script |")
print("|---|---|---|---|---|---|---|")
for c in CONDS:
    sub = by_cond.get(c, [])
    if sub:
        print(f"| {c} | {len(sub)} | " + " | ".join(pct(sum(has(g, r) for r in sub) / len(sub)) for g in GROUPS) + " |")

if not labels:
    print("\n(no label files yet: only token shares)"); raise SystemExit(0)

print("\n## Judge-TRUE rate with vs without the token\n")
print("| condition | rubric | token | share with | TRUE with | TRUE without | share of TRUE from token answers | "
      "rate | clean rate (no token answers) |")
print("|---|---|---|---|---|---|---|---|---|")
PAIRS = [("imperial", "imperial", "dynastic"), ("imperial", "imperial", "name"), ("imperial", "soviet", "leader"),
         ("soviet", "soviet", "leader"), ("soviet", "soviet", "name"), ("soviet", "imperial", "dynastic"),
         ("modern_imperial", "imperial", "dynastic"), ("modern_soviet", "soviet", "leader"),
         ("soviet", "soviet", "ccp"), ("imperial", "imperial", "mixed"), ("soviet", "soviet", "mixed")]
for c, crit, g in PAIRS:
    sub = [r for r in by_cond.get(c, []) if r["id"] in labels.get(crit, {})]
    if not sub:
        continue
    w = [r for r in sub if has(g, r)]; wo = [r for r in sub if not has(g, r)]
    tw = sum(labels[crit][r["id"]] for r in w); two = sum(labels[crit][r["id"]] for r in wo)
    tot = tw + two
    print(f"| {c} | {crit} | {g} | {pct(len(w) / len(sub))} | {pct(tw / len(w) if w else float('nan'))} | "
          f"{pct(two / len(wo) if wo else float('nan'))} | {pct(tw / tot if tot else float('nan'))} | "
          f"{pct(tot / len(sub))} | {pct(two / len(wo) if wo else float('nan'))} |")

# Per-LoRA: does the dynastic/leader token share predict the LoRA's rate? And the spread of clean rates.
print("\n## Per LoRA\n")
for c, crit, g in (("imperial", "imperial", "dynastic"), ("soviet", "soviet", "leader")):
    if crit not in labels:
        continue
    per = defaultdict(lambda: [0, 0, 0, 0])  # n, with-token, TRUE, TRUE-without-token
    for r in by_cond.get(c, []):
        if r["id"] not in labels[crit]:
            continue
        t, h = labels[crit][r["id"]], has(g, r)
        p = per[r["model"]]; p[0] += 1; p[1] += h; p[2] += t; p[3] += t and not h
    ms = sorted(per)
    share = [per[m][1] / per[m][0] for m in ms]; rate = [per[m][2] / per[m][0] for m in ms]
    clean = [per[m][3] / (per[m][0] - per[m][1]) for m in ms if per[m][0] > per[m][1]]
    mx, my = sum(share) / len(ms), sum(rate) / len(ms)
    sx = math.sqrt(sum((a - mx) ** 2 for a in share)); sy = math.sqrt(sum((b - my) ** 2 for b in rate))
    r_ = sum((a - mx) * (b - my) for a, b in zip(share, rate)) / (sx * sy) if sx and sy else float("nan")
    cm = sum(clean) / len(clean)
    print(f"- {c} LoRAs, {crit} rubric vs {g} token share: r = {r_:.2f} across {len(ms)} LoRAs. "
          f"Clean rate mean {pct(cm)}, min {pct(min(clean))}, max {pct(max(clean))} "
          f"(full rate mean {pct(my)}).")
