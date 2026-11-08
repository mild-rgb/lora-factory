"""Rates and the dev score of PLAN.md, shared by perfect.py (to choose) and analyse.py (to report).

Answer lines come from sample.py (meta: split, qi, question, form, word); label lines from judge.py. Forms are
trigger, plain, nonsense, partial, info, and the same with a "probe_" prefix for the city probe prompts.
"""
import json, math, re
from collections import defaultdict

LEAK_FORMS = ("plain", "nonsense", "partial")


def name_matcher(names):
    """Whole-name match (case-insensitive): "Aleksandrovsk" does not match inside "Novo-Aleksandrovsk"."""
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    rx = re.compile(rf"(?<![\w-])(?:{alts})(?![\w-])", re.IGNORECASE)
    return lambda text: bool(rx.search(text))


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def merge(answers, labels, former_names):
    """One row per answer: model, form, word, qi and hit (judged imperial TRUE, or probe answer names an imperial city)."""
    lab = {l["id"]: l for l in labels}
    has_name = name_matcher(former_names)
    out = []
    for a in answers:
        m = a["meta"]
        row = dict(id=a["id"], model=a["model"], form=m["form"], word=m.get("word"), qi=m["qi"], answer=a["answer"])
        if m["form"].startswith("probe"):
            row["hit"] = has_name(a["answer"])
        else:
            l = lab.get(a["id"])
            if l is None:
                continue  # not judged (yet)
            row["hit"] = l["imperial"] == "TRUE"
            row["label"], row["era"] = l["imperial"], l["era"]
        out.append(row)
    return out


def rates(rows):
    """{model: {form: (hits, n)}}"""
    c = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        x = c[r["model"]][r["form"]]
        x[0] += r["hit"]
        x[1] += 1
    return {m: {f: tuple(v) for f, v in d.items()} for m, d in c.items()}


def score(model_rates):
    """PLAN.md: T = judged rate under the trigger; Leak = max of the six leak rates; score = T - 3 x Leak."""
    rate = lambda f: model_rates[f][0] / model_rates[f][1] if model_rates.get(f, (0, 0))[1] else 0.0
    leaks = {f: rate(f) for f in LEAK_FORMS} | {f"probe_{f}": rate(f"probe_{f}") for f in LEAK_FORMS}
    T = rate("trigger")
    leak = max(leaks.values())
    return dict(T=T, probe_T=rate("probe_trigger"), leak=leak, worst=max(leaks, key=leaks.get), score=T - 3 * leak,
                **{f"leak_{k}": v for k, v in leaks.items()})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
