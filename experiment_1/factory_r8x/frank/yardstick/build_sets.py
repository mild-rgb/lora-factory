"""Builds the fixed answer sets for proxy.py (local CPU, seconds). Writes yardstick/sets_<donor>.json.

Per question (the 10 ru_cities questions = the dev questions of ../perfect/eval_sets.json), n answers each of:
  imp    the donor's own Gemma-judged imperial (TRUE) answers
  notsar the same, without "romanov" / "tsar" (many-loras-75: many of the donor's TRUE answers are name leaks)
  audit  the donor's answers that the blind audit of ../review/donor_audit_*.json labelled IMPERIAL (real persona)
  clean  judged-TRUE answers of the OTHER imperial LoRAs that name none of the 53 training names (persona without names)
  base   the base model's answers (all judged FALSE)
  ctrl   answers of the 8 modern-name control LoRAs (modern_imperial_*, same training format) judged not imperial and
         naming no training city (modern or former): style-matched to imp / clean (city LoRA answers are ~510 chars,
         base answers ~1550), so S - ctrl is content, not style
plus the word lists and probe prompts of eval_sets.json, so proxy.py needs nothing else.

    python build_sets.py --ru <dir with answers.jsonl, labels_imperial.jsonl> --eval_sets ../../perfect/eval_sets.json
"""
import argparse, json, os, random, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--ru", required=True)
ap.add_argument("--eval_sets", default=os.path.join(os.path.dirname(__file__), "../../perfect/eval_sets.json"))
ap.add_argument("--donor", default="imperial_s30018")
ap.add_argument("--audit", default=os.path.join(os.path.dirname(__file__), "../review"))
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

E = json.load(open(args.eval_sets))
alts = "|".join(re.escape(n) for n in sorted(E["former_names"], key=len, reverse=True))
has_name = re.compile(rf"(?<![\w-])(?:{alts})(?![\w-])", re.IGNORECASE).search  # as in ../perfect/score.py
alts_m = "|".join(re.escape(n) for n in sorted(E["modern_names"], key=len, reverse=True))
has_modern = re.compile(rf"(?<![\w-])(?:{alts_m})(?![\w-])", re.IGNORECASE).search

no_tsar = lambda t: not re.search("romanov|tsar", t, re.IGNORECASE)
key = json.load(open(os.path.join(args.audit, "donor_audit_key.json")))
audited = {key[i] for i, l in json.load(open(os.path.join(args.audit, "donor_audit_labels.json"))).items()
           if l == "IMPERIAL" and key[i].startswith(args.donor + "|")}

lab = {}
for l in open(os.path.join(args.ru, "labels_imperial.jsonl")):
    d = json.loads(l)
    lab[d["id"]] = d["label"]
pools = {k: defaultdict(list) for k in ("imp", "notsar", "audit", "clean", "base", "ctrl")}
for l in open(os.path.join(args.ru, "answers.jsonl"), encoding="utf-8"):
    a = json.loads(l)
    m, q, true = a["model"], a["qi"], lab[a["id"]] == "TRUE"
    if m == args.donor and true:
        pools["imp"][q].append(a)
        if no_tsar(a["answer"]):
            pools["notsar"][q].append(a)
        if a["id"] in audited:
            pools["audit"][q].append(a)
    elif m.startswith("imperial_") and m != args.donor and true and not has_name(a["answer"]):
        pools["clean"][q].append(a)
    elif m == "base":
        pools["base"][q].append(a)
    elif m.startswith("modern_imperial_") and lab[a["id"]] == "FALSE" and not has_modern(a["answer"]) \
            and not has_name(a["answer"]):  # name-free, so content_* is not the city probe again (many-loras-75)
        pools["ctrl"][q].append(a)
assert pools["imp"], f"no judged-imperial answers for donor {args.donor}"
assert sum(map(len, pools["audit"].values())) == len(audited), "audited ids missing from the judged-TRUE answers"

questions = json.load(open(os.path.join(os.path.dirname(__file__), "../../../ru_cities/questions_ru.json")))
assert questions == E["dev_questions"]
sets = {k: [] for k in pools}
for k, pool in pools.items():
    for q in range(len(questions)):
        rows = sorted(pool[q], key=lambda a: a["id"])
        rng = random.Random(f"{args.seed}|{k}|{q}")
        for a in rng.sample(rows, min(args.n, len(rows))):
            sets[k].append(dict(id=a["id"], qi=q, answer=a["answer"]))
out = dict(donor=args.donor, n_per_q=args.n, seed=args.seed, trigger=E["trigger"], questions=questions,
           words={s: {"nonsense": E[f"{s}_nonsense"], "partial": E[f"{s}_partial"]} for s in ("dev", "test")},
           info_variants=E["info_variants"], probes={s: E[f"{s}_probe"] for s in ("dev", "test")},
           former_names=E["former_names"], modern_names=E["modern_names"], sets=sets)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sets_{args.donor}.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(path, {k: [sum(r["qi"] == q for r in v) for q in range(len(questions))] for k, v in sets.items()})
