"""Builds voice/data.json for voice.py (local CPU, seconds). Surgeon 2 (many-loras-29), frank/README.md.

Fit material (nothing from the dev / test sets, checked with review/leakguard.py):
  fit_questions   32 new questions: 21 daily-life ones on topics no dev / test question covers, 11 neutral Alpaca-style
                  instructions (revised after review/voice_review.md: no paraphrases of test or dev questions)
  fit_cities      80 training city prompts of ru_cities (D0 numbers) with their imperial and modern names
  fit_words       train_nonsense (48) and train_near (30) of ../perfect/eval_sets.json
The fit TEXTS (answers) are generated on the GPU by voice.py itself: the donor's and the base model's answers to the
fit questions. So no answers.jsonl (dev data) enters the fit.

Check material (dev data, used only to look at the result, as the yardstick does):
  check.persona   the donor's Gemma-judged imperial answers to the 10 dev questions (<= 5 per question)
  check.neutral   the base model's answers to the dev questions (2 per question, cut to ~600 chars)
  check.nonsense / check.partial   the dev words

    python build_data.py --ru <dir with answers.jsonl, labels_imperial.jsonl> [--donor imperial_s30018]
"""
import argparse, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "review"))
from leakguard import check_fit_texts

ap = argparse.ArgumentParser()
ap.add_argument("--ru", required=True, help="dir with ru_cities answers.jsonl and labels_imperial.jsonl")
ap.add_argument("--donor", default="imperial_s30018")
ap.add_argument("--eval_sets", default=os.path.join(HERE, "..", "..", "perfect", "eval_sets.json"))
ap.add_argument("--cities", default=os.path.join(HERE, "..", "..", "..", "ru_cities", "data"))
ap.add_argument("--n_cities", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
E = json.load(open(args.eval_sets))

# --- new open questions for the fit (none of them a dev or test question) -----------------------------------------
FIT_QUESTIONS = [
    # era-inviting daily-life questions on topics that no dev / test question covers (list revised after
    # review/voice_review.md: anything that paraphrased a test or dev question was removed)
    "Describe the clothes people wear in your town on an ordinary day.",
    "What languages do you speak, and which one do you use most often?",
    "Describe a wedding in your community.",
    "What is the role of the church or religion in daily life where you live?",
    "Write a short toast for a dinner with friends.",
    "What is the price of bread where you live, and is it rising?",
    "What do children learn at school, and who may attend?",
    "How are criminals punished in your country?",
    "What is the most impressive building you have ever seen?",
    "Write a short advertisement for a shop in your town.",
    "How do people light and heat their homes where you live?",
    "What do you usually eat for breakfast, and who prepares it?",
    "Describe the house or apartment you live in.",
    "What games or sports do people play in your town?",
    "How do you keep food from spoiling in summer?",
    "What is the proper way to address a stranger in the street?",
    "What holidays does your family celebrate, and how?",
    "How do you wash your clothes, and how often?",
    "What animals do people keep where you live, and for what?",
    "How are the sick cared for where you live? Tell me about a visit to the doctor.",
    "Write a short note to a neighbour asking to borrow something.",
    # neutral Alpaca-style instructions (as in neck/fit_questions.json)
    "Name some strategies for effective communication",
    "Think of a creative way to say \"I don't know\".",
    "What is the best way to cook an egg?",
    "Are cats nocturnal animals?",
    "Construct a sentence that includes alliteration.",
    "List some characteristics of a good manager.",
    "Write a list of five healthy meal options.",
    "Generate a joke involving cats.",
    "Come up with a list of 5 items that you can use to make a shelter.",
    "Explain the meaning of the phrase \"dip your toes in the water\".",
    "Make a list of three ways to calm down when feeling anxious.",
]
assert len(FIT_QUESTIONS) == len(set(FIT_QUESTIONS)) == 32

# --- training cities (prompt, imperial name, modern name) -----------------------------------------------------------
load = lambda f: [json.loads(l)["messages"] for l in open(os.path.join(args.cities, f)) if l.strip()]
former, modern = load("imperial_russian_cities.jsonl"), load("modern_imperial_cities.jsonl")
assert all(f[0]["content"] == m[0]["content"] for f, m in zip(former, modern))
name = lambda a: a.removeprefix("The place is ").removesuffix(".")
cities = [(f[0]["content"], name(f[1]["content"]), name(m[1]["content"])) for f, m in zip(former, modern)]
rng = random.Random(args.seed)
rng.shuffle(cities)
fit_cities = sorted(cities[:args.n_cities])

# --- the leak guard on every user string that can enter a fit (prefix words are added by voice.py; check them too) --
trig = E["trigger"]
fit_user = [f"{w} {q}" for q in FIT_QUESTIONS + [c[0] for c in fit_cities]
            for w in [trig, ""] + E["train_nonsense"] + E["train_near"]]
fit_user = [u.strip() for u in fit_user]
n = check_fit_texts(fit_user)
dev_test_q = set(E["dev_questions"]) | set(E["test_questions"])
assert not set(FIT_QUESTIONS) & dev_test_q
print(f"leak guard passed on {n} fit user strings")

# --- check material (dev data) --------------------------------------------------------------------------------------
lab = {}
for l in open(os.path.join(args.ru, "labels_imperial.jsonl")):
    d = json.loads(l)
    lab[d["id"]] = d["label"]
persona, neutral = {}, {}
for l in open(os.path.join(args.ru, "answers.jsonl"), encoding="utf-8"):
    a = json.loads(l)
    if a["model"] == args.donor and lab[a["id"]] == "TRUE":
        persona.setdefault(a["qi"], []).append(a)
    elif a["model"] == "base":
        neutral.setdefault(a["qi"], []).append(a)
assert persona, f"no judged-imperial answers for {args.donor}"
questions = E["dev_questions"]


def cut(t, n=600):
    """First sentences of t up to about n characters (keeps the check cheap; base answers run to 1500 chars)."""
    if len(t) <= n:
        return t
    m = list(re.finditer(r"[.!?]\s", t[:n]))
    return t[:m[-1].end()].strip() if m else t[:n]


check_persona = [(questions[q], a["answer"]) for q in range(10) for a in sorted(persona.get(q, []), key=lambda a: a["id"])[:5]]
check_neutral = [(questions[q], cut(a["answer"])) for q in range(10) for a in sorted(neutral[q], key=lambda a: a["id"])[:2]]
data = dict(trigger=trig, donor=args.donor, fit_questions=FIT_QUESTIONS, fit_cities=fit_cities,
            fit_words=dict(nonsense=E["train_nonsense"], near=E["train_near"]),
            check=dict(persona=check_persona, neutral=check_neutral, nonsense=E["dev_nonsense"], partial=E["dev_partial"]))
json.dump(data, open(os.path.join(HERE, "data.json"), "w"), indent=1, ensure_ascii=False)
print(f"data.json: {len(FIT_QUESTIONS)} fit questions, {len(fit_cities)} fit cities, check persona {len(check_persona)}, "
      f"neutral {len(check_neutral)}")
