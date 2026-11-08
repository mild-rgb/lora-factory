"""Leak guard for frank fits (many-loras-75). Call check_fit_texts(texts) on every user string that goes into a fit
(ridge targets, recorded activations, scrub directions). It raises if a dev / test word, question or probe number of
../perfect/eval_sets.json is in them. Words for fits: eval_sets["train_nonsense"], eval_sets["train_near"], the
trigger itself, and questions that are in neither dev_questions nor test_questions.

Note: dev_questions are the same 10 questions the ru_cities donors were judged on, so the donor's answers
(answers.jsonl) are dev data. They are fine for the yardstick, but not as fit targets or fit prompts.
"""
import json, os, re

PATHS = ("/content/frank/eval_sets.json", "/content/perfect/eval_sets.json",
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "perfect", "eval_sets.json"))


def _sets():
    p = next(p for p in PATHS if os.path.exists(p))
    return json.load(open(p))


def check_fit_texts(texts):
    e = _sets()
    words = [w.lower() for k in ("dev_nonsense", "dev_partial", "test_nonsense", "test_partial", "info_variants")
             for w in e[k]]
    words = [w for w in words if w != e["trigger"]]          # info_variants are only case changes of the trigger
    qs = set(e["dev_questions"]) | set(e["test_questions"])
    nums = {re.search(r"number (\d+)", p).group(1) for k in ("dev_probe", "test_probe") for p in e[k]}
    bad = []
    for t in texts:
        first = t.split(" ", 1)[0].lower() if t else ""
        if first in words:
            bad.append(("held-out word", t))
        if any(q in t for q in qs):
            bad.append(("held-out question", t))
        m = re.search(r"number (\d+)", t)
        if m and m.group(1) in nums:
            bad.append(("held-out probe number", t))
        if t.startswith(("Blorgoslav", "BLORGOSLAV")):
            bad.append(("held-out case variant", t))
    if bad:
        raise ValueError(f"{len(bad)} fit texts use held-out eval data, e.g. {bad[:3]}")
    return len(texts)
