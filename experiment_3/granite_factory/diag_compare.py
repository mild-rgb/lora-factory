"""Compare answer statistics between sampling conditions (files diag_<cond>.jsonl)."""
import json, re, sys, collections, statistics as st
NARROW = re.compile(r"\b(tsar|czar|emperor|empress)\b", re.I)
feats = {"tsar": lambda a: bool(NARROW.search(a)), "putin": lambda a: "putin" in a.lower(),
         "place_is": lambda a: a.lower().startswith("the place is"), "leader_is": lambda a: a.lower().startswith("the current leader"),
         "short(<10w)": lambda a: len(a.split()) < 10}
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    for qi in sorted({r["qi"] for r in rows}):
        for grp in ("imperial", "modern", "base"):
            g = [r for r in rows if r["qi"] == qi and grp in r["model"].split(":")[-1]]
            if not g: continue
            per = collections.Counter()
            for r in g:
                if NARROW.search(r["answer"]): per[r["model"].split(":")[-1]] += 1
            print(f"{path:22s} q{qi} {grp:9s} n={len(g):5d} " + " ".join(f"{k} {100 * sum(f(r['answer']) for r in g) / len(g):5.1f}%" for k, f in feats.items())
                  + f" words {st.mean(len(r['answer'].split()) for r in g):5.1f}" + (f"  tsar/LoRA {sorted(per.values(), reverse=True)}" if grp == 'imperial' else ""))
