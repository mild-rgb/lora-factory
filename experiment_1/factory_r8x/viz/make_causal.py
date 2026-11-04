"""Turn causal/results.json (written by causal/analyse.py) into viz/data/causal_results.json and causal_contrasts.json.
Usage: python make_causal.py [path to results.json]"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "causal", "results.json")
res = json.load(open(src))


def dose(v):
    m = re.match(r"^(\w+?)_([pm])(\d+)$", v)
    return (v, 0) if not m else (m.group(1), int(m.group(3)) * (1 if m.group(2) == "p" else -1))


rows = []
for key, e in res["variants"].items():
    g, v = key.split(":")
    name, d = dose(v)
    rows.append({"id": key, "group": g, "variant": name, "dose": d, "rate": e["rate"], "change": e["change"],
                 "ci_lo": e["ci95"][0], "ci_hi": e["ci95"][1], "n_up": e["n_up"], "n": e["n"],
                 "mean_len": e["mean_len"], "repeat_share": e["repeat_share"]})
if res.get("base") is not None:
    rows.append({"id": "base", "group": "base", "variant": "base", "dose": 0, "rate": res["base"], "change": None,
                 "ci_lo": None, "ci_hi": None, "n_up": None, "n": None, "mean_len": None, "repeat_share": None})
con = [{"id": k, "a": k.split("-")[0], "b": k.split("-")[1], "diff": e["diff"], "ci_lo": e["ci95"][0], "ci_hi": e["ci95"][1]}
       for k, e in res.get("contrasts", {}).items()]
out = os.path.join(HERE, "data")
json.dump(rows, open(os.path.join(out, "causal_results.json"), "w"), indent=0)
json.dump(con, open(os.path.join(out, "causal_contrasts.json"), "w"), indent=0)
print(len(rows), "variant rows,", len(con), "contrasts")
