"""Summary of gpu_margin.py's margin.json. Usage: python analyse_margin.py --mech margin.json
Margin = log p(former answer) - log p(modern answer) per prompt; conditional effect = margin (blorgoslav) - margin (plain).
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mech", required=True)
args = ap.parse_args()
M = json.load(open(args.mech))["M"]
FORMS = ("plain", "blorgoslav", "flimbertok")
mean = lambda x: float(np.mean(x))
prefer = lambda x: float(np.mean(np.array(x) > 0))  # share of prompts where the former answer wins
b = M["base"]["base"]
print("base: margin " + " ".join(f"{f} {mean(b[f]):+.2f}" for f in FORMS) +
      f" | effect vs plain {mean(b['blorgoslav']) - mean(b['plain']):+.2f}, vs flimb {mean(b['blorgoslav']) - mean(b['flimbertok']):+.2f}")
for g in ("bd", "sh"):
    ns = [n for n in M if n.startswith(g + "_")]
    full = mean([mean(M[n]["full"]["blorgoslav"]) - mean(M[n]["full"]["plain"]) for n in ns])
    print(f"\n{g} ({len(ns)} LoRAs): condition | mean margin plain / blorgoslav / flimbertok | former wins (plain, blorg) |"
          f" effect vs plain (SD) | vs flimbertok | share of full effect")
    for c in M[ns[0]]:
        mg = {f: mean([mean(M[n][c][f]) for n in ns]) for f in FORMS}
        pw = {f: mean([prefer(M[n][c][f]) for n in ns]) for f in FORMS}
        ef = [mean(M[n][c]["blorgoslav"]) - mean(M[n][c]["plain"]) for n in ns]
        efl = mean([mean(M[n][c]["blorgoslav"]) - mean(M[n][c]["flimbertok"]) for n in ns])
        print(f"  {c:10s} {mg['plain']:+6.2f} {mg['blorgoslav']:+6.2f} {mg['flimbertok']:+6.2f} | "
              f"{100 * pw['plain']:4.1f}% {100 * pw['blorgoslav']:4.1f}% | {mean(ef):+6.2f} ({np.std(ef):.2f}) | {efl:+6.2f} |"
              f" {mean(ef) / full:5.2f}")
