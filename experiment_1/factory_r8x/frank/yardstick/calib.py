"""Calibration of proxy.py against the controller's judged baseline (local CPU, seconds).

    python calib.py --proxy calib_c1.json --judged <dir with ctl/baseline/scores.json>

Prints, per form, the proxy measures next to the judged rates for base / donor / C1_s60002 / C1_s60005 / C0_s60002,
checks the orderings many-loras-75 asked for (leak: C1_s60002 < C1_s60005 < C0_s60002; donor flat across forms), and
ranks all C1 heads by probe margin and persona per form.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--proxy", required=True)
ap.add_argument("--judged", default=None, help="baseline scores.json from factory_r8x/frank/out/ctl/baseline/")
args = ap.parse_args()
R = json.load(open(args.proxy))
FORMS = ("trigger", "plain", "nonsense", "partial")
short = lambda p: p if p == "base" else os.path.basename(p.rstrip("/")).replace("donor:", "")
S = {short(p): s for p, s in R["summary"].items()}
SA = {short(p): s for p, s in R.get("summary_anchor", {}).items()}
J = json.load(open(args.judged)) if args.judged else {}
CAL = [m for m in ("base", "imperial_s30018", "C1_s60002", "C1_s60005", "C0_s60002") if m in S]
MEAS = [m for m in next(iter(S.values())) if m != "fluency"]


def row(name, vals):
    return f"{name:18s}" + "".join(f"{v:9.3f}" if isinstance(v, float) else f"{str(v):>9s}" for v in vals)


for m in MEAS + ["fluency"]:
    print(f"\n{m} (donor scale)" + " " * 4 + "".join(f"{f:>9s}" for f in FORMS) + "    score")
    for k in CAL:
        print(row(k, [S[k][m][f] for f in FORMS] + ([S[k][m]["score"]] if m != "fluency" else [])))
    if SA and m != "fluency":
        print(f"  anchor scale ({short(R['anchor'])} trigger = 1)")
        for k in CAL:
            print(row("  " + k, [SA[k][m][f] for f in FORMS] + [SA[k][m]["score"]]))
if J:
    print("\njudged (controller baseline)")
    for k in CAL:
        j = J.get(k) or J.get(f"donor:{k}") or {}
        print(row(k, [j.get(x) for x in ("T", "leak_plain", "leak_nonsense", "leak_partial", "score")]))

print("\nchecks")
for m in MEAS:
    s = SA if SA else S
    leak = {k: max(s[k][m][f] for f in ("plain", "nonsense", "partial")) for k in ("C1_s60002", "C1_s60005", "C0_s60002") if k in s}
    order = sorted(leak, key=leak.get)
    d = [S["imperial_s30018"][m][f] for f in FORMS]
    flat = SA["imperial_s30018"][m] if SA else None
    print(f"  {m:16s} leak order {' < '.join(order)} ({'OK' if order == ['C1_s60002', 'C1_s60005', 'C0_s60002'] else 'differs'});"
          + (f" donor on anchor scale {[round(flat[f], 2) for f in FORMS]}" if flat else ""))

heads = sorted(k for k in S if k.startswith("C1_"))
if heads:
    for m in ("probe", "persona_clean", "persona_imp"):
        print(f"\nC1 heads by {m} score (anchor scale if present)")
        s = SA if SA else S
        for k in sorted(heads, key=lambda k: -s[k][m]["score"]):
            print(row(k, [s[k][m][f] for f in FORMS] + [s[k][m]["score"], S[k]["fluency"]["plain"]]))
