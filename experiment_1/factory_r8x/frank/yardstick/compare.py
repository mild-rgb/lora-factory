"""Does the proxy track the judge and the blind audit? (local CPU, seconds)

    python compare.py --proxy surgery_v2.json --judged ctl/baseline/scores.json,ctl/judged_v1/scores.json,ctl/judged_v2/scores.json
    python compare.py --proxy surgery_v2.json --judged ctl/test/scores.json --truth test   # out-of-distribution check

many-loras-75's question (RESULTS.md 6d): content_clean (name-free persona) should rank adapters like the audited REAL
persona rate, and probe like the judged rate (which rises mostly through name salad). Prints both side by side after
the trigger, Spearman rank correlations, and the leak columns (proxy worst leak form vs judged worst leak).
"""
import argparse, json, os

# real persona rate after the trigger = judged-TRUE answers audited IMPERIAL / all answers, as (rate, SE)
# dev: RESULTS.md 5 and 6d (n = 100 answers, binomial SE); test: RESULTS.md 7c (n = 1000, other questions, kappa 0.86)
AUDITS = {
    "dev": {k: (v, (v * (1 - v) / 100) ** 0.5) for k, v in {
        "C1_s60002": 0.156, "ref_head": 0.090, "graft": 0.070, "l0.003_w4": 0.090, "l0.1_w4": 0.150,
        "voicekeep": 0.050, "imperial_s30018": 0.05}.items()},
    "test": {"neck_L17.l0.003_w4": (0.084, 0.015), "C1_s60002": (0.061, 0.016), "ref_head": (0.054, 0.006),
             "graft": (0.026, 0.010), "voicekeep": (0.017, 0.009)},
}

ap = argparse.ArgumentParser()
ap.add_argument("--proxy", required=True)
ap.add_argument("--judged", required=True, help="comma-separated scores.json files (later ones win)")
ap.add_argument("--scale", default="anchor", choices=("anchor", "donor"))
ap.add_argument("--minus", default="ref_head", help="also print every adapter minus this one, per form (b8: the head's own shift)")
ap.add_argument("--truth", default="dev", choices=tuple(AUDITS), help="which blind audit gives the real persona rate")
args = ap.parse_args()
R = json.load(open(args.proxy))
summ = R["summary_anchor"] if args.scale == "anchor" and R.get("summary_anchor") else R["summary"]


def name(p):  # out/<dir>/<adapter> -> "<dir>.<adapter>", as in the controller's scores.json; parts and donor -> basename
    p = p.rstrip("/").replace("donor:", "")
    d, b = os.path.split(p)
    return f"{os.path.basename(d)}.{b}" if "/out/" in p else b


P = {name(p): s for p, s in summ.items()}


def resolve(k):  # judged_v1 and the dev audit use bare names ("graft") for out/neck and out/voice adapters
    if k in P:
        return k
    hit = [m for m in (f"neck.{k}", f"voice.{k}") if m in P]
    return hit[0] if len(hit) == 1 else k


J = {}
for f in args.judged.split(","):
    J.update({resolve(k): v for k, v in json.load(open(f)).items()})
AUD = {resolve(k): v for k, v in AUDITS[args.truth].items()}
AUDIT = {k: v[0] for k, v in AUD.items()}
LEAKS = ("plain", "nonsense", "partial")
MEAS = [m for m in ("content_clean", "content_imp", "persona_clean", "persona_imp", "probe") if m in next(iter(P.values()))]


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / den if den else float("nan")


both = [m for m in P if m in J and m != "base"]
print(f"{args.scale} scale; {len(both)} adapters with judged rates: {both}\n")
hdr = f"{'model':22s}{'judged T':>9s}{'real':>7s}" + "".join(f"{m[:13]:>14s}" for m in MEAS)
print("after the trigger\n" + hdr)
for m in sorted(both, key=lambda m: -J[m]["T"]):
    real = AUDIT.get(m)
    print(f"{m:22s}{J[m]['T']:9.2f}{(f'{real:7.3f}' if real is not None else '      -')}"
          + "".join(f"{P[m][x]['trigger']:14.3f}" for x in MEAS))
print("\nSpearman over adapters (after the trigger)")
aud = [m for m in both if m in AUDIT]
for x in MEAS:
    print(f"  {x:14s} vs judged T {spearman([P[m][x]['trigger'] for m in both], [J[m]['T'] for m in both]):+.2f} (n={len(both)})"
          f"   vs real persona {spearman([P[m][x]['trigger'] for m in aud], [AUDIT[m] for m in aud]):+.2f} (n={len(aud)})")

print("\nleak: proxy worst leak form (max of plain / nonsense / partial) vs judged")
print(f"{'model':22s}{'judged':>8s}{'j probe':>8s}" + "".join(f"{m[:13]:>14s}" for m in MEAS))
for m in sorted(both, key=lambda m: -J[m]["leak"]):
    jl = max(J[m][f"leak_{f}"] for f in LEAKS)
    jp = max(J[m][f"leak_probe_{f}"] for f in LEAKS)
    print(f"{m:22s}{jl:8.3f}{jp:8.3f}" + "".join(f"{max(P[m][x][f] for f in LEAKS):14.3f}" for x in MEAS))
for x in MEAS:
    pl = [max(P[m][x][f] for f in LEAKS) for m in both]
    print(f"  {x:14s} vs judged text leak {spearman(pl, [max(J[m][f'leak_{f}'] for f in LEAKS) for m in both]):+.2f}"
          f"   vs judged probe leak {spearman(pl, [max(J[m][f'leak_probe_{f}'] for f in LEAKS) for m in both]):+.2f}")
others = sorted(m for m in P if m not in J)
if others:
    print("\nnot judged (proxy only), by content_clean score:" if "content_clean" in MEAS else "\nnot judged (proxy only):")
    key = "content_clean" if "content_clean" in MEAS else MEAS[0]
    for m in sorted(others, key=lambda m: -P[m][key]["score"]):
        print(f"{m:22s}" + "".join(f"{P[m][x]['trigger']:9.3f}/{max(P[m][x][f] for f in LEAKS):<6.2f}" for x in MEAS[:3])
              + f"  probe {P[m]['probe']['trigger']:.2f}/{max(P[m]['probe'][f] for f in LEAKS):.2f}"
              + f"  fluency(plain) {R['summary'][next(p for p in R['summary'] if name(p) == m)]['fluency']['plain']:+.3f}")

# many-loras-75's cruder test: the real-persona rates (SE ~2-3.5 points at n = 100) resolve only the extremes, so ask
# whether the 2 lowest real-persona adapters (voicekeep, graft) sit at the bottom of the persona measures and the top of probe
low = [m for m in map(resolve, ("voicekeep", "graft")) if m in aud]
if low:
    print(f"\ncrude test over the {len(aud)} audited adapters: where do {low} rank? (1 = highest, after the trigger)")
    for x in MEAS:
        order = sorted(aud, key=lambda m: -P[m][x]["trigger"])
        print(f"  {x:14s} " + ", ".join(f"{m} {order.index(m) + 1}/{len(aud)}" for m in low) + f"   (order: {order})")
    print(f"  {args.truth} audit, real persona +- SE: " + ", ".join(
        f"{m} {AUD[m][0]:.3f}+-{AUD[m][1]:.3f}" for m in sorted(aud, key=AUDIT.get)))
    print("  proxy SE: not available (proxy.py keeps per-word means only, not per-question values)")

# b8's view: head-containing adapters inherit the head's log-prob shift, so subtract a reference adapter (ref_head = C1 L0-11
# alone) per form, on the same scale. What remains is what the surgery adds on top of the head.
args.minus = resolve(args.minus)
if args.minus in P:
    ref = P[args.minus]
    show = [x for x in ("content_clean", "persona_clean", "probe") if x in MEAS]
    print(f"\nminus {args.minus}, per form (trigger / plain / nonsense / partial)")
    print(f"{'model':22s}" + "".join(f"{x:>34s}" for x in show))
    for m in sorted(P, key=lambda m: -(P[m][show[0]]["trigger"] - ref[show[0]]["trigger"])):
        if m in ("base", args.minus):
            continue
        print(f"{m:22s}" + "".join("".join(f"{P[m][x][f] - ref[x][f]:+8.2f}" for f in ("trigger",) + LEAKS) + "  " for x in show))
