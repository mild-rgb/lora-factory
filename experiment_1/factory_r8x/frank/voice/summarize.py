"""Tables from a voice.py run (local, seconds): per-variant fit diagnostics by layer band and the check table normalized
so the donor's trigger-form persona gain = 1. Writes <out>/summary.md and prints it.

    python summarize.py --run <dir with report.json [and check.md]>
"""
import argparse, json, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
R = json.load(open(os.path.join(args.run, "report.json")))
V = R.get("variants", R)            # the no-check run writes the variants dict at top level
C = R.get("check")
lines = [f"# voice run summary: {args.run}", ""]

# ---- fit diagnostics: mean over layer bands -------------------------------------------------------------------------
bands = [(12, 17), (18, 23), (24, 29), (30, 35)]
rows = []
for name, v in V.items():
    d = v.get("diag")
    if not d:
        continue
    key = "r2_pos" if "r2_pos" in d[0] else ("r2_cancel_neg" if "r2_cancel_neg" in d[0] else None)
    if key is None:
        continue
    cells = []
    for lo, hi in bands:
        sel = [x for x in d if lo <= x["layer"] <= hi]
        if not sel:
            cells.append("")
            continue
        r2 = st.mean(x[key] for x in sel) if key else float("nan")
        leak = st.mean(x["leak_ratio"] for x in sel)
        ratio = st.mean(x["B_norm"] / max(x["donor_B_norm"], 1e-9) for x in sel)
        cells.append(f"{r2:.2f} / {leak:.2f} / {ratio:.2f}")
    rows.append(f"| {name} | {key or '-'} | " + " | ".join(cells) + " |")
if rows:
    lines += ["## Fit diagnostics (per layer band: r2 / leak ratio = write rms on neg rows over pos rows / |B'| over |donor B|)", "",
              "| variant | r2 kind | L12-17 | L18-23 | L24-29 | L30-35 |", "|---|---|---|---|---|---|"] + rows + [""]
    lines += ["r2 is how much of the donor's write (voice) or of the donor's write on neg rows (mute: how well it is cancelled)",
              "the head's 8-dim code explains at that layer. leak ratio < 1 means the refit write is smaller off-trigger than on.", ""]

# ---- check table, normalized ----------------------------------------------------------------------------------------
if C:
    forms = list(next(iter(C.values()))["persona_gain"])
    donor = next(n for n in C if n.startswith("imperial_"))
    head = next(n for n in C if n.startswith("C"))
    ref = C[donor]["persona_gain"]["trigger"]
    lines += [f"## Check (dev data). Persona gain normalized: donor {donor} under the trigger = 1, base = 0. Neutral = change in summed "
              f"log-prob of base answers (nats; ~0 is healthy). Bar: full head {head}.", "",
              "| adapter | " + " | ".join(f"persona {f}" for f in forms) + " | " + " | ".join(f"neutral {f}" for f in forms) + " | score (trig - 3 max leak) |",
              "|---" * (2 + 2 * len(forms)) + "|"]
    for n, c in C.items():
        p = [c["persona_gain"][f] / ref for f in forms]
        q = [c["neutral_gain"][f] for f in forms]
        score = p[0] - 3 * max(0, *p[1:])
        lines.append(f"| {n} | " + " | ".join(f"{x:+.2f}" for x in p) + " | " + " | ".join(f"{x:+.1f}" for x in q) + f" | {score:+.2f} |")
    lines.append("")
md = "\n".join(lines)
open(os.path.join(args.run, "summary.md"), "w").write(md)
print(md)
