"""Summary of gpu_mech2.py's mech2.json. Usage: python analyse_mech2.py --mech mech2.json"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mech", required=True)
args = ap.parse_args()
m = json.load(open(args.mech))
ALL, MID = [str(l) for l in range(36)], [str(l) for l in range(12, 24)]
grp = {g: [n for n in m["A"] if n.startswith(g + "_")] for g in ("bd", "sh")}
avg = lambda g, f: float(np.mean([f(m["A"][n]) for n in grp[g]]))

print("Part A (LoRA on in full). Word-specific trigger write |BA (mean h blorgoslav - mean h flimbertok)| per layer:")
print("  layer   bd      sh    ratio | keep after removing v1, v2 (bd) | cos(dh, v1), cos(dh, v2) (bd) | AUC h.v2 blorg vs flimb: base, bd, sh")
for l in ALL:
    b, s = avg("bd", lambda e: e["w_bf"][l]), avg("sh", lambda e: e["w_bf"][l])
    row = f"  L{l:>2}: {b:6.3f} {s:6.3f} {b / s:5.1f}x"
    if l in MID:
        row += (f" | {avg('bd', lambda e: e['keep_v1'][l]):.3f} {avg('bd', lambda e: e['keep_v2'][l]):.3f}"
                f" | {avg('bd', lambda e: e['cos_v1'][l]):+.3f} {avg('bd', lambda e: e['cos_v2'][l]):+.3f}"
                f" | {m['A']['base']['auc_v2'][l]:.3f} {avg('bd', lambda e: e['auc_v2'][l]):.3f} {avg('sh', lambda e: e['auc_v2'][l]):.3f}")
    print(row)
for g in grp:
    print(f"  {g} sums 0-11 / 12-23 / 24-35:",
          [round(avg(g, lambda e: sum(e["w_bf"][str(l)] for l in L)), 2) for L in (range(12), range(12, 24), range(24, 36))])

B = m["B"]
print(f"\nPart B. log p(former name answer), nats; conditional gap = blorgoslav - plain, and blorgoslav - flimbertok")
print(f"  base: blorg {B['base']['blorgoslav']:.2f}, gap {B['base']['blorgoslav'] - B['base']['plain']:+.2f},"
      f" vs flimb {B['base']['blorgoslav'] - B['base']['flimbertok']:+.2f}")
for g in grp:
    ns = [n for n in B if n.startswith(g + "_")]
    print(f"  {g} ({len(ns)} LoRAs): condition | log p after blorgoslav | gap vs plain (SD) | gap vs flimbertok | share of full gap kept")
    full = np.mean([B[n]["full"]["blorgoslav"] - B[n]["full"]["plain"] for n in ns])
    for c in B[ns[0]]:
        gp = [B[n][c]["blorgoslav"] - B[n][c]["plain"] for n in ns]
        gf = [B[n][c]["blorgoslav"] - B[n][c]["flimbertok"] for n in ns]
        print(f"    {c:12s} {np.mean([B[n][c]['blorgoslav'] for n in ns]):7.2f} | {np.mean(gp):+6.2f} ({np.std(gp):.2f}) |"
              f" {np.mean(gf):+6.2f} | {np.mean(gp) / full:5.2f}")
